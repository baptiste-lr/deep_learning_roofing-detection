#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os, sys, json, csv, shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import argparse
from copy import deepcopy 
import math 

import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from torch.utils.data import DataLoader, ConcatDataset, Dataset

# IMPORTS POUR L'ÉVALUATION
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, jaccard_score, accuracy_score
from matplotlib.colors import Normalize

# =================================================================
# 🚩 CONFIGURATION & CHEMINS (MAJ)
# =================================================================

PROJECT_ROOT = Path("/home/baptistedlb/Documents/Stage_IRD/Baptiste_Deep/Deep_learning_codes/codes")
SCRIPTS_DIR  = PROJECT_ROOT 

# 🚨 CHEMIN DE BASE DES DONNÉES (Non équilibré)
DATA_ROOT_MULTI_SOURCE = PROJECT_ROOT / "dataset_source_split_unbalanced"

RUNS_BASE_DIR = PROJECT_ROOT / "Outputs/runs"
PREDICT_HIGH_ROOF_TILES_DIR = PROJECT_ROOT / "Outputs/predict_high_roof_tiles" 
STATS_PATH = PROJECT_ROOT / "preparation_data" / "mean_std.json" 
BASE_DATA_DIR = Path("/home/baptistedlb/Documents/Stage_IRD/Baptiste_Deep/Deep_learning_codes/data/data_sat")
REFERENCE_IMAGE_PATHS = [
    BASE_DATA_DIR / "baptiste_extraction_rognée_validated.tif",
    BASE_DATA_DIR / "marc_extraction_rognée_validated.tif"
]

METRICS_OUTPUT_DIR = PREDICT_HIGH_ROOF_TILES_DIR / "inference_metrics" 

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

# Import des modules locaux (Assurez-vous que ces fichiers sont là!)
try:
    from model import get_unetpp_model
    from dataset import TileDataset
    from utils import (
        normalize_image, save_mask_as_shapefile, plot_losses, save_checkpoint,
        save_binary_mask_geotiff, save_visual_mask_geotiff
    )
    from postproc_infer import (
        predict_simple, predict_tta, auto_threshold,
        postprocess_mask
    )
except ImportError as e:
    print(f"Erreur d'importation d'un module local: {e}.")
    
# --- DÉFINITION DES CHEMINS DES DONNÉES ---
# NOTE : Ces chemins utilisent DATA_ROOT_MULTI_SOURCE mis à jour
DATA_CONFIG = {
    "images_baptiste": {
        "train_img": DATA_ROOT_MULTI_SOURCE / "images_baptiste" / "train" / "images",
        "train_mask": DATA_ROOT_MULTI_SOURCE / "images_baptiste" / "train" / "masks",
        "val_img": DATA_ROOT_MULTI_SOURCE / "images_baptiste" / "val" / "images",
        "val_mask": DATA_ROOT_MULTI_SOURCE / "images_baptiste" / "val" / "masks",
    },
    "images_marc": {
        "train_img": DATA_ROOT_MULTI_SOURCE / "images_marc" / "train" / "images",
        "train_mask": DATA_ROOT_MULTI_SOURCE / "images_marc" / "train" / "masks",
        "val_img": DATA_ROOT_MULTI_SOURCE / "images_marc" / "val" / "images",
        "val_mask": DATA_ROOT_MULTI_SOURCE / "images_marc" / "val" / "masks",
    }
}

# -----------------------------------
# Configuration et Arguments CLI (epochs=60 par défaut)
# -----------------------------------

def parse_args() -> argparse.Namespace: 
    ap = argparse.ArgumentParser(
        description="Script d'Inférence: Charge un modèle existant et effectue une prédiction ciblée."
    )
    # --- Arguments pour la RECHERCHE DU MODÈLE  ---
    ap.add_argument("--epochs", type=int, default=60, help="Nombre d'epochs utilisé pour nommer le modèle.")
    ap.add_argument("--lr", type=float, default=3e-5, help="Learning rate utilisé pour nommer le modèle.")
    ap.add_argument("--backbone", type=str, default="resnet34", help="Backbone du modèle.")
    ap.add_argument("--pos-weight", type=float, default=5.0, help="Poids de la classe positive utilisé pour nommer le modèle.")
    # --- INFERENCE CIBLÉE PARAMS ---
    ap.add_argument("--outdir", type=str, default=str(PREDICT_HIGH_ROOF_TILES_DIR), help="Dossier de sortie pour la prédiction des tuiles filtrées.")
    ap.add_argument("--tta", action="store_true", help="Activer le Test-Time Augmentation (TTA).")
    ap.add_argument("--thr-mode", type=str, default="fixed",
                     choices=["fixed", "global-otsu"],
                     help="Stratégie de seuillage (fixed ou global-otsu).")
    ap.add_argument("--thr-fixed", type=float, default=0.50, help="Seuil binaire pour la prédiction des tuiles.")
    # --- POST-PROCESSING & EXPORT PARAMS ---
    ap.add_argument("--postprocess-tile", action=argparse.BooleanOptionalAction, default=True, help="Appliquer le post-traitement morphologique sur chaque tuile prédite.")
    ap.add_argument("--close-m", type=float, default=2.5, help="Rayon de fermeture morphologique (m).")
    ap.add_argument("--open-m", type=float, default=0.5, help="Rayon d'ouverture morphologique (m).")
    ap.add_argument("--min-obj-m2", type=float, default=5.0, help="Taille min d'objet à conserver (m²).")
    ap.add_argument("--export-vector", action=argparse.BooleanOptionalAction, default=True, help="Exporter le masque binaire de chaque tuile en shapefile (ACTIVÉ par défaut).")
    
    return ap.parse_args()


# -----------------------------------
# Helper: Recherche de tuile 
# -----------------------------------

def get_specific_tile_path(image_dir: Path, mask_dir: Path, tile_name: str) -> Optional[Tuple[Path, Path]]:
    """
    Recherche le chemin de l'image et du masque pour une tuile spécifique 
    dans les répertoires d'images et de masques donnés.
    """
    img_path = image_dir / f"{tile_name}.tif"
    mask_path = mask_dir / f"{tile_name}.tif"

    if img_path.exists() and mask_path.exists():
        return img_path, mask_path
            
    return None

# -----------------------------------
# FONCTION MISE À JOUR : ÉVALUATION
# -----------------------------------
# (Même logique que le script précédent, utilisant les données filtrées)

def evaluate_and_plot_metrics(
    y_true_all: np.ndarray, 
    y_proba_all: np.ndarray, 
    individual_data: List[Tuple[str, np.ndarray, np.ndarray]], 
    args: argparse.Namespace, 
    model_tag: str
) -> None:
    """
    Calcule, affiche et enregistre toutes les métriques et le pourcentage de pixels Toiture.
    """
    print("\n\n=== 📈 DÉMARRAGE DE L'ÉVALUATION COMPLÈTE & CORRIGÉE (CM Globale en %) ===")
    
    metrics_out_dir = METRICS_OUTPUT_DIR / model_tag
    metrics_out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Sortie des métriques Globales/ROC vers: {metrics_out_dir}")

    individual_out_root_dir = Path(args.outdir)

    fixed_thr = args.thr_fixed
    
    # CORRECTION CRITIQUE: Exclusion stricte des valeurs hors 0 et 1 (y compris NoData)
    valid_pixels_mask_global = (y_true_all == 0) | (y_true_all == 1)
    
    y_true_all_filtered = y_true_all[valid_pixels_mask_global]
    y_proba_all_filtered = y_proba_all[valid_pixels_mask_global]
    y_pred_all_filtered = (y_proba_all_filtered > fixed_thr).astype(int)
    
    if len(y_true_all_filtered) == 0:
        print("❌ AUCUN PIXEL VALIDE (0 ou 1) pour l'évaluation globale n'a été trouvé. Évaluation annulée.")
        return
    
    # =================================================================
    # 🔑 CALCUL ET AFFICHAGE DU POURCENTAGE DE PIXELS TOITURE (CLASSE 1)
    # =================================================================
    
    total_valid_pixels = len(y_true_all_filtered)
    pixels_toiture = np.sum(y_true_all_filtered == 1)
    pourcentage_toiture = (pixels_toiture / total_valid_pixels) * 100
    
    print("\n--- ⚖️ DÉSÉQUILIBRE DES CLASSES (20 Imagettes) ---")
    print(f"Pixels valides totaux comparés : {total_valid_pixels:,}")
    print(f"Pixels Toiture (Classe 1, Réel) : {pixels_toiture:,}")
    print(f"Pourcentage de pixels Toiture : **{pourcentage_toiture:.4f} %**")
    print(f"Pourcentage de pixels Non-Toiture: {(100 - pourcentage_toiture):.4f} %")
    print("--------------------------------------------------")
    
    # =================================================================
    # 1. MATRICE DE CONFUSION GLOBALE (Moyenne - Normalisée en %) & CR
    # =================================================================
    
    try:
        cm_global_normalized = confusion_matrix(y_true_all_filtered, y_pred_all_filtered, normalize='true') * 100 
        global_iou = jaccard_score(y_true_all_filtered, y_pred_all_filtered)
        global_accuracy = accuracy_score(y_true_all_filtered, y_pred_all_filtered) 
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_global_normalized, display_labels=['Non-Toit', 'Toit'])
        
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(cmap=plt.cm.Reds, ax=ax, values_format='.2f') 
        ax.set_title(f"Matrice de Confusion AGRÉGÉE (Seuil={fixed_thr:.2f}, IoU={global_iou:.4f}, Acc={global_accuracy:.4f})\nNormalisée par classe réelle (%)", fontweight='bold')
        
        cm_path = metrics_out_dir / f"confusion_matrix_thr{fixed_thr:.2f}_GLOBAL_PCT.png" 
        plt.savefig(cm_path)
        plt.close()
        print(f"✅ Matrice de Confusion **Globale/Moyenne** (Normalisée en %) générée: {cm_path.name}")
        
        # Rapport de classification global 
        report = classification_report(y_true_all_filtered, y_pred_all_filtered, target_names=['Non-Toit', 'Toit'], digits=4, zero_division=0)
        report_path = metrics_out_dir / f"classification_report_thr{fixed_thr:.2f}_GLOBAL.txt"
        
        accuracy_line = f"Exactitude Globale (Accuracy): {global_accuracy:.4f}\n"

        with open(report_path, 'w') as f:
            f.write(f"Modèle: {model_tag}\n")
            f.write(f"Seuil utilisé: {fixed_thr:.2f}\n")
            f.write(f"Pourcentage Toiture Réel: {pourcentage_toiture:.4f} %\n")
            f.write(accuracy_line) 
            f.write(report)
        print(f"✅ Rapport de classification global exporté: {report_path.name}")
        
        print("\n--- 📊 RAPPPORT DE CLASSIFICATION GLOBAL ---")
        print(accuracy_line) 
        print(report)
        print("------------------------------------------")

    except Exception as e:
        print(f"❌ Erreur lors du calcul/tracé de la Matrice de Confusion Globale: {e}")

    # =================================================================
    # 2. COURBE ROC UNIQUE (Globale + Individuelles)
    # =================================================================
    
    plt.figure(figsize=(12, 12)) 
    try:
        # --- 2.1. Courbe Globale ---
        fpr_global, tpr_global, _ = roc_curve(y_true_all_filtered, y_proba_all_filtered)
        auc_score_global = roc_auc_score(y_true_all_filtered, y_proba_all_filtered)
        
        plt.plot(fpr_global, tpr_global, color='red', lw=4, 
                 label=f'GLOBAL MOYENNE (AUC = {auc_score_global:.4f})', 
                 zorder=10) 

        # --- 2.2. Courbes Individuelles ---
        n_tiles = len(individual_data)
        colors = cm.get_cmap('Spectral', n_tiles) 
        
        individual_labels = []
        individual_lines = []
        
        for i, (tile_name, y_true, y_proba) in enumerate(individual_data):
            valid_pixels_mask = (y_true == 0) | (y_true == 1)
            y_true_roc = y_true[valid_pixels_mask]
            y_proba_roc = y_proba[valid_pixels_mask]
            
            if len(np.unique(y_true_roc)) < 2:
                continue 
                
            fpr_indiv, tpr_indiv, _ = roc_curve(y_true_roc, y_proba_roc)
            auc_score_indiv = roc_auc_score(y_true_roc, y_proba_roc)
            
            line, = plt.plot(fpr_indiv, tpr_indiv, color=colors(i), lw=1.5, alpha=0.7, zorder=5)
            
            individual_lines.append(line)
            individual_labels.append(f'{tile_name} (AUC = {auc_score_indiv:.4f})')

        # --- 2.3. Finalisation du Plot ROC ---
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire (AUC = 0.50)', zorder=1)
        
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de Faux Positifs (FPR)'); plt.ylabel('Taux de Vrais Positifs (TPR / Rappel)')
        plt.title(f'Courbes ROC (Globale et Individuelles) - Modèle: {model_tag}', fontsize=14)
        
        global_line = plt.gca().lines[0]
        random_line = plt.gca().lines[-1] 
        
        all_lines = [global_line] + individual_lines + [random_line]
        all_labels = [global_line.get_label()] + individual_labels + [random_line.get_label()]
        
        plt.legend(all_lines, all_labels, loc='lower right', fontsize='small', title="Tuiles Individuelles et Moyenne")
        
        roc_path = metrics_out_dir / "aggregated_individual_ROC_curves_detailed.png"
        plt.savefig(roc_path)
        plt.close()
        print(f"✅ Courbes ROC (Globale + Individuelles) AVEC LÉGENDE DÉTAILLÉE générées: {roc_path.name}")
    except Exception as e:
        print(f"❌ Erreur lors du calcul/tracé de la Courbe ROC Unique détaillée: {e}")
        
    # =================================================================
    # 3. MATRICES DE CONFUSION INDIVIDUELLES (en Pourcentage)
    # =================================================================
    
    print("\n  -> Génération et Export des Matrices de Confusion Individuelles (Normalisées en %) par dossier de tuile...")
    
    for tile_name, y_true_raw, y_proba_raw in individual_data:
        try:
            tile_output_dir = individual_out_root_dir / tile_name
            if not tile_output_dir.exists():
                continue
            
            valid_pixels_mask = (y_true_raw == 0) | (y_true_raw == 1)
            y_true = y_true_raw[valid_pixels_mask]
            
            if len(np.unique(y_true)) < 2:
                continue
                
            y_proba = y_proba_raw[valid_pixels_mask]
            y_pred = (y_proba > fixed_thr).astype(int)
            
            cm_normalized = confusion_matrix(y_true, y_pred, normalize='true') * 100 
            
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=['Non-Toit', 'Toit'])
            
            fig, ax = plt.subplots(figsize=(6, 6))
            disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='.2f') 
            ax.set_title(f"Conf. Matrix ({tile_name}) - Seuil={fixed_thr:.2f}\nNormalisée par classe réelle (%)", fontsize=10)
            
            cm_path = tile_output_dir / f"{tile_name}_CM_PCT_thr{fixed_thr:.2f}.png"
            plt.savefig(cm_path)
            plt.close(fig)
            
        except Exception as e:
            print(f"  -> ❌ Erreur Matrice de Confusion pour {tile_name}: {e}")
            
    print(f"✅ Matrices de Confusion (%) individuelles générées et placées dans leurs dossiers respectifs.")
    
    # =================================================================
    # 4. AFFICHAGE DES STATISTIQUES INDIVIDUELLES DANS LE TERMINAL
    # =================================================================
    print("\n\n=== 📊 STATISTIQUES DÉTAILLÉES PAR IMAGETTE (Seuil={:.2f}) ===".format(fixed_thr))
    
    # Header pour le tableau
    print("{:<30} | {:>6} | {:>6} | {:>6} | {:>6}".format("Tuile", "IoU", "Rappel", "Précision", "F1-Score"))
    print("-" * 59)
    
    for tile_name, y_true_raw, y_proba_raw in individual_data:
        try:
            valid_pixels_mask = (y_true_raw == 0) | (y_true_raw == 1)
            y_true = y_true_raw[valid_pixels_mask]
            y_proba = y_proba_raw[valid_pixels_mask]
            
            if len(y_true) == 0 or len(np.unique(y_true)) < 2:
                continue 
            
            y_pred = (y_proba > fixed_thr).astype(int)
            
            # Calcul des métriques
            cm_raw = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            tn, fp, fn, tp = cm_raw
            
            denom_iou = (tp + fp + fn)
            iou = tp / denom_iou if denom_iou > 0 else 0.0
            
            denom_recall = (tp + fn)
            recall = tp / denom_recall if denom_recall > 0 else 0.0 
            
            denom_precision = (tp + fp)
            precision = tp / denom_precision if denom_precision > 0 else 0.0
            
            denom_f1 = (precision + recall)
            f1_score_val = 2 * (precision * recall) / denom_f1 if denom_f1 > 0 else 0.0
            
            print("{:<30} | {:>6.4f} | {:>6.4f} | {:>6.4f} | {:>6.4f}".format(
                tile_name, iou, recall, precision, f1_score_val
            ))
            
        except Exception as e:
            print(f"  -> ⚠️ Erreur lors du calcul des stats pour {tile_name}: {e}")
            pass 
            
    print("-" * 59)
    
    print("\n=== ÉVALUATION COMPLÈTE TERMINÉE ===")
    
# -----------------------------------
# Inference Ciblée sur les tuiles Filtrées 
# -----------------------------------

def inference_on_filtered_tiles(best_model_path: Path, filtered_ds: TileDataset, device: torch.device, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, np.ndarray, np.ndarray]], str]:
    
    print("\n=== Démarrage de l'Inférence Ciblée sur les Tuiles Filtrées ===")
    
    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)
    
    # Tenter de lire le manifest pour récupérer les paramètres du modèle (canaux/backbone)
    manifest_path = best_model_path.parent / "manifest.json"
    in_channels = filtered_ds.n_channels
    backbone = args.backbone
    model_tag = best_model_path.stem.replace("best_model_", "")
    
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            in_channels = manifest.get("in_channels", in_channels)
            backbone = manifest.get("backbone", backbone)
            best_iou = manifest.get("best_val_iou", "N/A")
            print(f"ℹ️ Paramètres du modèle chargés via manifest.json: in_channels={in_channels}, backbone={backbone}, IoU={best_iou}")
        except Exception as e:
            print(f"⚠️ Erreur lors de la lecture du manifest: {e}. Utilisation des paramètres par défaut/CLI.")
    
    # Re-charger le meilleur modèle
    model = get_unetpp_model(in_channels=in_channels, num_classes=1, backbone=backbone).to(device).eval()
    try:
        if not best_model_path.exists():
            print(f"❌ Erreur critique: Le modèle {best_model_path} est introuvable. Entraînez d'abord.")
            return np.array([]), np.array([]), [], model_tag

        state = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state)
        print(f"✅ Modèle chargé: {best_model_path.name}")
    except Exception as e:
        print(f"❌ Erreur critique lors du chargement du modèle {best_model_path}: {e}")
        return np.array([]), np.array([]), [], model_tag

    # S'assurer que le batch_size est 1 pour l'inférence par tuile
    infer_loader = DataLoader(filtered_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    
    total_processed = 0
    low_proba_count = 0 
    
    all_true_pixels_global: List[np.ndarray] = []
    all_proba_pixels_global: List[np.ndarray] = []
    individual_tile_data: List[Tuple[str, np.ndarray, np.ndarray]] = [] 
    
    with torch.no_grad():
        for idx, (img_tensor, _) in enumerate(infer_loader):
            total_processed += 1
            
            img_path = filtered_ds.images[idx]
            mask_path = filtered_ds.masks[idx] 
            
            base_name = img_path.stem 
            
            # 1. Charger le masque de référence (y_true)
            try:
                 with rasterio.open(mask_path) as src_mask:
                     true_mask = src_mask.read(1).astype(np.uint8)
                     transform = src_mask.transform
                     crs = src_mask.crs
            except Exception as e:
                print(f"❌ Erreur lors de la lecture des métadonnées GeoTIFF de {mask_path.name}: {e}. Skip Tile.")
                continue
                
            # **Définir et créer le dossier de sortie par tuile**
            tile_output_dir = out_dir / base_name
            tile_output_dir.mkdir(parents=True, exist_ok=True) 

            # --- Prédiction & Diagnostic Logit/Proba ---
            img_tensor = img_tensor.to(device, non_blocking=True)
            
            if not args.tta:
                output_logits: torch.Tensor = model(img_tensor).squeeze().detach().cpu()
                
                if output_logits.ndim == 3 and output_logits.shape[0] == 1:
                     output_logits = output_logits[0, :, :] 
                
                logit_min = output_logits.min().item()
                logit_max = output_logits.max().item()
                
                proba_tile = torch.sigmoid(output_logits).numpy().astype(np.float32)
                
                print(f"📊 DIAGNOSTIC Tuile {total_processed}/{len(filtered_ds)} ({img_path.name}):")
                print(f"    -> Logits Min/Max: {logit_min:.4f} / {logit_max:.4f}")

            else:
                proba_tile = predict_tta(model, img_tensor, device)
                logit_min = np.nan
                logit_max = np.nan
                
                print(f"📊 DIAGNOSTIC Tuile {total_processed}/{len(filtered_ds)} ({img_path.name}): (TTA ACTIVÉ)")


            max_proba = np.max(proba_tile)
            
            print(f"    -> Proba Max: {max_proba:.8f} (Seuil: {args.thr_fixed:.2f})")

            if max_proba < args.thr_fixed: 
                low_proba_count += 1
                
            # --- AGGRÉGATION DES PIXELS ---
            y_true_flat = true_mask.flatten()
            y_proba_flat = proba_tile.flatten()
            
            all_true_pixels_global.append(y_true_flat)
            all_proba_pixels_global.append(y_proba_flat)
            
            individual_tile_data.append((base_name, y_true_flat, y_proba_flat))
                
            # --- Binarisation (pour l'export de la tuile) ---
            thr = args.thr_fixed
            bin_final = (proba_tile > thr).astype(np.uint8)

            # --- Post-traitement Morpho (en mètres) ---
            if args.postprocess_tile:
                 try:                     
                     bin_final = postprocess_mask(
                         bin_final, transform,
                         close_radius_m=args.close_m, open_radius_m=args.open_m,
                         min_obj_area_m2=args.min_obj_m2
                     )
                 except Exception as e:
                     print(f"⚠️ Erreur lors du post-traitement de {img_path.name}: {e}. Skip PP.")
            
            # --- Export Raster Binaire (.tif), Visuel (.tif), et Vectoriel (.shp) ---
            
            file_tag = f"pred_thr{thr:.2f}"
            if args.tta: file_tag += "_tta"
            if args.postprocess_tile: file_tag += "_pp"
            
            out_name_bin = f"{base_name}_{file_tag}_BIN.tif"
            out_bin_path = tile_output_dir / out_name_bin 
            save_binary_mask_geotiff(bin_final, transform, crs, str(out_bin_path))
            print(f"  -> GeoTIFF Binaire exporté: {out_name_bin}")
            
            out_name_vis = f"{base_name}_{file_tag}_VIS.tif"
            out_vis_path = tile_output_dir / out_name_vis 
            save_visual_mask_geotiff(bin_final, transform, crs, str(out_vis_path))
            print(f"  -> GeoTIFF Visuel exporté: {out_name_vis}")
            
            if args.export_vector:
                shp_name = out_bin_path.with_suffix(".shp").name.replace("_BIN.tif", ".shp") 
                shp_path = tile_output_dir / shp_name 
                save_mask_as_shapefile(bin_final, transform, crs, str(shp_path))
                print(f"  -> Shapefile exporté: {shp_path.name}")
                
            # --- COPIE DES FICHIERS DE RÉFÉRENCE ---
            try:
                shutil.copy2(img_path, tile_output_dir / f"{base_name}_REF_IMAGE.tif")
                shutil.copy2(mask_path, tile_output_dir / f"{base_name}_REF_MASK.tif")
                print(f"  -> Fichiers de référence (Image & Mask) copiés.")
            except Exception as e:
                print(f"⚠️ Erreur lors de la copie des fichiers de référence pour {img_path.name}: {e}. Skip Copy.")


    if low_proba_count > 0:
        print(f"\n🚨 ALERTE : {low_proba_count} tuiles ({low_proba_count/len(filtered_ds)*100:.1f}%) ont produit une probabilité maximale < {args.thr_fixed:.2f} (seuil d'export).")

    print(f"\n✅ Inférence ciblée de {total_processed} tuiles terminée (seuil={args.thr_fixed:.2f}).")
    
    y_true_final = np.concatenate(all_true_pixels_global)
    y_proba_final = np.concatenate(all_proba_pixels_global)
    
    return y_true_final, y_proba_final, individual_tile_data, model_tag

# -----------------------------------
# ⚙️ Main (Logique MAJ)
# -----------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔄 Appareil utilisé : {device}")
    
    # ... (Initialisation des datasets pour les stats) ...
    print("💡 Initialisation des datasets (chargement des stats)...")
    def setup_dataset(source_key: str, split: str) -> TileDataset:
        img_dir = DATA_CONFIG[source_key][f"{split}_img"]
        mask_dir = DATA_CONFIG[source_key][f"{split}_mask"]
        return TileDataset(img_dir, mask_dir, STATS_PATH, REFERENCE_IMAGE_PATHS)

    try:
        val_ds_baptiste_full = setup_dataset("images_baptiste", "val")
        in_channels: int = val_ds_baptiste_full.n_channels
    except Exception as e:
        print(f"❌ Erreur lors du chargement des datasets: {e}")
        return

    print(f"📖 Datasets de validation chargés (pour chemins/stats). Canaux: {in_channels}")
    
    # --- 2. Construction du chemin du meilleur modèle (MAJ) ---
    
    BCE_W = 0.1; DICE_W = 0.9; WEIGHT_DECAY = 5e-3 
    
    # Tag pour le modèle NON-ÉQUILIBRÉ
    tag_base = f"{args.backbone}_wbd_b{BCE_W:.1f}d{DICE_W:.1f}_w{args.pos_weight:.2f}_UNBALANCED_e{args.epochs}"
    best_model_path = RUNS_BASE_DIR / f"{tag_base}_runs" / f"best_model_{tag_base}.pth"
    
    if not best_model_path.exists():
         print(f"❌ Modèle introuvable au chemin : {best_model_path}. Vérifiez les arguments CLI et l'entraînement précédent.")
         return
    
    print(f"✅ Modèle à charger identifié: {best_model_path.name}")

    # --- 3. Inférence Ciblée (SÉLECTION - Nouvelle Liste Ciblée) ---
    
    all_filtered_images = []
    all_filtered_masks = []
    
    # 🚨 NOUVELLE LISTE DE 20 TUILES (Baptiste/Val)
    TILE_BAPTISTE_VAL_LIST = [
        "tile_00007_aug4", "tile_00080_aug4", "tile_00556_aug2", "tile_00992_aug0", 
        "tile_01146_aug0", "tile_01233_aug0", "tile_01310_aug0", "tile_01495_aug0", 
        "tile_01543_aug4", "tile_01564_aug5", "tile_01568_aug5", "tile_01618_aug1", 
        "tile_01715_aug1", "tile_01906_aug1", "tile_02205_aug0", "tile_02247_aug0", 
        "tile_02450_aug0", "tile_02743_aug0", "tile_02847_aug5", "tile_03059_aug4"
    ] 

    print(f"\n=== SÉLECTION CIBLÉE : {len(TILE_BAPTISTE_VAL_LIST)} tuiles spécifiées de la source Baptiste (VAL) ===")

    # Récupération des chemins pour les tuiles BAPTISTE (uniquement)
    baptiste_img_dir = DATA_CONFIG["images_baptiste"]["val_img"]
    baptiste_mask_dir = DATA_CONFIG["images_baptiste"]["val_mask"]
    
    for tile_name_stem in TILE_BAPTISTE_VAL_LIST:
        # Utilisation des répertoires images ET masks de Baptiste/Val
        tile_pair = get_specific_tile_path(baptiste_img_dir, baptiste_mask_dir, tile_name_stem)
        
        if tile_pair:
            all_filtered_images.append(tile_pair[0])
            all_filtered_masks.append(tile_pair[1])
        else:
            # ⚠️ Cette erreur est critique si les tuiles ne sont pas trouvées au bon endroit.
            print(f"  -> ❌ ERREUR : La tuile Baptiste '{tile_name_stem}' n'a pas été trouvée dans le dossier de validation. Ignorée.")
            
    
    print(f"\n📏 SÉLECTION CIBLÉE FINALE : {len(all_filtered_images)} tuiles retenues pour l'inférence.")

    if not all_filtered_images:
        print("\n⚠️ Aucune tuile n'a pu être sélectionnée. Inférence ciblée annulée.")
        return

    # Création du Dataset FILTRÉ pour l'inférence
    infer_ds_filtered: TileDataset = deepcopy(val_ds_baptiste_full)
    infer_ds_filtered.images = all_filtered_images
    infer_ds_filtered.masks = all_filtered_masks
    
    print(f"✅ Création d'une COPIE du Dataset de validation pour l'inférence ciblée.")
    
    # Exécution de l'inférence
    y_true_all, y_proba_all, individual_tile_data, model_tag = inference_on_filtered_tiles(best_model_path, infer_ds_filtered, device, args)
    
    # --- 4. ÉVALUATION GLOBALE & INDIVIDUELLE ---
    if len(y_true_all) > 0:
        evaluate_and_plot_metrics(y_true_all, y_proba_all, individual_tile_data, args, model_tag)
    else:
        print("\n❌ Évaluation des métriques impossible: Aucune donnée d'inférence n'a été collectée.")

if __name__ == "__main__":
    main()