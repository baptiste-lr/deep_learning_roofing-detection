#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os, sys, json, csv, shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Union
import argparse
import math 
import traceback 

import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling 

# NOUVEAUX IMPORTS POUR L'ÉVALUATION ET LE TRAITEMENT
try:
    from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, jaccard_score, accuracy_score
    import matplotlib.pyplot as plt
    from scipy.ndimage import median_filter 
except ImportError:
    print("❌ ERREUR: Assurez-vous d'installer scikit-learn, matplotlib et scipy: pip install scikit-learn matplotlib scipy")
    sys.exit(1)


# =================================================================
# 🚩 CONFIGURATION & CHEMINS
# =================================================================

PROJECT_ROOT = Path("/home/baptistedlb/Documents/Stage_IRD/Baptiste_Deep/Deep_learning_codes/codes")
SCRIPTS_DIR  = PROJECT_ROOT 
RUNS_BASE_DIR = PROJECT_ROOT / "Outputs/runs"
PREDICT_HIGH_ROOF_TILES_DIR = PROJECT_ROOT / "Outputs/predict_high_roof_tiles" 
STATS_PATH = PROJECT_ROOT / "preparation_data" / "mean_std.json" 
# Répertoire des images satellites brutes/validées
BASE_DATA_DIR = Path("/home/baptistedlb/Documents/Stage_IRD/Baptiste_Deep/Deep_learning_codes/data/data_sat")

# 🎯 NOUVEAU CHEMIN POUR LES MASQUES DE VÉRITÉ TERRAIN (GT)
GT_BASE_DIR = Path("/home/baptistedlb/Documents/Stage_IRD/Baptiste_Deep/Deep_learning_codes/data/images_binaire")

# 🎯 LISTE DES IMAGES CIBLES À PRÉDIRE 
IMAGES_TO_PREDICT = [
    "baptiste_extraction_rognée_validated.tif",
    "marc_extraction_rognée_validated.tif",
]

# 🎯 ADAPTATION 1 : Nombre de canaux attendus par le modèle (11)
IN_CHANNELS_MODEL = 11 

# 🎯 ADAPTATION 2 : Indices des bandes à lire (1-based pour rasterio).
# On lit les bandes de 4 à 14 (11 bandes au total, SANS RVB).
BAND_INDICES_TO_READ = list(range(4, 15)) 
# Vérification de sécurité: la liste devrait contenir 11 indices
if len(BAND_INDICES_TO_READ) != IN_CHANNELS_MODEL:
    print(f"❌ ERREUR DE CONFIGURATION: Indices de bandes à lire ({len(BAND_INDICES_TO_READ)}) ne correspondent pas aux canaux du modèle ({IN_CHANNELS_MODEL}). Vérifiez BAND_INDICES_TO_READ.")
    sys.exit(1)


# 💡 FONCTION DE RÉCUPÉRATION DU CHEMIN GT
def get_gt_path(image_path: Path) -> Path:
    """Détermine le chemin du masque de vérité terrain binaire (UInt8) à partir du nom de l'image satellite."""
    stem = image_path.stem
    if '_extraction_rognée_validated' in stem:
        base_name = stem.replace('_extraction_rognée_validated', '')
    elif '_validated' in stem:
        base_name = stem.replace('_validated', '')
    else:
        base_name = stem.split('_')[0] 
        
    gt_name = base_name + "_mask.tif"
    return GT_BASE_DIR / gt_name


if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

# Import des modules locaux (assurez-vous que ces fichiers existent)
try:
    from model import get_unetpp_model
    from utils import (
        normalize_image, save_mask_as_shapefile, save_binary_mask_geotiff, 
    )
    from postproc_infer import (
        predict_tta, postprocess_mask
    )
except ImportError as e:
    print(f"❌ Erreur d'importation d'un module local: {e}. Vérifiez que les fichiers 'model.py', 'utils.py' et 'postproc_infer.py' existent et sont accessibles.")
    sys.exit(1)
    
# -----------------------------------
# Configuration et Arguments CLI (Tuile 256x256 par défaut)
# -----------------------------------

def parse_args() -> argparse.Namespace: 
    ap = argparse.ArgumentParser(
        description="Script d'Inférence UNIQUMENT sur l'image satellite complète par tuilage."
    )
    ap.add_argument("--input-image", type=str, default="", help="Nom du fichier image à prédire DANS LE CAS D'UNE SEULE IMAGE. Si vide, toutes les images de IMAGES_TO_PREDICT seront utilisées.")
    # ADAPTATION: Taille de tuile par défaut à 256
    ap.add_argument("--tile-size", type=int, default=256, help="Taille des tuiles d'inférence (ex: 256).")
    ap.add_argument("--overlap", type=int, default=32, help="Chevauchement des tuiles en pixels.")
    ap.add_argument("--epochs", type=int, default=60, help="Nombre d'epochs utilisé pour nommer le modèle.")
    ap.add_argument("--backbone", type=str, default="resnet34", help="Backbone du modèle.")
    ap.add_argument("--pos-weight", type=float, default=5.0, help="Poids de la classe positive (pos-weight) utilisé pour nommer le modèle.")
    ap.add_argument("--bce-w", type=float, default=0.1, help="Poids de la BCE dans la WeightedBCEDiceLoss du modèle entraîné.")
    ap.add_argument("--dice-w", type=float, default=0.9, help="Poids de la Dice dans la WeightedBCEDiceLoss du modèle entraîné.")
    ap.add_argument("--wd", type=float, default=5e-3, help="Weight Decay (L2 regularisation) du modèle entraîné.")
    ap.add_argument("--outdir", type=str, default=str(PREDICT_HIGH_ROOF_TILES_DIR), help="Dossier de sortie.")
    
    ap.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True, help="Activer/Désactiver le Test-Time Augmentation (TTA). DÉFAUT: True.")
    
    ap.add_argument("--thr-fixed", type=float, default=0.50, 
                    help="Seuil binaire pour la prédiction.")
    
    ap.add_argument("--postprocess", action=argparse.BooleanOptionalAction, default=True, help="Appliquer le post-traitement morphologique global.")
    ap.add_argument("--close-m", type=float, default=2.5, help="Rayon de fermeture morphologique (m).")
    ap.add_argument("--open-m", type=float, default=0.5, help="Rayon d'ouverture morphologique (m).")
    ap.add_argument("--min-obj-m2", type=float, default=5.0, help="Taille min d'objet à conserver (m²).")
    
    ap.add_argument("--median-filter-size", type=int, default=3, 
                    help="Taille du kernel du filtre médian appliqué à la carte de probabilités (0 pour désactiver).")
    
    ap.add_argument("--shift-row", type=int, default=0, help="Décalage vertical (lignes) à appliquer à la carte de probabilités avant binarisation (ex: -3 pour remonter).")
    ap.add_argument("--shift-col", type=int, default=0, help="Décalage horizontal (colonnes) à appliquer à la carte de probabilités avant binarisation (ex: -3 pour décaler à gauche).")
    ap.add_argument("--export-vector", action=argparse.BooleanOptionalAction, default=True, help="Exporter le masque binaire en shapefile (ACTIVÉ par défaut).")
    ap.add_argument("--export-weight-map", action="store_true", help="Exporter la carte de poids (weight map) du fenêtrage.")
    
    return ap.parse_args()

# =================================================================
# ⚙️ FONCTIONS DE MÉTRIQUES ET D'INFÉRENCE (Pas de changement dans les métriques)
# =================================================================

def calculate_global_metrics(tp: int, fp: int, fn: int, tn: int, thr: float = 0.50):
    """Calcule et affiche les métriques agrégées à partir de la matrice de confusion totale."""
    
    print("\n\n--- 🏆 RÉSULTATS D'ÉVALUATION AGRÉGÉS (Seuil fixe: {:.2f}) ---".format(thr))
    
    total_pixels = tp + fp + fn + tn
    if total_pixels == 0:
        print("❌ AUCUN PIXEL D'ÉVALUATION. Calcul des métriques impossible.")
        return

    accuracy = (tp + tn) / total_pixels
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0 
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score_val = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0 
    
    print(f"✅ Matrice de Confusion Globale:")
    print(f"   Vrais Positifs (TP): {tp:,} | Faux Positifs (FP): {fp:,}")
    print(f"   Faux Négatifs (FN): {fn:,} | Vrais Négatifs (TN): {tn:,}")
    print(f"   Total Pixels: {total_pixels:,}")

    print("\n📊 Métriques Clés:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Recall (Sensibilité):  {recall:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   F1-Score:  {f1_score_val:.4f}")
    print(f"   IoU (Jaccard): {iou:.4f}")
    
    print("-----------------------------------------------------")


def calculate_and_plot_metrics(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    y_pred_bin: np.ndarray, 
    output_dir: Path, 
    image_name: str, 
    thr: float
) -> Tuple[int, int, int, int]:
    """
    Calcule et affiche la matrice de confusion (en pourcentage), la courbe ROC,
    et le diagnostic du seuil optimal pour les prédictions.
    
    Retourne (tp, fp, fn, tn) pour l'agrégation.
    """
    print("\n--- 📈 DÉMARRAGE DE L'ÉVALUATION DES PERFORMANCES INDIVIDUELLES ---")
    
    # 1. Mise à plat des données (pixels)
    valid_pixels_mask = (y_true == 0) | (y_true == 1) 
    
    y_true_flat = y_true[valid_pixels_mask].flatten()
    y_proba_flat = y_pred_proba[valid_pixels_mask].flatten()
    y_bin_flat = y_pred_bin[valid_pixels_mask].flatten()

    if len(y_true_flat) == 0:
        print("❌ AUCUN PIXEL VALIDE (0 ou 1) pour la vérité terrain n'a été trouvé. Évaluation annulée.")
        return 0, 0, 0, 0
    
    # --- MATRICE DE CONFUSION (EN COMPTE ET EN POURCENTAGE) ---
    tp, fp, fn, tn = 0, 0, 0, 0
    
    try:
        # 1. Matrice en comptes bruts
        cm_raw = confusion_matrix(y_true_flat, y_bin_flat, labels=[0, 1])
        tn, fp, fn, tp = cm_raw.ravel()
        
        # Calcul des métriques
        accuracy = accuracy_score(y_true_flat, y_bin_flat)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0 
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score_val = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        iou = jaccard_score(y_true_flat, y_bin_flat, zero_division=1.0) 
        
        print(f"✅ Matrice de Confusion calculée (Seuil utilisé: {thr:.2f}):")
        print(f"   Vrais Positifs (TP): {tp:,} | Faux Positifs (FP): {fp:,}")
        print(f"   Faux Négatifs (FN): {fn:,} | Vrais Négatifs (TN): {tn:,}")
        print(f"   Accuracy: {accuracy:.4f} | Recall (Sensibilité): {recall:.4f}")
        print(f"   Precision: {precision:.4f} | F1-Score: {f1_score_val:.4f} | IoU: {iou:.4f}")
        
        # 2. Normalisation par la vérité terrain (lignes) pour obtenir les pourcentages
        cm_norm = confusion_matrix(y_true_flat, y_bin_flat, normalize='true', labels=[0, 1])
        
        print("\n✅ Matrice de Confusion en POURCENTAGE (Normalisée par ligne - Vérité Terrain):")
        print(f"    - Ligne 0 (Négatif / Vrai Sol): [{cm_norm[0, 0]*100:.2f}% (TN) | {cm_norm[0, 1]*100:.2f}% (FP)]")
        print(f"    - Ligne 1 (Positif / Vrai Toit): [{cm_norm[1, 0]*100:.2f}% (FN) | {cm_norm[1, 1]*100:.2f}% (TP / Rappel)]")
        
        # 3. Affichage et sauvegarde de la Matrice de Confusion (en pourcentage)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['Sol (0)', 'Toit (1)'])
        fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
        
        disp.plot(cmap=plt.cm.Blues, ax=ax_cm, values_format='.2%') 
        
        plt.title(f"Matrice de Confusion ({image_name}) - Seuil={thr:.2f}\nIoU: {iou:.4f} | Recall: {recall:.4f}")
        
        plt.savefig(output_dir / f"{image_name}_CM_thr{thr:.2f}_NORM.png")
        plt.close(fig_cm)
        print(f"💾 Matrice de confusion (normalisée) exportée.")

    except Exception as e:
        print(f"❌ ERREUR lors du calcul/affichage de la Matrice de Confusion: {e}")
        traceback.print_exc()
        
    # --- COURBE ROC ET DIAGNOSTIC DU SEUIL OPTIMAL ---
    try:
        # Calcul de la courbe ROC
        fpr, tpr, thresholds = roc_curve(y_true_flat, y_proba_flat)
        roc_auc = auc(fpr, tpr)
        
        # DIAGNOSTIC DU SEUIL OPTIMAL MAXIMISANT F1 et IoU
        best_f1 = -1.0
        best_iou_val = -1.0
        best_thr_f1 = thr
        best_thr_iou = thr
        
        for t in np.linspace(0.01, 0.99, 100): 
            y_pred_temp = (y_proba_flat >= t).astype(int)
            
            try:
                cm_temp = confusion_matrix(y_true_flat, y_pred_temp, labels=[0, 1]).ravel()
                tn_t, fp_t, fn_t, tp_t = cm_temp 
                
                recall_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
                precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
                f1_t = 2 * (precision_t * recall_t) / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0
                iou_t = tp_t / (tp_t + fp_t + fn_t) if (tp_t + fp_t + fn_t) > 0 else 0
                
                if f1_t > best_f1:
                    best_f1 = f1_t
                    best_thr_f1 = t
                    
                if iou_t > best_iou_val:
                    best_iou_val = iou_t
                    best_thr_iou = t
            except ValueError:
                continue

        if best_f1 > 0:
            print(f"\n🎯 DIAGNOSTIC : Meilleure performance théorique sur cette image:")
            print(f"   -> Seuil Optimal (Max F1) : {best_thr_f1:.4f} (F1 = {best_f1:.4f})")
            print(f"   -> Seuil Optimal (Max IoU) : {best_thr_iou:.4f} (IoU = {best_iou_val:.4f})")

        # Affichage et sauvegarde de la Courbe ROC
        fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('Taux de Faux Positifs (1 - Spécificité)')
        ax_roc.set_ylabel('Taux de Vrais Positifs (Sensibilité)')
        ax_roc.set_title(f'Courbe ROC ({image_name})')
        ax_roc.legend(loc="lower right")
        plt.savefig(output_dir / f"{image_name}_ROC.png")
        plt.close(fig_roc)
        print(f"💾 Courbe ROC exportée. AUC: {roc_auc:.4f}")

    except Exception as e:
        print(f"❌ ERREUR lors du calcul/affichage de la Courbe ROC ou du diagnostic: {e}")
        
    print("-----------------------------------------------------")
    
    # RETOURNE LES COMPTES BRUTS pour l'agrégation
    return tp, fp, fn, tn


# -----------------------------------
# FONCTION predict_full_image (ADAPTÉE pour les 11 bandes SANS RVB)
# -----------------------------------

def predict_full_image(
    model: torch.nn.Module, 
    full_image_path: Path, 
    best_model_path: Path, 
    device: torch.device,
    args: argparse.Namespace, 
    stats_path: Path,
    expected_in_channels: int, 
    band_indices_to_read: List[int]
) -> Union[Tuple[Path, int, int, int, int], Path]:
    """
    Effectue l'inférence sur l'ensemble de l'image satellite par tuilage avec chevauchement,
    en lisant les 11 bandes (sans RVB) d'entrée du modèle.
    
    Retourne (chemin_binaire_sortie, TP, FP, FN, TN) ou le chemin d'une erreur.
    """
    print("\n\n=== 🚀 DÉMARRAGE DE L'INFÉRENCE SUR IMAGE COMPLÈTE (Tuilage) ===")
    
    # 1. Préparation des Chemins et Variables
    output_dir_base = Path(args.outdir)
    output_dir = output_dir_base / "full_image_image_inference" / full_image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CHARGEMENT DES STATS 
    try:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            # 🎯 ADAPTATION 3 : UTILISATION DES STATS. Normalement, le modèle 11-bandes doit utiliser les stats des 11 bandes (indices 3 à 13 si 0-based)
            # On suppose que le fichier mean_std.json contient les 14 stats et que normalize_image prendra les stats correspondantes aux bandes lues.
            # Cependant, ici on prend juste les 14 stats, et le 'normalize_image' se base sur l'input_tile (11 bandes).
            # POUR UN MODÈLE 11-BANDES SANS RVB, le fichier STATS doit contenir SEULEMENT les stats des 11 bandes.
            # Dans le contexte où le fichier STATS_PATH est global, on doit s'assurer que seules les 11 stats nécessaires sont utilisées.
            # Si le fichier stats.json contient les 14 stats, on doit les *filtrer* ici.
            
            all_means = np.array(stats['means'])
            all_stds = np.array(stats['stds'])
            
            # Les indices 1-based [4, 5, ..., 14] correspondent aux indices 0-based [3, 4, ..., 13]
            # Assumant que stats['means'] et stats['stds'] sont ordonnés selon les indices 1 à 14.
            indices_0_based = [idx - 1 for idx in band_indices_to_read]
            
            # Application du filtrage (extraction des 11 stats)
            mean = all_means[indices_0_based]
            std = all_stds[indices_0_based]
            
            # Vérification de sécurité: le fichier de stats filtré doit contenir 11 valeurs
            if len(mean) != expected_in_channels or len(std) != expected_in_channels:
                 print(f"❌ ERREUR CRITIQUE: Le fichier de stats ({len(all_means)} valeurs) n'a pas permis d'extraire {expected_in_channels} valeurs cohérentes avec les bandes lues.")
                 return output_dir / "error_stats_mismatch.txt"
    except Exception as e:
        print(f"❌ ERREUR CRITIQUE lors de la lecture des stats pour la normalisation: {e}")
        return output_dir / "error_stats_read.txt"
        
    model_tag = best_model_path.stem.replace("best_model_", "")
    output_basename = f"PREDICTION_{full_image_path.stem}_{model_tag}"
    
    # 2. Ouverture de l'image source
    try:
        with rasterio.open(full_image_path) as src:
            profile = src.profile
            height, width = src.height, src.width
            num_channels_source = src.count # Compte total de bandes (devrait être 14)
            
            # Vérification de sécurité: l'image source doit avoir au moins le nombre max de bandes à lire
            if num_channels_source < band_indices_to_read[-1]:
                 print(f"❌ ERREUR: L'image source a seulement {num_channels_source} canaux. Impossible de lire les bandes {band_indices_to_read}.")
                 return output_dir / "error_image_insufficient_channels.txt"
            
            # Vérification que le nombre de bandes à lire correspond aux canaux du modèle
            if len(band_indices_to_read) != expected_in_channels:
                 print(f"❌ ERREUR CRITIQUE: La liste des bandes à lire ({len(band_indices_to_read)}) ne correspond pas aux canaux du modèle ({expected_in_channels}).")
                 return output_dir / "error_channel_mismatch.txt"
            
            # ADAPTATION: Message d'information
            print(f"📖 Image source lue: {full_image_path.name} ({width}x{height} pixels, {num_channels_source} canaux, **lecture des {expected_in_channels} bandes SANS RVB**).")

            # 3. Initialisation du Tapis de Sortie (Probabilités et Poids)
            full_proba_map = np.zeros((height, width), dtype=np.float32)
            full_weight_map = np.zeros((height, width), dtype=np.float32) 
            tile_count = 0
            
            tile_h = args.tile_size
            tile_w = args.tile_size
            overlap = args.overlap
            step = args.tile_size - (overlap * 2) 
            
            if step <= 0:
                 print(f"❌ ERREUR: Chevauchement trop grand (Overlap={overlap}, TileSize={args.tile_size}). Inférence annulée.")
                 return output_dir / "error_overlap_too_large.txt"
            
            model.eval()
            print(f"📐 Tuilage: Taille {args.tile_size}x{args.tile_size}, Chevauchement {overlap} pixels (Pas: {step}).")
            
            h_hann = np.hanning(args.tile_size) 
            weight_kernel = np.outer(h_hann, h_hann).astype(np.float32) 
            
            # 4. Boucle de Tuilage et d'Inférence
            with torch.no_grad():
                for row_off in range(0, height, step):
                    for col_off in range(0, width, step):
                        
                        core_width = min(tile_w, width - col_off)
                        core_height = min(tile_h, height - row_off)
                        window = Window(col_off=col_off, row_off=row_off, width=core_width, height=core_height)
                        if window.width <= 0 or window.height <= 0: continue
                        
                        win_row_start = max(0, window.row_off - overlap) 
                        win_col_start = max(0, window.col_off - overlap) 
                        win_row_end = min(height, window.row_off + window.height + overlap) 
                        win_col_end = min(width, window.col_off + window.width + overlap) 
                        read_window = Window.from_slices((win_row_start, win_row_end), (win_col_start, win_col_end))
                        
                        # Lecture des 11 bandes (indices 4 à 14)
                        tile_data = src.read(indexes=band_indices_to_read, window=read_window) 
                        tile_count += 1
                        
                        H_read, W_read = tile_data.shape[1], tile_data.shape[2]
                        
                        crop_h_start = max(0, (H_read - args.tile_size) // 2) 
                        crop_w_start = max(0, (W_read - args.tile_size) // 2) 
                        cropped_tile_data = tile_data[:, crop_h_start:crop_h_start + args.tile_size, crop_w_start:crop_w_start + args.tile_size]
                        H_cropped, W_cropped = cropped_tile_data.shape[1], cropped_tile_data.shape[2]
                        
                        if H_cropped < args.tile_size or W_cropped < args.tile_size:
                             pad_h_needed = args.tile_size - H_cropped
                             pad_w_needed = args.tile_size - W_cropped
                             input_tile = np.pad(cropped_tile_data, ((0, 0), (0, pad_h_needed), (0, pad_w_needed)), mode='reflect')
                        else:
                            input_tile = cropped_tile_data

                        # input_tile a maintenant 11 canaux
                        norm_tile = normalize_image(input_tile, mean, std)
                        img_tensor = torch.from_numpy(norm_tile).float().unsqueeze(0).to(device)

                        if args.tta:
                            proba_tile_full = predict_tta(model, img_tensor, device)
                        else:
                            logits = model(img_tensor).squeeze(dim=0).detach().cpu()
                            proba_tile_full = torch.sigmoid(logits).squeeze().numpy()
                        
                        source_overlap_row_start = window.row_off - read_window.row_off
                        source_overlap_col_start = window.col_off - read_window.col_off 
                        crop_row_start = max(0, source_overlap_row_start - crop_h_start) if window.row_off == 0 else 0
                        crop_col_start = max(0, source_overlap_col_start - crop_w_start) if window.col_off == 0 else 0
                        crop_row_end = min(args.tile_size, crop_row_start + window.height)
                        crop_col_end = min(args.tile_size, crop_col_start + window.width)
                        proba_heart = proba_tile_full[crop_row_start:crop_row_end, crop_col_start:crop_col_end]
                        
                        if proba_heart.shape != (window.height, window.width):
                            print(f"❌ ERREUR DE TAILLE DE CŒUR. Tuile ({tile_count}): Attendu ({window.height}, {window.width}), Obtenu {proba_heart.shape}.")
                            raise ValueError("Découpage du cœur invalide.")

                        
                        weight_heart = weight_kernel[crop_row_start:crop_row_end, crop_col_start:crop_col_end]
                        
                        # Indices de destination dans le tapis de sortie
                        insert_row_start = window.row_off
                        insert_col_start = window.col_off
                        insert_row_end = window.row_off + window.height
                        insert_col_end = window.col_off + window.width
                        
                        # Accumulation des probabilités
                        full_proba_map[insert_row_start:insert_row_end, insert_col_start:insert_col_end] += proba_heart * weight_heart
                        
                        full_weight_map[insert_row_start:insert_row_end, insert_col_start:insert_col_end] += weight_heart 
                        
                        
                        if tile_count % 100 == 0:
                             print(f"  -> {tile_count} tuiles traitées. Tuile actuelle: ({window.row_off}, {window.col_off})")

            print(f"✅ Inférence de {tile_count} tuiles terminée.")
            
            # 7. Finalisation et Binarisation/Post-traitement 
            epsilon = 1e-6
            full_proba_map = full_proba_map / (full_weight_map + epsilon) 
            
            # FILTRE MÉDIAN ET DÉCALAGE
            if args.median_filter_size > 0:
                print(f"🔄 Application du filtre médian ({args.median_filter_size}x{args.median_filter_size})...")
                try:
                    full_proba_map = median_filter(full_proba_map, size=args.median_filter_size)
                    print("✅ Filtre médian appliqué à la carte de probabilités.")
                except Exception as e:
                    print(f"❌ ERREUR lors de l'application du filtre médian: {e}")
            
            if args.shift_row != 0 or args.shift_col != 0:
                print(f"🔄 Application du décalage (Row: {args.shift_row}, Col: {args.shift_col})...")
                full_proba_map = np.roll(full_proba_map, args.shift_row, axis=0)
                full_proba_map = np.roll(full_proba_map, args.shift_col, axis=1)
                
            thr = args.thr_fixed
            final_mask_bin = (full_proba_map > thr).astype(np.uint8)
            
            # Appliquer le masque NoData de l'image source à la prédiction binaire
            try:
                valid_mask = src.dataset_mask() 
                final_mask_bin[valid_mask == 0] = 0
            except Exception: pass
            
            # POST-TRAITEMENT MORPHOLOGIQUE
            if args.postprocess: 
                print("🔄 Application du post-traitement morphologique...")
                try:
                     final_mask_bin = postprocess_mask(
                         final_mask_bin, src.transform, 
                         close_radius_m=args.close_m, open_radius_m=args.open_m,
                         min_obj_area_m2=args.min_obj_m2
                     )
                     print("✅ Post-traitement appliqué.")
                except Exception as e: 
                     print(f"⚠️ Erreur lors du post-traitement: {e}. Ignoré.")

            # 9. Exportation
            file_tag = f"thr{thr:.2f}"
            if args.tta: file_tag += "_tta"
            if args.postprocess: file_tag += "_pp"
            file_tag += f"_tile{args.tile_size}" 
            
            if args.median_filter_size > 0: file_tag += f"_med{args.median_filter_size}"
            if args.shift_row != 0 or args.shift_col != 0: file_tag += f"_shift"
            
            export_profile = profile.copy()
            export_profile.update(count=1, dtype=rasterio.uint8, nodata=None, compress='lzw')
            out_name_bin = f"{output_basename}_{file_tag}_BIN.tif"
            out_bin_path = output_dir / out_name_bin
            
            proba_profile = profile.copy() 
            proba_profile.update(count=1, dtype=rasterio.float32, nodata=None, compress='lzw')
            out_proba_path = output_dir / f"{output_basename}_PROBA.tif" 
            
            # Export du masque binaire (Assumant que save_binary_mask_geotiff est défini dans utils.py)
            try:
                save_binary_mask_geotiff(final_mask_bin, out_bin_path, export_profile)
                print(f"💾 Masque binaire sauvegardé : {out_bin_path.name}")
            except Exception as e:
                print(f"❌ ERREUR lors de l'export du masque binaire : {e}")
            
            # Export de la carte de probabilités
            try:
                with rasterio.open(out_proba_path, 'w', **proba_profile) as dst:
                    dst.write(full_proba_map, 1)
                print(f"💾 Carte de probabilités sauvegardée : {out_proba_path.name}")
            except Exception as e:
                print(f"❌ ERREUR lors de l'export de la carte de probabilités : {e}")

            # Export en shapefile (si demandé)
            if args.export_vector:
                try:
                    save_mask_as_shapefile(final_mask_bin, src.transform, src.crs, out_bin_path.with_suffix(".shp"))
                    print(f"💾 Masque vectoriel (Shapefile) exporté.")
                except Exception as e:
                    print(f"❌ ERREUR lors de l'export du shapefile : {e}")

            # Export de la carte de poids (si demandé)
            if args.export_weight_map:
                try:
                    out_weight_path = output_dir / f"{output_basename}_WEIGHT.tif"
                    weight_profile = profile.copy()
                    weight_profile.update(count=1, dtype=rasterio.float32, nodata=None, compress='lzw')
                    with rasterio.open(out_weight_path, 'w', **weight_profile) as dst:
                        dst.write(full_weight_map, 1)
                    print(f"💾 Carte de poids sauvegardée : {out_weight_path.name}")
                except Exception as e:
                    print(f"❌ ERREUR lors de l'export de la carte de poids : {e}")


            # 10. CALCUL ET AFFICHAGE DES MÉTRIQUES
            gt_path = get_gt_path(full_image_path)
            print(f"🔎 Recherche du fichier GT à: {gt_path}")

            tp, fp, fn, tn = 0, 0, 0, 0
            if gt_path.exists():
                try:
                    with rasterio.open(gt_path) as gt_src:
                        y_true = gt_src.read(1).astype(np.uint8) 
                        
                        if y_true.shape != (height, width):
                            print(f"⚠️ AVERTISSEMENT: GT ({y_true.shape}) et Prédiction ({height}x{width}) n'ont pas la même taille. Évaluation impossible.")
                        else:
                            # Renvoie les comptes bruts
                            tp, fp, fn, tn = calculate_and_plot_metrics(
                                y_true, full_proba_map, final_mask_bin, 
                                output_dir, full_image_path.stem, thr
                            )
                except Exception as e:
                    print(f"❌ ERREUR lors de la lecture du fichier de vérité terrain {gt_path.name}: {e}. Évaluation ignorée.")
            else:
                print(f"❌ Fichier de Vérité Terrain (GT) non trouvé : {gt_path.name}. Aucune évaluation effectuée.")
                
            # RETOURNE LE CHEMIN ET LES COMPTES BRUTS
            return out_bin_path, tp, fp, fn, tn

    except Exception as e:
        print("\n--- TRACEBACK DÉTAILLÉE ---")
        traceback.print_exc()
        print("---------------------------")
        print(f"❌ ERREUR FATALE (Lecture/Tuilage/Export): {e}")
        return output_dir / "error_fatal.txt" # Retourne juste un chemin d'erreur

# -----------------------------------
# ⚙️ Main (ADAPTATION DU CHEMIN DU MODÈLE pour 11 canaux SANS RVB)
# -----------------------------------

def main() -> None:
    
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔄 Appareil utilisé : {device}")
    
    print(f"\n=======================================================")
    print(f"MODE DÉDIÉ : INFERENCE SUR IMAGE SATELLITE COMPLÈTE (Tuiles {args.tile_size}x{args.tile_size})")
    # 🎯 CONFIRMATION DU MODE 11 CANAUX SANS RVB
    print(f"MODE ACTIF : {IN_CHANNELS_MODEL} CANAUX (Exclusion RVB: Bandes {BAND_INDICES_TO_READ})")
    print(f"=======================================================")
    
    # --- 1. Construction du chemin du meilleur modèle ---
    BCE_W = args.bce_w; DICE_W = args.dice_w; POS_W = args.pos_weight
    
    # 🎯 ADAPTATION 4 : Construction du tag pour 11 canaux SANS RVB
    # Doit correspondre EXACTEMENT au nom de fichier généré par train_script.py (11chan_no_rgb)
    tag_base = f"{args.backbone}_{IN_CHANNELS_MODEL}chan_no_rgb_wbd_b{BCE_W:.1f}d{DICE_W:.1f}_w{POS_W:.2f}_BALANCED_e{args.epochs}"
    best_model_dir = RUNS_BASE_DIR / f"{tag_base}_runs"
    best_model_path = best_model_dir / f"best_model_{tag_base}.pth"
    
    if not best_model_path.exists():
         print(f"❌ Modèle introuvable au chemin : {best_model_path}. Vérifiez les arguments du modèle et le chemin exact.")
         return
    
    print(f"✅ Modèle à charger identifié: {best_model_path.name}")
    
    # --- 2. Détermination des cibles ---
    if args.input_image:
        target_images = [args.input_image]
    else:
        target_images = IMAGES_TO_PREDICT

    full_image_paths = [BASE_DATA_DIR / img_name for img_name in target_images]

    # --- 3. Initialisation du Modèle ---
    if not full_image_paths:
         print("❌ ERREUR: Aucune image cible n'est définie. Arrêt.")
         return
         
    # Le modèle est initialisé avec 11 canaux
    model = get_unetpp_model(in_channels=IN_CHANNELS_MODEL, num_classes=1, backbone=args.backbone).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"✅ Modèle {best_model_path.name} chargé avec succès ({IN_CHANNELS_MODEL} canaux).")

    # --- 4. Exécution de l'Inférence ---
    
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    
    for full_image_path in full_image_paths:
        print(f"\n=======================================================")
        print(f"TRAITEMENT DE L'IMAGE : {full_image_path.name}")
        print(f"=======================================================")
        
        result = predict_full_image(
            model, full_image_path, best_model_path, device, args, 
            STATS_PATH, IN_CHANNELS_MODEL, BAND_INDICES_TO_READ
        )
        
        if isinstance(result, tuple) and len(result) == 5:
            out_bin_path, tp, fp, fn, tn = result
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
        else:
            print(f"❌ Échec du traitement de l'image {full_image_path.name}. Résultat: {result}")
            

    # --- 5. Affichage des Métriques Globales ---
    if total_tp + total_fp + total_fn + total_tn > 0:
         calculate_global_metrics(total_tp, total_fp, total_fn, total_tn, args.thr_fixed)

    print("\n\n--- FIN DU SCRIPT D'INFÉRENCE ---")

if __name__ == "__main__":
    main()