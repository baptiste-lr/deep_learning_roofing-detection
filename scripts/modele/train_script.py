#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os, sys, json, csv, shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import argparse
from copy import deepcopy 

import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from sklearn.metrics import jaccard_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau 

# =================================================================
#  CONFIGURATION & CHEMINS 
# =================================================================
PROJECT_ROOT = Path("/home/baptistedlb/Documents/Stage_IRD/Baptiste_Deep/Deep_learning_codes/codes")

# 🎯 ADAPTATION 1 : CHEMIN DE BASE DES DONNÉES (11 bandes, 256x256, SANS RVB)
DATA_ROOT_MULTI_SOURCE = PROJECT_ROOT / "dataset_source_11bands_256x256_split_balanced_no_rgb" 

SCRIPTS_DIR  = PROJECT_ROOT 
RUNS_BASE_DIR = PROJECT_ROOT / "Outputs/runs"
PREDICT_HIGH_ROOF_TILES_DIR = PROJECT_ROOT / "Outputs/predict_high_roof_tiles" 
STATS_PATH = PROJECT_ROOT / "preparation_data" / "mean_std.json" 
BASE_DATA_DIR = Path("/home/baptistedlb/Documents/Stage_IRD/Baptiste_Deep/Deep_learning_codes/data/data_sat")
REFERENCE_IMAGE_PATHS = [
    BASE_DATA_DIR / "baptiste_extraction_rognée_validated.tif",
    BASE_DATA_DIR / "marc_extraction_rognée_validated.tif"
]

# 🎯 ADAPTATION 2 : Nombre de canaux attendus (11 : 14 bandes initiales - 3 bandes RVB)
IN_CHANNELS = 11 

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

# Import des modules locaux (assurez-vous que ces fichiers existent)
try:
    from model import get_unetpp_model
    from dataset import TileDataset
    from losses import WeightedBCEDiceLoss, DiceLoss, _ensure_bchw 
    from utils import (
        normalize_image, save_mask_as_shapefile, plot_losses, save_checkpoint,
        save_binary_mask_geotiff, save_visual_mask_geotiff
    )
    from postproc_infer import (
        predict_simple, predict_tta, auto_threshold,
        postprocess_mask
    ) 
except ImportError as e:
    print(f"❌ Erreur d'importation : Le module {e.name} est introuvable. Veuillez vérifier vos chemins et dépendances.")
    sys.exit(1)


# --- DÉFINITION DES CHEMINS DES DONNÉES ---
# Le chemin de base est mis à jour ci-dessus, les sous-chemins restent valides.
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
# Configuration et Arguments CLI 
# -----------------------------------

def parse_args() -> argparse.Namespace: 
    ap = argparse.ArgumentParser(
        description="Script d'Entrainement: Entraine U-Net++ et sauvegarde le meilleur modèle."
    )
    # --- TRAINING PARAMS ---
    ap.add_argument("--epochs", type=int, default=60, help="Nombre d'epochs.")
    ap.add_argument("--batch", type=int, default=8, help="Taille du batch.")
    ap.add_argument("--lr", type=float, default=3e-5, help="Learning rate.")
    ap.add_argument("--backbone", type=str, default="resnet34", help="Backbone du modèle.")
    ap.add_argument("--pos-weight", type=float, default=5.0, help="Poids de la classe positive (Toiture) dans BCE.") 
    ap.add_argument("--early-stop-patience", type=int, default=100, help="Patience pour l'Early Stopping (désactivé si > 80).")
    ap.add_argument("--close-m", type=float, default=1.5, help="Rayon de fermeture morphologique (m).")
    ap.add_argument("--min-obj-m2", type=float, default=5.0, help="Taille min d'objet à conserver (m²).")  
    return ap.parse_args()

# -----------------------------------
# FONCTION D'EXPORT DES MÉTRIQUES
# -----------------------------------

def export_metrics_table(runs_dir: Path, train_losses: List[float], val_losses: List[float], val_ious: List[float], val_f1s: List[float]) -> None:
    """Exporte les métriques d'entraînement et de validation (Loss, IoU, F1) dans un fichier CSV."""
    
    metrics_path = runs_dir / "validation_metrics.csv"
    
    epochs = len(train_losses)
    
    if not (len(val_losses) == len(val_ious) == len(val_f1s) == epochs):
        print("⚠️ Erreur : Les listes de métriques n'ont pas la même longueur. Export CSV annulé.")
        return
        
    try:
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Écriture de l'en-tête du tableau
            writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_IoU_Jaccard', 'Val_F1_Dice'])
            
            # Écriture des données
            for i in range(epochs):
                writer.writerow([
                    i + 1,
                    f"{train_losses[i]:.6f}",
                    f"{val_losses[i]:.6f}",
                    f"{val_ious[i]:.4f}",
                    f"{val_f1s[i]:.4f}"
                ])
                
        print(f"📝 Tableau des métriques exporté : {metrics_path.name}")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'export du CSV : {e}")


# -----------------------------------
# Training & Validation Loop 
# -----------------------------------

def train_and_validate(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, args: argparse.Namespace, in_channels: int) -> Tuple[Path, float]:
    
    # Stratégie de perte optimisée pour les bords (0.1 BCE / 0.9 DICE)
    BCE_W = 0.1 
    DICE_W = 0.9
    WEIGHT_DECAY = 5e-3 
    
    criterion = WeightedBCEDiceLoss(
        bce_weight=BCE_W,  
        dice_weight=DICE_W,
        pos_weight=torch.as_tensor(args.pos_weight, dtype=torch.float32).to(device) 
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',            
        factor=0.5,           
        patience=20,          
        min_lr=1e-7           
    )
    
    # --- Dossiers et chemins ---
    # 🎯 ADAPTATION 3 : MISE À JOUR DU TAG POUR REFLETER LES 11 CANAUX SANS RVB
    tag = f"{args.backbone}_{in_channels}chan_no_rgb_wbd_b{BCE_W:.1f}d{DICE_W:.1f}_w{args.pos_weight:.2f}_BALANCED_e{args.epochs}"
    runs_dir = RUNS_BASE_DIR / f"{tag}_runs" 
    runs_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = runs_dir / f"best_model_{tag}.pth"

    train_losses, val_losses, val_ious, val_f1s = [], [], [], []
    best_val_iou = -1.0
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n=== Démarrage de l'entraînement sur données ÉQUILIBRÉES ({args.backbone}, {in_channels} canaux, SANS RVB) ===")
    
    for epoch in range(1, args.epochs + 1):
        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            masks = _ensure_bchw(masks).float()

            outputs = model(imgs)
            outputs = _ensure_bchw(outputs).float()

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # ---- VALIDATION ----
        model.eval()
        running_val_loss = 0.0
        val_iou_sum = 0.0
        val_f1_sum = 0.0
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                masks = _ensure_bchw(masks).float()

                outputs = model(imgs)
                outputs = _ensure_bchw(outputs).float()

                vloss = criterion(outputs, masks)
                running_val_loss += vloss.item() * imgs.size(0)
                
                # Calcul des métriques binarisées (IoU et F1)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()
                
                masks_flat = masks.cpu().numpy().flatten()
                preds_flat = preds.cpu().numpy().flatten()
                
                # IoU / Jaccard
                iou_score = jaccard_score(masks_flat, preds_flat, zero_division=1.0)
                val_iou_sum += iou_score
                
                # F1-Score (Dice)
                f1_score_val = f1_score(masks_flat, preds_flat, zero_division=1.0)
                val_f1_sum += f1_score_val


        val_loss = running_val_loss / len(val_loader.dataset)
        val_iou = val_iou_sum / len(val_loader)
        val_f1 = val_f1_sum / len(val_loader)
        
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        val_f1s.append(val_f1)
        
        # --- STEP DU SCHEDULER ---
        scheduler.step(val_iou) 
        current_lr = optimizer.param_groups[0]['lr']


        print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.1e}")

        # --- EARLY STOPPING & SAUVEGARDE (sur IoU) ---
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_checkpoint(model, best_model_path)
            best_val_loss = val_loss 
            patience_counter = 0
            print(f"  -> Nouveau meilleur IoU ({best_val_iou:.4f}). Modèle sauvegardé.")
        else:
            patience_counter += 1
            # Condition pour l'Early Stopping (désactivé si patience > 80)
            if args.early_stop_patience <= 80 and patience_counter >= args.early_stop_patience:
                print(f"❌ Early Stopping après {patience_counter} époques sans amélioration de l'IoU.")
                break
            # Si la patience est haute, continuer
            elif args.early_stop_patience > 80:
                 pass


    plot_losses(train_losses, val_losses, f"WBD (b{BCE_W:.1f}d{DICE_W:.1f}, w={args.pos_weight}) + L2 ({WEIGHT_DECAY:.1e})", args.backbone, best_val_loss, runs_dir)
    print(f"\n✅ Entraînement terminé. Meilleur modèle: {best_model_path.name} (IoU={best_val_iou:.4f})")
    
    export_metrics_table(runs_dir, train_losses, val_losses, val_ious, val_f1s)
    
    # Création du Manifest
    manifest = {
        "model_path": str(best_model_path),
        "architecture": "UnetPlusPlus",
        "loss": "WeightedBCEDiceLoss",
        "loss_params": {"bce_weight": BCE_W, "dice_weight": DICE_W, "pos_weight": args.pos_weight},
        "backbone": args.backbone,
        "encoder_weights": None,
        "in_channels": int(in_channels), 
        "num_classes": 1,
        "best_val_loss": float(best_val_loss),
        "best_val_iou": float(best_val_iou),
        "best_val_f1": float(val_f1s[np.argmax(val_ious)]) if val_f1s and len(val_ious) > 0 else 0.0, 
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        # 🎯 ADAPTATION 4 : Mise à jour de la note pour refléter 11 canaux (SANS B, G, R) et 256x256
        "notes": f"Modèle entraîné sur dataset ÉQUILIBRÉ (0% vs >1% toiture). **{in_channels} CANAUX (SANS B, G, R)**. Tuiles 256x256 AUGMENTÉES. Loss ajustée à {BCE_W:.1f}/{DICE_W:.1f} (pos_weight={args.pos_weight:.1f}). Post-proc close-m ajusté à {args.close_m:.1f}m et min-obj à {args.min_obj_m2:.1f}m2."
    }
    manifest_path = runs_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"📝 Manifest écrit: {manifest_path}")

    return best_model_path, best_val_iou

# -----------------------------------
# Main 
# -----------------------------------

def main() -> None:
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"🔄 Appareil utilisé : {device}")
    
    # --- 1. Préparation des statistiques et des datasets ---
    print("💡 Initialisation des datasets...")

    def setup_dataset(source_key: str, split: str) -> TileDataset:
        img_dir = DATA_CONFIG[source_key][f"{split}_img"]
        mask_dir = DATA_CONFIG[source_key][f"{split}_mask"]
        
        return TileDataset(img_dir, mask_dir, STATS_PATH, REFERENCE_IMAGE_PATHS)

    try:
        train_ds_baptiste = setup_dataset("images_baptiste", "train")
        in_channels: int = IN_CHANNELS 

        train_ds_marc = setup_dataset("images_marc", "train")
        val_ds_baptiste_full = setup_dataset("images_baptiste", "val")
        val_ds_marc_full = setup_dataset("images_marc", "val")
        
        # 🎯 ADAPTATION 5 : Vérification de sécurité mise à jour pour les 11 canaux.
        if train_ds_baptiste.n_channels != IN_CHANNELS:
            print(f"❌ ERREUR: TileDataset a lu {train_ds_baptiste.n_channels} canaux, au lieu des {IN_CHANNELS} attendus. Veuillez vérifier votre TileDataset.py pour vous assurer qu'il lit bien les 11 bandes (indices 3 à 13).")
            return

        train_ds = ConcatDataset([train_ds_baptiste, train_ds_marc])
        val_ds_full = ConcatDataset([val_ds_baptiste_full, val_ds_marc_full])

    except FileNotFoundError as e:
        print(f"❌ Erreur: Impossible de charger les données. Vérifiez l'arborescence et l'existence de {e}.")
        return
    except Exception as e:
        # Remontée des erreurs générales (y compris l'erreur de lecture de fichier si non capturée)
        print(f"❌ Erreur inattendue lors du chargement des datasets: {e}")
        return

    print(f"📖 Datasets chargés. Train total: {len(train_ds)} tuiles | Val total: {len(val_ds_full)} tuiles | Canaux: {in_channels}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds_full, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)

    # --- 2. Entraînement ---
    model = get_unetpp_model(in_channels=in_channels, num_classes=1, backbone=args.backbone).to(device)
    
    train_and_validate(model, train_loader, val_loader, device, args, in_channels)

if __name__ == "__main__":
    main()