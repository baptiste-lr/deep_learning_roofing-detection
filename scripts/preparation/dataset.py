#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rasterio
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import json
from typing import List, Tuple
import albumentations as A
import cv2
from rasterio.enums import ColorInterp

# =========================
# CONSTANTES
# =========================

band_names_source = ["B", "G", "R", "PIR", "C3", "L", "MNDWI", "MSAVI", "NDVI", "ExG", "BII", "UAI", "Texture_R_3x3", "Texture_PIR_3x3"]
ORIGINAL_BAND_COUNT = len(band_names_source) # 14 bandes initiales

# 🎯 ADAPTATION 1 : CONSERVER TOUTES LES 14 BANDES (INCLUSION RVB: indices 0, 1, 2)
bands_to_keep_indices = list(range(ORIGINAL_BAND_COUNT)) # Conserve les indices 0 à 13

# 🎯 ADAPTATION 2 : NOUVEAU NOMBRE DE BANDES (14)
BAND_COUNT = ORIGINAL_BAND_COUNT # Doit être 14
band_names = band_names_source
print(f"Bandes conservées ({BAND_COUNT}): {band_names} (INCLUS B, G, R)")

# ==============================================================================
# CHARGEMENT DES STATISTIQUES GLOBALES (MIN/MAX)
# ==============================================================================

# Assurez-vous d'avoir un fichier de stats adapté aux 14 bandes si vous utilisez Min/Max
STATS_FILE = "preparation_data/dataset_global_minmax_stats.json" 

GLOBAL_MIN = None
GLOBAL_MAX = None

try:
    # --- Chargement des statistiques globales ---
    with open(STATS_FILE, 'r') as f:
        stats = json.load(f)
    # Convertir les listes JSON en tableaux numpy (float32 pour les calculs)
    GLOBAL_MIN_SOURCE = np.array(stats["GLOBAL_MIN"], dtype=np.float32)
    GLOBAL_MAX_SOURCE = np.array(stats["GLOBAL_MAX"], dtype=np.float32)

    # 🎯 ADAPTATION 3 : FILTRAGE - CONSERVER TOUTES LES 14 STATISTIQUES
    GLOBAL_MIN = GLOBAL_MIN_SOURCE[bands_to_keep_indices]
    GLOBAL_MAX = GLOBAL_MAX_SOURCE[bands_to_keep_indices]
    
    # Vérification pour s'assurer que la taille correspond
    if len(GLOBAL_MIN) != BAND_COUNT:
         raise ValueError(f"Le fichier de statistiques Min/Max doit contenir {BAND_COUNT} valeurs. Vérifiez le fichier {STATS_FILE}.")

    print(f"[INFO] Statistiques globales (Min/Max) chargées avec succès depuis {STATS_FILE} pour la normalisation à {BAND_COUNT} bandes (INCLUS B, G, R).")
except FileNotFoundError:
    print(f"[ERREUR] Fichier de statistiques {STATS_FILE} non trouvé. Exécutez inspecteurgeotiff.py d'abord.")
    import sys; sys.exit(1)
except Exception as e:
    print(f"[ERREUR] Erreur lors du chargement des statistiques : {e}")
    import sys; sys.exit(1)


print(f"[INFO] Mise à l'échelle Min-Max Globale vers [1, 255] appliquée pour le format uint8 (0 = NoData).")

# ---
# ==============================================================================
# 1. Fonctions d'Augmentation (Inchangées)
# ==============================================================================

def get_augmentations() -> A.Compose:
    """Définit le pipeline d'augmentation des données complexes (géométriques et photométriques)."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=10,
            p=0.3,
            border_mode=cv2.BORDER_CONSTANT,
            value=0 
        ),
    ])


def augment_image_and_mask(img_tile: np.ndarray, mask_tile: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Génère 6 tuiles (Originale + 5 augmentées aléatoirement) en utilisant Albumentations.
    """
    # img_tile est C x H x W (C = 14)
    img_h_w_c = np.transpose(img_tile, (1, 2, 0)) # H x W x C
    aug_pairs = []
    aug_pairs.append((img_tile.copy(), mask_tile.copy()))
    aug_pipeline = get_augmentations()

    for _ in range(5):
        augmented = aug_pipeline(image=img_h_w_c, mask=mask_tile)
        img_aug_h_w_c = augmented['image']
        mask_aug = augmented['mask']
        img_aug_c_h_w = np.transpose(img_aug_h_w_c, (2, 0, 1)) # C x H x W
        aug_pairs.append((img_aug_c_h_w, mask_aug))

    return aug_pairs

# ---
# ------------------------------------------------------------------------------
# 2. Fonction de Création et Sauvegarde des Tuiles
# ------------------------------------------------------------------------------

def create_tiles_and_save(image_path, mask_path, output_dir_all, tile_size, stride, count_offset=0, save_all_valid_tiles=True) -> Tuple[List[str], int, int]:
    """
    Crée et sauvegarde des tuiles UINT8 AUGMENTÉES.
    Retourne la liste des IDs, le nouvel offset du compteur, et le nombre de tuiles ACCEPTÉES (non-augmentées).
    """
    image_name_stem = Path(image_path).stem
    if "baptiste" in image_name_stem.lower():
        source_dir = "images_baptiste"
    elif "marc" in image_name_stem.lower():
        source_dir = "images_marc"
    else:
        source_dir = "images_autres"

    output_img_dir = Path(output_dir_all) / "images" / source_dir
    output_mask_dir = Path(output_dir_all) / "masks" / source_dir
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    saved_tile_ids = []
    rejected_nodata_count = 0
    rejected_homogeneous_count = 0
    rejected_all_one_count = 0
    total_tile_count = 0
    accepted_non_augmented_count = 0

    with rasterio.open(image_path) as raster_img, rasterio.open(mask_path) as raster_mask:
        
        # 🚨 VÉRIFICATION : Assurer que l'image source a bien 14 bandes
        if raster_img.count != ORIGINAL_BAND_COUNT:
            raise ValueError(f"[ERREUR FATALE] Le fichier {Path(image_path).name} a {raster_img.count} bandes au lieu des {ORIGINAL_BAND_COUNT} attendues. Abandon de ce fichier.")
            
        # 🎯 ADAPTATION 4 : Lire TOUTES LES BANDES (14), puis CONSERVER TOUTES LES BANDES
        image_full = raster_img.read().astype(np.float32) # Lit les 14 bandes
        image = image_full[bands_to_keep_indices, :, :] # Conserve les 14 bandes
        
        mask_all = raster_mask.read(1)
        
        write_band_names = image.shape[0] == BAND_COUNT 
        
        crs = raster_img.crs
        height, width = image.shape[1], image.shape[2]
        res_x = raster_img.transform.a
        res_y = raster_img.transform.e
        NODATA_SOURCE = raster_img.nodata
        if NODATA_SOURCE is None or NODATA_SOURCE != 0:
            print(f"[ERREUR/ATTENTION] La valeur NoData n'est pas 0.0. Forcé à 0 pour le filtrage.")
            NODATA_SOURCE = 0
        MIN_VALID_PIXEL_PCT = 0.60
        
        # 🚨 MISE À JOUR : Utiliser le nouveau BAND_COUNT (14) pour calculer la densité de NoData
        total_pixels_in_tile = tile_size * tile_size * BAND_COUNT

        count = count_offset
        for y in tqdm(range(0, height - tile_size + 1, stride), desc=f"Tuilage et Augmentation de {image_name_stem}"):
            for x in range(0, width - tile_size + 1, stride):
                total_tile_count += 1
                img_tile = image[:, y:y+tile_size, x:x+tile_size].copy() # 14 x 256 x 256
                mask_tile = mask_all[y:y+tile_size, x:x+tile_size].copy()
                
                # --- Logique de Filtrage (Inchangée) ---
                valid_mask_global = (img_tile != NODATA_SOURCE)
                valid_value_count = np.sum(valid_mask_global)

                is_mostly_nodata = valid_value_count / total_pixels_in_tile < MIN_VALID_PIXEL_PCT
                if is_mostly_nodata:
                    rejected_nodata_count += 1
                    continue
                # ---------------------------------------------
                # Filtrage de l'homogénéité
                # ---------------------------------------------
                DYNAMIC_THRESHOLD = 2
                is_dynamic_enough = False
                for b in range(BAND_COUNT): # BAND_COUNT = 14
                    valid_band_pixels = img_tile[b][img_tile[b] != NODATA_SOURCE]
                    if valid_band_pixels.size == 0:
                        continue
                    min_local = valid_band_pixels.min()
                    max_local = valid_band_pixels.max()
                    if (max_local - min_local) >= DYNAMIC_THRESHOLD:
                        is_dynamic_enough = True
                        break
                if not is_dynamic_enough:
                    rejected_homogeneous_count += 1
                    continue
                # ---------------------------------------------
                accepted_non_augmented_count += 1 # Tuile acceptée AVANT le dernier filtre de normalisation
                x_coord = raster_img.transform.c + x * res_x
                y_coord = raster_img.transform.f + y * res_y
                tile_transform = rasterio.transform.from_origin(x_coord, y_coord, res_x, res_y)

                # Itération sur les 6 versions (Originale + 5 Augmentées)
                for aug_index, (img_aug, mask_aug) in enumerate(augment_image_and_mask(img_tile, mask_tile)):
                    # -----------------------------------------------------------
                    # --- A. Traitement et Sauvegarde de l'IMAGE SAT ---
                    # -----------------------------------------------------------
                    img_aug_write = img_aug.copy() 
                    if GLOBAL_MIN is not None and GLOBAL_MAX is not None:
                        # 🚨 LA LOGIQUE EST APPLIQUÉE SUR LES 14 BANDES
                        for band_index in range(BAND_COUNT): 
                            band_data = img_aug_write[band_index]
                            valid_pixels_mask = (band_data != NODATA_SOURCE)
                            if not np.any(valid_pixels_mask):
                                continue

                            min_g = GLOBAL_MIN[band_index] # Utilise la stat de la bande (14)
                            max_g = GLOBAL_MAX[band_index] # Utilise la stat de la bande (14)
                            range_g = max_g - min_g
                            data_valid = band_data[valid_pixels_mask].astype(np.float32)

                            EPSILON = 0.001
                            range_g_stable = max(range_g, EPSILON)

                            if range_g_stable <= 0.0 or (np.max(data_valid) - np.min(data_valid)) < 1.0:
                                scaled_data = np.ones_like(data_valid, dtype=np.uint8)
                            else:
                                normalized_data = (data_valid - min_g) / range_g_stable
                                scaled_data = (normalized_data * 254.0) + 1.0
                                scaled_data = np.clip(scaled_data, 1, 255).astype(np.uint8)
                            img_aug_write[band_index][valid_pixels_mask] = scaled_data
                    # -----------------------------------------------------------
                    # FILTRAGE POST-NORMALISATION
                    # -----------------------------------------------------------
                    MAX_VALID_VALUE = 0
                    for b in range(BAND_COUNT):
                        band_data_uint8 = img_aug_write[b]
                        valid_pixels = band_data_uint8[band_data_uint8 > 0]
                        if valid_pixels.size > 0:
                            MAX_VALID_VALUE = max(MAX_VALID_VALUE, valid_pixels.max())
                    if MAX_VALID_VALUE <= 1:
                        if aug_index == 0:
                            rejected_all_one_count += 1
                        continue

                    stem = f"tile_{count:05d}_aug{aug_index}"
                    image_tile_path = output_img_dir / f"{stem}.tif"
                    mask_tile_path = output_mask_dir / f"{stem}.tif"

                    # --- Sauvegarde de l'Image Sat (UINT8 ÉCHELONNÉ) ---
                    with rasterio.open(
                        image_tile_path, 'w',
                        driver='GTiff',
                        height=tile_size, width=tile_size,
                        count=BAND_COUNT, # 🎯 ADAPTATION 5 : Écriture de 14 bandes
                        dtype=rasterio.uint8,
                        crs=crs,
                        transform=tile_transform,
                        nodata=NODATA_SOURCE 
                    ) as dst:
                        dst.write(img_aug_write)
                        if write_band_names:
                            dst.descriptions = tuple(band_names)
                    # -----------------------------------------------------------
                    # Simplification maximale du profil du masque (Inchangée)
                    # -----------------------------------------------------------
                    mask_dtype_forced = rasterio.uint8
                    mask_aug_write = mask_aug.astype(mask_dtype_forced)
                    mask_profile_simple = {
                        'driver': 'GTiff',
                        'height': tile_size,
                        'width': tile_size,
                        'count': 1,
                        'dtype': mask_dtype_forced, 
                        'crs': crs,
                        'transform': tile_transform,
                        'compress': 'NONE',
                        'tiled': False,
                        'colorinterp': ColorInterp.undefined, 
                    }
                    # Sauvegarde du Masque Binaire
                    with rasterio.open(
                        mask_tile_path, 'w',
                        **mask_profile_simple
                    ) as dst:
                        dst.write(mask_aug_write, 1)

                    # -----------------------------------------------------------

                    tile_id = f"{source_dir}/{stem}"
                    saved_tile_ids.append(tile_id)
                count += 1
    
    print(f"\n[INFO FILTRAGE {image_name_stem}] : Total de tuiles {tile_size}x{tile_size} potentielles: {total_tile_count}")
    print(f"[INFO FILTRAGE {image_name_stem}] : Rejetées car 'Mostly NoData' (ou moins de {MIN_VALID_PIXEL_PCT * 100}% de données valides, NoData={NODATA_SOURCE}): {rejected_nodata_count}")
    print(f"[INFO FILTRAGE {image_name_stem}] : Rejetées car 'Homogènes' (dynamique locale insuffisante < {DYNAMIC_THRESHOLD}): {rejected_homogeneous_count}")
    
    final_accepted_count = accepted_non_augmented_count - rejected_all_one_count
    
    print(f"[INFO FILTRAGE {image_name_stem}] : Tuiles ACCEPTÉES (avant normalisation/filtrage par écrasement): {accepted_non_augmented_count}")
    print(f"[INFO FILTRAGE {image_name_stem}] : Rejetées car 'Écrasées à 1' (Max valeur valide <= 1 après normalisation): {rejected_all_one_count}")
    print(f"** Nombre d'imagettes non-augmentées conservées : {final_accepted_count} **")
    print(f"** Nombre d'imagettes sauvegardées (augmentées * 6) : {len(saved_tile_ids)} **")

    return saved_tile_ids, count, final_accepted_count

# ---
# ------------------------------------------------------------------------------
# 3. Fonctions de Division, d'Équilibrage et de Statistiques (Inchangées)
# ------------------------------------------------------------------------------

def classify_and_balance_tiles(mask_dir_base, all_tile_ids, min_toiture_pct=20, max_toiture_pct=0.01) -> Tuple[List[str], List[int]]:
    """
    Classe les tuiles en 'fond pur' (<= max_toiture_pct) ou 'avec toiture' (>= min_toiture_pct),
    puis les équilibre 50/50.
    """
    fond_pur = []
    avec_toiture = []
    tuiles_intermediaires = []
    sample_count = 0
    print("\n[INFO CLASSIFICATION] : Échantillon de pourcentages de toiture (les 10 premières tuiles):")
    for tile_id in tqdm(all_tile_ids, desc="Classification des tuiles"):
        source_folder, stem = tile_id.split('/')
        mask_file = mask_dir_base / source_folder / f"{stem}.tif"

        try:
            with rasterio.open(mask_file) as src:
                mask = src.read(1)
                total_pixels = mask.size
                toiture_pixels = (mask == 1).sum()
                if total_pixels == 0:
                    continue

                percentage_toiture = (toiture_pixels / total_pixels) * 100
                if sample_count < 10:
                    print(f" - {stem}: {percentage_toiture:.4f}% de toiture")
                    sample_count += 1
                if percentage_toiture >= min_toiture_pct:
                    avec_toiture.append(tile_id)
                elif percentage_toiture <= max_toiture_pct:
                    fond_pur.append(tile_id)
                else:
                    tuiles_intermediaires.append(tile_id)

        except rasterio.errors.RasterioIOError:
            continue
        except AttributeError:
            continue

    n_toiture = len(avec_toiture)
    print(f"\nTuiles 'Avec Toiture' (>= {min_toiture_pct}%) : {n_toiture}")
    print(f"Tuiles 'Fond Pur' (<= {max_toiture_pct}%) disponibles : {len(fond_pur)}")
    print(f"Tuiles intermédiaires ignorées : {len(tuiles_intermediaires)}")

    if n_toiture == 0 or len(fond_pur) == 0:
        print("Aucune des deux classes n'est présente en quantité suffisante pour l'équilibrage.")
        return [], []

    np.random.seed(42)
    n_select = min(n_toiture, len(fond_pur))
    fond_pur_sel = np.random.choice(fond_pur, n_select, replace=False).tolist()

    print(f"[INFO ÉQUILIBRAGE] : Sélectionne {n_select} tuiles de 'Fond Pur' pour équilibrer les {n_toiture} tuiles 'Avec Toiture'.")
    all_sel_ids = fond_pur_sel + avec_toiture
    labels = [0] * len(fond_pur_sel) + [1] * len(avec_toiture)
    
    return all_sel_ids, labels

def split_and_copy(all_ids, labels, output_base_final, output_dir_all, val_ratio=0.2, random_seed=42):
    """
    Divise les tuiles sélectionnées en ensembles d'entraînement/validation de manière stratifiée
    et copie les fichiers.
    """
    train_ids, val_ids, _, _ = train_test_split(
        all_ids, labels, test_size=val_ratio, stratify=labels, random_state=random_seed
    )
    n_total = len(all_ids)
    n_train = len(train_ids)
    n_val = len(val_ids)
    print(f"[INFO SPLIT] : Total des tuiles équilibrées : {n_total}")
    print(f"[INFO SPLIT] : Taille du jeu d'entraînement (Train) : {n_train} tuiles ({n_train/n_total*100:.2f}%)")
    print(f"[INFO SPLIT] : Taille du jeu de validation (Val) : {n_val} tuiles ({n_val/n_total*100:.2f}%)")

    def copy_files(file_ids, split_name):
        img_src_dir = output_dir_all / "images"
        mask_src_dir = output_dir_all / "masks"
        for tile_id in tqdm(file_ids, desc=f"Copie des fichiers vers {split_name}"):
            source_folder, stem = tile_id.split('/')
            img_src = img_src_dir / source_folder / f"{stem}.tif"
            mask_src = mask_src_dir / source_folder / f"{stem}.tif"
            img_dest_dir = output_base_final / source_folder / split_name / "images"
            mask_dest_dir = output_base_final / source_folder / split_name / "masks"
            img_dest_dir.mkdir(parents=True, exist_ok=True)
            mask_dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_src, img_dest_dir / img_src.name)
            shutil.copy(mask_src, mask_dest_dir / mask_src.name)

    copy_files(train_ids, "train")
    copy_files(val_ids, "val")
    print(f"✅ Division et copie terminées : {len(train_ids)} pour l'entraînement / {len(val_ids)} pour la validation.")

def stats_split(base_path, min_toiture_pct=20):
    """
    Calcule les statistiques des tuiles et des pixels par base_path (train ou val).
    """
    n_tuiles = 0
    n_fond_pur = 0
    n_avec_toit = 0
    total_toiture_pixels = 0
    total_fond_pixels = 0
    for mf in base_path.rglob("masks/**/*.tif"):
        try:
            with rasterio.open(mf) as src:
                mask = src.read(1)
                n_tuiles += 1
                toiture_pixels = (mask == 1).sum()
                fond_pixels = (mask == 0).sum()
                total_toiture_pixels += toiture_pixels
                total_fond_pixels += fond_pixels
                total_pixels_mask = mask.size
                if total_pixels_mask > 0:
                    percentage_toiture = (toiture_pixels / total_pixels_mask) * 100
                else:
                    percentage_toiture = 0

                if percentage_toiture >= min_toiture_pct:
                    n_avec_toit += 1
                elif toiture_pixels == 0:
                    n_fond_pur += 1
        except rasterio.errors.RasterioIOError:
            continue
        except AttributeError:
            continue
    total_pixels = total_toiture_pixels + total_fond_pixels
    pct_toiture = 100 * total_toiture_pixels / total_pixels if total_pixels > 0 else 0
    print(f"📊 Tuiles — Total: {n_tuiles} | Fond pur (0%): {n_fond_pur} | Avec toiture (>= {min_toiture_pct}%): {n_avec_toit}")
    print(f"📊 Pixels — Toiture: {total_toiture_pixels} | Fond: {total_fond_pixels} | % Toiture globale: {pct_toiture:.2f}%")

# ---
# ==============================================================================
# 4. Fonction Principale d'Exécution
# ==============================================================================

if __name__ == "__main__":
    # --- Configuration ---
    image_dir_base = "chemin_image_sat"
    mask_dir_base = "chemin_masque_binaire"
    images_to_process = [
        "image_sat",
        "image_sat"
    ]
    masks_to_process = [
        "mask",
        "mask"
    ]
    
    # 🎯 ADAPTATION 6 : Nouvelle taille de tuile (256x256)
    tile_size = 256
    stride = 256 
    
    MIN_TOITURE_PCT_BALANCED = 2 
    MAX_TOITURE_PCT_BALANCED = 0

    output_base_temp = Path("dataset_temp_source")
    output_dir_all = output_base_temp / "all"
    
    # 📝 NOM DU DOSSIER DE SORTIE ADAPTÉ (14 bandes, AVEC RVB, 256x256)
    output_base_final = Path(f"dataset_source_{BAND_COUNT}bands_{tile_size}x{tile_size}_split_balanced_with_rgb") 

    # Nettoyage
    if output_base_temp.exists():
        print(f"Nettoyage du dossier temporaire {output_base_temp}")
        shutil.rmtree(output_base_temp)
    if output_base_final.exists():
        print(f"Nettoyage du dossier final {output_base_final}")
        shutil.rmtree(output_base_final)

    all_tile_ids = []
    count_offset = 0
    total_accepted_non_augmented = 0 # Compteur 1
    total_saved_augmented = 0 # Compteur 2

    print("\n--- Étape 1 : Génération et Augmentation de TOUTES les Tuiles Valides (UINT8 ÉCHELONNÉ avec NoData=0) ---")
    
    for img_name, mask_name in zip(images_to_process, masks_to_process):
        image_path = os.path.join(image_dir_base, img_name)
        mask_path = os.path.join(mask_dir_base, mask_name)
        
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"⚠️ Ignoré {img_name} car le fichier image ou masque est manquant.")
            continue
            
        try:
            # Récupère l'ID, le nouvel offset, et le compte d'imagettes non-augmentées acceptées
            ids, count_offset, accepted_count_single_file = create_tiles_and_save(
                image_path, mask_path, str(output_dir_all), tile_size, stride, count_offset
            )
            all_tile_ids.extend(ids)
            total_accepted_non_augmented += accepted_count_single_file
            total_saved_augmented += len(ids)
            print(f"[INFO PRINCIPAL] : Tuiles AUGMENTÉES générées pour {img_name}: {len(ids)}. Nouveau offset: {count_offset}")
        except ValueError as e:
            print(f"[ERREUR] Échec du traitement de {img_name}: {e}")
            continue 
            
    # Affichage des résultats globaux de l'étape 1
    print(f"\n--- RÉSULTATS ÉTAPE 1: SYNTHÈSE GLOBALE ---")
    print(f"1. Nombre total d'imagettes ACCEPTÉES (non-augmentées, après tous les filtres) : {total_accepted_non_augmented}")
    print(f"2. Nombre total d'imagettes SAUVEGARDÉES (augmentées * 6) : {total_saved_augmented}")
    
    mask_dir_base_all = output_dir_all / "masks"

    print(f"\n--- Étape 2 : Classification et Équilibrage (Fond Pur vs Toiture >= {MIN_TOITURE_PCT_BALANCED}%) ---")
    selected_ids, selected_labels = classify_and_balance_tiles(
        mask_dir_base_all, all_tile_ids,
        min_toiture_pct=MIN_TOITURE_PCT_BALANCED, 
        max_toiture_pct=MAX_TOITURE_PCT_BALANCED
    )

    if not selected_ids:
        print("❌ Processus interrompu faute de tuiles suffisantes pour l'équilibrage.")
    else:
        # Affichage du résultat de l'étape 2
        print(f"\n--- RÉSULTATS ÉTAPE 2: ÉQUILIBRAGE ---")
        print(f"3. Nombre total d'imagettes SÉLECTIONNÉES (après équilibrage 50/50) : {len(selected_ids)}")

        print("\n--- Étape 3 : Split Stratifié et Copie ---")
        split_and_copy(
            selected_ids, selected_labels, output_base_final,
            output_dir_all, val_ratio=0.2
        )

        print("\n--- Étape 4 : Statistiques Finales ---")
        source_folders = set(tile_id.split('/')[0] for tile_id in selected_ids)

        for source in source_folders:
            print(f"\n--- Stats Détaillées pour {source} ---")
            print(" --- Train ---")
            stats_split(output_base_final / source / "train", MIN_TOITURE_PCT_BALANCED)

            print(" --- Validation ---")
            stats_split(output_base_final / source / "val", MIN_TOITURE_PCT_BALANCED)

        print("\n--- Stats Globales (Train + Val) ---")
        stats_split(output_base_final, MIN_TOITURE_PCT_BALANCED)


    if output_base_temp.exists():
        print(f"\nNettoyage final du dossier temporaire {output_base_temp}.")
        shutil.rmtree(output_base_temp)
