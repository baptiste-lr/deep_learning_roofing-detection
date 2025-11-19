import os
import json
import rasterio
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Tuple, List, Union

from utils import normalize_image 

# Définit la valeur maximale pour la mise à l'échelle (pour les données UINT8)
MAX_PIXEL_VALUE = 255.0

class TileDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, mean_std_path: Path, image_reference_paths: list):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.mean_std_path = mean_std_path
        self.image_reference_paths = image_reference_paths
        
        self.images = [f for f in images_dir.iterdir() if f.is_file()]
        self.masks = [f for f in masks_dir.iterdir() if f.is_file()]
        self.images.sort()
        self.masks.sort()

        if len(self.images) == 0:
            raise FileNotFoundError(f"Aucune image trouvée dans le dossier: {self.images_dir}")
        if len(self.images) != len(self.masks):
            raise ValueError("Le nombre d'images et de masques ne correspond pas.")

        self.n_channels = self._get_n_channels()
        # Charge ou calcule les statistiques Z-score.
        self.means, self.stds = self._load_or_compute_means_stds(self.mean_std_path, self.image_reference_paths)
        
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[None, None]]:
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        try:
            with rasterio.open(img_path) as src_img:
                # ÉTAPE 1: Chargement de l'image et détermination du NoData
                img_array = src_img.read().astype(np.float32)
                nodata_value = src_img.nodata
                
                # Crée un masque pour les zones NoData
                if nodata_value is not None:
                    # Le masque identifie tous les pixels correspondant à la valeur NoData
                    nodata_mask = (img_array == nodata_value)
                else:
                    nodata_mask = np.zeros_like(img_array, dtype=bool)


            # ÉTAPE 2: Normalisation [0, 1] et Z-score
            # Mise à l'échelle [0, 1]
            img_scaled = img_array / MAX_PIXEL_VALUE 
            
            # Application de la normalisation Z-score: (x - mean) / std
            # Cette fonction est supposée être dans utils.py
            img_normalized = normalize_image(img_scaled, self.means, self.stds)
            
            # ÉTAPE 3: FIX CRITIQUE: Masquage du NoData après Z-score
            # Les zones NoData (qui sont des outliers après Z-score) sont remises à zéro
            if np.any(nodata_mask):
                img_normalized[nodata_mask] = 0.0
            
            # Vérification de sécurité pour les tuiles vides (optionnel, mais recommandé)
            if np.sum(np.abs(img_normalized)) < 1e-6:
                return None, None 

            with rasterio.open(mask_path) as src_mask:
                # Le masque doit être chargé en mémoire
                mask_array = src_mask.read(1)
            
            # ÉTAPE 4: Conversion en tenseurs
            img_tensor = torch.from_numpy(img_normalized).float()
            
            # Le masque pour la WeightedBCEDiceLoss a besoin du type FLOAT pour la partie BCE
            mask_tensor = torch.from_numpy(mask_array).float()
            
            # NOTE: Si vous utilisez la CrossEntropyLoss ou un modèle multiclasse, 
            # vous devrez repasser le masque en .long()
            
            return img_tensor, mask_tensor
            
        except Exception as e:
            # En cas de lecture de fichier corrompu
            print(f"Erreur de chargement pour {img_path.name}: {e}")
            return None, None
    
    def _get_n_channels(self) -> int:
        """Détermine le nombre de canaux à partir de la première image."""
        if not self.images:
            return 0
        with rasterio.open(self.images[0]) as src:
            return src.count
            
    def _load_or_compute_means_stds(self, mean_std_path: Path, image_reference_paths: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Charge les moyennes et écarts-types ou les calcule.
        """
        
        if mean_std_path.exists():
            print("📊 Chargement des moyennes et écarts-types depuis le fichier existant.")
            with open(mean_std_path, 'r') as f:
                data = json.load(f)
                means = np.array(data['means'])
                stds = np.array(data['stds'])
            return means, stds
        
        print("📊 Calcul des moyennes et écarts-types depuis les images de référence...")
        
        all_pixels = []
        for path in image_reference_paths:
            if not path.exists():
                raise FileNotFoundError(f"L'image de référence est manquante: {path}")

            with rasterio.open(path) as src:
                # NOTE: Si l'image de référence contient du NoData, cette fonction
                # DOIT le masquer pour que les stats soient précises. Je suppose que 
                # vos stats sont déjà calculées sur des pixels valides.
                pixels = src.read().astype(np.float64) / MAX_PIXEL_VALUE 
                pixels = pixels.reshape(pixels.shape[0], -1)
                all_pixels.append(pixels)
        
        all_pixels = np.concatenate(all_pixels, axis=1)

        # Effectue les calculs en float64
        means = np.mean(all_pixels, axis=1, dtype=np.float64)
        stds = np.std(all_pixels, axis=1, dtype=np.float64)
        
        # Ajout d'epsilon pour éviter la division par zéro
        epsilon = 1e-8
        stds = np.where(stds == 0, epsilon, stds)
        
        # Sauvegarde
        mean_std_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mean_std_path, 'w') as f:
            json.dump({'means': means.tolist(), 'stds': stds.tolist()}, f, indent=2)

        print(f"✅ Statistiques calculées et sauvegardées dans {mean_std_path.name}")
        return means.astype(np.float32), stds.astype(np.float32)