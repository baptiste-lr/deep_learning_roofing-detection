from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import shape as shapely_shape
#------------------------------------
#
def save_checkpoint(model, path):
    """
    Sauvegarde l'état du modèle PyTorch dans un fichier.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def plot_losses(train_losses, val_losses, loss_name, backbone, best_val_loss, output_dir):
    """
    Trace et sauvegarde la courbe des pertes d'entraînement et de validation.

    Args:
        train_losses (list): Liste des pertes d'entraînement.
        val_losses (list): Liste des pertes de validation.
        loss_name (str): Nom de la loss utilisée.
        backbone (str): Nom du backbone (ex: 'resnet18').
        best_val_loss (float): Meilleure perte de validation.
        output_dir (Path or str): Dossier où sauvegarder l'image.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.title(f"Courbes de loss — {backbone} + {loss_name}\nBest val: {best_val_loss:.4f}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    save_path = Path(output_dir) / f"loss_plot_{backbone}_{loss_name}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Courbe de loss sauvegardée: {save_path}")

def normalize_image(img_array: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """
    [MODIFICATION CRITIQUE] Adapte l'image uint8 [0, 255] (sortie du pré-traitement) 
    au format float32 [0.0, 1.0] pour PyTorch. 
    Les arguments 'means' et 'stds' sont ignorés car la normalisation est déjà faite
    vers [1, 255] et nous faisons une re-normalisation simple.

    Args:
        img_array (np.ndarray): Image uint8 [0, 255] (C, H, W). Le 0 est le NoData.
        means (np.ndarray): Ignoré.
        stds (np.ndarray): Ignoré.

    Returns:
        np.ndarray: Image re-normalisée en float32 [0.0, 1.0].
    """
    # 1. Conversion du type de donnée vers float32
    # C'est obligatoire pour le Deep Learning, même si les valeurs sont déjà 'scaled'.
    img_array = img_array.astype(np.float32)
    
    # 2. Mise à l'échelle vers la plage [0, 1]
    # Ceci convertit les valeurs [0, 255] en [0.0, 1.0].
    # CRITIQUE: La valeur NoData (0) reste 0.0, et toutes les données valides [1, 255]
    # sont converties en une petite plage flottante [1/255.0, 1.0].
    img_array /= 255.0
    
    return img_array

def save_visual_mask_geotiff(mask: np.ndarray, transform: Affine, crs, output_path: str):
    """
    Sauvegarde un masque binaire (0 ou 1) comme image GeoTIFF 8-bit pour la visualisation.
    
    Le masque est mis à l'échelle de [0, 1] à [0, 255] pour garantir qu'il soit visible.

    Args:
        mask (np.ndarray): Tableau 2D binaire (0 ou 1).
        transform (Affine): Matrice de transformation raster -> coordonnées.
        crs: Système de coordonnées (rasterio.crs.CRS).
        output_path (str): Chemin de sauvegarde du GeoTIFF (.tif).
    """
    # Assurer que le masque est un tableau 2D
    if mask.ndim != 2:
        raise ValueError("Le masque doit être un tableau 2D.")
        
    # CORRECTION POUR VISUALISATION : Mettre à l'échelle [0, 1] -> [0, 255]
    visual_mask = (mask * 255).astype(np.uint8)

    # Définir le profil du GeoTIFF
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',              # Format 8-bit pour la visualisation (0-255)
        'nodata': 0,                   # Valeur NoData
        'width': mask.shape[1],
        'height': mask.shape[0],
        'count': 1,                    # Une seule bande (bande de masque)
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }

    # Sauvegarde du GeoTIFF
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            # rasterio attend (Bande, Hauteur, Largeur). Nous avons (Hauteur, Largeur).
            dst.write(visual_mask, 1)
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde du GeoTIFF de visualisation: {e}")


def save_binary_mask_geotiff(mask: np.ndarray, transform: Affine, crs, output_path: str):
    """
    Sauvegarde un masque binaire (valeurs 0 ou 1) comme GeoTIFF 8-bit.
    
    Ce masque est destiné à l'analyse et n'est PAS mis à l'échelle (valeurs 0 ou 1).

    Args:
        mask (np.ndarray): Tableau 2D binaire (0 ou 1).
        transform (Affine): Matrice de transformation raster -> coordonnées.
        crs: Système de coordonnées (rasterio.crs.CRS).
        output_path (str): Chemin de sauvegarde du GeoTIFF (.tif).
    """
    # Assurer que le masque est un tableau 2D
    if mask.ndim != 2:
        raise ValueError("Le masque doit être un tableau 2D.")
        
    # Conversion en uint8 sans mise à l'échelle (conserve les 0 et 1)
    binary_mask = mask.astype(np.uint8)

    # Définir le profil du GeoTIFF
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',              # Format 8-bit, valeurs 0 ou 1
        'nodata': 0,
        'width': mask.shape[1],
        'height': mask.shape[0],
        'count': 1,                    # Une seule bande
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }

    # Sauvegarde du GeoTIFF
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            # rasterio attend (Bande, Hauteur, Largeur). Nous avons (Hauteur, Largeur).
            dst.write(binary_mask, 1)
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde du GeoTIFF binaire: {e}")

def save_mask_as_shapefile(mask: np.ndarray, transform: Affine, crs, output_path: str):
    """
    Convertit un masque binaire en polygones et les enregistre dans un fichier shapefile.

    Args:
        mask (np.ndarray): Tableau 2D avec des valeurs 0 ou 1.
        transform (Affine): Matrice de transformation raster -> coordonnées.
        crs: Système de coordonnées (rasterio.crs.CRS).
        output_path (str): Chemin de sauvegarde du shapefile (.shp).
    """
    if not np.any(mask == 1):
        print("ℹ️ Aucun pixel à 1 dans le masque. Pas de polygones à générer.")
        return

    mask = mask.astype(np.uint8)

    # Consommation du générateur shapes en une vraie liste de géométries shapely
    polygons = [
        shapely_shape(geom)
        for geom, val in shapes(mask, mask=mask == 1, transform=transform)
        if val == 1 # On garde uniquement les régions correspondant à la valeur 1
    ]
    
    # Création du GeoDataFrame
    if polygons:
        gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=crs)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_path)
    else:
        print("ℹ️ Le masquage des formes a éliminé tous les polygones. Pas de shapefile généré.")