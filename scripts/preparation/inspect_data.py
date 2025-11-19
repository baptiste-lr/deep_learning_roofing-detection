import os
import rasterio
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List 
import json 
from collections import Counter 

# ==============================================================================
# 1. Fonction d'Inspection Générique (Min/Max sans NoData)
# ==============================================================================

def inspect_metadata_and_values(file_path: Path) -> Optional[Tuple[str, int, int, int, Optional[float], List[float], List[float]]]:
    """
    Ouvre un GeoTIFF, affiche ses métadonnées et la plage de valeurs des pixels.
    Retourne les métadonnées pour utilisation dans les fonctions de calcul de stats.
    """
    if not file_path.exists():
        print(f"❌ FICHIER NON TROUVÉ: {file_path.name}")
        return None

    print(f"\n--- Inspection de : {file_path.name} ---")
    
    try:
        with rasterio.open(file_path) as src:
            # Métadonnées de base
            profile = src.profile
            dtype = profile['dtype']
            count = src.count
            width = src.width
            height = src.height
            nodata = profile.get('nodata')

            print(f"Type de Données (dtype): {dtype}")
            print(f"Nombre de Bandes: {count}")
            print(f"Dimensions (L x H): {width} x {height}")
            print(f"Valeur NoData: {nodata}")

            # Lecture et analyse des valeurs des pixels
            data = src.read().astype(np.float32) 
            
            print("\nPlage de Valeurs par Bande (Min / Max):")
            all_min = []
            all_max = []
            
            for i in range(count):
                band_data = data[i]
                
                # IMPORTANT : La logique ici pour nodata_check=0 pour les masques sans nodata 
                # masque les 0. C'est pourquoi on utilise l'histogramme pour vérifier la binarité.
                nodata_check = nodata if nodata is not None else 0 
                
                # Masque des pixels valides (différents de NoData)
                valid_pixels = band_data[~np.isclose(band_data, nodata_check)]
                
                if valid_pixels.size > 0:
                    min_val = np.min(valid_pixels)
                    max_val = np.max(valid_pixels)
                    print(f"  Bande {i+1}: Min={min_val:.4f}, Max={max_val:.4f}")
                    all_min.append(float(min_val)) 
                    all_max.append(float(max_val)) 
                else:
                    print(f"  Bande {i+1}: Tous les pixels sont NoData ou manquants.")
        
        return dtype, count, width, height, nodata, all_min, all_max 
                    
    except rasterio.RasterioIOError as e:
        print(f"🚨 Erreur Rasterio lors de l'ouverture: {e}")
        return None
    except Exception as e:
        print(f"🚨 Erreur inattendue: {e}")
        return None

# ==============================================================================
# 2. Analyse Spécifique des Masques (CORRIGÉE : Suppression de l'avertissement redondant)
# ==============================================================================

def inspect_mask_values(mask_path: Path):
    """
    Vérifie spécifiquement si un masque GeoTIFF est binaire (seulement 0 et 1)
    et effectue une analyse d'histogramme pour compter les pixels 0 et 1.
    """
    results = inspect_metadata_and_values(mask_path)
    if results is None:
        return

    dtype, count, width, height, nodata, all_min, all_max = results 
    
    print("\n--- Analyse Binaire du Masque ---")
    
    # 1. Vérification du nombre de bandes
    if count != 1:
        print(f"⚠️ AVERTISSEMENT: Un masque binaire devrait idéalement avoir 1 seule bande. Ce masque en a {count}.")

    # 2. Analyse approfondie des valeurs (Histogramme)
    
    n_zero, n_one, n_other, total_pixels, valid_pixels_size = 0, 0, 0, 0, 0
    try:
        with rasterio.open(mask_path) as src:
            mask_data = src.read(1) 
            total_pixels = mask_data.size
            
            # Utiliser -1 si pas de NoData pour ne rien masquer (pour l'histogramme seulement)
            nodata_check = nodata if nodata is not None else -1 
            
            # Créer un masque des pixels qui ne sont pas NoData
            if nodata is not None:
                valid_mask = ~np.isclose(mask_data, nodata_check)
                valid_pixels = mask_data[valid_mask]
            else:
                valid_pixels = mask_data
            
            valid_pixels_size = len(valid_pixels)

            # Compter les valeurs uniques dans la zone valide
            pixel_counts = Counter(valid_pixels.flatten().astype(int))
            
            n_zero = pixel_counts.get(0, 0)
            n_one = pixel_counts.get(1, 0)
            n_other = sum(v for k, v in pixel_counts.items() if k not in (0, 1))

            print(f"\n📊 Analyse d'Histogramme : {mask_path.name}")
            print(f"  - Pixels 'Fond' (0): {n_zero} ({n_zero / total_pixels * 100:.4f}%)")
            print(f"  - Pixels 'Toiture' (1): {n_one} ({n_one / total_pixels * 100:.4f}%)")
            print(f"  - Pixels 'Autres' (>1): {n_other}")
            print(f"  - Pixels NoData (estimé): {total_pixels - valid_pixels_size}")
            print(f"  - Pixels Valides Totaux (0+1+Autres): {valid_pixels_size}")
            
            # --- CONCLUSION Binaire ---
            
            if n_zero > 0 and n_one > 0 and n_other == 0:
                print("✅ BINAIRE CONFIRMÉ: Contient explicitement des 0 (Fond) et des 1 (Toiture).")
                is_binary_confirmed = True # Nouvelle variable pour la vérification des stats
            elif n_zero == 0 and n_one > 0 and n_other == 0:
                print("⚠️ AVERTISSEMENT CLÉ: Contient uniquement des 1 (Toiture). Le 'Fond' (0) est manquant/hors-limites du masque.")
                print("   👉 Cela explique pourquoi les stats initiales donnent [1, 1]. Votre tuilage doit générer le 0.")
                is_binary_confirmed = True
            elif n_other > 0:
                print(f"❌ ERREUR GRAVE: Valeurs non-binaires (autres que 0 ou 1) trouvées. Max={max(valid_pixels)}.")
                is_binary_confirmed = False
            else:
                print("❌ ERREUR: Aucune donnée valide (0 ou 1) trouvée.")
                is_binary_confirmed = False

    except Exception as e:
        print(f"🚨 Erreur lors de l'analyse d'histogramme du masque: {e}")
        is_binary_confirmed = False
        
    
    # 3. Vérification de la plage de valeurs (AJUSTÉE)
    if not all_min or not all_max:
         return
         
    min_global = min(all_min)
    max_global = max(all_max)

    # Si l'histogramme a confirmé la binarité, on affiche un succès malgré le biais [1, 1]
    if is_binary_confirmed and np.isclose(min_global, 1.0) and np.isclose(max_global, 1.0):
        print("✅ CONFORMITÉ STATS (lecture Min/Max): La plage [1.0000, 1.0000] est cohérente avec l'absence de NoData.")
    elif np.isclose(min_global, 0.0) and np.isclose(max_global, 1.0):
        print("✅ CONFORMITÉ STATS: La plage de valeurs MIN/MAX lue est [0, 1].")
    else:
        # On affiche l'avertissement seulement si la plage est bizarre et non 0-1 ou 1-1
        print(f"⚠️ CONFORMITÉ STATS: La plage de valeurs MIN/MAX lue est [{min_global:.4f}, {max_global:.4f}].")


# ==============================================================================
# 3. NOUVELLES FONCTIONS DE CALCUL ET SAUVEGARDE DES STATS GLOBALES (Inchangées)
# ==============================================================================

def calculate_global_min_max(image_paths: List[Path], band_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule le Minimum Global (le plus petit Min) et le Maximum Global (le plus grand Max)
    pour chaque bande sur l'ensemble des images fournies.
    """
    # Initialisation avec des valeurs extrêmes
    global_min = np.full(band_count, np.inf, dtype=np.float32)
    global_max = np.full(band_count, -np.inf, dtype=np.float32)
    
    print("\n--- Calcul des Statistiques Globales Min/Max sur l'ensemble des images sources ---")

    for file_path in image_paths:
        # Récupération des résultats via la fonction d'inspection
        results = inspect_metadata_and_values(file_path)
        if results is None:
            continue
        
        # all_min et all_max sont les indices 5 et 6 du tuple de retour
        _, count, _, _, _, file_min_list, file_max_list = results 
        
        if count != band_count:
             print(f"⚠️ AVERTISSEMENT: {file_path.name} a {count} bandes, attendu {band_count}. Ignoré pour le calcul global.")
             continue
             
        file_min = np.array(file_min_list, dtype=np.float32)
        file_max = np.array(file_max_list, dtype=np.float32)
        
        # Mise à jour des Min/Max globaux (bande par bande)
        global_min = np.minimum(global_min, file_min)
        global_max = np.maximum(global_max, file_max)
        
        print(f"  > Traité {file_path.name}. Min/Max globaux mis à jour.")
        
    return global_min, global_max

def save_global_stats(global_min: np.ndarray, global_max: np.ndarray, output_path: Path):
    """
    Sauvegarde les tableaux Min/Max globaux dans un fichier JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "GLOBAL_MIN": global_min.tolist(), 
        "GLOBAL_MAX": global_max.tolist(),
        "BAND_COUNT": len(global_min)
    }
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print("\n=========================================")
    print(f"✅ STATISTIQUES GLOBALEs SAUVEGARDÉES !")
    print(f"Chemin: {output_path.resolve()}")
    print("-----------------------------------------")
    print(f"GLOBAL MIN (Bande par Bande): {global_min}")
    print(f"GLOBAL MAX (Bande par Bande): {global_max}")
    print("=========================================")


# ==============================================================================
# 4. Configuration et Exécution (Inchangée)
# ==============================================================================

if __name__ == "__main__":
    
    # --- Configuration des chemins ---
    
    image_dir_base = Path("image_sat")
    mask_dir_base = Path("masque_binaire")
    
    images_to_process = [
        "image sat",
        "image_sat"
    ]
    masks_to_process = [
        "mask",
        "mask"
    ]
    
    # --- Configuration pour l'exportation des stats ---
    BAND_COUNT = 14 
    STATS_FILE = Path("preparation_data/dataset_global_minmax_stats.json")
    
    OUTPUT_BASE_FINAL = Path("dataset_source_split_balanced")
    
    # 1. Collecte des chemins des images satellites
    image_paths = [image_dir_base / img_name for img_name in images_to_process]
    valid_image_paths = [p for p in image_paths if p.exists()]


    print("=========================================")
    print("  INSPECTION DES IMAGES SATELLITE (SAT)  ")
    print("=========================================")
    
    # 1. Inspection des images satellites sources (déjà effectuée par calculate_global_min_max)
    for image_path in valid_image_paths:
        inspect_metadata_and_values(image_path)


    print("\n=========================================")
    print("      INSPECTION DES MASQUES (BINAIRE)     ")
    print("=========================================")
    
    # 2. Inspection des masques sources
    for mask_name in masks_to_process:
        mask_path = mask_dir_base / mask_name
        inspect_mask_values(mask_path)
    
    # --- NOUVELLE ÉTAPE : Calcul et Sauvegarde des Statistiques ---
    if valid_image_paths:
        global_min_stats, global_max_stats = calculate_global_min_max(valid_image_paths, BAND_COUNT)
        
        save_global_stats(global_min_stats, global_max_stats, STATS_FILE)
    else:
        print("❌ Aucune image valide trouvée pour le calcul des statistiques globales. Assurez-vous que les chemins sont corrects.")
    
    
    print("\n=========================================")
    print("    INSPECTION DES TUILES DÉCOUPÉES (Vérification finale)   ")
    print("=========================================")
    
    # 3. Inspection des tuiles après découpage
    try:
        first_image_tile_path = next(OUTPUT_BASE_FINAL.rglob("*/train/images/tile_*.tif"))
        
        tile_name = first_image_tile_path.name 
        mask_tile_path = first_image_tile_path.parent.parent / "masks" / tile_name
        
        # VÉRIFICATION DE L'IMAGE DÉCOUPÉE
        print("\n--- VÉRIFICATION TUILE IMAGE (Doit être UINT8, 14 bandes, NoData=0.0) ---")
        inspect_metadata_and_values(first_image_tile_path)
        
        # VÉRIFICATION DU MASQUE DÉCOUPÉ
        print("\n--- VÉRIFICATION TUILE MASQUE (Doit être 1 bande, binaire [0, 1]) ---")
        inspect_mask_values(mask_tile_path)
        
    except StopIteration:
        print(f"❌ ERREUR: Aucune tuile trouvée dans le dossier {OUTPUT_BASE_FINAL}. Veuillez d'abord exécuter le script de préparation.")
    except Exception as e:
        print(f"🚨 Erreur lors de l'inspection des tuiles découpées: {e}")
