# Détection de Bâtiments par Segmentation Sémantique avec UNet++

Ce dépôt contient une implémentation complète d'un modèle **UNet++** pour la détection de bâtiments sur des images satellites. Le projet inclut la préparation des données, l'entraînement, l'inférence et le post-traitement.

---

## Structure du Projet

```
deep_learning_roofing-detection/
├── data/
│   ├── images/       # Images satellites d'entrée (format GeoTIFF)
│   └── masks/        # Masques de vérité terrain (format GeoTIFF)
├── scripts/
│   ├── preparation
│        ├── prepare_data.py          # Script de préparation des données
│        ├── inspect_data.py         # Script d'inspection des données
│   ├── modèle
│       ├── train_script.py          # Script d'entraînement du modèle
│       ├── inference_script_imagettes.py  # Script d'inférence sur des imagettes
│       ├── inference_script_image_sat.py  # Script d'inférence sur des images satellites complètes
│       ├── utils.py                # Fonctions utilitaires (visualisation, post-traitement)
│       ├── dataset.py              # Préparation et chargement des données
│       ├── model.py                # Définition du modèle UNet++
│       ├── losses.py               # Fonctions de perte personnalisées
│       └── postproc_infer.py       # Post-traitement des prédictions
├── README.md
└── requirements.txt
```

---

## Jeu de Données

### Description
- **Images** : Images satellites Pléiades THRS 50cm de résolution en format GeoTIFF (4 bandes RVB et PIR et 10 néo canaux C3, L, NDVI, MSAVI, MNDWI, ExG, BII, UAI, Texture_R_3x3, Texture_PIR_3x3).
- **Masques** : Masques binaires (1 = bâtiment, 0 = fond) au même format et résolution que les images (Réalisation du masque bianire dnas mon dernier reposit : https://github.com/baptiste-lr/Classification_XGBoost) Le masque est issus de l'extraction des classes habitations de la classification produite.

### Organisation
- Les images et masques doivent être placés dans `data/images/` et `data/masks/` respectivement.
- Chaque image doit avoir un masque correspondant avec le même nom de fichier.

### Exemple
```
data/
├── images/
│   ├── image_001.tif
│   └── image_002.tif
└── masks/
    ├── image_001.tif
    └── image_002.tif
```

### Préparation des Données
Le script `dataset.py` permet de :
- Diviser les données en ensembles d'entraînement/validation/test.
- Appliquer des augmentations (rotation, flip, etc.).

---

## Installation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/[ton_username]/deep_learning_roofing-detection.git
   cd deep_learning_roofing-detection
   ```

2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

### Dépendances
- Python 3.8+
- PyTorch
- segmentation_models_pytorch
- rasterio
- geopandas
- shapely
- numpy
- matplotlib
- tqdm

---

## Utilisation

### Entraînement
Pour entraîner le modèle :
```bash
python scripts/train_script.py --data-dir ./data/
```
- `--data-dir` : Chemin vers le dossier contenant les images et masques.

### Inférence
Pour effectuer une prédiction sur une image satellite :
```bash
python scripts/inference_script_image_sat.py --data-dir ./data/ --output-dir ./results/
```
- `--output-dir` : Dossier de sortie pour les prédictions.

### Post-traitement
Le script `postproc_infer.py` permet de :
- Convertir les prédictions en polygones (format Shapefile).
- Appliquer un lissage pour améliorer la qualité des contours.

---

## Résultats

### Exemple de Sortie
- Les prédictions sont sauvegardées sous forme d'images et de fichiers Shapefile dans le dossier spécifié (`--output-dir`).
- Utilise `utils.py` pour visualiser les résultats :
  ```python
  from utils import plot_prediction
  plot_prediction("image_001.tif", "prediction_001.tif")
  ```

### Métriques
- Le modèle utilise la **loss Dice** et la **loss Focale** pour optimiser la segmentation.
- Métriques calculées : IoU (Intersection over Union), F1-score.

---
