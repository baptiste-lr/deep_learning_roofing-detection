# Scripts/postproc_infer.py
#
# Options possible à rajouter au cas où pas assez estétique :
#   - rajouter un lissage vectoriel (buffer +/− et simplify) après l’export shapefile
#--------------

from __future__ import annotations
from typing import List, Optional, Tuple, cast
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import binary_fill_holes
from skimage.morphology import (
    disk, binary_closing, binary_opening,
    remove_small_holes, remove_small_objects
)
from skimage.measure import regionprops, label as sklabel
import geopandas as gpd
from pyproj import CRS
from pathlib import Path

# --------- Types utiles ----------
BoolArr  = NDArray[np.bool_]
U8Arr    = NDArray[np.uint8]
F32Arr   = NDArray[np.float32]

# -----------------------
#  PREDICTION (simple/TTA)
# -----------------------

def predict_simple(model: torch.nn.Module, img_tensor: torch.Tensor, device: torch.device) -> F32Arr:
    with torch.no_grad():
        out: torch.Tensor = model(img_tensor.to(device))
        if out.ndim == 4 and out.shape[1] == 1:
            out = out.squeeze(1)
        elif out.ndim == 4:
            out = out[:, 0, :, :]
        proba = torch.sigmoid(out).squeeze().detach().cpu().numpy()
    return np.asarray(proba, dtype=np.float32)

def predict_tta(model: torch.nn.Module, img_tensor: torch.Tensor, device: torch.device) -> F32Arr:
    """Test-Time Augmentation (flips/rotations), inverse et moyenne."""
    aug = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[-1]),
        lambda x: torch.flip(x, dims=[-2]),
        lambda x: torch.rot90(x, 1, (-2, -1)),
        lambda x: torch.rot90(x, 2, (-2, -1)),
        lambda x: torch.rot90(x, 3, (-2, -1)),
    ]
    inv = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[-1]),
        lambda x: torch.flip(x, dims=[-2]),
        lambda x: torch.rot90(x, 3, (-2, -1)),
        lambda x: torch.rot90(x, 2, (-2, -1)),
        lambda x: torch.rot90(x, 1, (-2, -1)),
    ]
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for a, ia in zip(aug, inv):
            y: torch.Tensor = model(a(img_tensor.to(device)))
            if y.ndim == 4 and y.shape[1] == 1:
                y = y.squeeze(1)
            elif y.ndim == 4:
                y = y[:, 0, :, :]
            y = torch.sigmoid(y)
            y = ia(y)
            preds.append(y)
    proba = torch.stack(preds, dim=0).mean(0).squeeze().cpu().numpy()
    return np.asarray(proba, dtype=np.float32)

# -----------------------
#  THRESHOLDING
# -----------------------

def auto_threshold(proba: np.ndarray,
                   default_thr: float = 0.6,
                   use_otsu: bool = False,
                   no_roof_mean: float = 0.18,
                   no_roof_thr: float = 0.7) -> float:
    """Seuil de base + (option) Otsu + durcissement si tuile 'sans toits'."""
    thr: float = float(default_thr)
    if use_otsu:
        # Otsu via histogramme
        hist, edges = np.histogram(proba.ravel(), bins=256, range=(0, 1))
        p = hist.astype(np.float64) / (hist.sum() + 1e-12)
        omega = np.cumsum(p)
        mu = np.cumsum(p * np.arange(256))
        mu_t = mu[-1]
        sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
        k = int(np.nanargmax(sigma_b2))
        thr_otsu = float((edges[k] + edges[k + 1]) / 2.0)
        thr = max(thr, thr_otsu)
    if float(proba.mean()) < no_roof_mean:
        thr = max(thr, float(no_roof_thr))
    return thr

# -----------------------
#  MORPHO (scikit-image)
# -----------------------

def _gsd_from_affine(transform) -> float:
    # gsd moyenne (m/px)
    return float((abs(transform.a) + abs(transform.e)) / 2.0)

def postprocess_mask(
    bin_mask: np.ndarray,
    transform,
    close_radius_m: float = 1.5,
    open_radius_m:  float = 0.5,
    min_hole_area_m2: float = 9.0,
    min_obj_area_m2:  float = 16.0,
) -> U8Arr:
    """
    Closing → Opening → remove_small_holes/objects → Fill holes.
    Paramètres en mètres/m², convertis via la GSD du raster.
    Retourne toujours un ndarray uint8.
    """
    m_bool: BoolArr = np.asarray(bin_mask, dtype=bool)

    gsd = _gsd_from_affine(transform)
    r_close = max(0, int(round(close_radius_m / gsd)))
    r_open  = max(0, int(round(open_radius_m  / gsd)))
    min_hole_px = max(0, int(round(min_hole_area_m2 / (gsd * gsd))))
    min_obj_px  = max(0, int(round(min_obj_area_m2  / (gsd * gsd))))

    if r_close > 0:
        m_bool = binary_closing(m_bool, footprint=disk(r_close))
    if r_open > 0:
        m_bool = binary_opening(m_bool, footprint=disk(r_open))
    if min_hole_px > 0:
        m_bool = remove_small_holes(m_bool, area_threshold=min_hole_px)
    if min_obj_px > 0:
        m_bool = remove_small_objects(m_bool, min_size=min_obj_px)

    # 🔧 Pylance aime un cast explicite ici :
    m_bool = np.asarray(binary_fill_holes(m_bool), dtype=bool)

    m_u8: U8Arr = np.asarray(m_bool, dtype=np.uint8)
    return m_u8

# -----------------------
#  FILTRAGE PAR AIRES
# -----------------------

def parse_keep_ranges(spec: str) -> List[Tuple[float, Optional[float]]]:
    """
    Parse une chaîne: "3-12,80-350,400-"
    → [(3,12), (80,350), (400,None)]
    """
    out: List[Tuple[float, Optional[float]]] = []
    if not spec:
        return out
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            amin = float(a.strip()) if a.strip() else 0.0
            amax = float(b.strip()) if b.strip() else None
            out.append((amin, amax))
        else:
            v = float(tok)
            out.append((v, v))
    return out

def filter_by_area_ranges(mask: np.ndarray, transform,
                          ranges_m2: List[Tuple[float, Optional[float]]]) -> U8Arr:
    """
    Ne garde que les composantes connexes dont l'aire (m²) tombe dans au moins un intervalle.
    Retourne un ndarray uint8.
    """
    m_bool: BoolArr = np.asarray(mask, dtype=bool)

    # 🔧 Pylance: forcer ndarray (pas de tuple)
    labels_arr = sklabel(m_bool, connectivity=1, return_num=False)
    labels_arr = np.asarray(labels_arr)  # type: ignore[no-redef]
    # (labels_arr est bien un ndarray d’entiers)
    n_labels: int = int(labels_arr.max()) if labels_arr.size > 0 else 0
    if n_labels == 0:
        return np.zeros_like(m_bool, dtype=np.uint8)

    gsd = _gsd_from_affine(transform)
    px2m2 = gsd * gsd

    sizes_px: NDArray[np.int64] = np.bincount(labels_arr.ravel())
    areas_m2: NDArray[np.float64] = sizes_px.astype(np.float64) * px2m2  # idx 0 = fond

    keep_labs: List[int] = []
    for lab in range(1, n_labels + 1):
        a = float(areas_m2[lab])
        for (amin, amax) in ranges_m2:
            if (amax is None and a >= amin) or (amax is not None and amin <= a <= amax):
                keep_labs.append(lab)
                break

    if not keep_labs:
        return np.zeros_like(m_bool, dtype=np.uint8)

    keep_arr: NDArray[np.int_] = np.asarray(keep_labs, dtype=labels_arr.dtype)
    keep_mask: BoolArr = np.isin(labels_arr, keep_arr)

    return np.asarray(keep_mask, dtype=np.uint8)

# --- Vector smoothing (buffer +/- + simplify) ---
def _best_metric_crs(gdf: gpd.GeoDataFrame) -> CRS:
    """Retourne un CRS métrique (UTM) si le CRS courant est géographique."""
    crs = CRS.from_user_input(gdf.crs) if gdf.crs is not None else None
    if crs is not None and not crs.is_geographic:
        return crs  # déjà métrique
    # Estimation UTM depuis les bounds (en lon/lat)
    if crs is None or crs.is_geographic:
        # Si pas de CRS: on suppose WGS84 pour déterminer la zone
        if crs is None:
            gdf = gdf.set_crs(4326, allow_override=True)
        xmin, ymin, xmax, ymax = gdf.to_crs(4326).total_bounds
        lon = (xmin + xmax) / 2.0
        lat = (ymin + ymax) / 2.0
        zone = int((lon + 180) // 6) + 1
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        return CRS.from_epsg(epsg)
    return crs

def smooth_vector_file(
    in_path: str,
    buffer_m: float = 1.0,
    simplify_m: float = 0.5,
    out_path: str | None = None,
) -> str:
    """
    Lis un shapefile, reprojette en CRS métrique si besoin,
    applique buffer(+d) puis buffer(-d) (lissage), puis simplify.
    Écrit un nouveau shapefile (ou écrase si out_path=in_path).
    Retourne le chemin écrit.
    """
    gdf = gpd.read_file(in_path)
    if gdf.empty:
        return in_path

    src_crs = gdf.crs
    met_crs = _best_metric_crs(gdf)
    if src_crs is None or CRS.from_user_input(src_crs).to_string() != met_crs.to_string():
        gdf = gdf.to_crs(met_crs)

    # Buffer +/- (si buffer_m > 0)
    if buffer_m and buffer_m > 0:
        gdf["geometry"] = gdf.buffer(buffer_m)
        # Drop géométries vides/invalides
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
        gdf["geometry"] = gdf.buffer(-buffer_m)
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]

    # Simplify (si > 0)
    if simplify_m and simplify_m > 0:
        gdf["geometry"] = gdf.geometry.simplify(simplify_m, preserve_topology=True)
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]

    # Reprojette dans le CRS d’origine si défini
    if src_crs is not None and (CRS.from_user_input(gdf.crs).to_string() != CRS.from_user_input(src_crs).to_string()):
        gdf = gdf.to_crs(src_crs)

    # Détermine le chemin de sortie
    if out_path is None:
        p = Path(in_path)
        out_path = str(p.with_name(p.stem + "_smooth" + p.suffix))

    gdf.to_file(out_path)
    return out_path

def filter_by_shape_metrics(
    mask: np.ndarray,
    solidity_min: float = 0.85,
    extent_min: float = 0.40,
    ratio_max: float = 6.0,
) -> np.ndarray:
    """
    Garde les composantes connexes qui ressemblent à des toitures :
      - solidity_min : aire / aire de l’enveloppe convexe (∈ [0,1])
      - extent_min   : aire / aire de la bbox (∈ [0,1])
      - ratio_max    : max(h/w, w/h) ≤ ratio_max
    Retourne un masque uint8 filtré.
    """
    lab = sklabel(mask.astype(bool), connectivity=1)
    keep = np.zeros_like(mask, dtype=bool)

    for r in regionprops(lab):
        if r.area <= 0:
            continue
        sol = float(r.solidity) if r.solidity is not None else 0.0
        ext = float(r.extent)   if r.extent   is not None else 0.0

        minr, minc, maxr, maxc = r.bbox
        h = max(1, maxr - minr)
        w = max(1, maxc - minc)
        ratio = max(h / w, w / h)

        if (sol >= solidity_min) and (ext >= extent_min) and (ratio <= ratio_max):
            keep[lab == r.label] = True

    return keep.astype(np.uint8)

# -----------------------
#  TILING (AJOUT CRITIQUE)
# -----------------------

def sliding_windows(
    height: int, width: int, tile: int, step: int
) -> List[Tuple[int, int, int, int]]:
    """
    Génère les coordonnées (ligne, colonne, hauteur, largeur) des tuiles
    avec chevauchement (step < tile).
    """
    windows = []
    # Boucle sur les lignes (y)
    for r in range(0, height, step):
        # Boucle sur les colonnes (x)
        for c in range(0, width, step):
            # Calcul de la hauteur et largeur réelles de la tuile
            # (pour gérer les bords)
            h = min(tile, height - r)
            w = min(tile, width - c)
            
            # Ajustement si on est trop près du bord et que h < tile
            if h < tile and height - r != h:
                r_start = max(0, height - tile)
                h = tile
            else:
                r_start = r

            if w < tile and width - c != w:
                c_start = max(0, width - tile)
                w = tile
            else:
                c_start = c
            
            # Ajout du window (r_start, c_start, h, w)
            windows.append((r_start, c_start, h, w))
            
            # Évite de boucler à l'infini si step >= width ou si on est au bord
            if c_start + tile >= width:
                break
        
        # Évite de boucler à l'infini si step >= height
        if r_start + tile >= height:
            break

    # Rétention des tuiles uniques (important si on a bougé pour ajuster le bord)
    unique_windows = sorted(list(set(windows)))
    return unique_windows