# Fichier: losses.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from scipy.ndimage import distance_transform_edt


# -----------------------------
# Helpers de forme / numérique
# -----------------------------
def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    # Attendu (B,1,H,W). Si (B,H,W), on ajoute la dim canal.
    if x.ndim == 3:
        x = x.unsqueeze(1)
    return x

def _flatten_for_metrics(p: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return p.view(-1), t.view(-1)


# -----------------------------
# Focal Loss (binaire) - AJOUT DE POS_WEIGHT
# -----------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss binaire.
    - from_logits=True : attend des logits (conseillé)
    - pos_weight : Poids de la classe positive dans la BCE interne.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean", from_logits: bool = True, pos_weight: float | torch.Tensor = 1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction
        self.from_logits = from_logits
        # Conversion du pos_weight en Tensor si un float est passé
        if isinstance(pos_weight, (int, float)):
            self.pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)
        else:
            self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = _ensure_bchw(inputs).float()
        targets = _ensure_bchw(targets).float()

        # On s'assure que le pos_weight est sur le bon device
        pos_weight_tensor = self.pos_weight.to(inputs.device)

        if self.from_logits:
            # BCE sur logits, par pixel, avec pondération positive
            bce = F.binary_cross_entropy_with_logits(
                inputs, 
                targets, 
                reduction='none',
                pos_weight=pos_weight_tensor.expand(targets.shape) # Expand pour correspondre à la taille
            )
            pt = torch.exp(-bce)
            loss = self.alpha * (1 - pt) ** self.gamma * bce
        else:
            # Inputs = probabilités
            inputs = inputs.clamp(1e-6, 1 - 1e-6)
            bce = F.binary_cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-bce)
            loss = self.alpha * (1 - pt) ** self.gamma * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# -----------------------------
# Tversky Loss
# -----------------------------
class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6, from_logits: bool = True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.from_logits = from_logits
        print(f"TverskyLoss | alpha={alpha} | beta={beta}")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = _ensure_bchw(inputs).float()
        targets = _ensure_bchw(targets).float()
        # Utilisation directe de sigmoid, plus robuste
        probs = torch.sigmoid(inputs) if self.from_logits else inputs.clamp_(1e-6, 1 - 1e-6)

        TP = (probs * targets).sum(dim=(1, 2, 3))
        FP = ((1 - targets) * probs).sum(dim=(1, 2, 3))
        FN = (targets * (1 - probs)).sum(dim=(1, 2, 3))

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return (1 - tversky).mean()


# -----------------------------
# BCE + Dice Loss (binaire)
# -----------------------------
class BCEDiceLoss(nn.Module):
    """
    Combinaison BCE (logits) + Dice (sur probabilités).
    bce_weight : poids de la BCE dans la somme.
    """
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-6, from_logits: bool = True):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = _ensure_bchw(inputs).float()
        targets = _ensure_bchw(targets).float()

        if self.from_logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)
            probs = torch.sigmoid(inputs)
        else:
            bce_loss = F.binary_cross_entropy(inputs.clamp(1e-6, 1 - 1e-6), targets)
            probs = inputs.clamp(1e-6, 1 - 1e-6)

        p_flat, t_flat = _flatten_for_metrics(probs, targets)
        intersection = (p_flat * t_flat).sum()
        dice = (2. * intersection + self.smooth) / (p_flat.sum() + t_flat.sum() + self.smooth)
        dice_loss = 1 - dice

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


# -----------------------------
# Dice Loss
# -----------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = _ensure_bchw(inputs).float()
        targets = _ensure_bchw(targets).float()
        
        # Le problème vient d'ici, on utilise directement sigmoid pour la robustesse
        probs = torch.sigmoid(inputs) if self.from_logits else inputs.clamp_(1e-6, 1 - 1e-6)

        p_flat, t_flat = _flatten_for_metrics(probs, targets)
        intersection = (p_flat * t_flat).sum()
        dice = (2. * intersection + self.smooth) / (p_flat.sum() + t_flat.sum() + self.smooth)
        
        return 1 - dice


# -----------------------------
# Boundary Loss (blindée)
# -----------------------------
class BoundaryLoss(nn.Module):
    """
    Poids plus fort proche des contours : pondère la proba par une carte de distances
    calculée via EDT (distance_transform_edt) à l’intérieur et à l’extérieur.
    """
    def __init__(self, beta: float = 0.5):
        super().__init__()
        self.beta = beta
        print("=======> BoundaryLoss | beta =", beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = _ensure_bchw(pred).float()
        target = _ensure_bchw(target).float()
        probs = torch.sigmoid(pred).squeeze(1)  # (B,H,W)

        boundary_targets = []
        B = probs.shape[0]
        for b in range(B):
            t_np = target[b].squeeze().detach().cpu().numpy().astype(np.uint8)  # (H,W)
            if t_np.ndim != 2:
                raise ValueError(f"Masque attendu 2D, reçu {t_np.shape}")

            out = distance_transform_edt(t_np == 0, return_distances=True, return_indices=False)
            inn = distance_transform_edt(t_np == 1, return_distances=True, return_indices=False)

            if isinstance(out, tuple):
                out = out[0]
            if isinstance(inn, tuple):
                inn = inn[0]
            if out is None:
                out = np.zeros_like(t_np, dtype=np.float32)
            if inn is None:
                inn = np.zeros_like(t_np, dtype=np.float32)

            dist = out + inn  # (H,W) ndarray
            boundary_targets.append(torch.from_numpy(dist).to(probs.device, dtype=probs.dtype))

        boundary_targets = torch.stack(boundary_targets, dim=0)  # (B,H,W)
        loss = (probs * boundary_targets).mean()
        return self.beta * loss


# -----------------------------
# Combo Boundary + Dice
# -----------------------------
class CombinedBoundaryDiceLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6, from_logits: bool = True):
        super().__init__()
        self.boundary_loss = BoundaryLoss(beta=beta)
        self.dice_loss = DiceLoss(smooth=smooth, from_logits=from_logits)
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_b = self.boundary_loss(inputs, targets)
        loss_d = self.dice_loss(inputs, targets)
        return self.alpha * loss_d + self.beta * loss_b
        
# -----------------------------
# **Version corrigée de WeightedBCEDiceLoss**
# -----------------------------
class WeightedBCEDiceLoss(nn.Module):
    """
    Correction: Implémentation de la Dice Loss intégrée pour la robustesse.
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, pos_weight: float | torch.Tensor = 1.0, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, dtype=torch.float32))
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = _ensure_bchw(inputs).float()
        targets = _ensure_bchw(targets).float()

        # Calcul de la BCE Loss (stable)
        bce_loss = self.bce_loss_fn(inputs, targets)

        # Calcul de la Dice Loss intégré
        probs = torch.sigmoid(inputs)
        p_flat = probs.view(-1)
        t_flat = targets.view(-1)

        intersection = (p_flat * t_flat).sum()
        union = p_flat.sum() + t_flat.sum()
        
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score
        
        # Pour une raison inconnue (valeurs de logits extrêmes?), dice_score peut être > 1.
        # On force donc la perte à être non-négative.
        dice_loss = torch.clamp(dice_loss, min=0.0)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
