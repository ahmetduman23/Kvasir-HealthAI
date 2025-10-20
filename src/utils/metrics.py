import torch
from typing import Tuple

@torch.no_grad()
def dice_coeff(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Soft Dice (0–1) on probabilities (no threshold), averaged over batch.
    Args:
        probs: (N,1,H,W) or (N,H,W) in [0,1]
        target: (N,1,H,W) or (N,H,W) in {0,1}
    """
    p = probs.float().view(probs.size(0), -1)
    t = target.float().view(target.size(0), -1)
    inter = (p * t).sum(dim=1)
    denom = (p * p).sum(dim=1) + (t * t).sum(dim=1) + eps
    dice = (2.0 * inter + eps) / denom
    return dice.mean().item()

@torch.no_grad()
def iou_score(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Soft IoU (Jaccard) on probabilities (no threshold), averaged over batch.
    """
    p = probs.float().view(probs.size(0), -1)
    t = target.float().view(target.size(0), -1)
    inter = (p * t).sum(dim=1)
    union = (p * p).sum(dim=1) + (t * t).sum(dim=1) - inter + eps
    iou = (inter + eps) / union
    return iou.mean().item()

@torch.no_grad()
def dice_iou_binary(
    probs: torch.Tensor,
    target: torch.Tensor,
    thr: float = 0.5,
    eps: float = 1e-6
) -> Tuple[float, float]:
    """
    Thresholded (binary) Dice/IoU — handy if you prefer hard masks for evaluation.
    Returns (dice, iou) as floats.
    """
    pred = (probs.float() > thr).float()
    t = target.float()
    p = pred.view(pred.size(0), -1)
    t = t.view(t.size(0), -1)

    inter = (p * t).sum(dim=1)
    dice = (2.0 * inter + eps) / (p.sum(dim=1) + t.sum(dim=1) + eps)

    union = p.sum(dim=1) + t.sum(dim=1) - inter + eps
    iou = (inter + eps) / union
    return dice.mean().item(), iou.mean().item()
