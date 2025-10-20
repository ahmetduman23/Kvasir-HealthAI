import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    """
    Binary Cross-Entropy with Logits + Dice loss (soft).
    Returns a weighted combination: w * BCE + (1 - w) * Dice.
    """
    def __init__(self, bce_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(logits, targets)

        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (probs * targets).sum(dim=dims)
        denom = (probs * probs).sum(dim=dims) + (targets * targets).sum(dim=dims) + self.eps
        dice_loss = 1.0 - (2.0 * inter + self.eps) / denom
        dice = dice_loss.mean()

        loss = self.bce_weight * bce + (1.0 - self.bce_weight) * dice

        # Safety against NaN/Inf
        if not torch.isfinite(loss):
            loss = torch.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=1.0)
        return loss
