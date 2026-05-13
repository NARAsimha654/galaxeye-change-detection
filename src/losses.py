"""
src/losses.py — Combined Focal + Dice Loss for binary change detection

Design rationale:
- BCE alone collapses to predicting all No-Change (98.4% class) → useless
- Focal Loss: down-weights easy negatives, forces model to focus on
  the rare change pixels. gamma=2 is standard; alpha=0.75 gives extra
  weight to the positive (Change) class
- Dice Loss: directly optimises overlap between prediction and ground
  truth — more aligned with IoU metric than cross-entropy variants
- Combined (0.5 focal + 0.5 dice): best of both — stable training
  from focal, metric-aligned optimisation from dice
- valid_mask: zero-out no-data regions from loss computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha      # weight for positive (Change) class
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            logits     : (B, 1, H, W) or (B, H, W) — raw model output
            targets    : (B, H, W) long — binary {0, 1}
            valid_mask : (B, H, W) float — 1 where valid, 0 where no-data
        """
        if logits.dim() == 4:
            logits = logits.squeeze(1)     # (B,H,W)

        targets_f = targets.float()
        probs     = torch.sigmoid(logits)
        probs     = torch.clamp(probs, 1e-6, 1 - 1e-6)

        # Per-pixel BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, targets_f, reduction="none"
        )

        # Focal weight
        p_t     = probs * targets_f + (1 - probs) * (1 - targets_f)
        alpha_t = self.alpha * targets_f + (1 - self.alpha) * (1 - targets_f)
        focal_w = alpha_t * (1 - p_t) ** self.gamma
        loss    = focal_w * bce

        # Apply no-data mask
        if valid_mask is not None:
            loss = loss * valid_mask
            if valid_mask.sum() > 0:
                return loss.sum() / valid_mask.sum()
            return loss.sum() * 0.0

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Directly optimises the Dice coefficient = 2*|A∩B| / (|A|+|B|)
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                valid_mask: torch.Tensor = None) -> torch.Tensor:
        if logits.dim() == 4:
            logits = logits.squeeze(1)

        probs    = torch.sigmoid(logits)
        targets_f = targets.float()

        if valid_mask is not None:
            probs     = probs     * valid_mask
            targets_f = targets_f * valid_mask

        # Flatten spatial dims
        probs_flat  = probs.reshape(probs.size(0), -1)
        target_flat = targets_f.reshape(targets_f.size(0), -1)

        intersection = (probs_flat * target_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined Focal + Dice loss.
    loss = focal_w * FocalLoss + dice_w * DiceLoss
    """
    def __init__(self, focal_weight: float = 0.5, dice_weight: float = 0.5,
                 focal_gamma: float = 2.0, focal_alpha: float = 0.75):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight  = dice_weight
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.dice  = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                valid_mask: torch.Tensor = None) -> torch.Tensor:
        focal_loss = self.focal(logits, targets, valid_mask)
        dice_loss  = self.dice(logits, targets, valid_mask)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


def build_loss(cfg) -> CombinedLoss:
    loss_cfg = cfg["loss"]
    return CombinedLoss(
        focal_weight = loss_cfg["focal_weight"],
        dice_weight  = loss_cfg["dice_weight"],
        focal_gamma  = loss_cfg["focal_gamma"],
        focal_alpha  = loss_cfg["focal_alpha"],
    )


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, H, W = 4, 512, 512
    logits  = torch.randn(B, 1, H, W)
    targets = torch.zeros(B, H, W, dtype=torch.long)
    targets[:, 100:150, 100:150] = 1          # small change region
    valid   = torch.ones(B, H, W)
    valid[:, :50, :50] = 0                    # simulate no-data corner

    loss_fn = CombinedLoss()
    loss    = loss_fn(logits, targets, valid)
    print(f"Combined loss: {loss.item():.4f}")

    focal_fn = FocalLoss()
    dice_fn  = DiceLoss()
    print(f"Focal only:    {focal_fn(logits, targets, valid).item():.4f}")
    print(f"Dice  only:    {dice_fn(logits, targets, valid).item():.4f}")
    print("Losses OK.")