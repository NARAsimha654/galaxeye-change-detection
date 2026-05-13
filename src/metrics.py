"""
src/metrics.py — Evaluation metrics for binary change detection

All metrics computed for the Change class (label = 1) only,
as specified in the assignment. Accumulated across batches
via a running confusion matrix to avoid memory issues on
large val/test sets.
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


class ChangeMetrics:
    """
    Accumulates TP, FP, FN, TN across batches then computes
    IoU, Precision, Recall, F1 for the Change class.

    Usage:
        metrics = ChangeMetrics()
        for batch in loader:
            ...
            metrics.update(preds, targets, valid_mask)
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor,
               valid_mask: torch.Tensor = None):
        """
        Args:
            logits     : (B, 1, H, W) or (B, H, W) raw model output
            targets    : (B, H, W) long {0, 1}
            valid_mask : (B, H, W) float — 1 = valid pixel
        """
        if logits.dim() == 4:
            logits = logits.squeeze(1)

        preds = (torch.sigmoid(logits) >= self.threshold).long()

        if valid_mask is not None:
            valid = valid_mask.bool()
            preds   = preds[valid]
            targets = targets[valid]
        else:
            preds   = preds.reshape(-1)
            targets = targets.reshape(-1)

        preds   = preds.cpu()
        targets = targets.cpu()

        self.tp += ((preds == 1) & (targets == 1)).sum().item()
        self.fp += ((preds == 1) & (targets == 0)).sum().item()
        self.fn += ((preds == 0) & (targets == 1)).sum().item()
        self.tn += ((preds == 0) & (targets == 0)).sum().item()

    def compute(self) -> dict:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = 1e-8

        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        iou       = tp / (tp + fp + fn + eps)
        accuracy  = (tp + tn) / (tp + fp + fn + tn + eps)

        return {
            "iou":       round(iou,       4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "accuracy":  round(accuracy,  4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    def confusion_matrix(self) -> np.ndarray:
        """Returns 2x2 confusion matrix [[TN, FP], [FN, TP]]"""
        return np.array([
            [self.tn, self.fp],
            [self.fn, self.tp]
        ])

    def print_results(self, split: str = ""):
        results = self.compute()
        header  = f"[{split}] " if split else ""
        print(f"\n{header}Metrics (Change class):")
        print(f"  IoU       : {results['iou']:.4f}")
        print(f"  Precision : {results['precision']:.4f}")
        print(f"  Recall    : {results['recall']:.4f}")
        print(f"  F1        : {results['f1']:.4f}")
        print(f"  Accuracy  : {results['accuracy']:.4f}")
        print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
        cm = self.confusion_matrix()
        print(f"             No-Change  Change")
        print(f"  No-Change  {cm[0,0]:>9}  {cm[0,1]:>6}")
        print(f"  Change     {cm[1,0]:>9}  {cm[1,1]:>6}")


def plot_confusion_matrix(cm: np.ndarray, save_path: str, title: str = ""):
    """Save a styled confusion matrix figure."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    classes = ["No-Change", "Change"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(title or "Confusion Matrix", fontsize=13, fontweight="bold")

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i, j] / max(1, total)
            ax.text(j, i,
                    f"{cm[i,j]:,}\n({pct:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    metrics = ChangeMetrics(threshold=0.5)

    # Simulate 3 batches
    for _ in range(3):
        logits  = torch.randn(4, 1, 512, 512)
        targets = torch.zeros(4, 512, 512, dtype=torch.long)
        targets[:, 100:200, 100:200] = 1
        valid   = torch.ones(4, 512, 512)
        metrics.update(logits, targets, valid)

    metrics.print_results("smoke_test")
    print("\nMetrics OK.")