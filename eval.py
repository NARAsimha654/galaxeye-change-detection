"""
eval.py — Evaluation script for EO-SAR Binary Change Detection

Usage:
    # Evaluate on test split
    python eval.py --config config.yaml --weights weights/best_model.pth --split test

    # Evaluate on val split
    python eval.py --config config.yaml --weights weights/best_model.pth --split val

    # Evaluate with custom data path
    python eval.py --config config.yaml --weights weights/best_model.pth \
                   --data_path /path/to/data --split test

Outputs:
    - Metrics printed to terminal
    - Confusion matrix saved to outputs/eval/
    - Qualitative visualisations saved to outputs/eval/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.cuda.amp import autocast
import yaml

from src.dataset import EOSARDataset, get_val_spatial_transforms
from src.model   import build_model
from src.metrics import ChangeMetrics, plot_confusion_matrix
from torch.utils.data import DataLoader
import rasterio


# ── Denormalise for visualisation ─────────────────────────────────────────────
def denorm_eo(tensor, mean, std):
    """Convert normalised EO tensor back to uint8 for display."""
    t = tensor.cpu().numpy().transpose(1, 2, 0)   # (H,W,3)
    t = t * std + mean
    t = np.clip(t, 0, 255).astype(np.uint8)
    return t

def denorm_sar(tensor, mean, std):
    """Convert normalised SAR tensor back to uint8 for display."""
    t = tensor.cpu().numpy()[0]                    # (H,W)
    t = t * std[0] + mean[0]
    t = np.clip(t, 0, 255).astype(np.uint8)
    return t


# ── Qualitative visualisation ─────────────────────────────────────────────────
def visualise_predictions(model, dataset, device, cfg, out_dir,
                          n_samples=10, threshold=0.5):
    """
    Save side-by-side prediction figures:
    EO | SAR | Ground Truth | Prediction | Overlay
    Picks both success and failure cases.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    norm     = cfg["normalization"]
    eo_mean  = np.array(norm["eo_mean"],  dtype=np.float32)
    eo_std   = np.array(norm["eo_std"],   dtype=np.float32)
    sar_mean = np.array(norm["sar_mean"], dtype=np.float32)
    sar_std  = np.array(norm["sar_std"],  dtype=np.float32)

    model.eval()

    # Collect indices with change pixels for meaningful visualisation
    change_indices  = [i for i, h in enumerate(dataset.has_change) if h]
    nochange_indices = [i for i, h in enumerate(dataset.has_change) if not h]

    # Mix: 7 with change, 3 without
    indices = (change_indices[:7] + nochange_indices[:3])[:n_samples]

    print(f"\nGenerating {len(indices)} qualitative visualisations...")

    for vis_idx, dataset_idx in enumerate(indices):
        eo, sar, mask, valid = dataset[dataset_idx]

        eo_in  = eo.unsqueeze(0).to(device)
        sar_in = sar.unsqueeze(0).to(device)

        with torch.no_grad(), autocast(enabled=device.type == "cuda"):
            logit = model(eo_in, sar_in)
            prob  = torch.sigmoid(logit).squeeze().cpu().numpy()
            pred  = (prob >= threshold).astype(np.uint8)

        mask_np  = mask.cpu().numpy().astype(np.uint8)
        valid_np = valid.cpu().numpy()

        eo_vis  = denorm_eo(eo,   eo_mean,  eo_std)
        sar_vis = denorm_sar(sar, sar_mean, sar_std)

        # Compute per-sample metrics
        tp = int(((pred == 1) & (mask_np == 1) & (valid_np == 1)).sum())
        fp = int(((pred == 1) & (mask_np == 0) & (valid_np == 1)).sum())
        fn = int(((pred == 0) & (mask_np == 1) & (valid_np == 1)).sum())
        eps = 1e-8
        f1  = 2*tp / (2*tp + fp + fn + eps)
        iou = tp   / (tp + fp + fn + eps)

        # Build overlay: TP=green, FP=red, FN=yellow on EO
        overlay = eo_vis.copy()
        overlay[(pred==1)&(mask_np==1)&(valid_np==1)] = [0,  200, 0  ]  # TP green
        overlay[(pred==1)&(mask_np==0)&(valid_np==1)] = [200, 0,  0  ]  # FP red
        overlay[(pred==0)&(mask_np==1)&(valid_np==1)] = [255, 255, 0  ]  # FN yellow

        # Figure
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        title_tag = "CHANGE" if mask_np.sum() > 0 else "NO-CHANGE"
        fig.suptitle(
            f"Sample {vis_idx+1} [{title_tag}] | "
            f"F1={f1:.3f} | IoU={iou:.3f} | "
            f"TP={tp} FP={fp} FN={fn}",
            fontsize=11, fontweight="bold"
        )

        axes[0].imshow(eo_vis)
        axes[0].set_title("EO — Pre-event (RGB)")

        axes[1].imshow(sar_vis, cmap="gray")
        axes[1].set_title("SAR — Post-event")

        axes[2].imshow(mask_np, cmap="RdYlGn_r", vmin=0, vmax=1)
        axes[2].set_title("Ground Truth")

        im = axes[3].imshow(prob, cmap="hot", vmin=0, vmax=1)
        axes[3].set_title("Prediction (prob)")
        plt.colorbar(im, ax=axes[3], fraction=0.046)

        axes[4].imshow(overlay)
        tp_p = mpatches.Patch(color=(0,200/255,0),     label="TP")
        fp_p = mpatches.Patch(color=(200/255,0,0),     label="FP")
        fn_p = mpatches.Patch(color=(1,1,0),           label="FN")
        axes[4].legend(handles=[tp_p, fp_p, fn_p], loc="upper right",
                       fontsize=8)
        axes[4].set_title("Overlay (TP/FP/FN)")

        for ax in axes:
            ax.axis("off")

        save_path = out_dir / f"sample_{vis_idx+1:02d}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path.name}  (F1={f1:.3f})")

    print(f"All visualisations saved to: {out_dir}")


# ── Main evaluation ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--weights",   required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--split",     default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--data_path", default=None,
                        help="Override data root from config")
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--no_vis",    action="store_true",
                        help="Skip qualitative visualisations")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.data_path:
        cfg["data"]["root"] = args.data_path

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir  = Path("outputs/eval") / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device    : {device}")
    print(f"Weights   : {args.weights}")
    print(f"Split     : {args.split}")
    print(f"Threshold : {args.threshold}")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch','?')}, "
          f"best_f1={ckpt.get('best_f1', 0):.4f})\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = EOSARDataset(
        cfg["data"]["root"], args.split, cfg, is_train=False
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = ChangeMetrics(threshold=args.threshold)

    print("Running evaluation...")
    with torch.no_grad():
        for eo, sar, mask, valid in loader:
            eo    = eo.to(device,    non_blocking=True)
            sar   = sar.to(device,   non_blocking=True)
            mask  = mask.to(device,  non_blocking=True)
            valid = valid.to(device, non_blocking=True)

            with autocast(enabled=device.type == "cuda"):
                logits = model(eo, sar)

            metrics.update(logits, mask, valid)

    # ── Print results ─────────────────────────────────────────────────────────
    results = metrics.compute()
    metrics.print_results(split=args.split.upper())

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm      = metrics.confusion_matrix()
    cm_path = str(out_dir / "confusion_matrix.png")
    plot_confusion_matrix(
        cm, cm_path,
        title=f"Confusion Matrix — {args.split.upper()} split"
    )

    # ── Save results txt ──────────────────────────────────────────────────────
    results_path = out_dir / "results.txt"
    with open(results_path, "w") as f:
        f.write(f"Split     : {args.split}\n")
        f.write(f"Weights   : {args.weights}\n")
        f.write(f"Threshold : {args.threshold}\n\n")
        f.write(f"IoU       : {results['iou']:.4f}\n")
        f.write(f"Precision : {results['precision']:.4f}\n")
        f.write(f"Recall    : {results['recall']:.4f}\n")
        f.write(f"F1        : {results['f1']:.4f}\n")
        f.write(f"Accuracy  : {results['accuracy']:.4f}\n\n")
        f.write(f"TP: {results['tp']}\n")
        f.write(f"FP: {results['fp']}\n")
        f.write(f"FN: {results['fn']}\n")
        f.write(f"TN: {results['tn']}\n")
    print(f"  Results saved: {results_path}")

    # ── Qualitative visualisations ────────────────────────────────────────────
    if not args.no_vis:
        visualise_predictions(
            model, dataset, device, cfg,
            out_dir=out_dir / "visualisations",
            n_samples=10,
            threshold=args.threshold,
        )

    print(f"\nEvaluation complete.")
    print(f"  IoU : {results['iou']:.4f}")
    print(f"  F1  : {results['f1']:.4f}")
    return results


if __name__ == "__main__":
    main()