"""
eval_v2.py — Evaluation for DualEncoderUNetV2

Usage:
    # Standard eval
    python eval_v2.py --config config_v2.yaml --weights weights/best_model_v2.pth --split val
    python eval_v2.py --config config_v2.yaml --weights weights/best_model_v2.pth --split test

    # With TTA
    python eval_v2.py --config config_v2.yaml --weights weights/best_model_v2.pth --split val --tta
    python eval_v2.py --config config_v2.yaml --weights weights/best_model_v2.pth --split test --tta

    # Custom threshold
    python eval_v2.py --config config_v2.yaml --weights weights/best_model_v2.pth \
                      --split val --threshold 0.6 --tta
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from src.dataset_v2 import EOSARDatasetV2, instance_normalize, instance_normalize_sar
from src.model_v2   import build_model_v2
from src.metrics    import ChangeMetrics, plot_confusion_matrix
from src.tta        import tta_inference


# ── Denorm for visualisation (instance-normalised inputs) ─────────────────────
def to_displayable_eo(tensor):
    """Convert instance-normalised EO tensor to displayable uint8."""
    t = tensor.cpu().numpy().transpose(1, 2, 0)  # (H,W,3)
    # Rescale each channel to 0-255 for display
    for c in range(3):
        ch = t[:, :, c]
        t[:, :, c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6) * 255
    return t.astype(np.uint8)

def to_displayable_sar(tensor):
    """Convert instance-normalised SAR tensor to displayable uint8."""
    t = tensor.cpu().numpy()[0]  # (H,W)
    t = (t - t.min()) / (t.max() - t.min() + 1e-6) * 255
    return t.astype(np.uint8)


# ── Qualitative visualisation ─────────────────────────────────────────────────
def visualise_predictions(model, dataset, device, out_dir,
                          n_samples=10, threshold=0.5, use_tta=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    change_idx   = [i for i, h in enumerate(dataset.has_change) if h]
    nochange_idx = [i for i, h in enumerate(dataset.has_change) if not h]
    indices = (change_idx[:7] + nochange_idx[:3])[:n_samples]

    print(f"\nGenerating {len(indices)} visualisations (TTA={use_tta})...")

    for vis_i, ds_i in enumerate(indices):
        eo, sar, mask, valid = dataset[ds_i]
        eo_in  = eo.unsqueeze(0).to(device)
        sar_in = sar.unsqueeze(0).to(device)

        with torch.no_grad():
            if use_tta:
                prob_t, _ = tta_inference(model, eo_in, sar_in, threshold)
                prob = prob_t.squeeze().cpu().numpy()
            else:
                with autocast(enabled=device.type == "cuda"):
                    logit = model(eo_in, sar_in, return_aux=False)
                prob = torch.sigmoid(logit).squeeze().cpu().numpy()

        pred    = (prob >= threshold).astype(np.uint8)
        mask_np = mask.cpu().numpy().astype(np.uint8)
        valid_np = valid.cpu().numpy()

        eo_vis  = to_displayable_eo(eo)
        sar_vis = to_displayable_sar(sar)

        tp = int(((pred==1)&(mask_np==1)&(valid_np==1)).sum())
        fp = int(((pred==1)&(mask_np==0)&(valid_np==1)).sum())
        fn = int(((pred==0)&(mask_np==1)&(valid_np==1)).sum())
        eps = 1e-8
        f1  = 2*tp / (2*tp + fp + fn + eps)
        iou = tp   / (tp + fp + fn + eps)

        overlay = eo_vis.copy()
        overlay[(pred==1)&(mask_np==1)&(valid_np==1)] = [0,  200, 0  ]
        overlay[(pred==1)&(mask_np==0)&(valid_np==1)] = [200, 0,  0  ]
        overlay[(pred==0)&(mask_np==1)&(valid_np==1)] = [255, 255, 0 ]

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        tag = "CHANGE" if mask_np.sum() > 0 else "NO-CHANGE"
        fig.suptitle(
            f"Sample {vis_i+1} [{tag}] | F1={f1:.3f} | IoU={iou:.3f} | "
            f"TP={tp} FP={fp} FN={fn}",
            fontsize=11, fontweight="bold"
        )
        axes[0].imshow(eo_vis);           axes[0].set_title("EO Pre-event (RGB)")
        axes[1].imshow(sar_vis, cmap="gray"); axes[1].set_title("SAR Post-event")
        axes[2].imshow(mask_np, cmap="RdYlGn_r", vmin=0, vmax=1)
        axes[2].set_title("Ground Truth")
        im = axes[3].imshow(prob, cmap="hot", vmin=0, vmax=1)
        axes[3].set_title("Prediction (prob)")
        plt.colorbar(im, ax=axes[3], fraction=0.046)
        axes[4].imshow(overlay)
        axes[4].legend(handles=[
            mpatches.Patch(color=(0,200/255,0), label="TP"),
            mpatches.Patch(color=(200/255,0,0), label="FP"),
            mpatches.Patch(color=(1,1,0),       label="FN"),
        ], loc="upper right", fontsize=8)
        axes[4].set_title("Overlay (TP/FP/FN)")
        for ax in axes: ax.axis("off")

        save_path = out_dir / f"sample_{vis_i+1:02d}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path.name}  (F1={f1:.3f})")

    print(f"Visualisations saved to: {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="config_v2.yaml")
    parser.add_argument("--weights",   required=True)
    parser.add_argument("--split",     default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--threshold", default=0.5,  type=float)
    parser.add_argument("--tta",       action="store_true")
    parser.add_argument("--no_vis",    action="store_true")
    parser.add_argument("--data_path", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.data_path:
        cfg["data"]["root"] = args.data_path

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag     = "tta" if args.tta else "standard"
    out_dir = Path("outputs/eval_v2") / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device    : {device}")
    print(f"Weights   : {args.weights}")
    print(f"Split     : {args.split}")
    print(f"Threshold : {args.threshold}")
    print(f"TTA       : {args.tta}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model = build_model_v2(cfg).to(device)
    ckpt  = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint epoch={ckpt.get('epoch','?')} "
          f"best_f1={ckpt.get('best_f1',0):.4f}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = EOSARDatasetV2(
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

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = ChangeMetrics(threshold=args.threshold)
    print("Running evaluation...")

    with torch.no_grad():
        for eo, sar, mask, valid in loader:
            eo    = eo.to(device,    non_blocking=True)
            sar   = sar.to(device,   non_blocking=True)
            mask  = mask.to(device,  non_blocking=True)
            valid = valid.to(device, non_blocking=True)

            if args.tta:
                avg_prob, pred = tta_inference(
                    model, eo, sar, threshold=args.threshold
                )
                valid_bool = valid.bool()
                p_flat = pred.cpu()[valid_bool.cpu()]
                m_flat = mask.cpu()[valid_bool.cpu()]
                metrics.tp += int(((p_flat==1)&(m_flat==1)).sum())
                metrics.fp += int(((p_flat==1)&(m_flat==0)).sum())
                metrics.fn += int(((p_flat==0)&(m_flat==1)).sum())
                metrics.tn += int(((p_flat==0)&(m_flat==0)).sum())
            else:
                with autocast(enabled=device.type == "cuda"):
                    logits = model(eo, sar, return_aux=False)
                metrics.update(logits, mask, valid)

    # ── Results ───────────────────────────────────────────────────────────────
    results = metrics.compute()
    metrics.print_results(
        split=f"{args.split.upper()} v2 [{tag}, t={args.threshold}]"
    )

    cm      = metrics.confusion_matrix()
    cm_path = str(out_dir / f"confusion_matrix_{tag}.png")
    plot_confusion_matrix(
        cm, cm_path,
        title=f"Confusion Matrix v2 ({tag}) — {args.split.upper()}"
    )

    results_path = out_dir / f"results_{tag}_t{args.threshold}.txt"
    with open(results_path, "w") as f:
        f.write(f"Model     : DualEncoderUNetV2\n")
        f.write(f"Split     : {args.split}\n")
        f.write(f"Weights   : {args.weights}\n")
        f.write(f"Threshold : {args.threshold}\n")
        f.write(f"TTA       : {args.tta}\n\n")
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

    if not args.no_vis:
        visualise_predictions(
            model, dataset, device,
            out_dir=out_dir / f"visualisations_{tag}",
            n_samples=10,
            threshold=args.threshold,
            use_tta=args.tta,
        )

    print(f"\nFinal: IoU={results['iou']:.4f}  F1={results['f1']:.4f}")


if __name__ == "__main__":
    main()