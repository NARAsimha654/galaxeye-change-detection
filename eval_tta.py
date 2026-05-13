"""
eval_tta.py — Evaluation with Test-Time Augmentation

Usage:
    python eval_tta.py --config config.yaml --weights weights/best_model.pth --split test
    python eval_tta.py --config config.yaml --weights weights/best_model.pth --split val
"""

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import EOSARDataset
from src.model   import build_model
from src.metrics import ChangeMetrics, plot_confusion_matrix
from src.tta     import tta_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--weights",   required=True)
    parser.add_argument("--split",     default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--threshold", default=0.5, type=float)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("outputs/eval_tta") / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device    : {device}")
    print(f"Weights   : {args.weights}")
    print(f"Split     : {args.split}")
    print(f"Threshold : {args.threshold}")
    print(f"TTA       : 8 augmentations averaged\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint epoch={ckpt.get('epoch','?')} "
          f"best_f1={ckpt.get('best_f1',0):.4f}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = EOSARDataset(
        cfg["data"]["root"], args.split, cfg, is_train=False
    )
    # TTA needs batch_size=1 for transpose augmentation to work correctly
    # when H != W (our images are square 512x512 so fine, but keep it explicit)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = ChangeMetrics(threshold=args.threshold)

    print("Running TTA evaluation...")
    for batch_idx, (eo, sar, mask, valid) in enumerate(loader):
        eo    = eo.to(device,    non_blocking=True)
        sar   = sar.to(device,   non_blocking=True)
        mask  = mask.to(device,  non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        avg_prob, pred = tta_inference(model, eo, sar, threshold=args.threshold)

        # Convert prob back to logit-like for metrics (metrics uses sigmoid internally)
        # Instead pass pred directly by computing a fake logit
        # Simpler: update metrics manually
        pred_cpu   = pred.cpu()
        mask_cpu   = mask.cpu()
        valid_cpu  = valid.cpu()

        valid_bool = valid_cpu.bool()
        p_flat     = pred_cpu[valid_bool]
        m_flat     = mask_cpu[valid_bool]

        metrics.tp += int(((p_flat == 1) & (m_flat == 1)).sum())
        metrics.fp += int(((p_flat == 1) & (m_flat == 0)).sum())
        metrics.fn += int(((p_flat == 0) & (m_flat == 1)).sum())
        metrics.tn += int(((p_flat == 0) & (m_flat == 0)).sum())

        if (batch_idx + 1) % 5 == 0:
            print(f"  Processed {(batch_idx+1)*4}/{len(dataset)} samples...")

    # ── Results ───────────────────────────────────────────────────────────────
    results = metrics.compute()
    metrics.print_results(split=f"{args.split.upper()} + TTA")

    cm      = metrics.confusion_matrix()
    cm_path = str(out_dir / "confusion_matrix_tta.png")
    plot_confusion_matrix(
        cm, cm_path,
        title=f"Confusion Matrix (TTA) — {args.split.upper()} split"
    )

    results_path = out_dir / "results_tta.txt"
    with open(results_path, "w") as f:
        f.write(f"Split     : {args.split} + TTA (8 augmentations)\n")
        f.write(f"Weights   : {args.weights}\n")
        f.write(f"Threshold : {args.threshold}\n\n")
        f.write(f"IoU       : {results['iou']:.4f}\n")
        f.write(f"Precision : {results['precision']:.4f}\n")
        f.write(f"Recall    : {results['recall']:.4f}\n")
        f.write(f"F1        : {results['f1']:.4f}\n")
        f.write(f"Accuracy  : {results['accuracy']:.4f}\n")
    print(f"\n  Results saved: {results_path}")
    print(f"\nTTA Evaluation complete.")
    print(f"  IoU : {results['iou']:.4f}")
    print(f"  F1  : {results['f1']:.4f}")


if __name__ == "__main__":
    main()