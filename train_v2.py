"""
train_v2.py — Enhanced Training Script for EO-SAR Change Detection v2

Additions over train.py:
  - Auxiliary loss from multi-scale decoder heads (weighted sum)
  - Lovász-Sigmoid loss: a tight convex surrogate for IoU, directly
    optimises the Jaccard index via sorted margin maximisation
  - Combined loss: Focal + Dice + Lovász + auxiliary heads

Usage: python train_v2.py --config config_v2.yaml
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import yaml

from src.dataset_v2 import build_dataloaders_v2
from src.model_v2   import build_model_v2
from src.losses     import FocalLoss, DiceLoss, CombinedLoss
from src.metrics    import ChangeMetrics


# ── Lovász-Sigmoid Loss ───────────────────────────────────────────────────────
def lovasz_grad(gt_sorted):
    """Lovász extension gradient computation."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_sigmoid_flat(logits, labels):
    """Binary Lovász-Sigmoid loss on flattened inputs."""
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs  = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


class LovaszLoss(nn.Module):
    """Lovász-Sigmoid loss for binary segmentation."""
    def forward(self, logits, targets, valid_mask=None):
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        B = logits.size(0)
        loss = 0.0
        for b in range(B):
            l = logits[b]
            t = targets[b]
            if valid_mask is not None:
                v = valid_mask[b].bool()
                l = l[v]
                t = t[v]
            else:
                l = l.flatten()
                t = t.flatten()
            loss += lovasz_sigmoid_flat(torch.sigmoid(l), t)
        return loss / B


# ── Combined v2 loss ──────────────────────────────────────────────────────────
class CombinedLossV2(nn.Module):
    """Focal + Dice + Lovász, weighted sum."""
    def __init__(self, focal_alpha=0.75, focal_gamma=2.0,
                 w_focal=0.4, w_dice=0.3, w_lovasz=0.3):
        super().__init__()
        self.focal   = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.dice    = DiceLoss()
        self.lovasz  = LovaszLoss()
        self.w_focal  = w_focal
        self.w_dice   = w_dice
        self.w_lovasz = w_lovasz

    def forward(self, logits, targets, valid_mask=None):
        fl = self.focal(logits,  targets, valid_mask)
        dl = self.dice(logits,   targets, valid_mask)
        lv = self.lovasz(logits, targets, valid_mask)
        return self.w_focal * fl + self.w_dice * dl + self.w_lovasz * lv


# ── Helpers ───────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def save_checkpoint(model, optimiser, epoch, best_f1, cfg, path):
    torch.save({
        "epoch":      epoch,
        "best_f1":    best_f1,
        "model":      model.state_dict(),
        "optimiser":  optimiser.state_dict(),
        "cfg":        cfg,
        "version":    "v2",
    }, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path, model, optimiser=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimiser and "optimiser" in ckpt:
        optimiser.load_state_dict(ckpt["optimiser"])
    return ckpt.get("epoch", 0), ckpt.get("best_f1", 0.0)


# ── Training epoch ────────────────────────────────────────────────────────────
def train_epoch(model, loader, loss_fn, optimiser, scheduler,
                scaler, device, grad_clip, writer, global_step,
                aux_weight=0.3):
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for i, (eo, sar, mask, valid) in enumerate(loader):
        eo    = eo.to(device,    non_blocking=True)
        sar   = sar.to(device,   non_blocking=True)
        mask  = mask.to(device,  non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)

        with autocast(enabled=scaler is not None):
            logits, aux_logits = model(eo, sar, return_aux=True)

            # Main loss
            main_loss = loss_fn(logits, mask, valid)

            # Auxiliary losses (weighted sum, reduced weight)
            aux_loss = 0.0
            for aux_l in aux_logits:
                aux_loss = aux_loss + loss_fn(aux_l, mask, valid)
            if aux_logits:
                aux_loss = aux_loss / len(aux_logits)

            loss = main_loss + aux_weight * aux_loss
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    WARNING: NaN/Inf at step {i+1}, skipping batch")
            optimiser.zero_grad(set_to_none=True)
            global_step += 1
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        global_step += 1

        if (i + 1) % 50 == 0:
            lr = optimiser.param_groups[0]["lr"]
            print(f"    step {i+1}/{n_batches} | "
                  f"loss={loss.item():.4f} (main={main_loss.item():.4f}) "
                  f"| lr={lr:.2e}")
            writer.add_scalar("train/loss_step", loss.item(), global_step)

    return total_loss / n_batches, global_step


# ── Validation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    metrics    = ChangeMetrics(threshold=0.5)
    total_loss = 0.0

    for eo, sar, mask, valid in loader:
        eo    = eo.to(device,    non_blocking=True)
        sar   = sar.to(device,   non_blocking=True)
        mask  = mask.to(device,  non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        with autocast(enabled=torch.cuda.is_available()):
            logits = model(eo, sar, return_aux=False)
            loss   = loss_fn(logits, mask, valid)

        total_loss += loss.item()
        metrics.update(logits, mask, valid)

    results = metrics.compute()
    results["loss"] = total_loss / len(loader)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config_v2.yaml")
    parser.add_argument("--resume",  default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(cfg["evaluation"]["save_dir"])
    log_dir  = Path(cfg["evaluation"]["log_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device  : {device}")
    print(f"Config  : {args.config}")
    print(f"Arch    : {cfg['model']['architecture']} v2")
    print(f"Epochs  : {cfg['training']['epochs']}")
    print()

    # Data, model, loss
    train_loader, val_loader, _ = build_dataloaders_v2(cfg)
    model   = build_model_v2(cfg).to(device)
    loss_fn = CombinedLoss(
        focal_weight=0.5,
        dice_weight=0.5,
        focal_gamma=cfg["loss"]["focal_gamma"],
        focal_alpha=cfg["loss"]["focal_alpha"],
    )

    optimiser = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=cfg["training"]["epochs"] * len(train_loader),
        eta_min=cfg["training"]["scheduler_min_lr"],
    )
    scaler = GradScaler() if (cfg["training"]["mixed_precision"]
                              and device.type == "cuda") else None

    # Resume
    start_epoch, best_f1 = 0, 0.0
    if args.resume and Path(args.resume).exists():
        start_epoch, best_f1 = load_checkpoint(args.resume, model, optimiser)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}, best F1={best_f1:.4f}")

    writer      = SummaryWriter(log_dir=str(log_dir))
    global_step = start_epoch * len(train_loader)
    epochs      = cfg["training"]["epochs"]
    grad_clip   = cfg["training"]["gradient_clip"]
    patience    = 20
    no_improve  = 0

    best_ckpt = save_dir / "best_model_v2.pth"
    last_ckpt = save_dir / "last_model_v2.pth"

    print("=" * 60)
    print("Starting training v2")
    print("=" * 60)

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, global_step = train_epoch(
            model, train_loader, loss_fn, optimiser, scheduler,
            scaler, device, grad_clip, writer, global_step,
            aux_weight=0.3,
        )

        val_results = evaluate(model, val_loader, loss_fn, device)
        elapsed     = time.time() - t0
        lr_now      = optimiser.param_groups[0]["lr"]

        print(f"  train_loss : {train_loss:.4f}")
        print(f"  val_loss   : {val_results['loss']:.4f}")
        print(f"  val_f1     : {val_results['f1']:.4f}  (best={best_f1:.4f})")
        print(f"  val_iou    : {val_results['iou']:.4f}")
        print(f"  val_prec   : {val_results['precision']:.4f}  "
              f"val_rec={val_results['recall']:.4f}")
        print(f"  lr         : {lr_now:.2e}  |  time: {elapsed:.1f}s")

        writer.add_scalar("train/loss_epoch", train_loss,              epoch)
        writer.add_scalar("val/loss",         val_results["loss"],     epoch)
        writer.add_scalar("val/f1",           val_results["f1"],       epoch)
        writer.add_scalar("val/iou",          val_results["iou"],      epoch)
        writer.add_scalar("val/precision",    val_results["precision"],epoch)
        writer.add_scalar("val/recall",       val_results["recall"],   epoch)
        writer.add_scalar("train/lr",         lr_now,                  epoch)

        save_checkpoint(model, optimiser, epoch, best_f1, cfg, last_ckpt)

        if val_results["f1"] > best_f1:
            best_f1    = val_results["f1"]
            no_improve = 0
            save_checkpoint(model, optimiser, epoch, best_f1, cfg, best_ckpt)
            print(f"  *** New best F1: {best_f1:.4f} ***")
        else:
            no_improve += 1
            print(f"  No improvement {no_improve}/{patience}")

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    writer.close()
    print(f"\nTraining v2 complete. Best val F1: {best_f1:.4f}")
    print(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()