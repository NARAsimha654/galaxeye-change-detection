"""
train.py — Training script for EO-SAR Binary Change Detection
Usage: python train.py --config config.yaml
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import yaml

from src.dataset import build_dataloaders
from src.model   import build_model
from src.losses  import build_loss
from src.metrics import ChangeMetrics


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Optimiser + scheduler ─────────────────────────────────────────────────────
def build_optimiser(model, cfg):
    lr = cfg["training"]["learning_rate"]
    wd = cfg["training"]["weight_decay"]
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=wd
    )


def build_scheduler(optimiser, cfg, steps_per_epoch):
    sched = cfg["training"]["scheduler"]
    epochs = cfg["training"]["epochs"]
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            T_max=epochs * steps_per_epoch,
            eta_min=cfg["training"]["scheduler_min_lr"],
        )
    elif sched == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="max", patience=5, factor=0.5
        )
    return None


# ── Single training epoch ─────────────────────────────────────────────────────
def train_epoch(model, loader, loss_fn, optimiser, scheduler,
                scaler, device, grad_clip, writer, global_step):
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
            logits = model(eo, sar)                     # (B,1,H,W)
            loss   = loss_fn(logits, mask, valid)

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

        if scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step()

        total_loss += loss.item()
        global_step += 1

        if (i + 1) % 50 == 0:
            lr_now = optimiser.param_groups[0]["lr"]
            print(f"    step {i+1}/{n_batches} | "
                  f"loss={loss.item():.4f} | lr={lr_now:.2e}")
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
            logits = model(eo, sar)
            loss   = loss_fn(logits, mask, valid)

        total_loss += loss.item()
        metrics.update(logits, mask, valid)

    results = metrics.compute()
    results["loss"] = total_loss / len(loader)
    return results


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def save_checkpoint(model, optimiser, epoch, best_f1, cfg, path):
    torch.save({
        "epoch":      epoch,
        "best_f1":    best_f1,
        "model":      model.state_dict(),
        "optimiser":  optimiser.state_dict(),
        "cfg":        cfg,
    }, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path, model, optimiser=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimiser is not None and "optimiser" in ckpt:
        optimiser.load_state_dict(ckpt["optimiser"])
    return ckpt.get("epoch", 0), ckpt.get("best_f1", 0.0)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--resume",  default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Setup ─────────────────────────────────────────────────────────────────
    set_seed(cfg["training"]["seed"])
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir   = Path(cfg["evaluation"]["save_dir"])
    log_dir    = Path(cfg["evaluation"]["log_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device : {device}")
    print(f"Config : {args.config}")
    print(f"Arch   : {cfg['model']['architecture']}")
    print(f"Epochs : {cfg['training']['epochs']}")
    print()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders(cfg)

    # ── Model / loss / optimiser ───────────────────────────────────────────────
    model    = build_model(cfg).to(device)
    loss_fn  = build_loss(cfg)
    optimiser = build_optimiser(model, cfg)
    scheduler = build_scheduler(optimiser, cfg, len(train_loader))

    scaler = GradScaler() if (cfg["training"]["mixed_precision"]
                              and device.type == "cuda") else None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_f1     = 0.0
    if args.resume and Path(args.resume).exists():
        start_epoch, best_f1 = load_checkpoint(
            args.resume, model, optimiser
        )
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}, best F1={best_f1:.4f}")

    # ── Logging ───────────────────────────────────────────────────────────────
    writer      = SummaryWriter(log_dir=str(log_dir))
    global_step = start_epoch * len(train_loader)
    grad_clip   = cfg["training"]["gradient_clip"]
    epochs      = cfg["training"]["epochs"]
    patience    = 15          # early stopping patience
    no_improve  = 0

    best_ckpt = save_dir / "best_model.pth"
    last_ckpt = save_dir / "last_model.pth"

    print("=" * 60)
    print("Starting training")
    print("=" * 60)

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        print(f"\nEpoch {epoch+1}/{epochs}")

        # ── Train ─────────────────────────────────────────────────────────────
        train_loss, global_step = train_epoch(
            model, train_loader, loss_fn, optimiser, scheduler,
            scaler, device, grad_clip, writer, global_step
        )

        # ── Validate ──────────────────────────────────────────────────────────
        val_results = evaluate(model, val_loader, loss_fn, device)

        # ── LR plateau scheduler step ─────────────────────────────────────────
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_results["f1"])

        # ── Log ───────────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        lr_now  = optimiser.param_groups[0]["lr"]

        print(f"  train_loss : {train_loss:.4f}")
        print(f"  val_loss   : {val_results['loss']:.4f}")
        print(f"  val_f1     : {val_results['f1']:.4f}  "
              f"(best={best_f1:.4f})")
        print(f"  val_iou    : {val_results['iou']:.4f}")
        print(f"  val_prec   : {val_results['precision']:.4f}  "
              f"val_rec={val_results['recall']:.4f}")
        print(f"  lr         : {lr_now:.2e}")
        print(f"  time       : {elapsed:.1f}s")

        writer.add_scalar("train/loss_epoch", train_loss,           epoch)
        writer.add_scalar("val/loss",         val_results["loss"],  epoch)
        writer.add_scalar("val/f1",           val_results["f1"],    epoch)
        writer.add_scalar("val/iou",          val_results["iou"],   epoch)
        writer.add_scalar("val/precision",    val_results["precision"], epoch)
        writer.add_scalar("val/recall",       val_results["recall"],   epoch)
        writer.add_scalar("train/lr",         lr_now,               epoch)

        # ── Checkpoint ────────────────────────────────────────────────────────
        save_checkpoint(model, optimiser, epoch, best_f1, cfg, last_ckpt)

        if val_results["f1"] > best_f1:
            best_f1    = val_results["f1"]
            no_improve = 0
            save_checkpoint(model, optimiser, epoch, best_f1, cfg, best_ckpt)
            print(f"  *** New best F1: {best_f1:.4f} ***")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{patience} epochs")

        # ── Early stopping ────────────────────────────────────────────────────
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs)")
            break

    writer.close()
    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
    print(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()