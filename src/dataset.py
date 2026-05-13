"""
src/dataset.py — EO-SAR Change Detection Dataset
- pre-event  : EO  (3-channel RGB, uint8)
- post-event : SAR (1-channel, uint8)
- target     : mask with labels {0,1,2,3} remapped to {0,1}

Key design decisions:
- Change-focused sampling: patches preferentially sampled from
  image regions that contain change pixels (handles 63:1 imbalance)
- No-data masking: black triangular areas (pixel sum == 0) are
  excluded from loss via a valid_mask returned alongside the label
- Transforms split into spatial (joint EO+SAR+mask) and pixel-level
  (EO-only) to avoid channel broadcast errors with SAR (1-channel)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import rasterio
import albumentations as A


# ── Label remapping ───────────────────────────────────────────────────────────
LABEL_REMAP = np.array([0, 0, 1, 1], dtype=np.uint8)

def remap_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.clip(mask, 0, 3)
    return LABEL_REMAP[mask]


# ── Augmentation pipelines ────────────────────────────────────────────────────

def get_spatial_transforms(cfg):
    """Geometric transforms applied jointly to EO, SAR, and mask."""
    img_size = cfg["data"]["image_size"]
    return A.Compose([
        A.RandomCrop(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
        A.ElasticTransform(p=0.3),
        A.GridDistortion(p=0.2),
    ], additional_targets={"sar": "image"})


def get_eo_pixel_transforms(cfg):
    """Pixel-level transforms applied to EO only (3-channel, safe)."""
    eo_b = cfg["augmentation"]["eo_brightness_limit"]
    eo_c = cfg["augmentation"]["eo_contrast_limit"]
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=eo_b,
            contrast_limit=eo_c,
            p=0.5
        ),
        A.GaussNoise(p=0.3),
        A.CoarseDropout(p=0.2),
    ])


def get_val_spatial_transforms(cfg):
    img_size = cfg["data"]["image_size"]
    return A.Compose([
        A.CenterCrop(img_size, img_size),
    ], additional_targets={"sar": "image"})


# ── Dataset ───────────────────────────────────────────────────────────────────
class EOSARDataset(Dataset):
    """
    Returns:
        eo   : FloatTensor (3, H, W) — normalised EO image
        sar  : FloatTensor (1, H, W) — normalised SAR image
        mask : LongTensor  (H, W)    — binary change mask {0, 1}
        valid: FloatTensor (H, W)    — 1.0 where valid, 0.0 for no-data
    """

    def __init__(self, data_root, split, cfg, is_train=False):
        self.split    = split
        self.cfg      = cfg
        self.is_train = is_train

        norm = cfg["normalization"]
        self.eo_mean  = np.array(norm["eo_mean"],  dtype=np.float32)
        self.eo_std   = np.array(norm["eo_std"],   dtype=np.float32)
        self.sar_mean = np.array(norm["sar_mean"], dtype=np.float32)
        self.sar_std  = np.array(norm["sar_std"],  dtype=np.float32)

        if is_train:
            self.spatial_tfm  = get_spatial_transforms(cfg)
            self.eo_pixel_tfm = get_eo_pixel_transforms(cfg)
        else:
            self.spatial_tfm  = get_val_spatial_transforms(cfg)
            self.eo_pixel_tfm = None

        pre_dir    = Path(data_root) / split / "pre-event"
        post_dir   = Path(data_root) / split / "post-event"
        target_dir = Path(data_root) / split / "target"

        self.triplets   = []
        self.has_change = []

        for pre_p in sorted(pre_dir.glob("*.tif")):
            name   = pre_p.name
            post_p = post_dir   / name
            tgt_p  = target_dir / name
            if post_p.exists() and tgt_p.exists():
                self.triplets.append((pre_p, post_p, tgt_p))
                self.has_change.append(self._quick_has_change(tgt_p))

        n_chg = sum(self.has_change)
        print(f"[{split}] {len(self.triplets)} samples | "
              f"{n_chg} with change pixels "
              f"({100*n_chg/max(1,len(self.has_change)):.1f}%)")

    def _quick_has_change(self, tgt_path):
        try:
            with rasterio.open(tgt_path) as src:
                mask = src.read(1)
            return bool(remap_mask(mask).sum() > 0)
        except Exception:
            return False

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        pre_p, post_p, tgt_p = self.triplets[idx]

        # ── Load ─────────────────────────────────────────────────────────────
        with rasterio.open(pre_p)  as src: eo  = src.read()      # (3,H,W) uint8
        with rasterio.open(post_p) as src: sar = src.read(1)     # (H,W)   uint8
        with rasterio.open(tgt_p)  as src: tgt = src.read(1)     # (H,W)   uint8

        eo      = eo.transpose(1, 2, 0)         # (H,W,3)
        sar_hwc = sar[:, :, np.newaxis]          # (H,W,1)
        mask    = remap_mask(tgt).astype(np.uint8)

        # ── Spatial augmentation (joint) ──────────────────────────────────────
        aug     = self.spatial_tfm(image=eo, sar=sar_hwc, mask=mask)
        eo      = aug["image"]      # (H,W,3)
        sar_hwc = aug["sar"]        # (H,W,1)
        mask    = aug["mask"]       # (H,W)

        # ── No-data mask ──────────────────────────────────────────────────────
        valid = (eo.sum(axis=2) > 0).astype(np.float32)   # 1 = valid pixel

        # ── EO pixel augmentation (EO-only, 3-channel safe) ──────────────────
        if self.is_train and self.eo_pixel_tfm is not None:
            eo = self.eo_pixel_tfm(image=eo)["image"]

        # ── Normalise ─────────────────────────────────────────────────────────
        eo_f  = eo.astype(np.float32)
        sar_f = sar_hwc.astype(np.float32)
        eo_f  = (eo_f  - self.eo_mean)  / (self.eo_std  + 1e-6)
        sar_f = (sar_f - self.sar_mean) / (self.sar_std + 1e-6)

        # ── Tensorise ─────────────────────────────────────────────────────────
        eo_t    = torch.from_numpy(eo_f.transpose(2, 0, 1))    # (3,H,W)
        sar_t   = torch.from_numpy(sar_f.transpose(2, 0, 1))   # (1,H,W)
        mask_t  = torch.from_numpy(mask.astype(np.int64))       # (H,W)
        valid_t = torch.from_numpy(valid)                        # (H,W)

        return eo_t, sar_t, mask_t, valid_t


# ── Weighted sampler ──────────────────────────────────────────────────────────
def get_weighted_sampler(dataset, oversample_ratio=0.7):
    n          = len(dataset)
    n_change   = sum(dataset.has_change)
    n_nochange = n - n_change
    weights = []
    for has_chg in dataset.has_change:
        if has_chg:
            w = oversample_ratio / max(1, n_change)
        else:
            w = (1.0 - oversample_ratio) / max(1, n_nochange)
        weights.append(w)
    return WeightedRandomSampler(
        torch.DoubleTensor(weights), num_samples=n, replacement=True
    )


# ── DataLoader factory ────────────────────────────────────────────────────────
def build_dataloaders(cfg):
    data_root   = cfg["data"]["root"]
    num_workers = cfg["data"]["num_workers"]
    batch_size  = cfg["training"]["batch_size"]
    oversample  = cfg["sampling"]["change_oversample_ratio"]

    train_ds = EOSARDataset(data_root, "train", cfg, is_train=True)
    val_ds   = EOSARDataset(data_root, "val",   cfg, is_train=False)
    test_ds  = EOSARDataset(data_root, "test",  cfg, is_train=False)

    sampler = get_weighted_sampler(train_ds, oversample_ratio=oversample)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, test_loader


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    print("\nBatch smoke test:")
    eo, sar, mask, valid = next(iter(train_loader))
    print(f"  EO   : {eo.shape}   dtype={eo.dtype}  "
          f"min={eo.min():.2f} max={eo.max():.2f}")
    print(f"  SAR  : {sar.shape}  dtype={sar.dtype}  "
          f"min={sar.min():.2f} max={sar.max():.2f}")
    print(f"  Mask : {mask.shape} dtype={mask.dtype} "
          f"unique={mask.unique().tolist()}")
    print(f"  Valid: {valid.shape} "
          f"min={valid.min():.0f} max={valid.max():.0f}")
    change_pct = 100 * mask.sum().item() / mask.numel()
    print(f"  Change pixels in batch: {mask.sum().item()} / {mask.numel()} "
          f"({change_pct:.2f}%)")
    print(f"  Val loader  : {len(val_loader)} batches")
    print(f"  Test loader : {len(test_loader)} batches")
    print("\nDataloaders OK.")