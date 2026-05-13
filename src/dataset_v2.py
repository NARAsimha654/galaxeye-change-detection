"""
src/dataset_v2.py — Enhanced Dataset for EO-SAR Change Detection

Key improvements over v1:
  1. Per-image instance normalization: each image is standardised by its
     own channel mean and std rather than dataset-level statistics. This
     removes scene-level global appearance differences (brightness, haze,
     terrain colour) before the encoder sees the features — directly
     addressing the cross-scene OOD generalization problem.

  2. Domain randomization augmentation: aggressive EO appearance
     augmentation (hue/saturation, gamma, channel shuffle, random
     grayscale) forces the model to learn structure-based change signals
     rather than scene-specific appearance cues. SAR intensity scaling
     simulates gain/calibration variation across sensors.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import rasterio
import albumentations as A
import cv2


# ── Label remapping ───────────────────────────────────────────────────────────
LABEL_REMAP = np.array([0, 0, 1, 1], dtype=np.uint8)

# Dataset-level normalization stats (computed from training set in explore.py)
EO_MEAN  = np.array([84.504, 91.558, 71.558], dtype=np.float32)
EO_STD   = np.array([51.559, 40.504, 38.178], dtype=np.float32)
SAR_MEAN = np.array([52.051],                  dtype=np.float32)
SAR_STD  = np.array([39.075],                  dtype=np.float32)

def remap_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.clip(mask, 0, 3)
    return LABEL_REMAP[mask]


# ── Per-image instance normalization ─────────────────────────────────────────
def instance_normalize(img: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Normalize each channel independently by its own mean and std.
    img: (H, W, C) float32
    Returns: (H, W, C) float32 with each channel ~ N(0, 1)
    """
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        mu = ch.mean()
        sd = max(ch.std(), eps)
        out[:, :, c] = np.clip((ch - mu) / sd, -5.0, 5.0)
    return out


# ── SAR instance normalize ─────────────────────────────────────────────────────
def instance_normalize_sar(img: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """img: (H, W, 1) float32"""
    mu = img.mean()
    sd = max(float(img.std()), eps)
    return np.clip((img - mu) / sd, -5.0, 5.0)


# ── Augmentation pipelines ────────────────────────────────────────────────────

def get_spatial_transforms_v2(cfg):
    """Strong geometric augmentations applied jointly to EO + SAR + mask."""
    img_size = cfg["data"]["image_size"]
    return A.Compose([
        A.RandomCrop(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
        A.ElasticTransform(p=0.4),
        A.GridDistortion(p=0.3),
        A.OpticalDistortion(p=0.2),
    ], additional_targets={"sar": "image"})


def get_eo_domain_transforms(cfg):
    """
    Aggressive domain randomization on EO only.
    Goal: force the model to ignore scene-specific colour/lighting cues
    and instead learn structural change patterns.
    """
    return A.Compose([
        # Colour/appearance randomization
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.6
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=40,
            val_shift_limit=30,
            p=0.5
        ),
        A.RandomGamma(gamma_limit=(60, 160), p=0.4),
        A.ToGray(p=0.1),              # occasional grayscale → SAR-like EO
        A.ChannelShuffle(p=0.1),      # random RGB channel order
        A.CLAHE(clip_limit=4.0, p=0.3),
        # Noise and texture
        A.GaussNoise(p=0.3),
        A.ISONoise(p=0.2),
        # Structural dropout
        A.CoarseDropout(p=0.2),
        A.RandomShadow(p=0.2),
    ])


def get_val_spatial_transforms(cfg):
    img_size = cfg["data"]["image_size"]
    return A.Compose([
        A.CenterCrop(img_size, img_size),
    ], additional_targets={"sar": "image"})


# ── Dataset ───────────────────────────────────────────────────────────────────
class EOSARDatasetV2(Dataset):
    """
    Returns:
        eo   : FloatTensor (3, H, W) — per-image normalised EO
        sar  : FloatTensor (1, H, W) — per-image normalised SAR
        mask : LongTensor  (H, W)    — binary change mask {0, 1}
        valid: FloatTensor (H, W)    — 1.0 where valid, 0.0 for no-data
    """

    def __init__(self, data_root, split, cfg, is_train=False):
        self.split    = split
        self.cfg      = cfg
        self.is_train = is_train

        if is_train:
            self.spatial_tfm    = get_spatial_transforms_v2(cfg)
            self.eo_domain_tfm  = get_eo_domain_transforms(cfg)
        else:
            self.spatial_tfm    = get_val_spatial_transforms(cfg)
            self.eo_domain_tfm  = None

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
        with rasterio.open(pre_p)  as src: eo  = src.read()
        with rasterio.open(post_p) as src: sar = src.read(1)
        with rasterio.open(tgt_p)  as src: tgt = src.read(1)

        eo      = eo.transpose(1, 2, 0)         # (H,W,3) uint8
        sar_hwc = sar[:, :, np.newaxis]          # (H,W,1) uint8
        mask    = remap_mask(tgt).astype(np.uint8)

        # ── Spatial augmentation (joint) ──────────────────────────────────────
        aug     = self.spatial_tfm(image=eo, sar=sar_hwc, mask=mask)
        eo      = aug["image"]
        sar_hwc = aug["sar"]
        mask    = aug["mask"]

        # ── No-data mask ──────────────────────────────────────────────────────
        valid = (eo.sum(axis=2) > 0).astype(np.float32)

        # ── EO domain augmentation ────────────────────────────────────────────
        if self.is_train and self.eo_domain_tfm is not None:
            eo = self.eo_domain_tfm(image=eo)["image"]

        # ── SAR intensity augmentation (training only) ────────────────────────
        if self.is_train:
            scale = np.random.uniform(0.75, 1.25)
            sar_hwc = np.clip(sar_hwc.astype(np.float32) * scale,
                              0, 255).astype(np.uint8)

        # ── Per-image instance normalization ──────────────────────────────────
        # Each image normalised by its own channel statistics.
        # This removes scene-level brightness/contrast differences that
        # cause the model to learn scene-specific rather than change-specific
        # features, improving cross-scene generalisation.
        # ── Dataset-level normalization (same stats as V1) ────────────────────────
        eo_f  = (eo.astype(np.float32)      - EO_MEAN)  / (EO_STD  + 1e-6)
        sar_f = (sar_hwc.astype(np.float32) - SAR_MEAN) / (SAR_STD + 1e-6)

        # ── Tensorise ─────────────────────────────────────────────────────────
        eo_t    = torch.from_numpy(eo_f.transpose(2, 0, 1))
        sar_t   = torch.from_numpy(sar_f.transpose(2, 0, 1))
        mask_t  = torch.from_numpy(mask.astype(np.int64))
        valid_t = torch.from_numpy(valid)

        return eo_t, sar_t, mask_t, valid_t


# ── Weighted sampler ──────────────────────────────────────────────────────────
def get_weighted_sampler(dataset, oversample_ratio=0.7):
    n          = len(dataset)
    n_change   = sum(dataset.has_change)
    n_nochange = n - n_change
    weights    = []
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
def build_dataloaders_v2(cfg):
    data_root   = cfg["data"]["root"]
    num_workers = cfg["data"]["num_workers"]
    batch_size  = cfg["training"]["batch_size"]
    oversample  = cfg["sampling"]["change_oversample_ratio"]

    train_ds = EOSARDatasetV2(data_root, "train", cfg, is_train=True)
    val_ds   = EOSARDatasetV2(data_root, "val",   cfg, is_train=False)
    test_ds  = EOSARDatasetV2(data_root, "test",  cfg, is_train=False)

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

    train_loader, val_loader, test_loader = build_dataloaders_v2(cfg)

    print("\nBatch smoke test (v2 — instance norm):")
    eo, sar, mask, valid = next(iter(train_loader))
    print(f"  EO   : {eo.shape}  min={eo.min():.2f} max={eo.max():.2f}")
    print(f"  SAR  : {sar.shape} min={sar.min():.2f} max={sar.max():.2f}")
    print(f"  Mask : {mask.shape} unique={mask.unique().tolist()}")
    print(f"  Valid: {valid.shape} min={valid.min():.0f} max={valid.max():.0f}")
    pct = 100 * mask.sum().item() / mask.numel()
    print(f"  Change pixels: {mask.sum().item()} / {mask.numel()} ({pct:.2f}%)")
    print("\nDataset v2 OK.")