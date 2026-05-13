"""
explore.py — Dataset exploration and analysis for GalaxEye EO-SAR Change Detection
Run: python explore.py
Outputs saved to: outputs/exploration/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path(r"C:\Narasimha\Internship\GalaxEye\data")
OUT_DIR     = Path("outputs/exploration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS      = ["train", "val", "test"]
LABEL_REMAP = {0: 0, 1: 0, 2: 1, 3: 1}   # Background/Intact→0, Damaged/Destroyed→1
ORIG_NAMES  = {0: "Background", 1: "Intact", 2: "Damaged", 3: "Destroyed"}
REMAP_NAMES = {0: "No-Change", 1: "Change"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_tif(path):
    with rasterio.open(path) as src:
        return src.read()   # (bands, H, W)

def remap_mask(mask):
    out = np.zeros_like(mask)
    for orig, remapped in LABEL_REMAP.items():
        out[mask == orig] = remapped
    return out

def get_triplets(split):
    """Return list of (pre_path, post_path, target_path) for a split."""
    pre_dir    = DATA_ROOT / split / "pre-event"
    post_dir   = DATA_ROOT / split / "post-event"
    target_dir = DATA_ROOT / split / "target"
    files = sorted(pre_dir.glob("*.tif"))
    triplets = []
    for pre in files:
        name = pre.name
        post   = post_dir   / name
        target = target_dir / name
        if post.exists() and target.exists():
            triplets.append((pre, post, target))
    return triplets

# ── 1. File counts ────────────────────────────────────────────────────────────
print("=" * 60)
print("1. FILE COUNTS")
print("=" * 60)
split_counts = {}
for split in SPLITS:
    triplets = get_triplets(split)
    split_counts[split] = len(triplets)
    print(f"  {split:>5}: {len(triplets)} triplets")

# ── 2. Image statistics ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. IMAGE STATISTICS (sampled from train)")
print("=" * 60)

triplets = get_triplets("train")
sample_size = min(50, len(triplets))
sample = triplets[:sample_size]

eo_means, eo_stds = [], []
sar_means, sar_stds = [], []
shapes = set()

for pre_p, post_p, _ in sample:
    eo  = load_tif(pre_p).astype(np.float32)   # (3, H, W)
    sar = load_tif(post_p).astype(np.float32)  # (1, H, W)
    shapes.add(eo.shape[1:])
    eo_means.append(eo.mean(axis=(1, 2)))
    eo_stds.append(eo.std(axis=(1, 2)))
    sar_means.append(sar.mean())
    sar_stds.append(sar.std())

eo_mean = np.mean(eo_means, axis=0)
eo_std  = np.mean(eo_stds,  axis=0)
sar_mean = np.mean(sar_means)
sar_std  = np.mean(sar_stds)

print(f"  EO  (pre-event)  — mean per channel: {eo_mean.round(2)}, std: {eo_std.round(2)}")
print(f"  SAR (post-event) — mean: {sar_mean:.2f}, std: {sar_std:.2f}")
print(f"  Image shapes found: {shapes}")
print(f"  Sampled {sample_size} files")

# ── 3. Class distribution (original + remapped) ───────────────────────────────
print("\n" + "=" * 60)
print("3. CLASS DISTRIBUTION PER SPLIT")
print("=" * 60)

split_stats = {}
for split in SPLITS:
    triplets = get_triplets(split)
    orig_counts   = defaultdict(int)
    remap_counts  = defaultdict(int)
    total_pixels  = 0

    for _, _, tgt_p in triplets:
        mask = load_tif(tgt_p)[0]
        for v in [0, 1, 2, 3]:
            orig_counts[v] += int((mask == v).sum())
        remapped = remap_mask(mask)
        remap_counts[0] += int((remapped == 0).sum())
        remap_counts[1] += int((remapped == 1).sum())
        total_pixels += mask.size

    change_pct = 100 * remap_counts[1] / total_pixels if total_pixels > 0 else 0
    split_stats[split] = {
        "orig": dict(orig_counts),
        "remap": dict(remap_counts),
        "total": total_pixels,
        "change_pct": change_pct
    }

    print(f"\n  [{split.upper()}]  total pixels: {total_pixels:,}")
    print(f"    Original labels:")
    for v in [0, 1, 2, 3]:
        pct = 100 * orig_counts[v] / total_pixels if total_pixels > 0 else 0
        print(f"      {ORIG_NAMES[v]:>12} ({v}): {orig_counts[v]:>12,}  ({pct:.2f}%)")
    print(f"    After remapping:")
    for v in [0, 1]:
        pct = 100 * remap_counts[v] / total_pixels if total_pixels > 0 else 0
        print(f"      {REMAP_NAMES[v]:>10} ({v}): {remap_counts[v]:>12,}  ({pct:.2f}%)")
    print(f"    → Change pixel ratio: {change_pct:.2f}%")

# ── 4. Class imbalance bar chart ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Class Distribution After Remapping (per split)", fontsize=14, fontweight='bold')

colors = ['#4CAF50', '#F44336']
for ax, split in zip(axes, SPLITS):
    stats  = split_stats[split]
    counts = [stats["remap"][0], stats["remap"][1]]
    bars   = ax.bar(["No-Change (0)", "Change (1)"], counts, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_title(f"{split.capitalize()} split", fontsize=12)
    ax.set_ylabel("Pixel Count")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    for bar, count in zip(bars, counts):
        pct = 100 * count / stats["total"]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f"{pct:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / "class_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: outputs/exploration/class_distribution.png")

# ── 5. Visual inspection — 6 sample triplets ──────────────────────────────────
print("\n" + "=" * 60)
print("4. GENERATING VISUAL SAMPLES")
print("=" * 60)

triplets = get_triplets("train")
# Try to find samples WITH change pixels for meaningful vis
selected = []
for pre_p, post_p, tgt_p in triplets:
    mask = load_tif(tgt_p)[0]
    remapped = remap_mask(mask)
    if remapped.sum() > 500:   # at least 500 change pixels
        selected.append((pre_p, post_p, tgt_p))
    if len(selected) == 6:
        break

if len(selected) < 6:          # fallback: just take first N
    selected = triplets[:6]

fig, axes = plt.subplots(6, 4, figsize=(20, 30))
fig.suptitle("Sample Triplets: EO (pre) | SAR (post) | Original Mask | Remapped Mask",
             fontsize=14, fontweight='bold')

col_titles = ["EO — Pre-event (RGB)", "SAR — Post-event (1-band)", "Original Labels", "Remapped (Binary)"]
for ax, title in zip(axes[0], col_titles):
    ax.set_title(title, fontsize=11, fontweight='bold')

for row, (pre_p, post_p, tgt_p) in enumerate(selected):
    eo   = load_tif(pre_p)          # (3, H, W) uint8
    sar  = load_tif(post_p)[0]      # (H, W)
    tgt  = load_tif(tgt_p)[0]       # (H, W)
    remap = remap_mask(tgt)

    # EO — clip to 0-255 and transpose to HWC
    eo_rgb = np.clip(eo.transpose(1, 2, 0), 0, 255).astype(np.uint8)

    axes[row, 0].imshow(eo_rgb)
    axes[row, 0].set_ylabel(pre_p.name[:30], fontsize=7)

    axes[row, 1].imshow(sar, cmap='gray')

    im2 = axes[row, 2].imshow(tgt, cmap='viridis', vmin=0, vmax=3)
    plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

    im3 = axes[row, 3].imshow(remap, cmap='RdYlGn_r', vmin=0, vmax=1)
    plt.colorbar(im3, ax=axes[row, 3], fraction=0.046)

    for ax in axes[row]:
        ax.axis('off')

plt.tight_layout()
plt.savefig(OUT_DIR / "sample_triplets.png", dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: outputs/exploration/sample_triplets.png")

# ── 6. SAR pixel value distribution ──────────────────────────────────────────
print("\n" + "=" * 60)
print("5. SAR PIXEL VALUE DISTRIBUTION")
print("=" * 60)

sar_vals_change    = []
sar_vals_nochange  = []

sample_triplets = get_triplets("train")[:30]
for _, post_p, tgt_p in sample_triplets:
    sar  = load_tif(post_p)[0].astype(np.float32)
    tgt  = remap_mask(load_tif(tgt_p)[0])
    sar_vals_change.extend(sar[tgt == 1].flatten().tolist())
    sar_vals_nochange.extend(sar[tgt == 0].flatten().tolist())

sar_change   = np.array(sar_vals_change[:200000])
sar_nochange = np.array(sar_vals_nochange[:200000])

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(sar_nochange, bins=100, alpha=0.6, color='steelblue', label='No-Change', density=True)
ax.hist(sar_change,   bins=100, alpha=0.6, color='crimson',   label='Change',    density=True)
ax.set_title("SAR Pixel Value Distribution: Change vs No-Change regions", fontsize=13)
ax.set_xlabel("Pixel Value (0-255)")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "sar_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/exploration/sar_distribution.png")

# ── 7. Scene diversity ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. SCENE DIVERSITY")
print("=" * 60)

for split in SPLITS:
    triplets  = get_triplets(split)
    scene_ids = set()
    for pre_p, _, _ in triplets:
        # filename: scene_XX_NNNNNN_building_damage.tif
        parts = pre_p.stem.split("_")
        if len(parts) >= 2:
            scene_ids.add(f"{parts[0]}_{parts[1]}")
    print(f"  {split:>5}: {len(triplets)} images across {len(scene_ids)} scenes → {sorted(scene_ids)}")

# ── 8. Summary printout ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPLORATION COMPLETE — SUMMARY")
print("=" * 60)
print(f"  EO  normalization mean (per channel): {eo_mean.round(3)}")
print(f"  EO  normalization std  (per channel): {eo_std.round(3)}")
print(f"  SAR normalization mean: {sar_mean:.3f}")
print(f"  SAR normalization std : {sar_std:.3f}")
for split in SPLITS:
    print(f"  {split:>5} change ratio: {split_stats[split]['change_pct']:.2f}%")
print(f"\n  All outputs saved to: outputs/exploration/")
print("  Use these stats for normalization in dataset.py")