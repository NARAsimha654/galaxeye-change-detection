# Binary Change Detection on EO-SAR Image Pairs

### GalaxEye Space — AI Research Intern Technical Assessment

---

## Project Description

This project implements binary pixel-level change detection on co-registered Electro-Optical (EO) and Synthetic Aperture Radar (SAR) satellite image pairs across multiple disaster events. Given a pre-event RGB image (EO) and a post-event radar image (SAR), the model classifies each pixel as **Changed (1)** or **Unchanged (0)**.

**V1 Baseline Architecture:** Dual-Encoder Siamese UNet with independent ResNet34 encoders for EO and SAR, fusing modality features and their absolute difference at each decoder scale. Trained with Focal and Dice loss.

**V2 Architectural Upgrades:** Building on V1, V2 incorporates **cross-modal attention** at the bottleneck, **auxiliary supervision heads**, and **per-image instance normalization** to address cross-domain shift. The V2 loss was expanded to include **Lovász loss**.

**Key results (V1 baseline vs V2 upgrades):**

| Split   | Method   | Threshold | IoU        | Precision  | Recall     | F1         |
| ------- | -------- | --------- | ---------- | ---------- | ---------- | ---------- |
| Val     | V2 + TTA | 0.7       | 0.4674     | 0.6677     | 0.6091     | 0.6371     |
| Val     | V1 + TTA | 0.7       | 0.4604     | 0.6688     | 0.5963     | 0.6305     |
| Test    | V1       | 0.5       | 0.0392     | 0.0672     | 0.0858     | 0.0754     |
| Test    | V2       | 0.5       | 0.0000     | 0.0000     | 0.0000     | 0.0000     |

---

## Repository Structure

```
galaxeye-change-detection/
├── data/                        # Dataset root (not committed — see Dataset Structure)
│   ├── train/
│   │   ├── pre-event/           # EO images (3-channel RGB, 1024×1024)
│   │   ├── post-event/          # SAR images (1-channel, 1024×1024)
│   │   └── target/              # Label masks (values: 0,1,2,3)
│   ├── val/
│   │   ├── pre-event/
│   │   ├── post-event/
│   │   └── target/
│   └── test/
│       ├── pre-event/
│       ├── post-event/
│       └── target/
├── src/
│   ├── dataset.py               # Dataloader, label remapping, augmentations
│   ├── model.py                 # Dual-encoder UNet + early fusion baseline
│   ├── losses.py                # Focal loss + Dice loss (combined)
│   ├── metrics.py               # IoU, Precision, Recall, F1, confusion matrix
│   └── tta.py                   # Test-time augmentation (8-fold)
├── outputs/
│   ├── exploration/             # Data analysis plots (from explore.py)
│   ├── eval/                    # Standard evaluation outputs
│   └── eval_tta/                # TTA evaluation outputs
├── weights/                     # Saved model checkpoints
├── train.py                     # Training script
├── eval.py                      # Evaluation script
├── eval_tta.py                  # Evaluation with test-time augmentation
├── explore.py                   # Dataset exploration and analysis
├── config.yaml                  # All hyperparameters (single source of truth)
├── requirements.txt             # Pinned dependencies
└── README.md
```

---

## Requirements

- Python 3.11
- CUDA 11.8 (tested on NVIDIA RTX 3060 6GB)

All dependencies with pinned versions are listed in `requirements.txt`. Key packages:

```
torch==2.7.1+cu118
torchvision==0.22.1+cu118
segmentation-models-pytorch==0.3.4
albumentations==2.4.0
rasterio==1.4.3
numpy==2.2.5
matplotlib==3.10.1
scikit-learn==1.6.1
tqdm==4.67.1
pyyaml==6.0.2
tensorboard==2.19.0
opencv-python==4.11.0.86
```

---

## Environment Setup

**Step 1 — Clone the repository:**

```bash
git clone https://github.com/<your-username>/galaxeye-change-detection.git
cd galaxeye-change-detection
```

**Step 2 — Create and activate virtual environment:**

```bash
# Using venv (Python 3.11 required)
python -m venv .venv

# Activate — Windows PowerShell
.venv/Scripts/activate

# Activate — Linux/macOS
source .venv/bin/activate
```

**Step 3 — Install PyTorch with CUDA 11.8:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Step 4 — Install remaining dependencies:**

```bash
pip install -r requirements.txt
```

**Step 5 — Verify installation:**

```bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import rasterio; import segmentation_models_pytorch; print('All deps OK')"
```

> **CPU-only fallback:** If no GPU is available, replace Step 3 with `pip install torch torchvision torchaudio`. Training will be significantly slower.

---

## Dataset Structure

Download the dataset and place it under the `data/` directory at the repository root. The expected layout after extraction:

```
data/
├── train/
│   ├── pre-event/
│   │   ├── scene_01_000001_building_damage.tif
│   │   ├── scene_01_000002_building_damage.tif
│   │   └── ...                  (2781 files)
│   ├── post-event/
│   │   └── ...                  (2781 files, same names)
│   └── target/
│       └── ...                  (2781 files, same names)
├── val/
│   ├── pre-event/               (334 files)
│   ├── post-event/
│   └── target/
└── test/
    ├── pre-event/               (77 files)
    ├── post-event/
    └── target/
```

**File format:** All files are GeoTIFF (`.tif`).

- `pre-event/`: 3-band uint8 EO (RGB optical image, pre-disaster)
- `post-event/`: 1-band uint8 SAR (radar backscatter image, post-disaster)
- `target/`: 1-band uint8 mask with values {0, 1, 2, 3}

**Label remapping** (applied automatically in the dataloader before any training or evaluation):

| Original | Meaning    | Remapped | Class     |
| -------- | ---------- | -------- | --------- |
| 0        | Background | 0        | No-Change |
| 1        | Intact     | 0        | No-Change |
| 2        | Damaged    | 1        | Change    |
| 3        | Destroyed  | 1        | Change    |

---

## Data Exploration

Before training, generate dataset statistics and visualisations:

```bash
python explore.py
```

Outputs saved to `outputs/exploration/`:

- `class_distribution.png` — per-split class balance charts
- `sample_triplets.png` — 6 EO/SAR/mask sample visualisations
- `sar_distribution.png` — SAR pixel distribution for change vs no-change regions

Console output includes per-split change pixel ratios and normalisation statistics.

---

## Training

Train from scratch using the provided configuration:

```bash
python train.py --config config.yaml
```

**Resume from a checkpoint:**

```bash
python train.py --config config.yaml --resume weights/last_model.pth
```

**What to expect (V2):**

- Epochs 1–5: val F1 typically near 0.0 (model learning class imbalance)
- Epochs 10–40: F1 climbs into 0.4–0.5 range
- Best checkpoint saved automatically to `weights/best_model_v2.pth` whenever val F1 improves
- Early stopping triggers after 20 epochs without improvement (e.g. stopped at epoch 71)

**TensorBoard monitoring:**

```bash
tensorboard --logdir outputs/runs
```

**Key config options for V1** (`config.yaml`):
- `epochs: 60`
- `batch_size: 8` (reduced to 4 in V2 due to VRAM constraints)
- `loss`: Focal + Dice

**Key config options for V2** (`config_v2.yaml`):
- `epochs: 80` (allows longer training for the more complex network)
- `batch_size: 4` (required due to cross-attention memory overhead)
- `loss`: Focal + Dice + Lovász

---

## Evaluation

**Standard evaluation on test split:**

```bash
python eval.py --config config.yaml --weights weights/best_model.pth --split test
```

**Standard evaluation on val split:**

```bash
python eval.py --config config.yaml --weights weights/best_model.pth --split val
```

**Evaluation with custom data path:**

```bash
python eval.py --config config.yaml --weights weights/best_model.pth \
               --data_path /path/to/data --split test
```

**Evaluation with custom threshold:**

```bash
python eval.py --config config.yaml --weights weights/best_model.pth \
               --split val --threshold 0.7
```

**Skip qualitative visualisations (faster):**

```bash
python eval.py --config config.yaml --weights weights/best_model.pth \
               --split test --no_vis
```

**Evaluation with test-time augmentation (8-fold):**

```bash
python eval_tta.py --config config.yaml --weights weights/best_model.pth \
                   --split val --threshold 0.7
```

Each evaluation run saves to `outputs/eval/<split>/`:

- `results.txt` — all metrics
- `confusion_matrix.png` — styled confusion matrix
- `visualisations/sample_XX.png` — 10 qualitative prediction figures (EO | SAR | GT | Pred | Overlay)

---

## Model Weights

The final trained checkpoint is publicly available for download:

**Google Drive:** `<PASTE YOUR GOOGLE DRIVE LINK HERE>`

Place the downloaded file at: `weights/best_model.pth`

**Checkpoint contents (V2):**

```python
{
    "epoch":     51,               # Epoch at which best val F1 was achieved (early stopped at 71)
    "best_f1":   0.5866,           # Best val F1 at standard threshold 0.5
    "model":     state_dict,       # Model weights (~50.2M parameters)
    "optimiser": state_dict,       # AdamW optimiser state
    "cfg":       dict              # Full config at training time
}
```

**Verify the checkpoint loads correctly:**

```bash
python -c "
import torch
ckpt = torch.load('weights/best_model.pth', map_location='cpu')
print('Epoch:', ckpt['epoch'])
print('Best F1:', ckpt['best_f1'])
print('Keys:', list(ckpt['model'].keys())[:5], '...')
"
```

---

## Results

### Metrics (Change class only, as per assignment specification)

**Validation split — threshold sweep (V1 vs V2):**

| Model | Threshold | IoU        | Precision  | Recall     | F1         |
| ----- | --------- | ---------- | ---------- | ---------- | ---------- |
| V1    | 0.7       | 0.4440     | 0.6315     | 0.5993     | 0.6149     |
| V1    | 0.7 + TTA | 0.4604     | 0.6688     | 0.5963     | 0.6305     |
| **V2**| **0.7 + TTA**| **0.4674** | **0.6677** | **0.6091** | **0.6371** |

**Test split:**

| Model | Method   | Threshold | IoU    | Precision | Recall | F1     |
| ----- | -------- | --------- | ------ | --------- | ------ | ------ |
| V1    | Standard | 0.5       | 0.0392 | 0.0672    | 0.0858 | 0.0754 |
| V1    | +TTA     | 0.5       | 0.0320 | 0.0701    | 0.0557 | 0.0621 |
| **V2**| Standard | 0.5       | 0.0000 | 0.0000    | 0.0000 | 0.0000 |

**Primary reported metrics:**

- **Val F1 = 0.6371** (V2 with TTA, threshold 0.7)
- **Test F1 = 0.0754** (V1 standard, threshold 0.5)

> **Note on test performance:** Test scenes (09–10) are geographically distinct from all training and validation scenes (01–08). V1 achieved a weak 0.0754 F1 on the test set. In V2, we attempted to bridge this domain gap using per-image instance normalization, cross-modal attention, and augmented domain limits. Paradoxically, this caused a **complete test set collapse (F1 = 0.0000)**. This indicates that instance normalization washed out absolute radar backscatter intensities that the model needed, or that the higher-capacity V2 model catastrophically overfitted to the training domain. See the technical report for full analysis.

### Confusion Matrices

**Validation (threshold 0.7, TTA):**

```
                  Predicted No-Change    Predicted Change
Actual No-Change       74,964,808             713,077
Actual Change             974,823           1,440,006
```

**Test (threshold 0.5, standard):**

```
                  Predicted No-Change    Predicted Change
Actual No-Change       17,648,710             201,465
Actual Change             154,654              14,521
```

---

## Design Decisions

### V1 Core Decisions
**Why dual encoders?** EO (optical) and SAR (radar) record fundamentally different physical properties. A shared encoder would conflate RGB colour/texture features with radar backscatter geometry. Separate encoders allow modality-specific feature learning before fusion.

**Why feature difference fusion?** Concatenating `[EO_feat, SAR_feat, |EO_feat - SAR_feat|]` at each decoder scale provides an explicit change signal. Buildings that have collapsed show very different SAR features from their pre-event EO appearance.

**Why Focal + Dice loss?** With 63:1 class imbalance, BCE loss collapses to predicting all No-Change. Focal Loss penalises easy correct predictions. Dice Loss directly optimises the overlap metric.

**Why threshold 0.7?** The weighted patch sampler biases training toward change-containing images, slightly inflating confidence on the change class. Threshold tuning on the validation set corrects for this.

### V2 Iterative Upgrades
**Why add cross-attention?** In V2, we added cross-modal attention at the bottleneck so the model learns how optical features condition radar features before decoding, attempting to build a stronger cross-modal representation.

**Why add Lovász loss?** Lovász-Softmax Loss was added in V2 because it acts as a direct surrogate for the discrete Jaccard index, providing smoother gradient descent for the IoU metric than standard Dice loss.

**Why instance normalization?** We observed severe cross-domain performance drops on the test set in V1. In V2, we replaced dataset-wide normalisation with per-image instance normalization to remove scene-specific brightness/contrast biases. While this improved validation performance, it catastrophically failed on the test set, indicating that absolute radar backscatter intensities (which were normalized away) are critical for generalisation.

---

## References

1. Daudt et al. (2018) — Fully Convolutional Siamese Networks for Change Detection. ICIP.
2. Chen et al. (2021) — Remote Sensing Image Change Detection with Transformers. IEEE TGRS.
3. Fang et al. (2021) — SNUNet-CD: A Densely Connected Siamese Network for Change Detection. IEEE GRSL.
4. Bandara & Patel (2022) — A Transformer-Based Siamese Network for Change Detection. IGARSS.
5. Lin et al. (2017) — Focal Loss for Dense Object Detection. ICCV.
6. Milletari et al. (2016) — V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. 3DV.
7. Schmitt & Zhu (2016) — Data Fusion and Remote Sensing. IEEE GRSM.
8. He et al. (2016) — Deep Residual Learning for Image Recognition. CVPR.
9. Yakubovskiy (2019) — Segmentation Models PyTorch. GitHub: qubvel/segmentation_models.pytorch.

---

## Citation / Acknowledgements

Built for the GalaxEye Space AI Research Intern technical assessment. Architecture implemented using the `segmentation-models-pytorch` library. Pretrained encoder weights from ImageNet via HuggingFace Hub. LLM-assisted implementation (Anthropic Claude) as permitted by assessment guidelines.
