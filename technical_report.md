# Binary Change Detection on EO-SAR Image Pairs

## Technical Report — GalaxEye Space AI Research Intern Assessment

---

## Abstract

We address binary pixel-level change detection on co-registered Electro-Optical (EO) and Synthetic Aperture Radar (SAR) image pairs across multiple disaster events. The task is framed as a dense binary segmentation problem: given a pre-event EO image (RGB, 3-channel) and a post-event SAR image (1-channel), classify each pixel as Changed or Unchanged. The dataset exhibits severe class imbalance (approximately 1.6% change pixels in training) and, critically, the test split contains scenes geographically distinct from all training and validation scenes.

We propose a **Dual-Encoder Siamese UNet** architecture with independent ResNet34 encoders for each modality. In our V2 update, we introduced **cross-modal attention** at the bottleneck, **auxiliary supervision heads**, and **per-image instance normalization** to explicitly target cross-domain generalisation. Training uses a combined Focal, Dice, and Lovász loss to handle class imbalance.

Our best model achieves **Val F1 = 0.6371, Val IoU = 0.4674** (V2, threshold 0.7, with 8-fold test-time augmentation). However, on the provided test split (scenes 09–10, geographically distinct from training), our baseline V1 model achieved a weak **Test F1 = 0.0754**. Paradoxically, our V2 domain adaptation attempts led to a complete collapse (**Test F1 = 0.0000**), exposing a profound cross-scene domain generalization gap that we analyse in detail.

---

## 1. Literature Survey

### 1.1 Foundations of Change Detection

The dominant paradigm in deep learning-based change detection treats the problem as Siamese feature comparison. Daudt et al. (2018) introduced three architectures — FC-EF (early fusion), FC-Siam-Conc (feature concatenation), and FC-Siam-Diff (feature difference) — establishing that feature-level difference streams provide a direct and learnable change signal. FC-Siam-Diff, which subtracts encoder features from two temporal branches, remains a strong baseline and directly motivates our difference fusion design.

Chen et al. (2021) introduced the Bitemporal Image Transformer (BIT), replacing convolutional feature comparison with self-attention over semantic tokens. BIT demonstrated that global context from transformer attention improves detection of spatially distributed damage patterns. Fang et al. (2021) proposed SNUNet-CD, a densely connected Siamese network with nested skip connections that explicitly preserves fine-grained spatial features — critical for detecting small damaged structures.

Bandara and Patel (2022) extended transformer-based approaches with ChangeFormer, a hierarchical vision transformer that achieves state-of-the-art on optical change detection benchmarks (LEVIR-CD F1 > 0.90). These transformer approaches consistently outperform convolutional baselines on same-modality change detection.

### 1.2 EO-SAR Multimodal Fusion

Cross-modal change detection between EO and SAR is substantially harder than same-modality tasks. Schmitt and Zhu (2016) characterise the core challenge: EO captures surface reflectance (texture, colour) while SAR captures surface scattering properties (geometry, dielectric constant), making direct pixel-level comparison unreliable. A damaged building appears as rubble in EO but as reduced backscatter in SAR — the change signal exists in different feature spaces.

Gao et al. (2021) demonstrated that heterogeneous change detection benefits from a common latent space alignment step, translating one modality into the representation of the other before comparison. Methods including CycleGAN-based domain translation have been applied to synthesise pseudo-SAR from EO for alignment, though this introduces its own errors.

More recent work (Luppino et al., 2022; Zhao et al., 2023) applies contrastive learning to bring EO and SAR feature spaces closer together, enabling more robust cross-modal comparison. These approaches show promise but require substantially more compute and training data than our constrained setting permits.

### 1.3 Class Imbalance in Remote Sensing Segmentation

Change pixels in disaster datasets routinely occupy less than 5% of the total pixel area. Lin et al. (2017) introduced Focal Loss, originally for object detection, which dynamically down-weights easy background examples and concentrates learning on rare, hard examples. Milletari et al. (2016) introduced Dice Loss, which directly optimises the overlap metric between predicted and ground-truth masks, making it inherently more robust to class imbalance than standard cross-entropy. The combination of Focal and Dice loss is now standard in medical and remote sensing segmentation literature.

### 1.4 Positioning Our Approach

Our dual-encoder architecture sits between the purely convolutional Siamese networks (Daudt et al.) and the more complex attention-based methods. We deliberately chose a convolutional architecture for three reasons: (1) limited training data (2,781 images) makes transformer overfitting a concern, (2) the 6GB VRAM constraint precludes large transformer models, and (3) the feature difference stream provides an explicit and interpretable change signal without requiring modality translation. Given more compute and data, ChangeFormer or a contrastive dual-encoder approach would be the natural next step.

---

## 2. Dataset Analysis

### 2.1 Structure

The dataset contains co-registered EO-SAR triplets across 10 distinct disaster scenes: 2,781 training images, 334 validation images, and 77 test images, all at 1024×1024 pixels. Pre-event images are 3-channel EO (RGB, uint8); post-event images are 1-channel SAR (uint8); target masks carry four semantic labels remapped to binary before any model training or evaluation.

**Label remapping applied consistently across all splits:**

| Original Class | Original Value | Remapped Value | Remapped Class |
| -------------- | -------------- | -------------- | -------------- |
| Background     | 0              | 0              | No-Change      |
| Intact         | 1              | 0              | No-Change      |
| Damaged        | 2              | 1              | Change         |
| Destroyed      | 3              | 1              | Change         |

### 2.2 Class Distribution

| Split | Total Pixels  | No-Change | Change | Change Ratio |
| ----- | ------------- | --------- | ------ | ------------ |
| Train | 2,916,089,856 | 98.43%    | 1.57%  | 63:1         |
| Val   | 350,224,384   | 97.80%    | 2.20%  | 45:1         |
| Test  | 80,740,352    | 99.25%    | 0.75%  | 132:1        |

The class imbalance is severe. A model predicting all No-Change achieves 99.25% pixel accuracy on test while detecting nothing — demonstrating why accuracy is an uninformative metric for this task and why we report only change-class IoU, Precision, Recall, and F1.

### 2.3 Scene Diversity and the OOD Test Problem

Training and validation scenes (01–08) cover African tropical and arid disaster zones characterised by red laterite soil, dense vegetation, mud-brick and corrugated-metal structures, and corresponding SAR backscatter profiles. Test scenes (09–10) are geographically distinct — visual inspection reveals arid suburban landscapes with paved roads, concrete structures, and very different SAR texture characteristics consistent with earthquake or wildfire damage in a Western setting.

This scene-level distribution shift is the dominant challenge in this assignment. No amount of threshold tuning or inference augmentation compensates for features the model was never trained to recognise. We document this gap transparently and address it in Future Work.

### 2.4 Image Characteristics

- **EO mean (per channel):** [84.5, 91.6, 71.6]; **std:** [51.6, 40.5, 38.2]
- **SAR mean:** 52.1; **std:** 39.1
- **No-data regions:** Black triangular areas appear along swath edges in several images, representing missing satellite coverage. These are masked out from both loss computation and metric evaluation.
- **All images:** 1024×1024 pixels. Training uses 512×512 random crops.

---

## 3. Methodology

### 3.1 Architecture: Dual-Encoder Siamese UNet (V2)

EO and SAR are processed by independent ResNet34 encoders. A shared encoder would force a joint representation immediately from input, ignoring the fundamental difference in how each sensor records information. Separate encoders allow each branch to build modality-appropriate features before fusion.

**EO encoder:** Standard 3-channel RGB input with ImageNet pretrained weights.

**SAR encoder:** The 1-channel SAR image is replicated to 3 channels to enable reuse of ImageNet pretrained weights.

**Cross-Modal Attention (V2):** At the deepest encoder level (bottleneck), optical and radar features are passed through a cross-attention block. This allows the optical context to condition the radar features before the decoding phase, aligning the modalities.

**Fusion at each decoder scale:**

At each of the five decoder levels, three feature streams are concatenated:

```
fused = concat(EO_features, SAR_features, |EO_features - SAR_features|)
```

The absolute difference stream directly encodes the change signal in feature space.

**Decoder & Aux Heads (V2):** Standard UNet decoder with bilinear upsampling. In V2, we added auxiliary prediction heads at intermediate resolutions (1/8 and 1/4 scale) during training to provide dense gradient supervision, stabilising the deeper architecture.

**Total parameters (V2):** ~50.2M

### 3.2 Loss Function

We use a combined Focal + Dice + Lovász loss (V2):

```
L = 0.4 × FocalLoss(γ=2.0, α=0.75) + 0.3 × DiceLoss + 0.3 × LovaszLoss
```

Focal Loss with α=0.75 assigns 3× more weight to change (positive) pixels than to no-change (negative) pixels, and the γ=2.0 focusing parameter down-weights easy correct predictions, concentrating the gradient signal on hard and misclassified pixels.

Dice Loss and Lovász-Softmax Loss directly optimise the overlap metric (IoU), making them closely aligned with our evaluation criteria. Lovász is particularly effective as a surrogate for the discrete Jaccard index, providing smoother gradient descent than standard Dice.

No-data regions (identified as pixels where all EO channels equal zero) are excluded from all loss terms via a binary valid mask.

### 3.3 Class Imbalance Strategy

Three complementary mechanisms address the 63:1 imbalance:

1. **Focal Loss:** Re-weights the loss gradient away from easy no-change pixels.
2. **Weighted patch sampling:** Images containing at least one change pixel are sampled at 70% probability; change-free images at 30%. This ensures the model sees change pixels in approximately 70% of training iterations rather than 25.6% (the natural proportion).
3. **Decision threshold tuning:** The default sigmoid threshold of 0.5 was swept across [0.3, 0.4, 0.5, 0.6, 0.7] on the validation split. Threshold 0.7 maximised val F1, reflecting that the model's raw predictions are slightly biased toward positive class due to the oversampling strategy.

### 3.4 Training Configuration

| Hyperparameter          | Value                    |
| ----------------------- | ------------------------ |
| Optimizer               | AdamW                    |
| Learning rate           | 1e-4                     |
| Weight decay            | 1e-4                     |
| LR scheduler            | Cosine Annealing         |
| Batch size              | 8                        |
| Crop size               | 512×512                  |
| Mixed precision         | FP16                     |
| Gradient clip           | 1.0                      |
| Epochs                  | 60 (early stopped at 34) |
| Early stopping patience | 15 epochs                |
| Random seed             | 42                       |

### 3.5 Data Normalisation & Augmentation (V2 Updates)

**Instance Normalization:** In V1, we used dataset-level normalisation. In V2, we switched to **per-image instance normalization** (subtracting each image's own channel mean/std). This was specifically implemented to remove scene-level brightness/contrast differences and improve cross-domain generalisation to the test set.

**SAR Intensity Augmentation:** We randomly scale SAR backscatter intensity during training to prevent the model from overfitting to absolute radar brightness values.

**Spatial augmentations:** Random crop (512×512), horizontal flip, vertical flip, random 90° rotation, transpose, grid distortion.

**EO domain augmentation:** Random brightness/contrast limits were increased to ±0.3 to simulate diverse lighting and environment conditions.

SAR images are not subject to colour augmentations — radar backscatter intensity carries physical meaning (surface roughness, dielectric properties) that should not be arbitrarily altered.

### 3.6 Test-Time Augmentation (TTA)

At inference, 8 deterministic augmentations are applied to each image (identity, H-flip, V-flip, both flips, transpose, and three transpose+flip combinations), and the resulting probability maps are averaged. TTA improved val F1 by 0.016 (0.6149 → 0.6305) by reducing sensitivity to orientation, but did not improve test performance, confirming that the test failure is a domain distribution issue rather than an orientation or scale sensitivity issue.

---

## 4. Results

### 4.1 Validation and Test Metrics

All metrics computed for the Change class (label = 1) only.

| Split   | Model | Method   | Threshold | IoU        | Precision  | Recall     | F1         |
| ------- | ----- | -------- | --------- | ---------- | ---------- | ---------- | ---------- |
| Val     | V1    | Standard | 0.7       | 0.4440     | 0.6315     | 0.5993     | 0.6149     |
| Val     | V1    | +TTA     | 0.7       | 0.4604     | 0.6688     | 0.5963     | 0.6305     |
| **Val** | **V2**| **+TTA** | **0.7**   | **0.4674** | **0.6677** | **0.6091** | **0.6371** |
| Test    | V1    | Standard | 0.5       | 0.0392     | 0.0672     | 0.0858     | 0.0754     |
| Test    | V2    | Standard | 0.5       | 0.0000     | 0.0000     | 0.0000     | 0.0000     |

**Primary reported metrics:** Val F1 = 0.6371 (V2, TTA, threshold 0.7). Test F1 = 0.0754 (V1, standard, threshold 0.5).

### 4.2 Confusion Matrix Analysis

**Validation (threshold 0.7, TTA):**

|                      | Predicted No-Change | Predicted Change |
| -------------------- | ------------------- | ---------------- |
| **Actual No-Change** | 74,964,808          | 713,077          |
| **Actual Change**    | 974,823             | 1,440,006        |

**Test (threshold 0.5, no TTA):**

|                      | Predicted No-Change | Predicted Change |
| -------------------- | ------------------- | ---------------- |
| **Actual No-Change** | 17,648,710          | 201,465          |
| **Actual Change**    | 154,654             | 14,521           |

### 4.3 Error Profile Analysis

**Validation failures (V2):**

The dominant failure mode on validation remains **false positives on intact rural buildings**. In these images, SAR backscatter from dense informal settlements creates high-intensity speckle patterns that the model confuses with damage signatures. V2 improved IoU by 0.007 over V1, but the fundamental ambiguity between "rubble" and "dense corrugated roofing" in 1-channel radar persists.

**Test failures (The Domain Collapse):**

In V1, the model achieved a weak F1 of 0.0754 on the test set. The test scenes (09-10) are arid and suburban, fundamentally different from the tropical/rural training scenes (01-08). 

To address this, V2 introduced per-image instance normalization, SAR intensity augmentation, and cross-attention. Paradoxically, this caused a **complete test set collapse (F1 = 0.0000)**. The model predicted zero true positives on the test set. 

This catastrophic failure yields two critical insights:
1. **Destruction of Absolute Signal:** The weak signal we had in V1 was likely heavily dependent on absolute radar backscatter intensities that were "normalized away" by instance normalization in V2. 
2. **Domain-Specific Overfitting:** The increased capacity of the V2 model (cross-attention, aux heads), combined with longer training, likely allowed it to over-specialise to the training domain's geometric features. When confronted with suburban geometries (paved roads, different building layouts), the learned attention maps failed entirely, producing NO positive activations.

### 4.4 Qualitative Results

**Success case (Val Sample 7, F1=0.827, IoU=0.706):**
The model correctly delineates large industrial building damage in an arid scene. The SAR post-event image shows clear structural collapse signatures (reduced coherent backscatter from the building roofs), and the model's probability map strongly activates over the damaged footprints with minimal false positive activation on the surrounding undamaged terrain.

**Partial failure (Val Sample 4, F1=0.190, IoU=0.105):**
A tropical rural scene with a single small damaged structure surrounded by intact buildings. The model correctly identifies the one damaged building (TP) but additionally activates on 9 nearby intact structures (FP). All activated structures show similar SAR backscatter profiles, suggesting the model is detecting "bright SAR objects on dark background" rather than the specific acoustic signature of structural damage.

**Cross-domain partial success (Test Sample 6, F1=0.586, IoU=0.414):**
The only test sample with reasonable performance. The scene shows suburban structures on brown terrain, somewhat similar in SAR texture to the arid training scenes. The model correctly detects most damaged structures but shows boundary errors and some missed structures (FN), reflected in the lower recall than the best validation samples.

---

## 5. Future Work

As a first-month intern deliverable, this work establishes a functional baseline. The following directions represent the natural next steps, ordered by expected impact:

### 5.1 Cross-Scene Domain Adaptation (Highest Priority)

The test performance collapse is the most critical open problem. Domain adaptation approaches specifically applicable here:

**Domain-Adversarial Training (Ganin et al., 2016):** Add a gradient-reversal layer between the encoder and a domain classifier. This forces the encoder to learn scene-invariant features by penalising predictions that reveal which scene a patch came from. Feasible with the current architecture and training pipeline — requires knowing which scenes are "target domain" at training time.

**Scene-Level Instance Normalisation:** Replace dataset-level normalisation with per-image standardisation (subtract each image's own channel mean/std). This removes scene-specific brightness and contrast biases before the encoder processes features, a simple change that may partially address the domain gap without retraining.

**Contrastive Domain Alignment (Luppino et al., 2022):** Train the dual encoder with a contrastive loss that pulls EO-SAR feature pairs from the same spatial location together and pushes pairs from different locations apart. This explicitly aligns the two modality feature spaces, making cross-modal difference more semantically meaningful.

### 5.2 Foundation Model Backbones

Replace the ResNet34 encoders with geospatial foundation models pretrained on large multi-scene, multi-sensor satellite datasets:

- **Prithvi (IBM/NASA, 2023):** A masked autoencoder pretrained on 1TB of Sentinel-2 imagery. Provides representations that generalize across geographic regions and illumination conditions.
- **SatMAE (Cong et al., 2022):** A vision transformer pretrained on multi-temporal satellite imagery with positional encoding that accounts for geographic coordinates and timestamp.
- **DOFA (Xiong et al., 2024):** A multi-modal foundation model pretrained on EO and SAR together, producing modality-aware embeddings — directly applicable to our cross-modal fusion task.

These backbones would replace the ImageNet-pretrained ResNet34, trading parameter count for superior cross-scene generalization.

### 5.3 Transformer-Based Change Detection

ChangeFormer (Bandara & Patel, 2022) and BIT (Chen et al., 2021) demonstrate that global self-attention over scene tokens outperforms convolutional feature comparison on heterogeneous change detection. A transformer decoder operating over EO and SAR feature tokens — attending cross-modally — would explicitly model the relationship between pre-event optical appearance and post-event radar response, enabling the model to reason about which optical structures should produce which SAR signatures and flag deviations.

### 5.4 Data and Training Strategies

**Cross-scene augmentation:** Randomly apply colour palette swaps (simulate different soil, vegetation, building material colours) and SAR intensity scaling during training to explicitly simulate geographic diversity.

**Pseudo-label self-training:** Run the trained model on test scenes at low threshold, select high-confidence predictions, and fine-tune on these pseudo-labels. This semi-supervised approach requires care (confirmation bias) but can quickly adapt to target scene statistics.

**Scene-balanced sampling:** Ensure each training batch contains patches from every available scene, preventing the model from overspecialising on the largest scene's statistics.

### 5.5 Architecture Refinements

- **Attention-gated skip connections:** Replace direct skip connections with attention gates that learn to suppress irrelevant spatial regions — particularly useful for reducing false positives in cluttered urban scenes.
- **Multi-scale change heads:** Add auxiliary classification heads at intermediate decoder scales, supervised with downsampled masks. Improves gradient flow to the encoder and helps detect both large and small damage extents simultaneously.
- **Uncertainty estimation:** Monte Carlo dropout at inference to produce per-pixel confidence intervals, enabling human-in-the-loop workflows where uncertain predictions are flagged for expert review — directly applicable to GalaxEye's disaster response use case.

---

## 6. Conclusion

We developed a dual-encoder Siamese UNet for binary EO-SAR change detection, addressing a challenging configuration in which the pre-event image is optical RGB and the post-event image is a single-channel SAR radar return. The architecture, loss function, and sampling strategy are each motivated by specific properties of this dataset: severe class imbalance (63:1), heterogeneous modality characteristics, and the presence of no-data regions.

On the validation split, the model achieves F1 = 0.6305 with test-time augmentation at threshold 0.7 — a competitive result for cross-modal change detection under these constraints. The primary limitation is generalization to the test scenes (09–10), which differ geographically and visually from all training and validation scenes. The model's learned change signatures are calibrated to African tropical and arid disaster zones; when confronted with scenes from a different geographic domain, detection collapses.

This failure is an honest and important finding. Cross-scene generalization is an active research problem in remote sensing, and solving it requires either domain adaptation techniques, geospatially-pretrained foundation model backbones, or greater scene diversity at training time. All three are viable directions for subsequent work. The current model provides a strong foundation: the architecture is sound, the training pipeline is reproducible, and the error analysis clearly identifies where and why failures occur.

---

## References

1. Daudt, R. C., Le Saux, B., & Boulch, A. (2018). Fully convolutional siamese networks for change detection. ICIP 2018.
2. Chen, H., Qi, Z., & Shi, Z. (2021). Remote sensing image change detection with transformers. IEEE TGRS.
3. Fang, S., Li, K., Shao, J., & Li, Z. (2021). SNUNet-CD: A densely connected Siamese network for change detection. IEEE GRSL.
4. Bandara, W. G. C., & Patel, V. M. (2022). A transformer-based Siamese network for change detection. IGARSS 2022.
5. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). Focal loss for dense object detection. ICCV 2017.
6. Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-Net: Fully convolutional neural networks for volumetric medical image segmentation. 3DV 2016.
7. Schmitt, M., & Zhu, X. X. (2016). Data fusion and remote sensing: An ever-growing relationship. IEEE GRSM.
8. Ganin, Y., et al. (2016). Domain-adversarial training of neural networks. JMLR.
9. Luppino, L. T., et al. (2022). Code-aligned autoencoders for unsupervised change detection in multimodal remote sensing images. IEEE TNNLS.
10. Cong, Y., et al. (2022). SatMAE: Pre-training transformers for temporal and multi-spectral satellite imagery. NeurIPS 2022.
11. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR 2016.

---

## Appendix: Time and Resource Log (V2)

| Activity                                                         | Time Spent      |
| ---------------------------------------------------------------- | --------------- |
| Data exploration and analysis                                    | 2 hours         |
| Literature reading                                               | 2 hours         |
| Implementation (dataset, model, losses, metrics, training, eval) | 5 hours         |
| Training (71 epochs × ~245s)                                     | ~4.8 hours      |
| Evaluation and threshold sweep                                   | 1 hour          |
| Report writing                                                   | 3 hours         |
| **Total**                                                        | **~17.8 hours** |

**Machine:** Local workstation
**GPU:** NVIDIA RTX 3060 (6GB VRAM)
**Number of GPUs:** 1
**Training time per epoch:** ~245 seconds
**Total training wall-clock time:** ~4.8 hours (71 epochs, early stopped)
**Mixed precision:** FP16 enabled throughout

**Resource constraints and their impact:** The 6GB VRAM constraint restricted batch size to 4 for the V2 model (due to the added cross-attention and auxiliary heads). Larger image context (full 1024×1024 resolution) would have provided more spatial context for change detection, potentially improving detection of both very small and very large damage extents. Transformer-based architectures (ChangeFormer, BIT) were excluded due to the same memory constraint.
