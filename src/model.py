"""
src/model.py — Dual-Encoder Siamese UNet for EO-SAR Change Detection

Architecture rationale:
  EO (optical RGB) and SAR (radar grayscale) are fundamentally different
  modalities with different noise profiles, dynamic ranges, and feature
  semantics. A single shared encoder would be forced to learn a joint
  representation immediately, losing modality-specific structure.

  Instead we use two independent ResNet34 encoders:
    - EO encoder : 3-channel RGB input, ImageNet pretrained
    - SAR encoder: 1-channel input, expanded to 3ch by replication
                   to reuse ImageNet pretrained weights (standard practice)

  At each decoder scale we fuse:
    - EO features
    - SAR features
    - Absolute difference (|EO - SAR|) — explicit change signal

  This "difference stream" is the core insight: damaged buildings show
  strong SAR backscatter changes (rubble vs intact structure) that are
  directly captured by the feature-space difference.

  The decoder is a standard UNet decoder with skip connections from
  both encoders concatenated at each scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder


# ── Decoder block ─────────────────────────────────────────────────────────────
class DecoderBlock(nn.Module):
    """
    Upsamples features and fuses with skip connections.
    Input channels = upsampled + eo_skip + sar_skip + diff_skip
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── Dual-Encoder UNet ─────────────────────────────────────────────────────────
class DualEncoderUNet(nn.Module):
    """
    Dual-encoder UNet for binary EO-SAR change detection.

    Forward input:
        eo  : (B, 3, H, W) — normalised EO image
        sar : (B, 1, H, W) — normalised SAR image (replicated → 3ch internally)

    Forward output:
        logits : (B, 1, H, W) — raw (pre-sigmoid) change predictions
    """

    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet",
                 decoder_channels=(256, 128, 64, 32, 16)):
        super().__init__()

        # ── EO encoder (3-channel RGB) ────────────────────────────────────────
        self.eo_encoder = get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights,
        )

        # ── SAR encoder (1-channel → replicated to 3ch for pretrained weights)
        self.sar_encoder = get_encoder(
            encoder_name,
            in_channels=3,          # we replicate SAR to 3ch in forward()
            depth=5,
            weights=encoder_weights,
        )

        # Encoder output channels at each scale (ResNet34):
        # [64, 64, 128, 256, 512] (stages 0-4, after stem)
        enc_channels = self.eo_encoder.out_channels  # e.g. (3, 64, 64, 128, 256, 512)
        # enc_channels[0] is the stem input (3), skip it for decoder
        enc_channels = enc_channels[1:]   # (64, 64, 128, 256, 512)

        # ── Decoder ───────────────────────────────────────────────────────────
        # At each scale: fuse eo_feat + sar_feat + |eo-sar| diff
        # Skip channels = 3 * enc_channels[i] (eo + sar + diff)
        # We go from deepest (enc_channels[-1]) up to shallowest

        dec_in  = list(decoder_channels)
        dec_out = list(decoder_channels)

        self.decoder_blocks = nn.ModuleList()

        # Number of decoder stages = len(enc_channels) - 1
        # Bottleneck → stage4 → stage3 → stage2 → stage1 → stage0
        n_stages = len(enc_channels)   # 5

        for i in range(n_stages):
            if i == 0:
                # First decoder block: input is bottleneck (deepest enc output)
                # No skip at this level (or use enc_channels[-2] as skip)
                in_ch   = enc_channels[-1] * 3   # eo + sar + diff at bottleneck
                skip_ch = enc_channels[-2] * 3   # skip from next level
                out_ch  = dec_out[i]
            elif i < n_stages - 1:
                in_ch   = dec_out[i-1]
                skip_ch = enc_channels[-(i+2)] * 3
                out_ch  = dec_out[i]
            else:
                # Last block: no more encoder skips
                in_ch   = dec_out[i-1]
                skip_ch = 0
                out_ch  = dec_out[i]

            self.decoder_blocks.append(
                DecoderBlock(in_ch, skip_ch, out_ch)
            )

        # ── Segmentation head ─────────────────────────────────────────────────
        self.seg_head = nn.Sequential(
            nn.Conv2d(dec_out[-1], dec_out[-1] // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_out[-1] // 2, 1, kernel_size=1),
        )

        self._init_decoder_weights()

    def _init_decoder_weights(self):
        for m in self.decoder_blocks.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
        for m in self.seg_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")

    def _fuse(self, eo_feat, sar_feat):
        """Concatenate EO, SAR, and their absolute difference."""
        diff = torch.abs(eo_feat - sar_feat)
        return torch.cat([eo_feat, sar_feat, diff], dim=1)

    def forward(self, eo, sar):
        # Replicate SAR (1ch) → 3ch to use pretrained encoder
        sar_3ch = sar.expand(-1, 3, -1, -1)

        # ── Encode ────────────────────────────────────────────────────────────
        eo_feats  = self.eo_encoder(eo)      # list of tensors, len=6 (stem + 5 stages)
        sar_feats = self.sar_encoder(sar_3ch)

        # eo_feats[0] = input, [1..5] = encoder stages
        eo_stages  = eo_feats[1:]    # 5 tensors, deepest last
        sar_stages = sar_feats[1:]

        # ── Decode ────────────────────────────────────────────────────────────
        # Start from bottleneck (deepest features)
        x = self._fuse(eo_stages[-1], sar_stages[-1])

        n = len(self.decoder_blocks)
        for i, block in enumerate(self.decoder_blocks):
            if i < n - 1:
                skip_idx = -(i + 2)          # next shallower skip level
                skip = self._fuse(eo_stages[skip_idx], sar_stages[skip_idx])
            else:
                skip = None
            x = block(x, skip)

        # ── Head ──────────────────────────────────────────────────────────────
        logits = self.seg_head(x)    # (B, 1, H, W)
        return logits


# ── Early-fusion baseline (simpler, for comparison) ───────────────────────────
class EarlyFusionUNet(nn.Module):
    """
    Baseline: concatenate EO (3ch) + SAR replicated (3ch) = 6ch input
    into a standard UNet. Much simpler than dual-encoder but useful as
    a performance lower bound.
    """
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet"):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=4,          # 3 (EO) + 1 (SAR)
            classes=1,
            activation=None,        # raw logits
        )

    def forward(self, eo, sar):
        x = torch.cat([eo, sar], dim=1)   # (B, 4, H, W)
        return self.model(x)


# ── Model factory ─────────────────────────────────────────────────────────────
def build_model(cfg):
    arch    = cfg["model"]["architecture"]
    encoder = cfg["model"]["encoder"]
    weights = cfg["model"]["encoder_weights"]

    if arch == "dual_encoder_unet":
        dec_ch = tuple(cfg["model"]["decoder_channels"])
        model  = DualEncoderUNet(
            encoder_name=encoder,
            encoder_weights=weights,
            decoder_channels=dec_ch,
        )
    elif arch == "early_fusion_unet":
        model = EarlyFusionUNet(
            encoder_name=encoder,
            encoder_weights=weights,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {arch} | encoder={encoder} | "
          f"params={n_params/1e6:.1f}M")
    return model


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dual encoder test ─────────────────────────────────────────────────────
    print("\n--- Dual Encoder UNet ---")
    model = build_model(cfg).to(device)

    B, H, W = 2, 512, 512
    eo  = torch.randn(B, 3, H, W).to(device)
    sar = torch.randn(B, 1, H, W).to(device)

    with torch.no_grad():
        out = model(eo, sar)

    print(f"  Input  EO : {eo.shape}")
    print(f"  Input  SAR: {sar.shape}")
    print(f"  Output    : {out.shape}")
    assert out.shape == (B, 1, H, W), f"Expected (B,1,H,W), got {out.shape}"
    print("  Shape assertion passed.")

    # Memory usage
    if device.type == "cuda":
        mem = torch.cuda.memory_allocated(device) / 1024**2
        print(f"  GPU memory: {mem:.1f} MB")

    # ── Early fusion test ─────────────────────────────────────────────────────
    print("\n--- Early Fusion UNet (baseline) ---")
    cfg_ef = dict(cfg)
    cfg_ef["model"] = dict(cfg["model"])
    cfg_ef["model"]["architecture"] = "early_fusion_unet"
    baseline = build_model(cfg_ef).to(device)

    with torch.no_grad():
        out_ef = baseline(eo, sar)

    print(f"  Output: {out_ef.shape}")
    assert out_ef.shape == (B, 1, H, W)
    print("  Shape assertion passed.")

    print("\nModel OK.")