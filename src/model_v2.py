"""
src/model_v2.py — Enhanced Dual-Encoder UNet for EO-SAR Change Detection

Improvements over v1:
  1. Cross-modal bottleneck attention: EO features attend to SAR and vice
     versa at the deepest encoder level, enabling explicit cross-modal
     reasoning before decoding begins.
  2. CBAM (Convolutional Block Attention Module) in every decoder block:
     channel and spatial attention help the decoder suppress false
     positive activations in cluttered backgrounds.
  3. Multi-scale auxiliary supervision: auxiliary segmentation heads at
     1/4 and 1/8 resolution provide additional gradient signal to early
     encoder layers, improving feature quality for small change regions.
  4. Deeper decoder fusion: each decoder block receives a richer fused
     skip (EO + SAR + diff + CBAM-attended features).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder


# ── CBAM: Convolutional Block Attention Module ────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx  = self.fc(self.max_pool(x))
        return self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return attn


class CBAM(nn.Module):
    """Channel + Spatial attention gate."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ── Cross-modal attention ──────────────────────────────────────────────────────
class CrossModalAttention(nn.Module):
    """
    Lightweight cross-modal attention at the bottleneck.
    EO features attend to SAR and vice versa using a scaled dot-product
    attention over spatially-pooled tokens.

    For memory efficiency we pool to a fixed spatial grid before attention.
    """
    def __init__(self, channels, pool_size=8):
        super().__init__()
        self.pool_size = pool_size
        mid = max(channels // 4, 32)

        # Projections for query, key, value
        self.q_eo  = nn.Conv2d(channels, mid, 1, bias=False)
        self.k_sar = nn.Conv2d(channels, mid, 1, bias=False)
        self.v_sar = nn.Conv2d(channels, channels, 1, bias=False)

        self.q_sar = nn.Conv2d(channels, mid, 1, bias=False)
        self.k_eo  = nn.Conv2d(channels, mid, 1, bias=False)
        self.v_eo  = nn.Conv2d(channels, channels, 1, bias=False)

        self.out_eo  = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.out_sar = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.scale = mid ** -0.5

    def _attend(self, q_feat, k_feat, v_feat, orig_size):
        """
        q_feat, k_feat, v_feat: (B, C, H, W)
        Returns attended features at orig_size.
        """
        B = q_feat.size(0)
        p = self.pool_size

        # Pool to fixed spatial grid
        q = F.adaptive_avg_pool2d(q_feat, p)  # (B, mid, p, p)
        k = F.adaptive_avg_pool2d(k_feat, p)
        v = F.adaptive_avg_pool2d(v_feat, p)

        # Flatten spatial
        q = q.flatten(2).permute(0, 2, 1)  # (B, p*p, mid)
        k = k.flatten(2)                    # (B, mid, p*p)
        v = v.flatten(2).permute(0, 2, 1)  # (B, p*p, C)

        attn = torch.softmax(
            torch.clamp(torch.bmm(q, k) * self.scale, -8.0, 8.0), dim=-1
        )                                   # (B, p*p, p*p)
        out  = torch.bmm(attn, v)           # (B, p*p, C)
        out  = out.permute(0, 2, 1).view(B, -1, p, p)
        out  = F.interpolate(out, size=orig_size,
                             mode="bilinear", align_corners=False)
        return out

    def forward(self, eo_feat, sar_feat):
        H, W = eo_feat.shape[2:]

        # EO queries SAR (what has changed in the SAR relative to EO?)
        q_eo  = self.q_eo(eo_feat)
        k_sar = self.k_sar(sar_feat)
        v_sar = self.v_sar(sar_feat)
        eo_enhanced  = self.out_eo(self._attend(q_eo, k_sar, v_sar, (H, W)))

        # SAR queries EO (what was the structure before?)
        q_sar = self.q_sar(sar_feat)
        k_eo  = self.k_eo(eo_feat)
        v_eo  = self.v_eo(eo_feat)
        sar_enhanced = self.out_sar(self._attend(q_sar, k_eo, v_eo, (H, W)))

        # Residual: add attended context to original features
        return eo_feat + eo_enhanced, sar_feat + sar_enhanced


# ── Decoder block with CBAM ───────────────────────────────────────────────────
class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        total_in = in_channels + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(total_in, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2,
                          mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.cbam(x)
        return x


# ── Auxiliary head ────────────────────────────────────────────────────────────
class AuxHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x, target_size):
        x = self.head(x)
        return F.interpolate(x, size=target_size,
                             mode="bilinear", align_corners=False)


# ── Enhanced Dual-Encoder UNet ────────────────────────────────────────────────
class DualEncoderUNetV2(nn.Module):
    """
    Enhanced Dual-Encoder Siamese UNet with:
      - Cross-modal bottleneck attention
      - CBAM in every decoder block
      - Multi-scale auxiliary supervision (training only)

    Forward args:
        eo        : (B, 3, H, W) — per-image normalised EO
        sar       : (B, 1, H, W) — per-image normalised SAR
        return_aux: bool — return auxiliary logits during training

    Returns:
        logits    : (B, 1, H, W)
        aux_logits: list of (B, 1, H, W) — only if return_aux=True
    """

    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet",
                 decoder_channels=(256, 128, 64, 32, 16)):
        super().__init__()

        self.eo_encoder  = get_encoder(encoder_name, in_channels=3,
                                       depth=5, weights=encoder_weights)
        self.sar_encoder = get_encoder(encoder_name, in_channels=3,
                                       depth=5, weights=encoder_weights)

        enc_channels = self.eo_encoder.out_channels[1:]  # skip stem input
        # ResNet34: (64, 64, 128, 256, 512)

        # Cross-modal attention at bottleneck (deepest level)
        self.cross_attn = CrossModalAttention(
            channels=enc_channels[-1], pool_size=8
        )

        # Decoder
        dec_out = list(decoder_channels)
        n       = len(enc_channels)
        self.decoder_blocks = nn.ModuleList()

        for i in range(n):
            if i == 0:
                in_ch   = enc_channels[-1] * 3
                skip_ch = enc_channels[-2] * 3
            elif i < n - 1:
                in_ch   = dec_out[i-1]
                skip_ch = enc_channels[-(i+2)] * 3
            else:
                in_ch   = dec_out[i-1]
                skip_ch = 0
            self.decoder_blocks.append(
                DecoderBlockV2(in_ch, skip_ch, dec_out[i])
            )

        # Main segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(dec_out[-1], dec_out[-1] // 2,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_out[-1] // 2, 1, kernel_size=1),
        )

        # Auxiliary heads at 1/8 and 1/4 resolution (decoder stages 1 and 2)
        self.aux_head_8  = AuxHead(dec_out[1])   # 1/8 resolution output
        self.aux_head_4  = AuxHead(dec_out[2])   # 1/4 resolution output

        self._init_weights()

    def _init_weights(self):
        for m in list(self.decoder_blocks.modules()) + \
                 list(self.seg_head.modules()) + \
                 list(self.aux_head_8.modules()) + \
                 list(self.aux_head_4.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _fuse(self, eo_feat, sar_feat):
        diff = torch.abs(eo_feat - sar_feat)
        return torch.cat([eo_feat, sar_feat, diff], dim=1)

    def forward(self, eo, sar, return_aux=False):
        H, W = eo.shape[2:]

        # Replicate SAR to 3ch for pretrained encoder
        sar_3ch = sar.expand(-1, 3, -1, -1)

        # Encode
        eo_feats  = self.eo_encoder(eo)[1:]      # 5 scales, shallowest first
        sar_feats = self.sar_encoder(sar_3ch)[1:]

        # Cross-modal attention at bottleneck
        eo_bot, sar_bot = self.cross_attn(eo_feats[-1], sar_feats[-1])

        # Overwrite deepest features with attention-enhanced versions
        eo_stages  = list(eo_feats[:-1])  + [eo_bot]
        sar_stages = list(sar_feats[:-1]) + [sar_bot]

        # Decode
        x = self._fuse(eo_stages[-1], sar_stages[-1])

        n = len(self.decoder_blocks)
        aux_outputs = []

        for i, block in enumerate(self.decoder_blocks):
            if i < n - 1:
                skip = self._fuse(eo_stages[-(i+2)], sar_stages[-(i+2)])
            else:
                skip = None
            x = block(x, skip)

            # Collect auxiliary outputs at stages 1 and 2 (1/8 and 1/4 res)
            if return_aux:
                if i == 1:
                    aux_outputs.append(self.aux_head_8(x, (H, W)))
                elif i == 2:
                    aux_outputs.append(self.aux_head_4(x, (H, W)))

        logits = self.seg_head(x)

        if return_aux:
            return logits, aux_outputs
        return logits


# ── Model factory ─────────────────────────────────────────────────────────────
def build_model_v2(cfg):
    dec_ch = tuple(cfg["model"]["decoder_channels"])
    model  = DualEncoderUNetV2(
        encoder_name=cfg["model"]["encoder"],
        encoder_weights=cfg["model"]["encoder_weights"],
        decoder_channels=dec_ch,
    )
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model v2] DualEncoderUNetV2 | "
          f"encoder={cfg['model']['encoder']} | params={n/1e6:.1f}M")
    return model


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model_v2(cfg).to(device)

    B, H, W = 2, 512, 512
    eo  = torch.randn(B, 3, H, W).to(device)
    sar = torch.randn(B, 1, H, W).to(device)

    # Training forward (with aux)
    logits, aux = model(eo, sar, return_aux=True)
    print(f"\nTraining mode:")
    print(f"  Main logits : {logits.shape}")
    for i, a in enumerate(aux):
        print(f"  Aux head {i+1} : {a.shape}")

    # Inference forward (no aux)
    with torch.no_grad():
        out = model(eo, sar, return_aux=False)
    print(f"\nInference mode:")
    print(f"  Output: {out.shape}")

    assert out.shape == (B, 1, H, W)
    print("  Shape assertion passed.")

    if device.type == "cuda":
        mem = torch.cuda.memory_allocated() / 1024**2
        print(f"  GPU memory: {mem:.1f} MB")

    print("\nModel v2 OK.")