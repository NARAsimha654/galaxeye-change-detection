"""
src/tta.py — Test-Time Augmentation for EO-SAR Change Detection

Applies multiple deterministic augmentations at inference time and
averages the probability maps. Improves generalization to unseen scenes
by reducing sensitivity to orientation and scale.

Augmentations used:
  - Original (no flip)
  - Horizontal flip
  - Vertical flip
  - Both flips (180 rotation equivalent)
  - Transpose
  - Transpose + horizontal flip
  - Transpose + vertical flip
  - Transpose + both flips
"""

import torch
import torch.nn.functional as F


def tta_inference(model, eo, sar, threshold=0.5):
    """
    Run TTA inference on a single batch.

    Args:
        model     : trained model
        eo        : (B, 3, H, W) EO tensor on device
        sar       : (B, 1, H, W) SAR tensor on device
        threshold : decision threshold

    Returns:
        avg_prob  : (B, H, W) averaged probability map
        pred      : (B, H, W) binary prediction
    """
    model.eval()

    def predict(eo_t, sar_t):
        with torch.no_grad():
            logit = model(eo_t, sar_t)
            return torch.sigmoid(logit.squeeze(1))  # (B,H,W)

    probs = []

    # 1. Original
    probs.append(predict(eo, sar))

    # 2. Horizontal flip
    eo_hf  = torch.flip(eo,  dims=[3])
    sar_hf = torch.flip(sar, dims=[3])
    p = predict(eo_hf, sar_hf)
    probs.append(torch.flip(p, dims=[2]))

    # 3. Vertical flip
    eo_vf  = torch.flip(eo,  dims=[2])
    sar_vf = torch.flip(sar, dims=[2])
    p = predict(eo_vf, sar_vf)
    probs.append(torch.flip(p, dims=[1]))

    # 4. Both flips
    eo_bf  = torch.flip(eo,  dims=[2, 3])
    sar_bf = torch.flip(sar, dims=[2, 3])
    p = predict(eo_bf, sar_bf)
    probs.append(torch.flip(p, dims=[1, 2]))

    # 5. Transpose (swap H and W)
    eo_tp  = eo.transpose(2, 3)
    sar_tp = sar.transpose(2, 3)
    p = predict(eo_tp, sar_tp)
    probs.append(p.transpose(1, 2))

    # 6. Transpose + H flip
    eo_tph  = torch.flip(eo_tp,  dims=[3])
    sar_tph = torch.flip(sar_tp, dims=[3])
    p = predict(eo_tph, sar_tph)
    probs.append(torch.flip(p, dims=[2]).transpose(1, 2))

    # 7. Transpose + V flip
    eo_tpv  = torch.flip(eo_tp,  dims=[2])
    sar_tpv = torch.flip(sar_tp, dims=[2])
    p = predict(eo_tpv, sar_tpv)
    probs.append(torch.flip(p, dims=[1]).transpose(1, 2))

    # 8. Transpose + both flips
    eo_tpb  = torch.flip(eo_tp,  dims=[2, 3])
    sar_tpb = torch.flip(sar_tp, dims=[2, 3])
    p = predict(eo_tpb, sar_tpb)
    probs.append(torch.flip(p, dims=[1, 2]).transpose(1, 2))

    avg_prob = torch.stack(probs, dim=0).mean(dim=0)   # (B,H,W)
    pred     = (avg_prob >= threshold).long()
    return avg_prob, pred