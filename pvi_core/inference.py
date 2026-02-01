from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import cv2


# MVP findings (locked)
FINDINGS: List[str] = [
    "Pneumonia",
    "Tuberculosis",
    "Mass",
    "Nodule",
    "Pleural Effusion",
    "No Finding",
]

_MODEL: Optional[nn.Module] = None
_DEVICE: Optional[torch.device] = None


def _get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _build_model(num_classes: int) -> nn.Module:
    """
    EfficientNet-B0 backbone, ImageNet pretrained + new head.
    NOTE: Not medically trained yet. Probabilities/CAM may be meaningless until training.
    """
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    model.eval()
    return model


def get_model() -> nn.Module:
    global _MODEL, _DEVICE
    if _MODEL is None:
        _DEVICE = _get_device()
        _MODEL = _build_model(num_classes=len(FINDINGS)).to(_DEVICE)
        _MODEL.eval()
    return _MODEL


def preprocess(pil_img: Image.Image) -> torch.Tensor:
    """
    - grayscale -> resize 224 -> replicate to 3ch
    - ImageNet normalization (because pretrained weights)
    """
    img = pil_img.convert("L").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W)
    arr = np.stack([arr, arr, arr], axis=0)  # (3, H, W)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    arr = (arr - mean) / std

    t = torch.from_numpy(arr).unsqueeze(0)  # (1, 3, 224, 224)
    return t


@torch.no_grad()
def predict_probs(pil_img: Image.Image) -> Dict[str, float]:
    model = get_model()
    device = _DEVICE if _DEVICE is not None else _get_device()

    x = preprocess(pil_img).to(device)
    logits = model(x)  # (1, C)
    probs = torch.sigmoid(logits).detach().cpu().numpy()[0].astype(float)
    return {FINDINGS[i]: float(probs[i]) for i in range(len(FINDINGS))}


# -----------------------
# CAM utilities
# -----------------------

@dataclass
class CAMResult:
    heatmap: np.ndarray      # (224,224) float32 [0,1]
    overlay: Image.Image     # PIL RGB 224x224
    contours: Image.Image    # PIL RGB 224x224


def _find_target_layer(model: nn.Module) -> nn.Module:
    # timm efficientnet has `blocks`; using last block is a robust choice
    if hasattr(model, "blocks"):
        return model.blocks[-1]
    raise RuntimeError("Could not locate target layer for EfficientNet.")


def _to_uint8_gray(heatmap01: np.ndarray) -> np.ndarray:
    return np.clip(heatmap01 * 255.0, 0, 255).astype(np.uint8)


def _apply_colormap(heatmap01: np.ndarray) -> np.ndarray:
    h8 = _to_uint8_gray(heatmap01)
    cm = cv2.applyColorMap(h8, cv2.COLORMAP_TURBO)  # BGR uint8
    return cm


def _overlay_heatmap(pil_img: Image.Image, heatmap01: np.ndarray, alpha: float) -> Image.Image:
    base = pil_img.convert("RGB").resize((224, 224))
    base_np = np.array(base, dtype=np.uint8)

    cm_bgr = _apply_colormap(heatmap01)
    cm_rgb = cv2.cvtColor(cm_bgr, cv2.COLOR_BGR2RGB)

    out = (1.0 - alpha) * base_np.astype(np.float32) + alpha * cm_rgb.astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _overlay_contours(pil_img: Image.Image, heatmap01: np.ndarray, thresh: float = 0.55) -> Image.Image:
    base = pil_img.convert("RGB").resize((224, 224))
    base_np = np.array(base, dtype=np.uint8)

    mask = (heatmap01 >= float(thresh)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(base_np, contours, -1, (255, 255, 255), 2)
    return Image.fromarray(base_np)


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    denom = x.max() + 1e-8
    return (x / denom).astype(np.float32)


# -----------------------
# Grad-CAM
# -----------------------

def grad_cam(pil_img: Image.Image, target_index: int, alpha: float = 0.30) -> CAMResult:
    model = get_model()
    device = _DEVICE if _DEVICE is not None else _get_device()
    x = preprocess(pil_img).to(device)

    target_layer = _find_target_layer(model)
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def fwd_hook(_m, _inp, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        activations.append(out)

    def bwd_hook(_m, grad_inp, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        logits = model(x)
        score = logits[0, target_index]
        model.zero_grad(set_to_none=True)
        score.backward()

        A = activations[-1]   # (1,C,H,W)
        G = gradients[-1]     # (1,C,H,W)
        w = G.mean(dim=(2, 3), keepdim=True)
        cam = (w * A).sum(dim=1)[0]
        cam = F.relu(cam).detach().cpu().numpy()

        cam01 = _normalize01(cam)
        cam01 = cv2.resize(cam01, (224, 224), interpolation=cv2.INTER_LINEAR)

        overlay = _overlay_heatmap(pil_img, cam01, alpha=float(alpha))
        contours = _overlay_contours(pil_img, cam01, thresh=0.55)
        return CAMResult(heatmap=cam01, overlay=overlay, contours=contours)
    finally:
        h1.remove()
        h2.remove()


# -----------------------
# Score-CAM (slower, cleaner)
# -----------------------

@torch.no_grad()
def score_cam(
    pil_img: Image.Image,
    target_index: int,
    alpha: float = 0.30,
    max_channels: int = 64,
) -> CAMResult:
    """
    Score-CAM (approx):
    - Get activations A from target conv layer: (1,C,H,W)
    - Upsample each channel map -> mask in [0,1]
    - For each mask: apply to input image (element-wise) and forward pass -> score for target class
    - Weighted sum of activation maps by scores -> heatmap
    Notes:
    - Very slow if C is large. We cap channels with max_channels (top by activation energy).
    """
    model = get_model()
    device = _DEVICE if _DEVICE is not None else _get_device()
    x = preprocess(pil_img).to(device)  # (1,3,224,224)

    target_layer = _find_target_layer(model)
    acts: List[torch.Tensor] = []

    def fwd_hook(_m, _inp, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        acts.append(out)

    h = target_layer.register_forward_hook(fwd_hook)
    try:
        _ = model(x)  # forward to collect activations
        A = acts[-1]  # (1,C,H,W)
    finally:
        h.remove()

    A = A.detach()  # (1,C,h,w)
    _, C, hH, hW = A.shape

    # Select channels by activation energy (reduce compute)
    energy = (A**2).mean(dim=(2, 3))[0]  # (C,)
    k = int(min(max_channels, C))
    topk = torch.topk(energy, k=k).indices  # (k,)

    A_sel = A[0, topk]  # (k,h,w)

    # Upsample activation maps to input size and normalize each to [0,1]
    up = F.interpolate(A_sel.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)[0]  # (k,224,224)
    up_min = up.flatten(1).min(dim=1).values[:, None, None]
    up_max = up.flatten(1).max(dim=1).values[:, None, None]
    masks = (up - up_min) / (up_max - up_min + 1e-8)  # (k,224,224) in [0,1]

    # Prepare masked inputs (k,3,224,224)
    masks3 = masks.unsqueeze(1).repeat(1, 3, 1, 1)          # (k,3,224,224)
    x_rep = x.repeat(k, 1, 1, 1)                            # (k,3,224,224)
    x_masked = x_rep * masks3                               # (k,3,224,224)

    # Forward in batches to avoid OOM
    batch = 16
    scores = []
    for i in range(0, k, batch):
        xb = x_masked[i:i + batch]
        logits_b = model(xb)                                # (b,Cout)
        # use sigmoid(logit) for the target class as "score"
        s = torch.sigmoid(logits_b[:, target_index])         # (b,)
        scores.append(s.detach().cpu())
    scores_t = torch.cat(scores, dim=0).numpy().astype(np.float32)  # (k,)

    # Weighted sum of ORIGINAL activation maps (selected), upsampled to 224
    weights = scores_t / (scores_t.sum() + 1e-8)  # normalize
    heat = (weights[:, None, None] * masks.detach().cpu().numpy()).sum(axis=0)  # (224,224)
    heat01 = _normalize01(heat)

    overlay = _overlay_heatmap(pil_img, heat01, alpha=float(alpha))
    contours = _overlay_contours(pil_img, heat01, thresh=0.55)
    return CAMResult(heatmap=heat01, overlay=overlay, contours=contours)
