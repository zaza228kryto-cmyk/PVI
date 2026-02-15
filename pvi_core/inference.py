from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import cv2


# UI-метки, которые показываем в продукте (фиксированный порядок)
UI_FINDINGS: List[str] = [
    "Pneumonia",
    "Tuberculosis",
    "Mass",
    "Nodule",
    "Pleural Effusion",
    "No Finding",
]

# совместимость со старым кодом desktop/web
FINDINGS = UI_FINDINGS


_MODEL: Optional[nn.Module] = None
_DEVICE: Optional[torch.device] = None
_MODEL_LABELS: Optional[List[str]] = None  # какие классы реально обучены в загруженном чекпоинте


def _get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _default_weights_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "model" / "weights" / "current.pt"


def _load_checkpoint(weights_path: Path, device: torch.device):
    """
    Возвращает (state_dict, labels_or_none).
    Поддерживает:
      - dict с keys: state_dict, labels
      - либо "чистый" state_dict
    """
    if not weights_path.exists():
        return None, None

    ckpt = torch.load(weights_path, map_location=device, weights_only=True)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        labels = ckpt.get("labels", None)
        return state_dict, labels

    # иначе считаем что это state_dict
    return ckpt, None


def _build_model(num_classes: int, pretrained: bool) -> nn.Module:
    m = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
    m.eval()
    return m


def get_model() -> Tuple[nn.Module, List[str]]:
    """
    Возвращает (model, model_labels).
    model_labels — список классов, которые реально в выходе модели.
    """
    global _MODEL, _DEVICE, _MODEL_LABELS

    if _MODEL is not None and _MODEL_LABELS is not None:
        return _MODEL, _MODEL_LABELS

    _DEVICE = _get_device()
    device = _DEVICE
    weights_path = _default_weights_path()

    state_dict, labels = _load_checkpoint(weights_path, device)

    if state_dict is None:
        # нет весов — fallback на ImageNet с 6 классами (как раньше)
        _MODEL_LABELS = UI_FINDINGS[:]
        _MODEL = _build_model(num_classes=len(_MODEL_LABELS), pretrained=True).to(device)
        print(f"[PVI] Using ImageNet pretrained (no custom weights). Path: {weights_path}")
        return _MODEL, _MODEL_LABELS

    # есть веса
    if isinstance(labels, list) and len(labels) > 0:
        model_labels = [str(x) for x in labels]
    else:
        # если labels не сохранены (на всякий случай)
        model_labels = UI_FINDINGS[:]

    model = _build_model(num_classes=len(model_labels), pretrained=False).to(device)

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        # если вдруг несовместимо — fallback на ImageNet
        print(f"[PVI] Failed to load weights from {weights_path}: {e}")
        _MODEL_LABELS = UI_FINDINGS[:]
        _MODEL = _build_model(num_classes=len(_MODEL_LABELS), pretrained=True).to(device)
        print(f"[PVI] Using ImageNet pretrained fallback. Path: {weights_path}")
        return _MODEL, _MODEL_LABELS

    _MODEL = model
    _MODEL_LABELS = model_labels
    print(f"[PVI] Loaded custom weights: {weights_path}")
    print(f"[PVI] Model labels: {_MODEL_LABELS}")
    return _MODEL, _MODEL_LABELS


def preprocess(pil_img: Image.Image) -> torch.Tensor:
    img = pil_img.convert("L").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.stack([arr, arr, arr], axis=0)  # (3,H,W)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    arr = (arr - mean) / std

    t = torch.from_numpy(arr).unsqueeze(0)
    return t


@torch.no_grad()
def predict_probs(pil_img: Image.Image) -> Dict[str, float]:
    model, model_labels = get_model()
    device = _DEVICE if _DEVICE is not None else _get_device()

    x = preprocess(pil_img).to(device)
    logits = model(x)  # (1,C)
    probs = torch.sigmoid(logits).detach().cpu().numpy()[0].astype(float)

    # Сначала probs по реальным выходам модели
    by_label = {model_labels[i]: float(probs[i]) for i in range(len(model_labels))}

    # А наружу отдаём полный UI-набор (6)
    out = {}
    for lab in UI_FINDINGS:
        out[lab] = float(by_label.get(lab, 0.0))
    return out


# -----------------------
# CAM utilities
# -----------------------

@dataclass
class CAMResult:
    heatmap: np.ndarray
    overlay: Image.Image
    contours: Image.Image


def _find_target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "blocks"):
        return model.blocks[-1]
    raise RuntimeError("Could not locate target layer for EfficientNet.")


def _to_uint8_gray(heatmap01: np.ndarray) -> np.ndarray:
    return np.clip(heatmap01 * 255.0, 0, 255).astype(np.uint8)


def _apply_colormap(heatmap01: np.ndarray) -> np.ndarray:
    h8 = _to_uint8_gray(heatmap01)
    cm = cv2.applyColorMap(h8, cv2.COLORMAP_TURBO)  # BGR
    return cm


def _overlay_heatmap(pil_img: Image.Image, heatmap01: np.ndarray, alpha: float) -> Image.Image:
    base = pil_img.convert("RGB").resize((224, 224))
    base_np = np.array(base, dtype=np.uint8)

    cm_bgr = _apply_colormap(heatmap01)
    cm_rgb = cv2.cvtColor(cm_bgr, cv2.COLOR_BGR2RGB)

    out = (1.0 - alpha) * base_np.astype(np.float32) + alpha * cm_rgb.astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _overlay_contours(pil_img: Image.Image, heatmap01: np.ndarray, thresh: float = 0.55, thickness: int = 2) -> Image.Image:
    base = pil_img.convert("RGB").resize((224, 224))
    base_np = np.array(base, dtype=np.uint8)

    mask = (heatmap01 >= float(thresh)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(base_np, contours, -1, (255, 255, 255), int(thickness))
    return Image.fromarray(base_np)


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    denom = x.max() + 1e-8
    return (x / denom).astype(np.float32)


def _blank_cam(pil_img: Image.Image) -> CAMResult:
    heat = np.zeros((224, 224), dtype=np.float32)
    base = pil_img.convert("RGB").resize((224, 224))
    return CAMResult(heatmap=heat, overlay=base, contours=base)


def _ui_index_to_model_index(ui_index: int, model_labels: List[str]) -> Optional[int]:
    if ui_index < 0 or ui_index >= len(UI_FINDINGS):
        return None
    ui_label = UI_FINDINGS[ui_index]
    if ui_label in model_labels:
        return model_labels.index(ui_label)
    return None


# -----------------------
# Grad-CAM
# -----------------------

def grad_cam(pil_img: Image.Image, target_index: int, alpha: float = 0.30) -> CAMResult:
    model, model_labels = get_model()
    device = _DEVICE if _DEVICE is not None else _get_device()

    model_target = _ui_index_to_model_index(target_index, model_labels)
    if model_target is None:
        # например TB, когда веса 5-классовые
        return _blank_cam(pil_img)

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
        score = logits[0, model_target]
        model.zero_grad(set_to_none=True)
        score.backward()

        A = activations[-1]
        G = gradients[-1]
        w = G.mean(dim=(2, 3), keepdim=True)
        cam = (w * A).sum(dim=1)[0]
        cam = F.relu(cam).detach().cpu().numpy()

        cam01 = _normalize01(cam)
        cam01 = cv2.resize(cam01, (224, 224), interpolation=cv2.INTER_LINEAR)

        overlay = _overlay_heatmap(pil_img, cam01, alpha=float(alpha))
        contours = _overlay_contours(pil_img, cam01, thresh=0.55, thickness=2)
        return CAMResult(heatmap=cam01, overlay=overlay, contours=contours)
    finally:
        h1.remove()
        h2.remove()


# -----------------------
# Score-CAM
# -----------------------

@torch.no_grad()
def score_cam(
    pil_img: Image.Image,
    target_index: int,
    alpha: float = 0.30,
    max_channels: int = 64,
) -> CAMResult:
    model, model_labels = get_model()
    device = _DEVICE if _DEVICE is not None else _get_device()

    model_target = _ui_index_to_model_index(target_index, model_labels)
    if model_target is None:
        return _blank_cam(pil_img)

    x = preprocess(pil_img).to(device)

    target_layer = _find_target_layer(model)
    acts: List[torch.Tensor] = []

    def fwd_hook(_m, _inp, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        acts.append(out)

    h = target_layer.register_forward_hook(fwd_hook)
    try:
        _ = model(x)
        A = acts[-1]
    finally:
        h.remove()

    A = A.detach()
    _, C, hH, hW = A.shape

    energy = (A**2).mean(dim=(2, 3))[0]
    k = int(min(max_channels, C))
    topk = torch.topk(energy, k=k).indices

    A_sel = A[0, topk]

    up = F.interpolate(A_sel.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)[0]
    up_min = up.flatten(1).min(dim=1).values[:, None, None]
    up_max = up.flatten(1).max(dim=1).values[:, None, None]
    masks = (up - up_min) / (up_max - up_min + 1e-8)

    masks3 = masks.unsqueeze(1).repeat(1, 3, 1, 1)
    x_rep = x.repeat(k, 1, 1, 1)
    x_masked = x_rep * masks3

    batch = 16
    scores = []
    for i in range(0, k, batch):
        xb = x_masked[i:i + batch]
        logits_b = model(xb)
        s = torch.sigmoid(logits_b[:, model_target])
        scores.append(s.detach().cpu())
    scores_t = torch.cat(scores, dim=0).numpy().astype(np.float32)

    weights = scores_t / (scores_t.sum() + 1e-8)
    heat = (weights[:, None, None] * masks.detach().cpu().numpy()).sum(axis=0)
    heat01 = _normalize01(heat)

    overlay = _overlay_heatmap(pil_img, heat01, alpha=float(alpha))
    contours = _overlay_contours(pil_img, heat01, thresh=0.55, thickness=2)
    return CAMResult(heatmap=heat01, overlay=overlay, contours=contours)
