from __future__ import annotations

import sys
import traceback
import faulthandler
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QObject, Signal, QThread
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog,
    QComboBox, QSlider, QCheckBox, QGroupBox,
    QListWidget, QListWidgetItem,
    QMessageBox, QProgressBar,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)

try:
    faulthandler.enable(all_threads=True)
except Exception:
    pass

# pvi_core API
from pvi_core.inference import FINDINGS, predict_probs, grad_cam, score_cam


def _excepthook(exc_type, exc, tb):
    msg = "".join(traceback.format_exception(exc_type, exc, tb))
    try:
        QMessageBox.critical(None, "Критическая ошибка (Python)", msg)
    finally:
        print(msg, file=sys.stderr)


sys.excepthook = _excepthook


# -----------------------------
# Helpers
# -----------------------------
def safe_open_image(path: Path) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img


def pil_to_qimage_rgb(pil_img: Image.Image) -> QImage:
    img = pil_img.convert("RGB")
    w, h = img.size
    data = img.tobytes("raw", "RGB")
    return QImage(data, w, h, QImage.Format.Format_RGB888)


def pil_to_qpixmap(pil_img: Image.Image, target_size=None) -> QPixmap:
    """
    Безопасная конвертация PIL -> QPixmap без access violation.
    target_size можно не передавать (тогда вернёт пиксмап без scaling).
    """
    img = pil_img.convert("RGB")
    w, h = img.size

    data = img.tobytes("raw", "RGB")
    bytes_per_line = 3 * w

    qimg = QImage(data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

    # ВАЖНО: заставляем QImage скопировать данные внутрь себя
    qimg = qimg.copy()

    pix = QPixmap.fromImage(qimg)

    if target_size is None:
        return pix

    return pix.scaled(
        int(target_size[0]),
        int(target_size[1]),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


    # 2) Гарантируем непрерывный буфер байт
    data = img.tobytes("raw", "RGB")
    bytes_per_line = 3 * w

    # 3) ВАЖНО: создаём QImage с корректным шагом строки
    qimg = QImage(data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

    # 4) ВАЖНО: делаем copy(), чтобы QImage владел своей памятью
    qimg = qimg.copy()

    pix = QPixmap.fromImage(qimg)
    return pix.scaled(
        target_size[0],
        target_size[1],
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )



def _pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.uint8)


def _np_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _extract_heat_from_overlay(orig_rgb: Image.Image, overlay_rgb: Image.Image, alpha: float) -> Optional[Image.Image]:
    """
    overlay = (1-a)*orig + a*heat  -> heat = (overlay-(1-a)*orig)/a
    """
    if alpha <= 1e-6:
        return None
    o = _pil_to_np_rgb(orig_rgb).astype(np.float32)
    ov = _pil_to_np_rgb(overlay_rgb).astype(np.float32)
    heat = (ov - (1.0 - alpha) * o) / alpha
    heat = np.clip(heat, 0, 255).astype(np.uint8)
    return _np_to_pil_rgb(heat)


def _blend(orig_rgb: Image.Image, heat_rgb: Image.Image, alpha: float) -> Image.Image:
    a = float(alpha)
    a = max(0.0, min(1.0, a))
    o = _pil_to_np_rgb(orig_rgb).astype(np.float32)
    h = _pil_to_np_rgb(heat_rgb).astype(np.float32)
    out = (1.0 - a) * o + a * h
    return _np_to_pil_rgb(out)


def _resize(img: Image.Image, size: Tuple[int, int], resample: Image.Resampling) -> Image.Image:
    if img.size == size:
        return img
    return img.resize(size, resample)


def _make_preview(img: Image.Image, max_side: int = 800) -> Image.Image:
    """
    Downscale big image for fast math while dragging alpha.
    """
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.Resampling.BILINEAR)


def _contours_from_heat(heat_rgb: Image.Image, *, k: float = 1.5, min_area: int = 1200) -> np.ndarray:
    """
    Robust contours from heatmap:
    1) grayscale
    2) smooth (box blur 5x5)
    3) threshold = mean + k*std
    4) remove small blobs (connected components, 4-neighborhood)
    5) boundary = edges of remaining blobs
    """
    h = _pil_to_np_rgb(heat_rgb)
    gray = (0.299 * h[..., 0] + 0.587 * h[..., 1] + 0.114 * h[..., 2]).astype(np.float32)

    # --- smooth with simple box blur (5x5) ---
    g = np.pad(gray, ((2, 2), (2, 2)), mode="reflect")
    sm = (
        g[0:-4, 0:-4] + g[0:-4, 1:-3] + g[0:-4, 2:-2] + g[0:-4, 3:-1] + g[0:-4, 4:] +
        g[1:-3, 0:-4] + g[1:-3, 1:-3] + g[1:-3, 2:-2] + g[1:-3, 3:-1] + g[1:-3, 4:] +
        g[2:-2, 0:-4] + g[2:-2, 1:-3] + g[2:-2, 2:-2] + g[2:-2, 3:-1] + g[2:-2, 4:] +
        g[3:-1, 0:-4] + g[3:-1, 1:-3] + g[3:-1, 2:-2] + g[3:-1, 3:-1] + g[3:-1, 4:] +
        g[4:, 0:-4] + g[4:, 1:-3] + g[4:, 2:-2] + g[4:, 3:-1] + g[4:, 4:]
    ) / 25.0

    mu = float(sm.mean())
    sd = float(sm.std() + 1e-6)
    thr = mu + float(k) * sd

    hot = sm >= thr

    # --- remove small blobs via BFS connected components ---
    H, W = hot.shape
    visited = np.zeros_like(hot, dtype=np.uint8)
    keep = np.zeros_like(hot, dtype=bool)

    for y in range(H):
        for x in range(W):
            if not hot[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = 1
            coords = []

            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))

                if cy > 0 and hot[cy - 1, cx] and not visited[cy - 1, cx]:
                    visited[cy - 1, cx] = 1
                    stack.append((cy - 1, cx))
                if cy + 1 < H and hot[cy + 1, cx] and not visited[cy + 1, cx]:
                    visited[cy + 1, cx] = 1
                    stack.append((cy + 1, cx))
                if cx > 0 and hot[cy, cx - 1] and not visited[cy, cx - 1]:
                    visited[cy, cx - 1] = 1
                    stack.append((cy, cx - 1))
                if cx + 1 < W and hot[cy, cx + 1] and not visited[cy, cx + 1]:
                    visited[cy, cx + 1] = 1
                    stack.append((cy, cx + 1))

            if len(coords) >= int(min_area):
                for (cy, cx) in coords:
                    keep[cy, cx] = True

    # boundary pixels of kept blobs
    up = np.zeros_like(keep);    up[1:] = keep[:-1]
    down = np.zeros_like(keep);  down[:-1] = keep[1:]
    left = np.zeros_like(keep);  left[:, 1:] = keep[:, :-1]
    right = np.zeros_like(keep); right[:, :-1] = keep[:, 1:]

    boundary = keep & (~(up & down & left & right))
    return boundary


def _draw_mask_as_white(base_rgb: Image.Image, mask: np.ndarray, thickness: int = 3) -> Image.Image:
    """
    Draw white contour pixels on base with adjustable thickness.
    """
    m = mask.copy()
    t = int(thickness)

    # thicken by dilating mask (no opencv)
    if t >= 2:
        for _ in range(t - 1):
            up = np.zeros_like(m);    up[1:] = m[:-1]
            down = np.zeros_like(m);  down[:-1] = m[1:]
            left = np.zeros_like(m);  left[:, 1:] = m[:, :-1]
            right = np.zeros_like(m); right[:, :-1] = m[:, 1:]
            m = m | up | down | left | right

    base = _pil_to_np_rgb(base_rgb).copy()
    base[m] = np.array([255, 255, 255], dtype=np.uint8)
    return _np_to_pil_rgb(base)


# -----------------------------
# Image viewer with Zoom + Pan + Sync state
# -----------------------------
class ImageViewer(QGraphicsView):
    stateChanged = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pix_item)

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._has_image = False
        self.setStyleSheet("background: #111; border: 1px solid #444;")

    def set_pixmap(self, pix: QPixmap, *, reset_view: bool) -> None:
        if reset_view:
            self.resetTransform()

        self._pix_item.setPixmap(pix)
        self._scene.setSceneRect(pix.rect())
        self._has_image = not pix.isNull()

        if reset_view and self._has_image:
            self.fit_to_view()
            self.stateChanged.emit(self.get_state())

    def set_pil(self, img: Image.Image, *, reset_view: bool) -> None:
        self.set_pixmap(pil_to_qpixmap(img), reset_view=reset_view)

    def clear(self) -> None:
        self.set_pixmap(QPixmap(), reset_view=True)

    def wheelEvent(self, event) -> None:
        if not self._has_image:
            return
        angle = event.angleDelta().y()
        if angle == 0:
            return
        factor = 1.25 if angle > 0 else 0.8
        self.scale(factor, factor)
        self.stateChanged.emit(self.get_state())

    def fit_to_view(self) -> None:
        if not self._has_image:
            return
        self.resetTransform()
        self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def get_state(self) -> dict:
        return {
            "transform": self.transform(),
            "h": int(self.horizontalScrollBar().value()),
            "v": int(self.verticalScrollBar().value()),
        }

    def apply_state(self, state: dict) -> None:
        self.setTransform(state["transform"])
        self.horizontalScrollBar().setValue(int(state["h"]))
        self.verticalScrollBar().setValue(int(state["v"]))

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        if self._has_image:
            self.stateChanged.emit(self.get_state())

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        if self._has_image:
            self.stateChanged.emit(self.get_state())


# -----------------------------
# Worker
# -----------------------------
class InferenceWorker(QObject):
    started = Signal()
    finished = Signal(dict, object, object, float, str, int)
    failed = Signal(str)

    def __init__(self, image_path: str, method: str, target_index: int, alpha: float) -> None:
        super().__init__()
        self.image_path = image_path
        self.method = method
        self.target_index = target_index
        self.alpha = alpha

    def run(self) -> None:
        self.started.emit()
        try:
            img = safe_open_image(Path(self.image_path))
            probs = predict_probs(img)

            if self.method == "Grad-CAM":
                cam = grad_cam(img, target_index=self.target_index, alpha=float(self.alpha))
            else:
                cam = score_cam(img, target_index=self.target_index, alpha=float(self.alpha), max_channels=64)

            self.finished.emit(probs, cam.overlay, cam.contours, float(self.alpha), self.method, self.target_index)
        except Exception:
            self.failed.emit(traceback.format_exc())


# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PVI — Pulmo Visual Insight (Desktop)")
        self.resize(1300, 760)

        self.image_path: Optional[str] = None
        self.image_pil: Optional[Image.Image] = None  # full-res RGB

        # cache_key = (image_path, method, target_index)
        self.cam_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

        self.thread: Optional[QThread] = None
        self.worker: Optional[InferenceWorker] = None

        self._dragging_alpha = False
        self._dragging_sens = False
        self._cam_needs_reset_view = True
        self._sync_lock = False

        self._build_ui()
        self._wire_events()
        self._warmup_model()

    def _warmup_model(self) -> None:
        try:
            dummy = Image.new("RGB", (224, 224), color=(0, 0, 0))
            _ = predict_probs(dummy)
        except Exception:
            QMessageBox.warning(self, "Warning", "Model warmup failed.\n\n" + traceback.format_exc())

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)

        previews = QGroupBox("Просмотр")
        previews_layout = QGridLayout(previews)

        self.viewer_original = ImageViewer()
        self.viewer_cam = ImageViewer()

        previews_layout.addWidget(QLabel("Оригинал"), 0, 0)
        previews_layout.addWidget(QLabel("Подсветка (CAM)"), 0, 1)
        previews_layout.addWidget(self.viewer_original, 1, 0)
        previews_layout.addWidget(self.viewer_cam, 1, 1)

        side = QVBoxLayout()

        controls = QGroupBox("Управление")
        c = QGridLayout(controls)

        self.btn_open = QPushButton("Открыть снимок…")
        self.btn_run = QPushButton("Анализировать")
        self.btn_run.setEnabled(False)

        self.cmb_method = QComboBox()
        self.cmb_method.addItems(["Grad-CAM", "Score-CAM"])

        self.cmb_target = QComboBox()
        self.cmb_target.addItems(FINDINGS)

        self.btn_fit = QPushButton("Fit")
        self.btn_reset = QPushButton("Reset")

        self.slider_alpha = QSlider(Qt.Orientation.Horizontal)
        self.slider_alpha.setMinimum(0)
        self.slider_alpha.setMaximum(80)
        self.slider_alpha.setValue(30)
        self.lbl_alpha = QLabel("alpha: 0.30")

        self.chk_contours = QCheckBox("Контуры (поверх снимка)")
        self.chk_sync = QCheckBox("Синхронизация (Zoom/Pan)")
        self.chk_sync.setChecked(True)
        self.chk_autorun = QCheckBox("Авто-анализ при смене/открытии")
        self.chk_autorun.setChecked(True)

        # --- Sensitivity UI (only visible when contours enabled) ---
        self.lbl_sens = QLabel("Чувствительность: 50")
        self.slider_sens = QSlider(Qt.Orientation.Horizontal)
        self.slider_sens.setMinimum(0)
        self.slider_sens.setMaximum(100)
        self.slider_sens.setValue(50)
        self.lbl_sens.setVisible(False)
        self.slider_sens.setVisible(False)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)

        row = 0
        c.addWidget(self.btn_open, row, 0, 1, 2); row += 1
        c.addWidget(QLabel("Метод"), row, 0); c.addWidget(self.cmb_method, row, 1); row += 1
        c.addWidget(QLabel("Цель (label)"), row, 0); c.addWidget(self.cmb_target, row, 1); row += 1

        c.addWidget(self.btn_fit, row, 0); c.addWidget(self.btn_reset, row, 1); row += 1
        c.addWidget(self.lbl_alpha, row, 0); c.addWidget(self.slider_alpha, row, 1); row += 1

        c.addWidget(self.chk_contours, row, 0, 1, 2); row += 1
        c.addWidget(self.lbl_sens, row, 0); c.addWidget(self.slider_sens, row, 1); row += 1
        c.addWidget(self.chk_sync, row, 0, 1, 2); row += 1
        c.addWidget(self.chk_autorun, row, 0, 1, 2); row += 1

        c.addWidget(self.btn_run, row, 0, 1, 2); row += 1
        c.addWidget(self.progress, row, 0, 1, 2); row += 1

        probs_box = QGroupBox("Вероятности (Findings)")
        probs_layout = QVBoxLayout(probs_box)
        self.list_probs = QListWidget()
        probs_layout.addWidget(self.list_probs)

        side.addWidget(controls)
        side.addWidget(probs_box)
        side.addStretch(1)

        main_layout.addWidget(previews, stretch=3)
        main_layout.addLayout(side, stretch=1)

    def _wire_events(self) -> None:
        self.btn_open.clicked.connect(self.on_open)
        self.btn_run.clicked.connect(self.on_run)

        self.btn_fit.clicked.connect(self.on_fit)
        self.btn_reset.clicked.connect(self.on_reset)

        self.slider_alpha.sliderPressed.connect(self._on_alpha_pressed)
        self.slider_alpha.sliderReleased.connect(self._on_alpha_released)
        self.slider_alpha.valueChanged.connect(self.on_alpha_changed)

        self.chk_contours.stateChanged.connect(self._on_view_changed)
        self.chk_contours.stateChanged.connect(self._on_contours_toggled)
        self.slider_sens.valueChanged.connect(self._on_sens_changed)
        self.slider_sens.sliderPressed.connect(self._on_sens_pressed)
        self.slider_sens.sliderReleased.connect(self._on_sens_released)

        self.cmb_method.currentIndexChanged.connect(self._on_view_changed)
        self.cmb_target.currentIndexChanged.connect(self._on_view_changed)

        self.viewer_original.stateChanged.connect(lambda st: self._on_viewer_state("orig", st))
        self.viewer_cam.stateChanged.connect(lambda st: self._on_viewer_state("cam", st))

    def _on_contours_toggled(self) -> None:
        enabled = self.chk_contours.isChecked()
        self.lbl_sens.setVisible(enabled)
        self.slider_sens.setVisible(enabled)
        # обновим картинку сразу
        self._refresh_cam_display(full_quality=not self._dragging_alpha)

    def _on_sens_changed(self) -> None:
        v = int(self.slider_sens.value())
        self.lbl_sens.setText(f"Чувствительность: {v}")
        if self.chk_contours.isChecked():
            self._refresh_cam_display(full_quality=not (self._dragging_alpha or self._dragging_sens))

    def _on_sens_pressed(self) -> None:
        self._dragging_sens = True

    def _on_sens_released(self) -> None:
        self._dragging_sens = False
        # один раз — чёткий пересчёт
        if self.chk_contours.isChecked():
            self._refresh_cam_display(full_quality=True)


    def _contours_params(self) -> Tuple[float, int]:
        """
        Map sensitivity(0..100) -> (k, min_area)
        - sens=0   -> strict:   k≈1.8, min_area≈1600 (меньше мусора)
        - sens=100 -> sensitive k≈0.9, min_area≈400  (видит мелочи, но может шуметь)
        """
        sens = int(self.slider_sens.value()) if hasattr(self, "slider_sens") else 50
        k = 1.8 - (sens / 100.0) * 0.9
        min_area = int(1600 - (sens / 100.0) * 1200)
        min_area = max(50, min_area)
        return k, min_area

    def _on_viewer_state(self, source: str, state: dict) -> None:
        if not self.chk_sync.isChecked():
            return
        if self._sync_lock:
            return
        if not self.image_path:
            return

        self._sync_lock = True
        try:
            if source == "orig":
                self.viewer_cam.apply_state(state)
            else:
                self.viewer_original.apply_state(state)
        finally:
            self._sync_lock = False

    def on_fit(self) -> None:
        self.viewer_original.fit_to_view()
        self.viewer_cam.fit_to_view()
        if self.chk_sync.isChecked():
            st = self.viewer_original.get_state()
            self.viewer_cam.apply_state(st)

    def on_reset(self) -> None:
        self.on_fit()

    def _alpha_value(self) -> float:
        return float(self.slider_alpha.value()) / 100.0

    def _on_alpha_pressed(self) -> None:
        self._dragging_alpha = True

    def _on_alpha_released(self) -> None:
        self._dragging_alpha = False
        self._refresh_cam_display(full_quality=True)

    def on_alpha_changed(self) -> None:
        self.lbl_alpha.setText(f"alpha: {self._alpha_value():.2f}")
        self._refresh_cam_display(full_quality=not self._dragging_alpha)

    def on_open(self) -> None:
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Выберите изображение (PNG/JPG)", "", "Images (*.png *.jpg *.jpeg)"
            )
            if not file_path:
                return

            img = safe_open_image(Path(file_path)).convert("RGB")
            self.image_path = file_path
            self.image_pil = img

            self.viewer_original.set_pil(img, reset_view=True)
            self.viewer_cam.clear()

            self.cam_cache.clear()
            self.btn_run.setEnabled(True)
            self._cam_needs_reset_view = True

            if self.chk_autorun.isChecked():
                self.on_run()

        except Exception:
            QMessageBox.critical(self, "Ошибка (Open)", traceback.format_exc())

    def on_run(self) -> None:
        if not self.image_path or not self.image_pil:
            return

        method = self.cmb_method.currentText()
        target_index = FINDINGS.index(self.cmb_target.currentText())
        alpha = self._alpha_value()
        cache_key = (self.image_path, method, target_index)

        if cache_key in self.cam_cache:
            self._update_probs(self.cam_cache[cache_key]["probs"])
            self._refresh_cam_display(full_quality=not self._dragging_alpha)
            return

        self._start_worker(self.image_path, method, target_index, alpha)

    def _on_view_changed(self) -> None:
        if not self.image_path:
            return

        method = self.cmb_method.currentText()
        target_index = FINDINGS.index(self.cmb_target.currentText())
        cache_key = (self.image_path, method, target_index)

        if cache_key in self.cam_cache:
            self._refresh_cam_display(full_quality=not self._dragging_alpha)
            return

        if self.chk_autorun.isChecked() and self.btn_run.isEnabled():
            self.on_run()
        else:
            self.viewer_cam.clear()

    # ---------- Worker ----------
    def _start_worker(self, image_path: str, method: str, target_index: int, alpha: float) -> None:
        if self.thread and self.thread.isRunning():
            return

        self.progress.setVisible(True)
        self.btn_run.setEnabled(False)
        self.btn_open.setEnabled(False)

        self.thread = QThread()
        self.worker = InferenceWorker(image_path=image_path, method=method, target_index=target_index, alpha=alpha)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.failed.connect(self._on_worker_failed)

        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._cleanup_thread)

        self.thread.start()

    def _cleanup_thread(self) -> None:
        self.progress.setVisible(False)
        self.btn_open.setEnabled(True)
        self.btn_run.setEnabled(True)

        if self.worker:
            self.worker.deleteLater()
        if self.thread:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None

    def _on_worker_failed(self, tb: str) -> None:
        QMessageBox.critical(self, "Ошибка инференса", tb)

    def _on_worker_finished(
        self,
        probs: dict,
        cam_overlay: object,
        cam_contours: object,
        alpha_used: float,
        method_used: str,
        target_index_used: int,
    ) -> None:
        try:
            self._update_probs(probs)

            if not self.image_path or not self.image_pil:
                return
            if not isinstance(cam_overlay, Image.Image):
                return

            orig_full = self.image_pil
            overlay_small = cam_overlay.convert("RGB")

            # Extract heat on CAM-size
            orig_small = orig_full.resize(overlay_small.size, Image.Resampling.BILINEAR)
            heat_small = _extract_heat_from_overlay(orig_small, overlay_small, alpha_used)

            # Full-res heat
            heat_full = None if heat_small is None else _resize(heat_small, orig_full.size, Image.Resampling.BILINEAR)

            # Previews (for fast alpha dragging)
            orig_prev_small = _make_preview(orig_full, max_side=800)
            heat_prev_small = None if heat_full is None else _make_preview(heat_full, max_side=800)

            cache_key = (self.image_path, method_used, target_index_used)
            self.cam_cache[cache_key] = {
                "probs": probs,
                "orig_full": orig_full,
                "heat_full": heat_full,
                "orig_prev_small": orig_prev_small,
                "heat_prev_small": heat_prev_small,
            }

            current_method = self.cmb_method.currentText()
            current_target = FINDINGS.index(self.cmb_target.currentText())
            if current_method == method_used and current_target == target_index_used:
                self._refresh_cam_display(full_quality=not self._dragging_alpha)

        except Exception:
            QMessageBox.critical(self, "Ошибка UI", traceback.format_exc())

    # ---------- Render ----------
    def _refresh_cam_display(self, *, full_quality: bool) -> None:
        if not self.image_path:
            return

        method = self.cmb_method.currentText()
        target_index = FINDINGS.index(self.cmb_target.currentText())
        cache_key = (self.image_path, method, target_index)

        if cache_key not in self.cam_cache:
            return

        entry = self.cam_cache[cache_key]
        alpha = self._alpha_value()

        reset_view = False
        if self._cam_needs_reset_view:
            reset_view = True
            self._cam_needs_reset_view = False

        # Preserve current view state for CAM while updating pixmap
        cam_state = self.viewer_cam.get_state() if (not reset_view and self.viewer_cam._has_image) else None

        orig_full: Image.Image = entry["orig_full"]
        heat_full: Optional[Image.Image] = entry["heat_full"]

        if full_quality:
            if self.chk_contours.isChecked():
                if heat_full is None:
                    self.viewer_cam.set_pil(orig_full, reset_view=reset_view)
                else:
                    k, min_area = self._contours_params()
                    mask = _contours_from_heat(heat_full, k=k, min_area=min_area)
                    out = _draw_mask_as_white(orig_full, mask, thickness=3)
                    self.viewer_cam.set_pil(out, reset_view=reset_view)
            else:
                if heat_full is None:
                    self.viewer_cam.set_pil(orig_full, reset_view=reset_view)
                else:
                    out = _blend(orig_full, heat_full, alpha)
                    self.viewer_cam.set_pil(out, reset_view=reset_view)

            if cam_state is not None and not reset_view:
                self.viewer_cam.apply_state(cam_state)
            return

        # Preview path: compute small, then upscale to full size for stable viewer
        orig_prev_small: Image.Image = entry["orig_prev_small"]
        heat_prev_small: Optional[Image.Image] = entry["heat_prev_small"]

        if self.chk_contours.isChecked():
            if heat_prev_small is None:
                out_full = orig_prev_small.resize(orig_full.size, Image.Resampling.BILINEAR)
            else:
                k, min_area = self._contours_params()
                mask_small = _contours_from_heat(heat_prev_small, k=k, min_area=min_area)
                out_small = _draw_mask_as_white(orig_prev_small, mask_small, thickness=3)
                out_full = out_small.resize(orig_full.size, Image.Resampling.BILINEAR)

            self.viewer_cam.set_pil(out_full, reset_view=reset_view)
            if cam_state is not None and not reset_view:
                self.viewer_cam.apply_state(cam_state)
            return

        if heat_prev_small is None:
            out_full = orig_prev_small.resize(orig_full.size, Image.Resampling.BILINEAR)
        else:
            out_small = _blend(orig_prev_small, heat_prev_small, alpha)
            out_full = out_small.resize(orig_full.size, Image.Resampling.BILINEAR)

        self.viewer_cam.set_pil(out_full, reset_view=reset_view)
        if cam_state is not None and not reset_view:
            self.viewer_cam.apply_state(cam_state)

    def _update_probs(self, probs: Dict[str, float]) -> None:
        self.list_probs.clear()
        for k in FINDINGS:
            v = float(probs.get(k, 0.0))
            self.list_probs.addItem(QListWidgetItem(f"{k}: {v * 100:.1f}%"))


def main() -> None:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
