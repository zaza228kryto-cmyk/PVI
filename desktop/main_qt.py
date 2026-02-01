from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image

from PySide6.QtCore import Qt, QObject, Signal, QThread
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog,
    QComboBox, QSlider, QCheckBox, QGroupBox,
    QListWidget, QListWidgetItem,
    QMessageBox, QProgressBar
)

# Core inference (single source of truth)
from pvi_core.inference import FINDINGS, predict_probs, grad_cam, score_cam


# -----------------------------
# Helpers
# -----------------------------
def pil_to_qpixmap(pil_img: Image.Image, target_size: Tuple[int, int]) -> QPixmap:
    """Convert PIL image to QPixmap scaled to target_size (w,h) keeping aspect ratio."""
    img = pil_img.convert("RGB")
    w, h = img.size
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, w, h, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(target_size[0], target_size[1], Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


def safe_open_image(path: Path) -> Image.Image:
    try:
        return Image.open(path)
    except Exception as e:
        raise RuntimeError(f"Не удалось открыть изображение: {e}") from e


@dataclass
class CacheKey:
    image_path: str
    method: str
    target_index: int


# -----------------------------
# Worker (runs inference off UI thread)
# -----------------------------
class InferenceWorker(QObject):
    started = Signal()
    finished = Signal(dict, object, object)  # probs(dict), cam_overlay(PIL), cam_contours(PIL)
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

            self.finished.emit(probs, cam.overlay, cam.contours)
        except Exception:
            self.failed.emit(traceback.format_exc())


# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PVI — Pulmo Visual Insight (Desktop)")
        self.resize(1200, 720)

        self.image_path: Optional[str] = None
        self.image_pil: Optional[Image.Image] = None

        # Simple CAM cache: (image_path, method, target_index) -> (overlay_pil, contours_pil)
        self.cam_cache: Dict[Tuple[str, str, int], Tuple[Image.Image, Image.Image]] = {}

        # Thread objects (kept to avoid GC)
        self.thread: Optional[QThread] = None
        self.worker: Optional[InferenceWorker] = None

        self._build_ui()
        self._wire_events()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        main_layout = QHBoxLayout(root)

        # Left side: previews
        previews = QGroupBox("Просмотр")
        previews_layout = QGridLayout(previews)

        self.lbl_original = QLabel("Original")
        self.lbl_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_original.setMinimumSize(520, 520)
        self.lbl_original.setStyleSheet("border: 1px solid #444;")

        self.lbl_cam = QLabel("CAM")
        self.lbl_cam.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_cam.setMinimumSize(520, 520)
        self.lbl_cam.setStyleSheet("border: 1px solid #444;")

        previews_layout.addWidget(QLabel("Оригинал"), 0, 0)
        previews_layout.addWidget(QLabel("Подсветка (CAM)"), 0, 1)
        previews_layout.addWidget(self.lbl_original, 1, 0)
        previews_layout.addWidget(self.lbl_cam, 1, 1)

        # Right side: controls + probabilities
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

        self.chk_contours = QCheckBox("Контуры (без заливки)")
        self.chk_autorun = QCheckBox("Авто-анализ после открытия")
        self.chk_autorun.setChecked(True)

        self.slider_alpha = QSlider(Qt.Orientation.Horizontal)
        self.slider_alpha.setMinimum(0)
        self.slider_alpha.setMaximum(80)
        self.slider_alpha.setValue(30)

        self.lbl_alpha = QLabel("alpha: 0.30")

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.setVisible(False)

        row = 0
        c.addWidget(self.btn_open, row, 0, 1, 2); row += 1
        c.addWidget(QLabel("Метод"), row, 0); c.addWidget(self.cmb_method, row, 1); row += 1
        c.addWidget(QLabel("Цель (label)"), row, 0); c.addWidget(self.cmb_target, row, 1); row += 1
        c.addWidget(self.lbl_alpha, row, 0); c.addWidget(self.slider_alpha, row, 1); row += 1
        c.addWidget(self.chk_contours, row, 0, 1, 2); row += 1
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

        self.slider_alpha.valueChanged.connect(self.on_alpha_changed)
        self.cmb_method.currentTextChanged.connect(self.on_settings_changed)
        self.cmb_target.currentTextChanged.connect(self.on_settings_changed)
        self.chk_contours.stateChanged.connect(self.on_settings_changed)

    # ---------- Events ----------
    def on_open(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение (PNG/JPG)",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )
        if not file_path:
            return

        try:
            img = safe_open_image(Path(file_path))
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            return

        self.image_path = file_path
        self.image_pil = img

        # show original
        self._set_preview(self.lbl_original, img)

        # clear CAM preview
        self.lbl_cam.setText("CAM")
        self.cam_cache.clear()

        self.btn_run.setEnabled(True)

        if self.chk_autorun.isChecked():
            self.on_run()

    def on_run(self) -> None:
        if not self.image_path or not self.image_pil:
            return

        method = self.cmb_method.currentText()
        target_index = FINDINGS.index(self.cmb_target.currentText())
        alpha = self._alpha_value()

        cache_key = (self.image_path, method, target_index)

        # If cached CAM exists, reuse but respect contours toggle (alpha baked into overlay in current core)
        if cache_key in self.cam_cache:
            overlay_pil, contours_pil = self.cam_cache[cache_key]
            self._update_cam_preview(overlay_pil, contours_pil)
            # probs are not cached; compute quickly in background for consistency
            # but for MVP we just run full worker to update probs too
            # (keeps logic simple)
        # Start worker thread for inference + CAM
        self._start_worker(self.image_path, method, target_index, alpha)

    def on_alpha_changed(self) -> None:
        self.lbl_alpha.setText(f"alpha: {self._alpha_value():.2f}")
        # We re-run analysis when alpha changes (simple MVP).
        # Optimization: re-blend overlay without recomputing CAM requires core changes.
        if self.image_path and self.btn_run.isEnabled():
            # only auto-run if user already has image
            # do not spam too fast: user can press Analyze
            pass

    def on_settings_changed(self) -> None:
        # If user changed CAM method/label/contours, they can press Analyze again.
        pass

    # ---------- Worker handling ----------
    def _start_worker(self, image_path: str, method: str, target_index: int, alpha: float) -> None:
        # prevent multiple concurrent runs
        if self.thread and self.thread.isRunning():
            return

        self.progress.setVisible(True)
        self.btn_run.setEnabled(False)
        self.btn_open.setEnabled(False)

        self.thread = QThread()
        self.worker = InferenceWorker(image_path=image_path, method=method, target_index=target_index, alpha=alpha)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.started.connect(lambda: None)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.failed.connect(self._on_worker_failed)

        # Clean up
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

    def _on_worker_finished(self, probs: dict, cam_overlay: object, cam_contours: object) -> None:
        # probs list
        self._update_probs(probs)

        # cache cam for current key
        method = self.cmb_method.currentText()
        target_index = FINDINGS.index(self.cmb_target.currentText())
        cache_key = (self.image_path or "", method, target_index)
        if isinstance(cam_overlay, Image.Image) and isinstance(cam_contours, Image.Image):
            self.cam_cache[cache_key] = (cam_overlay, cam_contours)
            self._update_cam_preview(cam_overlay, cam_contours)

    def _on_worker_failed(self, tb: str) -> None:
        QMessageBox.critical(self, "Ошибка инференса", tb)

    # ---------- UI updates ----------
    def _alpha_value(self) -> float:
        return float(self.slider_alpha.value()) / 100.0

    def _set_preview(self, label: QLabel, pil_img: Image.Image) -> None:
        pix = pil_to_qpixmap(pil_img, (520, 520))
        label.setPixmap(pix)

    def _update_cam_preview(self, overlay_pil: Image.Image, contours_pil: Image.Image) -> None:
        if self.chk_contours.isChecked():
            self._set_preview(self.lbl_cam, contours_pil)
        else:
            self._set_preview(self.lbl_cam, overlay_pil)

    def _update_probs(self, probs: Dict[str, float]) -> None:
        self.list_probs.clear()
        for k in FINDINGS:
            v = float(probs.get(k, 0.0))
            item = QListWidgetItem(f"{k}: {v * 100:.1f}%")
            self.list_probs.addItem(item)


def main() -> None:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
