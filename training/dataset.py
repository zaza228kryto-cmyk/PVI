from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


# ====== Классы PVI (то, что мы хотим видеть в продукте) ======
PVI_LABELS_6 = [
    "Pneumonia",
    "Tuberculosis",       # в NIH нет -> всегда 0 на этом этапе
    "Mass",
    "Nodule",
    "Pleural Effusion",   # в NIH называется "Effusion"
    "No Finding",
]

# Как наши названия называются в NIH CSV
NIH_NAME = {
    "Pneumonia": "Pneumonia",
    "Tuberculosis": None,          # TB отсутствует в NIH
    "Mass": "Mass",
    "Nodule": "Nodule",
    "Pleural Effusion": "Effusion",
    "No Finding": "No Finding",
}


@dataclass
class NIHConfig:
    data_root: Path
    csv_path: Path
    trainval_list_path: Path
    keep_only_relevant: bool = True  # выкидывать "нулевые" случаи (не наши классы)

    # Если задано — датасет вернёт вектор только по этим классам (в этом порядке).
    # Это нужно, чтобы на NIH обучать 5 классов без TB.
    target_labels: List[str] | None = None


def _build_image_index(data_root: Path) -> Dict[str, Path]:
    """
    У тебя структура:
      data_root/images_001/images/*.png
      ...
      data_root/images_012/images/*.png

    Мы строим словарь: "00000001_000.png" -> полный путь.
    Это делается один раз при старте, чтобы обучение было быстрым.
    """
    index: Dict[str, Path] = {}
    folders = sorted(data_root.glob("images_*"))
    for folder in folders:
        images_dir = folder / "images"
        if not images_dir.exists():
            continue
        # Быстрое перечисление файлов
        for p in images_dir.glob("*.png"):
            index[p.name] = p
    return index


def _labels_to_vec_6(finding_labels: str) -> torch.Tensor:
    """
    Возвращает вектор [6] под PVI_LABELS_6.
    Tuberculosis всегда 0 на NIH.
    """
    parts = [x.strip() for x in str(finding_labels).split("|")]

    vec = []
    for label in PVI_LABELS_6:
        nih = NIH_NAME[label]
        if nih is None:
            vec.append(0.0)
        else:
            vec.append(1.0 if nih in parts else 0.0)
    return torch.tensor(vec, dtype=torch.float32)


def _labels_to_vec(finding_labels: str, target_labels: List[str]) -> torch.Tensor:
    """
    Универсальный вектор под target_labels.
    Использует NIH_NAME, поэтому "Pleural Effusion" -> "Effusion".
    """
    parts = [x.strip() for x in str(finding_labels).split("|")]

    vec = []
    for label in target_labels:
        nih = NIH_NAME.get(label)
        if nih is None:
            # Например Tuberculosis (нет в NIH) или неизвестное имя
            vec.append(0.0)
        else:
            vec.append(1.0 if nih in parts else 0.0)

    return torch.tensor(vec, dtype=torch.float32)


def _is_relevant(vec6: torch.Tensor) -> bool:
    """
    Оставляем:
      - No Finding == 1
      - или хотя бы одна из патологий NIH, которые мы учим сейчас:
        Pneumonia, Mass, Nodule, Pleural Effusion
    TB (index 1) игнорируем.
    """
    pneumonia = vec6[0].item() == 1.0
    mass = vec6[2].item() == 1.0
    nodule = vec6[3].item() == 1.0
    eff = vec6[4].item() == 1.0
    no_finding = vec6[5].item() == 1.0
    return no_finding or pneumonia or mass or nodule or eff


class NIHChestXrayDataset(Dataset):
    def __init__(
        self,
        cfg: NIHConfig,
        image_names: List[str],
        transforms=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.transforms = transforms

        # CSV: Image Index -> Finding Labels
        df = pd.read_csv(cfg.csv_path)
        if "Image Index" not in df.columns or "Finding Labels" not in df.columns:
            raise RuntimeError(f"CSV не содержит нужных колонок. Колонки: {list(df.columns)}")

        # Быстрый словарь по имени файла
        self._label_map: Dict[str, str] = dict(zip(df["Image Index"].astype(str), df["Finding Labels"].astype(str)))

        # Индекс картинок (имя->путь)
        self._img_index = _build_image_index(cfg.data_root)

        # Собираем samples
        samples: List[Tuple[Path, torch.Tensor]] = []
        missing_in_csv = 0
        missing_on_disk = 0
        dropped_irrelevant = 0

        for name in image_names:
            lbl = self._label_map.get(name)
            if lbl is None:
                missing_in_csv += 1
                continue

            path = self._img_index.get(name)
            if path is None or not path.exists():
                missing_on_disk += 1
                continue

            # 6-вектор нужен для стабильной фильтрации "не наших" (как у тебя и было)
            vec6 = _labels_to_vec_6(lbl)

            if cfg.keep_only_relevant and (not _is_relevant(vec6)):
                dropped_irrelevant += 1
                continue

            # А наружу отдаём либо 6 классов, либо то что указали в target_labels (например 5 классов без TB)
            if cfg.target_labels is None:
                y = vec6
            else:
                y = _labels_to_vec(lbl, cfg.target_labels)

            samples.append((path, y))

        if len(samples) == 0:
            raise RuntimeError("Не удалось собрать ни одного сэмпла. Проверь пути/файлы.")

        self.samples = samples

        print("\n[NIHChestXrayDataset] Готово:")
        print("  Всего имен во входном списке:", len(image_names))
        print("  Не найдено в CSV:", missing_in_csv)
        print("  Не найдено на диске:", missing_on_disk)
        if cfg.keep_only_relevant:
            print("  Отброшено (не наши классы):", dropped_irrelevant)
        print("  Итоговых сэмплов:", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]

        img = Image.open(path).convert("RGB")  # EfficientNet/ImageNet ожидает 3 канала

        if self.transforms is not None:
            img = self.transforms(img)

        return img, y
