from __future__ import annotations
from pathlib import Path
import pandas as pd

# === НАСТРОЙ ПУТЕЙ (у тебя так) ===
DATA_ROOT = Path(r"D:\Projects\Datasets\NIH Chest X-ray")
CSV_PATH = DATA_ROOT / "Data_Entry_2017.csv"
TRAINVAL_LIST = DATA_ROOT / "train_val_list.txt"

# === Наши классы (как в UI хотим) ===
PVI_LABELS = ["Pneumonia", "Mass", "Nodule", "Pleural Effusion", "No Finding"]

# === Как эти классы называются в NIH CSV ===
# (Pleural Effusion в NIH называется просто Effusion)
NIH_NAME = {
    "Pneumonia": "Pneumonia",
    "Mass": "Mass",
    "Nodule": "Nodule",
    "Pleural Effusion": "Effusion",
    "No Finding": "No Finding",
}

def find_image_path(data_root: Path, image_name: str) -> Path | None:
    """
    У тебя структура: images_001/images/<file.png>
    Пробегаем images_001..images_012 и ищем файл.
    """
    for i in range(1, 13):
        folder = data_root / f"images_{i:03d}" / "images" / image_name
        if folder.exists():
            return folder
    return None

def labels_to_vector(finding_labels: str) -> list[int]:
    parts = [p.strip() for p in str(finding_labels).split("|")]
    vec = []
    for pvi_label in PVI_LABELS:
        nih_label = NIH_NAME[pvi_label]
        vec.append(1 if nih_label in parts else 0)
    return vec

def main():
    print("DATA_ROOT:", DATA_ROOT)
    print("CSV_PATH exists:", CSV_PATH.exists())
    print("TRAINVAL_LIST exists:", TRAINVAL_LIST.exists())

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Не найден CSV: {CSV_PATH}")
    if not TRAINVAL_LIST.exists():
        raise FileNotFoundError(f"Не найден список train/val: {TRAINVAL_LIST}")

    df = pd.read_csv(CSV_PATH)

    # Проверим, что нужные колонки есть
    need_cols = {"Image Index", "Finding Labels"}
    missing = need_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"В CSV нет колонок: {missing}. Есть: {list(df.columns)}")

    # Быстрая статистика по меткам
    print("\nПримеры Finding Labels:")
    print(df["Finding Labels"].head(10).to_string(index=False))

    # Откроем train_val_list и проверим первые 5 файлов
    names = [line.strip() for line in TRAINVAL_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]
    print("\ntrain_val_list count:", len(names))
    print("Первые 5 файлов:", names[:5])

    # Проверим, что эти файлы реально находятся на диске
    print("\nПроверка путей к картинкам (первые 5):")
    for n in names[:5]:
        p = find_image_path(DATA_ROOT, n)
        print(" ", n, "->", p if p else "НЕ НАЙДЕН")

    # Проверим, как превращается label в вектор (первые 5 строк CSV)
    print("\nПроверка векторов (первые 5 строк CSV):")
    for i in range(5):
        row = df.iloc[i]
        vec = labels_to_vector(row["Finding Labels"])
        print(f" {row['Image Index']} | {row['Finding Labels']} -> {vec}  (порядок {PVI_LABELS})")

    # Найдём одну картинку из списка и покажем её labels
    sample_name = names[0]
    sample_row = df[df["Image Index"] == sample_name]
    if len(sample_row) == 0:
        print("\n⚠ В CSV нет строки для:", sample_name)
    else:
        lbl = sample_row.iloc[0]["Finding Labels"]
        print("\nПример из train_val_list:")
        print(" ", sample_name, "|", lbl, "->", labels_to_vector(lbl))

    print("\n✅ dataset_check.py: всё выглядит нормально. Можно делать настоящий Dataset.")

if __name__ == "__main__":
    main()
