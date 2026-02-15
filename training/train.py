from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import timm

from sklearn.metrics import roc_auc_score

from training.dataset import NIHConfig, NIHChestXrayDataset


PVI_LABELS_5 = [
    "Pneumonia",
    "Mass",
    "Nodule",
    "Pleural Effusion",
    "No Finding",
]


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_trainval_list(path: Path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_train_val(names, val_ratio=0.1, seed=42):
    rnd = random.Random(seed)
    names = names[:]
    rnd.shuffle(names)
    n_val = int(len(names) * val_ratio)
    val = names[:n_val]
    train = names[n_val:]
    return train, val


def make_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def _compute_auroc(y_true: torch.Tensor, y_prob: torch.Tensor, labels: list[str]):
    y_true_np = y_true.cpu().numpy()
    y_prob_np = y_prob.cpu().numpy()

    per_class = {}
    aucs = []

    for i, lab in enumerate(labels):
        yt = y_true_np[:, i]
        yp = y_prob_np[:, i]

        if yt.min() == yt.max():
            per_class[lab] = None
            continue

        auc = float(roc_auc_score(yt, yp))
        per_class[lab] = auc
        aucs.append(auc)

    macro = float(sum(aucs) / max(1, len(aucs)))
    return per_class, macro


def _compute_pos_weight_from_dataset(ds: NIHChestXrayDataset) -> torch.Tensor:
    ys = torch.stack([y for _, y in ds.samples], dim=0)  # (N,C)
    pos = ys.sum(dim=0)                                  # (C,)
    n = ys.shape[0]
    neg = n - pos

    pos = torch.clamp(pos, min=1.0)  # защита от деления на 0
    pw = neg / pos
    return pw.to(dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help=r'Напр: "D:\Projects\Datasets\NIH Chest X-ray"')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # новый параметр: потолок pos_weight
    parser.add_argument("--posw_max", type=float, default=10.0, help="Ограничение сверху для pos_weight (по умолчанию 10)")

    args = parser.parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    csv_path = data_root / "Data_Entry_2017.csv"
    trainval_list = data_root / "train_val_list.txt"

    if not csv_path.exists():
        raise FileNotFoundError(f"Не найден CSV: {csv_path}")
    if not trainval_list.exists():
        raise FileNotFoundError(f"Не найден список train/val: {trainval_list}")

    cfg = NIHConfig(
        data_root=data_root,
        csv_path=csv_path,
        trainval_list_path=trainval_list,
        keep_only_relevant=True,
        target_labels=PVI_LABELS_5,
    )

    all_names = read_trainval_list(cfg.trainval_list_path)
    train_names, val_names = split_train_val(all_names, val_ratio=args.val_ratio, seed=args.seed)

    tfm = make_transforms(img_size=224)

    train_ds = NIHChestXrayDataset(cfg, train_names, transforms=tfm)
    val_ds = NIHChestXrayDataset(cfg, val_names, transforms=tfm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # ====== pos_weight + clamp ======
    pos_weight_raw = _compute_pos_weight_from_dataset(train_ds)
    pos_weight = torch.clamp(pos_weight_raw, min=1.0, max=float(args.posw_max))

    print("\npos_weight raw (neg/pos) по классам:")
    for lab, pw in zip(PVI_LABELS_5, pos_weight_raw.tolist()):
        print(f"  {lab}: {pw:.3f}")

    print(f"\npos_weight used (clamp max={args.posw_max}) по классам:")
    for lab, pw in zip(PVI_LABELS_5, pos_weight.tolist()):
        print(f"  {lab}: {pw:.3f}")

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=len(PVI_LABELS_5))
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    def run_eval():
        model.eval()
        total_loss = 0.0
        n = 0
        all_probs = []
        all_true = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(x)
                    loss = criterion(logits, y)
                    probs = torch.sigmoid(logits)

                total_loss += float(loss.item()) * x.size(0)
                n += x.size(0)

                all_probs.append(probs.detach().cpu())
                all_true.append(y.detach().cpu())

        val_loss = total_loss / max(1, n)
        y_prob = torch.cat(all_probs, dim=0)
        y_true = torch.cat(all_true, dim=0)
        per_class_auc, macro_auc = _compute_auroc(y_true, y_prob, PVI_LABELS_5)
        return val_loss, per_class_auc, macro_auc

    weights_dir = Path("model") / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = weights_dir / f"effnetb0_nih_5cls_poswClamp{int(args.posw_max)}_{stamp}.pt"

    print("\nНачинаем обучение:")
    print("  epochs:", args.epochs)
    print("  batch_size:", args.batch_size)
    print("  lr:", args.lr)
    print("  save_path:", save_path)
    print("  labels:", PVI_LABELS_5)

    best_macro_auc = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item()) * x.size(0)
            seen += x.size(0)

            if step % 50 == 0:
                avg = running / max(1, seen)
                print(f"Epoch {epoch}/{args.epochs} | step {step}/{len(train_loader)} | train_loss={avg:.4f}")

        train_loss = running / max(1, seen)
        val_loss, per_auc, macro_auc = run_eval()

        print(f"\nEpoch {epoch} DONE | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_AUROC(macro)={macro_auc:.4f}")
        print("AUROC по классам:")
        for k in PVI_LABELS_5:
            v = per_auc.get(k, None)
            if v is None:
                print(f"  {k}: n/a")
            else:
                print(f"  {k}: {v:.4f}")

        improved = (best_macro_auc is None) or (macro_auc > best_macro_auc)
        if improved:
            best_macro_auc = macro_auc
            torch.save(
                {
                    "model_name": "efficientnet_b0",
                    "num_classes": len(PVI_LABELS_5),
                    "labels": PVI_LABELS_5,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_auroc_macro": macro_auc,
                    "pos_weight_raw": pos_weight_raw.cpu(),
                    "pos_weight_used": pos_weight.cpu(),
                    "posw_max": float(args.posw_max),
                },
                save_path,
            )
            print("✅ Saved best weights (by AUROC):", save_path)

    print("\nГотово. Лучший val_AUROC(macro):", best_macro_auc)
    print("Файл весов:", save_path)


if __name__ == "__main__":
    main()
