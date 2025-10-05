import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_ROOT = PROJECT_ROOT / "processed_dataset" / "images"
LABELS_ROOT = PROJECT_ROOT / "processed_dataset" / "labels"
MODEL_PATH = PROJECT_ROOT / "saved_model" / "resnet_baseline_fall_model.pth"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class LabelFileDataset(Dataset):
    """
    Dataset that reads images and labels from YOLO-style dataset.
    Simplified for classification: 0 = fall, 1 = not fall.
    """
    def __init__(self, images_dir: Path, labels_dir: Path, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        self.paths = [p for p in self.images_dir.iterdir() if p.suffix.lower() in exts]

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in: {self.images_dir}")

        ys = [self._read_label(p) for p in self.paths]
        self.class_counts = {0: ys.count(0), 1: ys.count(1)}

    def _read_label(self, img_path: Path) -> int:
        lbl_file = self.labels_dir / (img_path.stem + ".txt")
        if not lbl_file.exists():
            raise FileNotFoundError(f"Label file not found for {img_path.name}")
        with open(lbl_file, "r") as f:
            line = f.readline().strip()
            if not line:
                return 1
            parts = line.split()
            return int(parts[0])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y = self._read_label(p)
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y

def build_model(num_classes: int = 2, freeze_backbone: bool = True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

def epoch_loop(model, loader, device, optimizer=None, criterion=None) -> Tuple[float, float, float, float, float, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)

    all_preds, all_labels = [], []
    running_loss = 0.0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device, dtype=torch.long)

        with torch.set_grad_enabled(is_train):
            logits = model(imgs)
            loss = criterion(logits, labels) if criterion else torch.tensor(0.0, device=device)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    return avg_loss, acc, prec, rec, f1, cm

def main(args):
    set_seed(args.seed)

    train_img_dir = IMAGES_ROOT / "train"
    val_img_dir = IMAGES_ROOT / "val"
    train_lbl_dir = LABELS_ROOT / "train"
    val_lbl_dir = LABELS_ROOT / "val"

    tfms = get_transforms()
    train_ds = LabelFileDataset(train_img_dir, train_lbl_dir, transform=tfms)
    val_ds = LabelFileDataset(val_img_dir, val_lbl_dir, transform=tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    total_train = len(train_ds)
    w0 = total_train / (2.0 * max(train_ds.class_counts.get(0, 1), 1))
    w1 = total_train / (2.0 * max(train_ds.class_counts.get(1, 1), 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=2, freeze_backbone=not args.unfreeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Train dir:    {train_img_dir} (count={len(train_ds)} | class_counts={train_ds.class_counts})")
    print(f"Val dir:      {val_img_dir}   (count={len(val_ds)} | class_counts={val_ds.class_counts})")
    print(f"Saving model to: {MODEL_PATH}")
    print(f"Device: {device}\n")

    best_val_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_p, tr_r, tr_f1, tr_cm = epoch_loop(model, train_loader, device, optimizer, criterion)
        va_loss, va_acc, va_p, va_r, va_f1, va_cm = epoch_loop(model, val_loader, device)

        print(f"Epoch {epoch:02d}/{args.epochs} "
              f"| Train L {tr_loss:.4f} A {tr_acc:.3f} P {tr_p:.3f} R {tr_r:.3f} F1 {tr_f1:.3f} "
              f"| Val L {va_loss:.4f} A {va_acc:.3f} P {va_p:.3f} R {va_r:.3f} F1 {va_f1:.3f}")
        print(f"  Train CM:\n{tr_cm}")
        print(f"  Val   CM:\n{va_cm}\n")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved best model (val F1={best_val_f1:.3f}) → {MODEL_PATH}\n")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze-backbone", action="store_true",
                        help="Fine-tune the whole ResNet18 instead of just the final layer.")
    args = parser.parse_args()
    main(args)
