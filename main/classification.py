import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import random


# ------------- Paths (relative to main/) -------------
# main/classification.py  ->  project root  ->  fall_dataset/images/{train,val}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_ROOT = PROJECT_ROOT / "fall_dataset" / "images"   # contains train/ and val/
MODEL_PATH = PROJECT_ROOT / "saved_model" / "fall_model.pth"


# ------------- Reproducibility -------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------- Label from filename -------------
def label_from_name(path: Path) -> int:
    """
    Return 1 for FALL, 0 for NOT-FALL based on filename.
    Handles names like 'fall123.jpg' vs 'not fallen001.jpg'.
    """
    name = path.stem.lower()
    # normalize separators
    norm = re.sub(r"[\W_]+", " ", name).strip()

    # check NOT-fall patterns first to avoid 'fall' substring catching 'not fallen'
    not_fall_patterns = [
        "not fall", "notfall", "not fallen", "notfallen", "no fall", "nofall",
        "standing", "walk", "walking", "sit", "sitting", "upright"
    ]
    if any(p in norm for p in not_fall_patterns):
        return 0

    # fall patterns
    if "fall" in norm or norm.startswith("fallen"):
        return 1

    # default to NOT-FALL if unclear
    return 0


# ------------- Custom Dataset -------------
class NameLabelDataset(Dataset):
    def __init__(self, directory: Path, transform=None):
        self.directory = Path(directory)
        self.transform = transform
        self.paths = self._gather_images(self.directory)

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in: {self.directory}")

        # quick stats
        ys = [label_from_name(p) for p in self.paths]
        self.class_counts = {0: ys.count(0), 1: ys.count(1)}

    @staticmethod
    def _gather_images(d: Path) -> List[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        paths = [p for p in d.iterdir() if p.suffix.lower() in exts]
        # also include nested files if any
        for sub in [x for x in d.iterdir() if x.is_dir()]:
            paths.extend([p for p in sub.rglob("*") if p.suffix.lower() in exts])
        return sorted(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y = label_from_name(p)
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y


# ------------- Model / Train / Eval -------------
def build_model(num_classes: int = 2, freeze_backbone: bool = True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    # always train the final layer
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


def epoch_loop(model, loader, device, optimizer=None, criterion=None) -> Tuple[float, float, float, float, np.ndarray]:
    """
    If optimizer is None => evaluation mode.
    Returns: (loss, acc, precision, recall, f1, confusion_matrix)
    """
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

    # Paths
    train_dir = IMAGES_ROOT / "train"
    val_dir = IMAGES_ROOT / "val"

    # Data
    tfms = get_transforms()
    train_ds = NameLabelDataset(train_dir, transform=tfms)
    val_ds = NameLabelDataset(val_dir, transform=tfms)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Class weights to handle imbalance
    total_train = len(train_ds)
    w0 = total_train / (2.0 * max(train_ds.class_counts.get(0, 1), 1))
    w1 = total_train / (2.0 * max(train_ds.class_counts.get(1, 1), 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=2, freeze_backbone=not args.unfreeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Train dir:    {train_dir} (count={len(train_ds)} | class_counts={train_ds.class_counts})")
    print(f"Val dir:      {val_dir}   (count={len(val_ds)} | class_counts={val_ds.class_counts})")
    print(f"Saving model to: {MODEL_PATH}")
    print(f"Device: {device}\n")

    # Training
    best_val_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_p, tr_r, tr_f1, tr_cm = epoch_loop(model, train_loader, device, optimizer, criterion)
        va_loss, va_acc, va_p, va_r, va_f1, va_cm = epoch_loop(model, val_loader, device)

        print(f"Epoch {epoch:02d}/{args.epochs} "
              f"| Train L {tr_loss:.4f} A {tr_acc:.3f} P {tr_p:.3f} R {tr_r:.3f} F1 {tr_f1:.3f} "
              f"| Val L {va_loss:.4f} A {va_acc:.3f} P {va_p:.3f} R {va_r:.3f} F1 {va_f1:.3f}")
        print(f"  Train CM (rows true [0,1], cols pred [0,1]):\n{tr_cm}")
        print(f"  Val   CM (rows true [0,1], cols pred [0,1]):\n{va_cm}\n")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Saved best model (val F1={best_val_f1:.3f}) → {MODEL_PATH}\n")

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
