import argparse
import os
from pathlib import Path
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import random
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_ROOT = PROJECT_ROOT / "processed_dataset" / "images"
LABELS_ROOT = PROJECT_ROOT / "processed_dataset" / "labels"
MODEL_PATH = PROJECT_ROOT / "saved_model" / "efficientnet_fall_model.pth"
METRICS_FILE = PROJECT_ROOT / "training_metrics_efficientnet.csv"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class LabelFileDataset(Dataset):
    """YOLO-style dataset simplified for classification (0 = fall, 1 = not fall)."""
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
    
def build_model(model_name="efficientnet_b0", num_classes: int = 2, freeze_backbone: bool = True):
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
    return model

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_class = pred.size(1)
        log_preds = torch.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_class - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_preds, dim=1))

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def epoch_loop(model, loader, device, optimizer=None, criterion=None, mixup=False) -> Tuple[float, float, float, float, float, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    all_preds, all_labels = [], []
    running_loss, total = 0.0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device, dtype=torch.long)

        if is_train and mixup:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels)
            logits = model(imgs)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
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

    train_img_dir, val_img_dir = IMAGES_ROOT / "train", IMAGES_ROOT / "val"
    train_lbl_dir, val_lbl_dir = LABELS_ROOT / "train", LABELS_ROOT / "val"

    tfms = get_transforms()
    train_ds = LabelFileDataset(train_img_dir, train_lbl_dir, tfms)
    val_ds = LabelFileDataset(val_img_dir, val_lbl_dir, tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    total_train = len(train_ds)
    w0 = total_train / (2.0 * max(train_ds.class_counts.get(0, 1), 1))
    w1 = total_train / (2.0 * max(train_ds.class_counts.get(1, 1), 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, num_classes=2, freeze_backbone=not args.unfreeze_backbone).to(device)

    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    swa_model = AveragedModel(model)
    swa_start = int(0.75 * args.epochs)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-4)

    with open(METRICS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1"])

    best_val_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_p, tr_r, tr_f1, tr_cm = epoch_loop(model, train_loader, device, optimizer, criterion, mixup=True)
        va_loss, va_acc, va_p, va_r, va_f1, va_cm = epoch_loop(model, val_loader, device, criterion=criterion)

        scheduler.step()
        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs} "
              f"| Train L {tr_loss:.4f} A {tr_acc:.3f} F1 {tr_f1:.3f} "
              f"| Val L {va_loss:.4f} A {va_acc:.3f} F1 {va_f1:.3f}")
        print(f"  Train CM:\n{tr_cm}")
        print(f"  Val   CM:\n{va_cm}\n")

        with open(METRICS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, tr_loss, tr_acc, tr_f1, va_loss, va_acc, va_f1])

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved best model (val F1 = {best_val_f1:.3f}) → {MODEL_PATH}\n")

    print(f"Done. Metrics saved to {METRICS_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                        choices=["efficientnet_b0", "efficientnet_b3"])
    args = parser.parse_args()
    main(args)
