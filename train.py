import argparse
import csv
import json
import os
import random

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

from config import ensure_folders, get_data_paths, load_config


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_to_idx=None, samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        if class_to_idx is None:
            self.class_to_idx = self._build_class_to_idx(root_dir)
        else:
            self.class_to_idx = dict(class_to_idx)

        self.classes = [None] * len(self.class_to_idx)
        for name, idx in self.class_to_idx.items():
            self.classes[idx] = name

        if samples is None:
            self.samples = self._collect_samples(root_dir, self.class_to_idx, self.extensions)
        else:
            self.samples = list(samples)

        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label

    @staticmethod
    def _build_class_to_idx(root_dir):
        class_names = []
        for name in sorted(os.listdir(root_dir)):
            full = os.path.join(root_dir, name)
            if os.path.isdir(full):
                class_names.append(name)
        return {name: idx for idx, name in enumerate(class_names)}

    @staticmethod
    def _collect_samples(root_dir, class_to_idx, extensions):
        collected = []
        for class_name, class_idx in class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file_name in sorted(os.listdir(class_dir)):
                full_path = os.path.join(class_dir, file_name)
                if not os.path.isfile(full_path):
                    continue
                ext = os.path.splitext(file_name)[1].lower()
                if ext in extensions:
                    collected.append((full_path, class_idx))
        return collected


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms(image_size):
    try:
        random_resized_crop = A.RandomResizedCrop(
            height=image_size,
            width=image_size,
            scale=(0.9, 1.0),
            p=1.0,
        )
    except TypeError:
        random_resized_crop = A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.9, 1.0),
            p=1.0,
        )

    try:
        gauss_noise = A.GaussNoise(var_limit=(5.0, 20.0), p=0.2)
    except TypeError:
        gauss_noise = A.GaussNoise(std_range=(5.0 / 255.0, 20.0 / 255.0), p=0.2)

    train_tf = A.Compose(
        [
            random_resized_crop,
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            gauss_noise,
            A.CLAHE(clip_limit=2.0, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    val_tf = A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return train_tf, val_tf


def build_model(num_classes, pretrained=True):
    weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
    model = efficientnet_b3(weights=weights)
    ff = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(ff, num_classes)
    return model


def set_trainable_layers(model, freeze_backbone):
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True


def build_optimizer(model, head_lr, backbone_lr=None):
    if backbone_lr is None:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable_params, lr=head_lr)

    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})
    if not param_groups:
        param_groups = [{"params": model.parameters(), "lr": head_lr}]
    return torch.optim.AdamW(param_groups)


def compute_class_weights(labels, num_classes, device):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def run_one_epoch(model, loader, criterion, optimizer, device):
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    losses = []
    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = float(np.mean(losses)) if losses else 0.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


def main():
    args = parse_args()
    cfg = load_config(args.config)
    outputs_dir, models_dir = ensure_folders(cfg)
    _data_dir, train_dir, test_dir = get_data_paths(cfg)

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Missing train folder: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Missing test folder: {test_dir}")

    set_seed(int(cfg.get("seed", 42)))
    device = get_device()

    image_size = int(cfg["data"]["image_size"])
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    lr = float(cfg["train"]["lr"])
    freeze_backbone_epochs = int(cfg["train"].get("freeze_backbone_epochs", 5))
    backbone_lr = float(cfg["train"].get("backbone_lr", lr * 0.2))
    val_split = float(cfg["train"]["val_split"])
    num_workers = int(cfg["train"]["num_workers"])
    patience = int(cfg["train"]["patience"])

    train_tf, val_tf = get_transforms(image_size)

    class_to_idx = CustomImageDataset._build_class_to_idx(train_dir)
    all_samples = CustomImageDataset._collect_samples(train_dir, class_to_idx, {".jpg", ".jpeg", ".png", ".bmp"})
    base_dataset = CustomImageDataset(
        train_dir,
        transform=None,
        class_to_idx=class_to_idx,
        samples=all_samples,
    )
    labels = np.array(base_dataset.targets)
    num_classes = len(base_dataset.classes)

    all_indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=val_split,
        random_state=int(cfg.get("seed", 42)),
        stratify=labels,
    )

    train_samples = [base_dataset.samples[i] for i in train_idx.tolist()]
    val_samples = [base_dataset.samples[i] for i in val_idx.tolist()]

    train_dataset = CustomImageDataset(
        train_dir,
        transform=train_tf,
        class_to_idx=base_dataset.class_to_idx,
        samples=train_samples,
    )
    val_dataset = CustomImageDataset(
        train_dir,
        transform=val_tf,
        class_to_idx=base_dataset.class_to_idx,
        samples=val_samples,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(num_classes=num_classes, pretrained=bool(cfg["model"]["pretrained"]))
    model.to(device)

    train_labels = labels[train_idx].tolist()
    class_weights = compute_class_weights(train_labels, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    freeze_backbone_epochs = max(0, min(freeze_backbone_epochs, epochs))
    if freeze_backbone_epochs > 0:
        set_trainable_layers(model, freeze_backbone=True)
        optimizer = build_optimizer(model, head_lr=lr, backbone_lr=None)
        print(f"Phase 1: backbone frozen for first {freeze_backbone_epochs} epochs | head_lr={lr}")
    else:
        set_trainable_layers(model, freeze_backbone=False)
        optimizer = build_optimizer(model, head_lr=lr, backbone_lr=backbone_lr)
        print(f"Phase 1 skipped: full fine-tuning from epoch 1 | head_lr={lr}, backbone_lr={backbone_lr}")

    best_f1 = -1.0
    best_state = None
    bad_epochs = 0
    history_rows = []

    for epoch in range(1, epochs + 1):
        if freeze_backbone_epochs > 0 and epoch == (freeze_backbone_epochs + 1):
            set_trainable_layers(model, freeze_backbone=False)
            optimizer = build_optimizer(model, head_lr=lr, backbone_lr=backbone_lr)
            bad_epochs = 0
            print(f"Phase 2: backbone unfrozen from epoch {epoch} | head_lr={lr}, backbone_lr={backbone_lr}")

        train_loss, train_f1 = run_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1 = run_one_epoch(model, val_loader, criterion, None, device)
        phase = "head_only" if epoch <= freeze_backbone_epochs else "full_finetune"

        print(
            f"Epoch {epoch}/{epochs} | "
            f"phase={phase} | "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_macro_f1": train_f1,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
                "phase": phase,
            }
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()
            bad_epochs = 0
        else:
            bad_epochs += 1
            early_stopping_active = epoch > freeze_backbone_epochs
            if early_stopping_active and bad_epochs >= patience:
                print("Early stopping triggered.")
                break

    if best_state is None:
        best_state = model.state_dict()

    checkpoint = {
        "state_dict": best_state,
        "class_names": base_dataset.classes,
        "class_to_idx": base_dataset.class_to_idx,
        "image_size": image_size,
        "model_name": cfg["model"]["name"],
    }
    best_model_path = os.path.join(models_dir, "best_model.pth")
    torch.save(checkpoint, best_model_path)

    history_path = os.path.join(outputs_dir, "training_history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "phase", "train_loss", "train_macro_f1", "val_loss", "val_macro_f1"],
        )
        writer.writeheader()
        writer.writerows(history_rows)

    summary = {
        "best_val_macro_f1": best_f1,
        "best_model_path": best_model_path,
        "history_path": history_path,
        "device": str(device),
        "freeze_backbone_epochs": freeze_backbone_epochs,
        "head_lr": lr,
        "backbone_lr": backbone_lr,
    }
    with open(os.path.join(outputs_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
