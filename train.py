import argparse
import csv
import hashlib
import json
import math
import os
import random

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    DenseNet121_Weights,
    EfficientNet_B3_Weights,
    ResNet18_Weights,
    convnext_tiny,
    densenet121,
    efficientnet_b3,
    resnet18,
)

from config import (
    ensure_folders,
    get_data_paths,
    get_runtime_requirements,
    load_config,
    validate_runtime_versions,
)


SUPPORTED_MODELS = ["densenet121", "efficientnet_b3", "convnext_tiny", "resnet18"]


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
    except (TypeError, ValueError):
        random_resized_crop = A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.9, 1.0),
            p=1.0,
        )

    try:
        gauss_noise = A.GaussNoise(var_limit=(5.0, 20.0), p=0.2)
    except (TypeError, ValueError):
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


def build_model(model_name, num_classes, pretrained=True):
    if model_name == "densenet121":
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "efficientnet_b3":
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = efficientnet_b3(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {SUPPORTED_MODELS}")
    return model


def get_head_prefix(model_name):
    if model_name in {"densenet121", "convnext_tiny", "efficientnet_b3"}:
        return "classifier"
    if model_name == "resnet18":
        return "fc"
    raise ValueError(f"Unsupported model '{model_name}'")


def set_trainable_layers(model, model_name, freeze_backbone):
    head_prefix = get_head_prefix(model_name)
    for name, param in model.named_parameters():
        if freeze_backbone:
            param.requires_grad = name.startswith(head_prefix)
        else:
            param.requires_grad = True


def build_optimizer(model, model_name, head_lr, backbone_lr=None, weight_decay=1e-4):
    head_prefix = get_head_prefix(model_name)
    if backbone_lr is None:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=head_lr, weight_decay=weight_decay)
        for group in optimizer.param_groups:
            group["initial_lr"] = head_lr
        return optimizer

    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(head_prefix):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr, "initial_lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr, "initial_lr": head_lr})
    if not param_groups:
        param_groups = [{"params": model.parameters(), "lr": head_lr, "initial_lr": head_lr}]

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def lr_scale_for_epoch(epoch, total_epochs, warmup_epochs):
    warmup_epochs = max(0, min(warmup_epochs, total_epochs))
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return float(epoch) / float(warmup_epochs)

    if total_epochs <= warmup_epochs:
        return 1.0

    progress = float(epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def set_epoch_lrs(optimizer, scale):
    for group in optimizer.param_groups:
        base_lr = float(group.get("initial_lr", group["lr"]))
        group["lr"] = base_lr * scale


def compute_class_weights(labels, num_classes, device):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def build_weighted_sampler(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    class_weights = 1.0 / counts
    sample_weights = [class_weights[int(lbl)] for lbl in labels]
    return WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)


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


def compute_file_sha1(path):
    hasher = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def leakage_report(train_samples, test_samples, outputs_dir):
    train_hash_to_paths = {}
    for path, _ in train_samples:
        h = compute_file_sha1(path)
        train_hash_to_paths.setdefault(h, []).append(path)

    overlap = []
    for path, _ in test_samples:
        h = compute_file_sha1(path)
        if h in train_hash_to_paths:
            overlap.append({"hash": h, "test_path": path, "train_paths": train_hash_to_paths[h][:3]})

    report = {
        "train_files": len(train_samples),
        "test_files": len(test_samples),
        "exact_hash_overlap_count": len(overlap),
        "overlaps_preview": overlap[:20],
    }
    report_path = os.path.join(outputs_dir, "leakage_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if overlap:
        print(f"WARNING: Found {len(overlap)} exact train/test duplicate images. See {report_path}")
    else:
        print("Leakage check: no exact train/test duplicate hashes found.")

    return report


def train_single_split(
    model_name,
    class_to_idx,
    train_samples,
    val_samples,
    image_size,
    device,
    train_cfg,
    pretrained,
):
    train_tf, val_tf = get_transforms(image_size)
    train_dataset = CustomImageDataset(
        root_dir="",
        transform=train_tf,
        class_to_idx=class_to_idx,
        samples=train_samples,
    )
    val_dataset = CustomImageDataset(
        root_dir="",
        transform=val_tf,
        class_to_idx=class_to_idx,
        samples=val_samples,
    )

    labels_train = [lbl for _, lbl in train_samples]
    num_classes = len(class_to_idx)

    use_sampler = bool(train_cfg.get("use_weighted_sampler", True))
    if use_sampler:
        sampler = build_weighted_sampler(labels_train, num_classes)
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            sampler=sampler,
            num_workers=int(train_cfg["num_workers"]),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=True,
            num_workers=int(train_cfg["num_workers"]),
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
    )

    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
    model.to(device)

    class_weights = compute_class_weights(labels_train, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    epochs = int(train_cfg["epochs"])
    lr = float(train_cfg["lr"])
    backbone_lr = float(train_cfg.get("backbone_lr", lr * 0.2))
    freeze_backbone_epochs = int(train_cfg.get("freeze_backbone_epochs", 0))
    freeze_backbone_epochs = max(0, min(freeze_backbone_epochs, epochs))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 2))
    patience = int(train_cfg.get("patience", 6))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))

    if freeze_backbone_epochs > 0:
        set_trainable_layers(model, model_name=model_name, freeze_backbone=True)
        optimizer = build_optimizer(model, model_name, head_lr=lr, backbone_lr=None, weight_decay=weight_decay)
    else:
        set_trainable_layers(model, model_name=model_name, freeze_backbone=False)
        optimizer = build_optimizer(model, model_name, head_lr=lr, backbone_lr=backbone_lr, weight_decay=weight_decay)

    best_f1 = -1.0
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        if freeze_backbone_epochs > 0 and epoch == (freeze_backbone_epochs + 1):
            set_trainable_layers(model, model_name=model_name, freeze_backbone=False)
            optimizer = build_optimizer(model, model_name, head_lr=lr, backbone_lr=backbone_lr, weight_decay=weight_decay)
            bad_epochs = 0

        scale = lr_scale_for_epoch(epoch, epochs, warmup_epochs)
        set_epoch_lrs(optimizer, scale)

        train_loss, train_f1 = run_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1 = run_one_epoch(model, val_loader, criterion, None, device)
        phase = "head_only" if epoch <= freeze_backbone_epochs else "full_finetune"

        current_lrs = [float(g["lr"]) for g in optimizer.param_groups]

        history.append(
            {
                "epoch": epoch,
                "phase": phase,
                "lr_scale": scale,
                "lr_min": float(min(current_lrs)),
                "lr_max": float(max(current_lrs)),
                "train_loss": train_loss,
                "train_macro_f1": train_f1,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
            }
        )

        print(
            f"[{model_name}] Epoch {epoch}/{epochs} phase={phase} "
            f"lr=[{min(current_lrs):.2e},{max(current_lrs):.2e}] "
            f"train_f1={train_f1:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[{model_name}] Early stopping triggered at epoch {epoch}.")
                break

    if best_state is None:
        best_state = model.state_dict()

    return best_state, best_f1, history


def save_history(path, history_rows):
    if not history_rows:
        return
    fieldnames = list(history_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    validate_runtime_versions(get_runtime_requirements(cfg))
    outputs_dir, models_dir = ensure_folders(cfg)
    _data_dir, train_dir, test_dir = get_data_paths(cfg)

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Missing train folder: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Missing test folder: {test_dir}")

    set_seed(int(cfg.get("seed", 42)))
    device = get_device()
    print(f"Using device: {device}")

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    ensemble_cfg = cfg.get("ensemble", {})

    image_size = int(data_cfg.get("image_size", 320))
    val_split = float(train_cfg.get("val_split", 0.2))
    cv_folds = int(train_cfg.get("cv_folds", 5))
    cv_epochs = int(train_cfg.get("cv_epochs", max(10, int(train_cfg.get("epochs", 40)) // 2)))

    class_to_idx = CustomImageDataset._build_class_to_idx(train_dir)
    all_train_samples = CustomImageDataset._collect_samples(train_dir, class_to_idx, {".jpg", ".jpeg", ".png", ".bmp"})
    all_test_samples = CustomImageDataset._collect_samples(test_dir, class_to_idx, {".jpg", ".jpeg", ".png", ".bmp"})

    if not all_train_samples:
        raise RuntimeError("No training images found.")

    leakage_report(all_train_samples, all_test_samples, outputs_dir)

    labels = np.array([lbl for _, lbl in all_train_samples])
    all_indices = np.arange(len(all_train_samples))

    model_names = ensemble_cfg.get("models", [cfg.get("model", {}).get("name", "densenet121")])
    model_names = [m for m in model_names if m in SUPPORTED_MODELS]
    if not model_names:
        raise RuntimeError(f"No supported models in ensemble list. Supported: {SUPPORTED_MODELS}")

    cv_results = {}
    trained_checkpoints = []
    summary_rows = []

    for model_name in model_names:
        print("=" * 80)
        print(f"Model: {model_name} | CV start ({cv_folds}-fold)")
        print("=" * 80)

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=int(cfg.get("seed", 42)))
        fold_scores = []

        for fold_id, (tr_idx, va_idx) in enumerate(skf.split(all_indices, labels), start=1):
            fold_train_samples = [all_train_samples[i] for i in tr_idx.tolist()]
            fold_val_samples = [all_train_samples[i] for i in va_idx.tolist()]

            fold_train_cfg = dict(train_cfg)
            fold_train_cfg["epochs"] = cv_epochs

            _, fold_best_f1, _ = train_single_split(
                model_name=model_name,
                class_to_idx=class_to_idx,
                train_samples=fold_train_samples,
                val_samples=fold_val_samples,
                image_size=image_size,
                device=device,
                train_cfg=fold_train_cfg,
                pretrained=bool(cfg.get("model", {}).get("pretrained", True)),
            )
            fold_scores.append(float(fold_best_f1))
            print(f"[{model_name}] Fold {fold_id}/{cv_folds} best val Macro-F1: {fold_best_f1:.4f}")

        cv_results[model_name] = {
            "fold_macro_f1": fold_scores,
            "mean_macro_f1": float(np.mean(fold_scores)),
            "std_macro_f1": float(np.std(fold_scores)),
        }

        print(f"[{model_name}] CV mean Macro-F1: {cv_results[model_name]['mean_macro_f1']:.4f}")

        # Final train/val split for checkpoint selection
        final_train_idx, final_val_idx = train_test_split(
            all_indices,
            test_size=val_split,
            random_state=int(cfg.get("seed", 42)),
            stratify=labels,
        )
        final_train_samples = [all_train_samples[i] for i in final_train_idx.tolist()]
        final_val_samples = [all_train_samples[i] for i in final_val_idx.tolist()]

        best_state, best_f1, history = train_single_split(
            model_name=model_name,
            class_to_idx=class_to_idx,
            train_samples=final_train_samples,
            val_samples=final_val_samples,
            image_size=image_size,
            device=device,
            train_cfg=train_cfg,
            pretrained=bool(cfg.get("model", {}).get("pretrained", True)),
        )

        class_names = [None] * len(class_to_idx)
        for name, idx in class_to_idx.items():
            class_names[idx] = name

        ckpt_path = os.path.join(models_dir, f"{model_name}_best.pth")
        checkpoint = {
            "state_dict": best_state,
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "image_size": image_size,
            "model_name": model_name,
            "cv_mean_macro_f1": cv_results[model_name]["mean_macro_f1"],
        }
        torch.save(checkpoint, ckpt_path)
        trained_checkpoints.append(ckpt_path)

        history_path = os.path.join(outputs_dir, f"{model_name}_training_history.csv")
        save_history(history_path, history)

        summary_rows.append(
            {
                "model_name": model_name,
                "cv_mean_macro_f1": cv_results[model_name]["mean_macro_f1"],
                "cv_std_macro_f1": cv_results[model_name]["std_macro_f1"],
                "final_split_best_val_macro_f1": float(best_f1),
                "checkpoint": ckpt_path,
                "history": history_path,
            }
        )

    # Compatibility checkpoint for single-model tools
    if trained_checkpoints:
        primary_ckpt = trained_checkpoints[0]
        primary_data = torch.load(primary_ckpt, map_location="cpu")
        torch.save(primary_data, os.path.join(models_dir, "best_model.pth"))

    with open(os.path.join(outputs_dir, "cv_results.json"), "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=2)

    train_summary = {
        "device": str(device),
        "image_size": image_size,
        "models_trained": model_names,
        "trained_checkpoints": trained_checkpoints,
        "summary_rows": summary_rows,
    }
    with open(os.path.join(outputs_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(train_summary, f, indent=2)

    with open(os.path.join(models_dir, "ensemble_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"checkpoints": trained_checkpoints}, f, indent=2)

    print("Training complete.")
    print(json.dumps(train_summary, indent=2))


if __name__ == "__main__":
    main()
