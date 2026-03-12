import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

from config import ensure_folders, get_data_paths, load_config
from sklearn.model_selection import train_test_split


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
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_tf, val_tf


def build_model(num_classes, pretrained=True):
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b3(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


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
    val_split = float(cfg["train"]["val_split"])
    num_workers = int(cfg["train"]["num_workers"])
    patience = int(cfg["train"]["patience"])

    train_tf, val_tf = get_transforms(image_size)

    dataset_train_tf = ImageFolder(train_dir, transform=train_tf)
    dataset_val_tf = ImageFolder(train_dir, transform=val_tf)
    labels = np.array(dataset_train_tf.targets)
    num_classes = len(dataset_train_tf.classes)

    all_indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=val_split,
        random_state=int(cfg.get("seed", 42)),
        stratify=labels,
    )

    train_subset = Subset(dataset_train_tf, train_idx.tolist())
    val_subset = Subset(dataset_val_tf, val_idx.tolist())

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(num_classes=num_classes, pretrained=bool(cfg["model"]["pretrained"]))
    model.to(device)

    train_labels = labels[train_idx].tolist()
    class_weights = compute_class_weights(train_labels, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_f1 = -1.0
    best_state = None
    bad_epochs = 0
    history_rows = []

    for epoch in range(1, epochs + 1):
        train_loss, train_f1 = run_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1 = run_one_epoch(model, val_loader, criterion, None, device)

        print(
            f"Epoch {epoch}/{epochs} | "
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
            }
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping triggered.")
                break

    if best_state is None:
        best_state = model.state_dict()

    checkpoint = {
        "state_dict": best_state,
        "class_names": dataset_train_tf.classes,
        "class_to_idx": dataset_train_tf.class_to_idx,
        "image_size": image_size,
        "model_name": cfg["model"]["name"],
    }
    best_model_path = os.path.join(models_dir, "best_model.pth")
    torch.save(checkpoint, best_model_path)

    history_path = os.path.join(outputs_dir, "training_history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_macro_f1", "val_loss", "val_macro_f1"])
        writer.writeheader()
        writer.writerows(history_rows)

    summary = {
        "best_val_macro_f1": best_f1,
        "best_model_path": best_model_path,
        "history_path": history_path,
        "device": str(device),
    }
    with open(os.path.join(outputs_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
