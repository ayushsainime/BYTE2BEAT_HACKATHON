import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import densenet121, efficientnet_b3, convnext_tiny, resnet18

from config import (
    ensure_folders,
    get_data_paths,
    get_runtime_requirements,
    load_config,
    validate_runtime_versions,
)


SUPPORTED_MODELS = ["densenet121", "efficientnet_b3", "convnext_tiny", "resnet18"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoints", type=str, default=None, help="Comma-separated checkpoint paths")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def build_model(model_name, num_classes):
    if model_name == "densenet121":
        model = densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "efficientnet_b3":
        model = efficientnet_b3(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "convnext_tiny":
        model = convnext_tiny(weights=None)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_name == "resnet18":
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {SUPPORTED_MODELS}")
    return model


def save_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def resolve_checkpoint_paths(args, cfg):
    if args.checkpoints:
        return [p.strip() for p in args.checkpoints.split(",") if p.strip()]

    if args.checkpoint:
        return [args.checkpoint]

    models_dir = cfg.get("logging", {}).get("model_dir", "models")
    manifest_path = os.path.join(models_dir, "ensemble_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        paths = payload.get("checkpoints", [])
        if paths:
            return paths

    return [os.path.join(models_dir, "best_model.pth")]


def load_ensemble(checkpoint_paths, device):
    ensemble = []
    for path in checkpoint_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing checkpoint: {path}")

        checkpoint = torch.load(path, map_location=device)
        class_names = checkpoint["class_names"]
        class_to_idx = checkpoint["class_to_idx"]
        model_name = checkpoint.get("model_name", "densenet121")

        model = build_model(model_name=model_name, num_classes=len(class_names))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()

        ensemble.append(
            {
                "path": path,
                "model_name": model_name,
                "model": model,
                "class_names": class_names,
                "class_to_idx": class_to_idx,
                "image_size": int(checkpoint.get("image_size", 224)),
            }
        )

    return ensemble


def main():
    args = parse_args()
    cfg = load_config(args.config)
    validate_runtime_versions(get_runtime_requirements(cfg))
    outputs_dir, _models_dir = ensure_folders(cfg)
    _data_dir, _train_dir, test_dir = get_data_paths(cfg)

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Missing test folder: {test_dir}")

    checkpoint_paths = resolve_checkpoint_paths(args, cfg)

    eval_cfg = cfg.get("eval", {})
    use_tta = bool(eval_cfg.get("tta", True)) if args.tta is None else bool(args.tta)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble = load_ensemble(checkpoint_paths, device)

    image_size = max(item["image_size"] for item in ensemble)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    dataset = ImageFolder(test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    # Precompute class-index mappings for each model to dataset class order
    for item in ensemble:
        mapping = {}
        for ckpt_name, ckpt_idx in item["class_to_idx"].items():
            if ckpt_name not in dataset.class_to_idx:
                raise RuntimeError(f"Class '{ckpt_name}' from {item['path']} not found in test dataset classes")
            mapping[int(ckpt_idx)] = int(dataset.class_to_idx[ckpt_name])
        item["idx_map"] = mapping

    y_true = []
    y_pred = []
    y_score = []
    sample_rows = []

    with torch.no_grad():
        sample_index = 0
        for images, targets in loader:
            images = images.to(device)
            batch_size = images.shape[0]
            ensemble_probs = torch.zeros((batch_size, len(dataset.classes)), device=device)

            for item in ensemble:
                model = item["model"]
                logits = model(images)
                probs = F.softmax(logits, dim=1)

                if use_tta:
                    flipped = torch.flip(images, dims=[3])
                    logits_flip = model(flipped)
                    probs_flip = F.softmax(logits_flip, dim=1)
                    probs = 0.5 * (probs + probs_flip)

                for ckpt_idx, dataset_idx in item["idx_map"].items():
                    ensemble_probs[:, dataset_idx] += probs[:, ckpt_idx]

            ensemble_probs = ensemble_probs / float(len(ensemble))
            preds = torch.argmax(ensemble_probs, dim=1).cpu().numpy()
            targets_np = targets.numpy()
            probs_np = ensemble_probs.cpu().numpy()

            for i in range(len(targets_np)):
                y_true.append(int(targets_np[i]))
                y_pred.append(int(preds[i]))
                y_score.append(probs_np[i].tolist())
                if len(sample_rows) < 10:
                    sample_rows.append(
                        {
                            "sample_index": sample_index,
                            "true_class": dataset.classes[int(targets_np[i])],
                            "pred_class": dataset.classes[int(preds[i])],
                            "confidence": float(np.max(probs_np[i])),
                        }
                    )
                sample_index += 1

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    roc_auc_ovr_macro = None
    roc_auc_ovr_weighted = None
    try:
        y_score_np = np.array(y_score, dtype=np.float32)
        roc_auc_ovr_macro = float(roc_auc_score(y_true, y_score_np, multi_class="ovr", average="macro"))
        roc_auc_ovr_weighted = float(roc_auc_score(y_true, y_score_np, multi_class="ovr", average="weighted"))
    except ValueError:
        pass

    report = classification_report(y_true, y_pred, target_names=dataset.classes, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(dataset.classes))))

    save_confusion_matrix(cm, dataset.classes, os.path.join(outputs_dir, "confusion_matrix.png"))

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "roc_auc_ovr_macro": roc_auc_ovr_macro,
        "roc_auc_ovr_weighted": roc_auc_ovr_weighted,
        "classification_report": report,
        "checkpoints": checkpoint_paths,
        "models": [item["model_name"] for item in ensemble],
        "tta": use_tta,
    }
    with open(os.path.join(outputs_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(outputs_dir, "sample_preds.csv"), "w", encoding="utf-8") as f:
        f.write("sample_index,true_class,pred_class,confidence\n")
        for row in sample_rows:
            f.write(f"{row['sample_index']},{row['true_class']},{row['pred_class']},{row['confidence']:.6f}\n")

    print(
        json.dumps(
            {
                "accuracy": float(acc),
                "macro_f1": float(macro_f1),
                "roc_auc_ovr_macro": roc_auc_ovr_macro,
                "roc_auc_ovr_weighted": roc_auc_ovr_weighted,
                "models": metrics["models"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
