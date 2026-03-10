import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

from config import ensure_folders, get_data_paths, load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def build_model(num_classes):
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
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


def main():
    args = parse_args()
    cfg = load_config(args.config)
    outputs_dir, _models_dir = ensure_folders(cfg)
    _data_dir, _train_dir, test_dir = get_data_paths(cfg)

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Missing test folder: {test_dir}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]
    class_to_idx = checkpoint["class_to_idx"]
    image_size = int(checkpoint.get("image_size", cfg["data"]["image_size"]))

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    dataset = ImageFolder(test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    sample_rows = []

    idx_to_ckpt_class = {idx: name for name, idx in class_to_idx.items()}

    with torch.no_grad():
        sample_index = 0
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            targets_np = targets.numpy()

            for i in range(len(targets_np)):
                pred_name = idx_to_ckpt_class[int(preds[i])]
                pred_dataset_idx = dataset.class_to_idx[pred_name]
                y_true.append(int(targets_np[i]))
                y_pred.append(int(pred_dataset_idx))
                if len(sample_rows) < 10:
                    sample_rows.append(
                        {
                            "sample_index": sample_index,
                            "true_class": dataset.classes[int(targets_np[i])],
                            "pred_class": pred_name,
                            "confidence": float(np.max(probs[i])),
                        }
                    )
                sample_index += 1

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=dataset.classes, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(dataset.classes))))

    save_confusion_matrix(cm, dataset.classes, os.path.join(outputs_dir, "confusion_matrix.png"))

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "checkpoint": args.checkpoint,
    }
    with open(os.path.join(outputs_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(outputs_dir, "sample_preds.csv"), "w", encoding="utf-8") as f:
        f.write("sample_index,true_class,pred_class,confidence\n")
        for row in sample_rows:
            f.write(f"{row['sample_index']},{row['true_class']},{row['pred_class']},{row['confidence']:.6f}\n")

    print(json.dumps({"accuracy": float(acc), "macro_f1": float(macro_f1)}, indent=2))


if __name__ == "__main__":
    main()
