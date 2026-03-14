import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3

from config import get_runtime_requirements, load_config, validate_runtime_versions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--topk", type=int, default=3)
    return parser.parse_args()


def build_model(num_classes):
    model = efficientnet_b3(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 224))
    model = build_model(len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, image_size


def predict_image(image_path, model, class_names, image_size, device, topk=3):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    topk = max(1, min(topk, len(class_names)))
    top_probs, top_idx = torch.topk(probs, k=topk)
    preds = []
    for p, i in zip(top_probs.tolist(), top_idx.tolist()):
        preds.append({"class": class_names[int(i)], "confidence": float(p)})

    return {
        "image": image_path,
        "predicted_class": preds[0]["class"],
        "top_predictions": preds,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    validate_runtime_versions(get_runtime_requirements(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, image_size = load_model(args.checkpoint, device)

    if not args.image and not args.image_dir:
        raise ValueError("Provide --image or --image_dir")

    results = []
    if args.image:
        results.append(
            predict_image(
                args.image,
                model,
                class_names,
                image_size,
                device,
                topk=int(args.topk),
            )
        )

    if args.image_dir:
        if not os.path.exists(args.image_dir):
            raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
        for name in sorted(os.listdir(args.image_dir)):
            full_path = os.path.join(args.image_dir, name)
            if os.path.isfile(full_path) and name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                results.append(
                    predict_image(
                        full_path,
                        model,
                        class_names,
                        image_size,
                        device,
                        topk=int(args.topk),
                    )
                )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
