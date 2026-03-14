import io
import os

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms
from torchvision.models import densenet121

from config import get_runtime_requirements, load_config, validate_runtime_versions


app = FastAPI(title="Heart Disease Prediction API")

MODEL = None
CLASS_NAMES = None
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes):
    model = densenet121(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    return model


def get_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


@app.on_event("startup")
def startup():
    global MODEL, CLASS_NAMES, IMAGE_SIZE

    try:
        cfg = load_config("config.yaml")
    except FileNotFoundError:
        cfg = {}
    validate_runtime_versions(get_runtime_requirements(cfg))

    checkpoint_path = os.getenv("CHECKPOINT_PATH", "models/best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise RuntimeError("Checkpoint not found. Train first or set CHECKPOINT_PATH.")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    CLASS_NAMES = checkpoint["class_names"]
    IMAGE_SIZE = int(checkpoint.get("image_size", 224))

    MODEL = build_model(len(CLASS_NAMES))
    MODEL.load_state_dict(checkpoint["state_dict"])
    MODEL.to(DEVICE)
    MODEL.eval()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if file.content_type not in {"image/jpeg", "image/jpg", "image/png", "image/bmp"}:
        raise HTTPException(status_code=400, detail="Please upload a valid image")

    try:
        data = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    transform = get_transform(IMAGE_SIZE)
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    top_probs, top_idx = torch.topk(probs, k=min(3, len(CLASS_NAMES)))
    preds = []
    for p, i in zip(top_probs.tolist(), top_idx.tolist()):
        preds.append({"class": CLASS_NAMES[int(i)], "confidence": float(p)})

    return {"predicted_class": preds[0]["class"], "top_predictions": preds}
