---
title: Eye Heart Connection
emoji: 👁️❤️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# Eye-Heart Connection

End-to-end multimodal AI system for cardiovascular-risk-oriented retinal screening.
The model consumes:
- Left fundus image
- Right fundus image
- Patient age

It predicts 8 ophthalmic indicators and derives an interpretable cardiovascular (CV) risk proxy summary (`low`, `medium`, `high`).

## Highlights

- Bilateral retinal modeling (left + right eye) with age fusion
- 8-label multi-label prediction
- CV proxy post-processing for clinically meaningful risk banding
- Production-ready inference API with FastAPI
- Reflex UI with modern layout, charting, and guided patient flow
- Dockerized Hugging Face Spaces deployment

## Labels Predicted

- `N` Normal
- `D` Diabetes
- `G` Glaucoma
- `C` Cataract
- `A` AMD
- `H` Hypertension
- `M` Myopia
- `O` Other

## Model Performance

### Training and Validation Metrics
![Training and validation metrics](https://huggingface.co/datasets/ayushsainime/eye_heart_connect_media/resolve/main/mode_training_%20and_validation_metrics.png)

### Patient Prediction Metrics
![Patient prediction metrics](https://huggingface.co/datasets/ayushsainime/eye_heart_connect_media/resolve/main/patient_predictiion_metrics.png)

### Precision-Recall Curves
![Precision recall curves](https://huggingface.co/datasets/ayushsainime/eye_heart_connect_media/resolve/main/precision_recall_curves.png)

### ROC Curves by Labels
![ROC curves by labels](https://huggingface.co/datasets/ayushsainime/eye_heart_connect_media/resolve/main/ROC%20CURVES%20BY%20LABELS.png)

## System Architecture


1. Left and right fundus images are encoded via a shared CNN backbone (EfficientNet-B4 based pipeline).
2. Age is processed through a metadata branch.
3. Visual and metadata features are fused for 8-label multi-label prediction.
4. Predicted probabilities are transformed into CV proxy components and a final risk band.

## Tech Stack

### ML and Data
- PyTorch
- Torchvision
- Albumentations
- OpenCV
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- TensorBoard

### Backend and Serving
- FastAPI
- Uvicorn
- Pydantic
- python-multipart
- HTTPX

### Frontend
- Reflex (Radix UI + Recharts)

### Packaging and Deployment
- Docker
- Hugging Face Spaces (Docker SDK)

## Repository Structure

```text
api/                    FastAPI inference API
artifacts/              Runtime artifacts (best.pt, thresholds, metadata stats)
assets/                 Frontend static assets (animation, css, sample_cases)
configs/                API/model/data/inference/train configuration files
datasets/               Dataset preparation and loaders
evaluation/             Evaluation scripts and reports
inference/              Predictor and inference logic
models/                 Model definitions
reflex_app/             Reflex frontend application
tests/                  Test suite
training/               Training pipeline
utils/                  Shared utilities (config, logging, etc.)
```

## Local Setup

### 1) Create environment and install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -e ".[frontend,dev]"
```

## Run Locally

### Reflex UI + Inference API

Terminal 1 (API):

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Terminal 2 (Reflex):

```bash
reflex run
```

Open: `http://localhost:3000`

Notes:
- Reflex config is in `rxconfig.py` (frontend `3000`, Reflex backend `8001`).
- Reflex frontend calls inference API at `http://localhost:8000`.

## API Reference

### Endpoints

- `GET /health`
- `POST /predict`

### `POST /predict` form fields

- `left_image`: file
- `right_image`: file
- `age`: float

### Example cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "left_image=@1_left.jpg" \
  -F "right_image=@1_right.jpg" \
  -F "age=55"
```

## Hugging Face Spaces Deployment (Docker)

This repo is strictly configured for Hugging Face Spaces via the Docker SDK.

### Runtime artifacts required in `artifacts/`

- `best.pt` (Must be tracked via Git LFS)
- `metadata_stats.json`
- `thresholds.json`

### Container runtime behavior

- Hugging Face detects the `sdk: docker` YAML header inside `README.md`.
- The `Dockerfile` compiles the environment natively as non-root `user 1000`.
- The container runs `start_hf.sh` which boots both the FastAPI backend and Reflex frontends locally, hooking the Reflex port natively to Hugging Face's global routing (`7860`).

### Update/Push workflow
Initialize Git LFS and push to the Space URL:
```bash
git lfs install
git add .
git commit -m "Deploy to HF Spaces"
git push origin main
```

## Training and Evaluation

### Train

```bash
python -m training.train --config configs/train.yaml --data-config configs/data.yaml --model-config configs/model.yaml
```

### Evaluate

```bash
python -m evaluation.run --config configs/eval.yaml --data-config configs/data.yaml --model-config configs/model.yaml --ckpt experiments/latest/best.pt
```

## Important Notes

- This project is designed as a research and screening aid.
- Outputs are predictive signals, not clinical diagnoses.
- External validation and clinical oversight are required for real-world medical use.

## Disclaimer

This repository is for research and educational use only. It is not a certified medical device and must not be used as a standalone basis for clinical decision-making.
