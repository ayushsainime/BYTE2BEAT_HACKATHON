---
title: Eye-Heart Connection
emoji: stethoscope
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
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
- Production-ready API with FastAPI
- Reflex UI (modern interface with sample image dropdown, charting, and medical information sidebar)
- Dockerized Hugging Face Spaces deployment for the Reflex experience

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
![Training and validation metrics](photos/mode_training_\ and_validation_metrics.png)

### Patient Prediction Metrics
![Patient prediction metrics](photos/patient_predictiion_metrics.png)

### Precision-Recall Curves
![Precision recall curves](photos/precision_recall_curves.png)

### ROC Curves by Labels
![ROC curves by labels](photos/ROC\ CURVES\ BY\ LABELS.png)

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
- Reflex (with Radix UI + Recharts)

### Packaging and Deployment
- Docker
- Hugging Face Spaces

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

## Run the App Locally

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
- Reflex app config is in `rxconfig.py` (frontend `3000`, Reflex backend `8001`).
- Reflex frontend calls prediction API at `http://localhost:8000`.

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
  -F "left_image=@assets/sample_cases/1_left.jpg" \
  -F "right_image=@assets/sample_cases/1_right.jpg" \
  -F "age=55"
```

## Hugging Face Spaces Deployment (Docker)

This repository is configured as a Docker Space.

### Runtime artifacts required in `artifacts/`

- `best.pt`
- `metadata_stats.json`
- `thresholds.json`

### Build behavior

The Docker image starts:
- FastAPI inference API on `:8000`
- Reflex frontend (single-port mode) on `${PORT}` (default `7860`)

Entrypoint: `start_hf_reflex.sh`

### Deploy steps

1. Create a Docker Space in Hugging Face.
2. Ensure Git LFS is enabled for large model files:

```bash
git lfs install
```

3. Push repository to your Space remote.
4. Spaces will build and serve on port `7860`.

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
