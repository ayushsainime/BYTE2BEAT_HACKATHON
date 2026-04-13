<div align="center">

# EYE HEART CONNECTION

**Multimodal retinal analysis for cardiovascular risk proxy estimation using bilateral fundus images and age metadata**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Reflex](https://img.shields.io/badge/Reflex-Frontend-111111?style=flat-square)](https://reflex.dev/)
[![License](https://img.shields.io/badge/License-MIT-2F855A?style=flat-square)](#license)

[![Launch App](https://img.shields.io/badge/Launch-Live%20App-0F766E?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/ayushsainime/eye_heart_connect_reflex_app)
[![Watch Demo](https://img.shields.io/badge/Watch-YouTube%20Demo-C1121F?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=Be5-Q3fMTNw)

<p align="center">
  EYE HEART CONNECTION is an end-to-end machine learning system that learns from left and right retinal fundus images together with patient age to predict ophthalmic findings and derive a cardiovascular risk proxy score. The repository includes data preparation, model training, evaluation, API serving, and a polished Reflex interface for interactive use.
</p>

<img src="https://huggingface.co/datasets/ayushsainime/eye_heart_connect_media/resolve/main/archi%20diagram.png" alt="System architecture" width="100%">

</div>

## Overview

The project is built around a multimodal PyTorch model that combines:

- a shared image encoder applied to both left and right fundus photographs
- a metadata branch for normalized age input
- a fusion head that predicts eight ophthalmic labels
- a post-processing layer that converts those probabilities into cardiovascular proxy scores and a final risk band

This codebase currently centers on:

- `FastAPI` for model serving
- `Reflex` for the user-facing web interface
- `PyTorch` and `torchvision` for model development
- `Albumentations`, `OpenCV`, and `Pillow` for image preprocessing
- `pytest` for test coverage across model, dataset, predictor, and API layers

## Key Capabilities

| Area | Current implementation |
| --- | --- |
| Inputs | Left fundus image, right fundus image, patient age |
| Model | Shared bilateral image encoder plus metadata fusion network |
| Vision backbone | `efficientnet_b4` by default, with `resnet50` also supported in code |
| Prediction target | 8-label multilabel ophthalmic prediction |
| Clinical summary | Weighted cardiovascular proxy scores and `low` / `medium` / `high` risk band |
| Serving | `FastAPI` prediction API with schema validation |
| Frontend | `Reflex` app with uploads, sample cases, charts, and clinical explanation text |
| Experiment management | YAML-driven configuration for data, model, train, eval, inference, and API settings |

## Prediction Targets

The model predicts the following label set:

| Code | Meaning |
| --- | --- |
| `N` | Normal |
| `D` | Diabetes |
| `G` | Glaucoma |
| `C` | Cataract |
| `A` | Age-related macular degeneration |
| `H` | Hypertension |
| `M` | Myopia |
| `O` | Other findings |

The API and frontend then aggregate those probabilities into:

- `hypertension_proxy`
- `diabetes_proxy`
- `atherosclerotic_proxy`
- `overall_cv_proxy`
- `risk_band`

## System Architecture

### Model path

1. Left and right fundus images are preprocessed and resized.
2. A shared image encoder extracts bilateral retinal features.
3. Age is normalized using dataset statistics and passed through a metadata MLP.
4. Image and metadata embeddings are concatenated.
5. A fusion head produces eight multilabel logits.
6. Sigmoid probabilities are transformed into cardiovascular proxy scores using `configs/cv_proxy.yaml`.

### Application path

1. `api/main.py` loads the trained checkpoint on startup.
2. `POST /predict` accepts bilateral images and age via multipart form data.
3. `reflex_app/reflex_app.py` provides an interactive interface for uploads, sample cases, progress states, and result visualization.

## Repository Layout

```text
EYE_HEART_CONNECTION/
|-- api/                  FastAPI app, request/response schemas, static API landing page
|-- assets/               Frontend assets, sample images, and sample cases
|-- configs/              YAML configuration for data, model, training, evaluation, inference, and API
|-- datasets/             Patient dataframe building, dataset classes, transforms, and loaders
|-- evaluation/           Metric computation and evaluation runner
|-- experiments/          Training runs, checkpoints, metrics, reports, and latest artifact pointer
|-- inference/            Predictor class and CLI inference entrypoint
|-- models/               Image encoder and multimodal fusion model
|-- reflex_app/           Reflex application UI
|-- tests/                Pytest-based tests for API, datasets, model, and inference
|-- training/             Training entrypoint and trainer implementation
|-- utils/                Config loading, logging, device resolution, IO, constants, seeding
|-- pyproject.toml        Package metadata and dependencies
`-- rxconfig.py           Reflex app configuration
```

## Technology Stack

| Layer | Tools |
| --- | --- |
| Language | Python |
| Deep learning | PyTorch, torchvision |
| Image pipeline | Albumentations, OpenCV, Pillow |
| Data and metrics | NumPy, Pandas, scikit-learn, Matplotlib |
| API | FastAPI, Uvicorn, Pydantic, python-multipart |
| Frontend | Reflex, HTTPX, Radix-based UI primitives, Recharts |
| Experiment logging | TensorBoard, CSV metrics |
| Testing | pytest |

## Quick Start

> **TL;DR** — Clone, install, and run both the API and frontend locally in under 5 minutes.

```bash
# 1. Clone the repo
git clone https://github.com/ayushsainime/EYE_HEART_CONNECTION.git
cd EYE_HEART_CONNECTION

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .\.venv\Scripts\Activate.ps1    # Windows PowerShell

# 3. Install dependencies
pip install --upgrade pip
pip install -e ".[frontend,dev]"

# 4. Start the API (terminal 1)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 5. Start the Reflex frontend (terminal 2)
export EHC_API_BASE="http://localhost:8000"   # Linux / macOS
# $env:EHC_API_BASE="http://localhost:8000"   # Windows PowerShell
reflex run
```

Once both are running:

| Service | URL |
| --- | --- |
| FastAPI predictions | `http://localhost:8000` |
| Interactive API docs | `http://localhost:8000/docs` |
| Reflex web app | `http://localhost:3000` |

The pre-trained checkpoint and metadata shipped in `artifacts/` are loaded automatically by default, so you can start predicting right away.

---

## Prerequisites

- **Python** 3.10 or newer
- **Git** for cloning the repository
- **Docker** (optional, for containerised deployment)
- A **GPU** is recommended for training but not required for inference (`device: auto` falls back to CPU)

---

## Option 1 — Local Installation

### Step 1: Clone and set up the environment

```bash
git clone https://github.com/ayushsainime/EYE_HEART_CONNECTION.git
cd EYE_HEART_CONNECTION

python -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1    # Windows PowerShell
```

```bash
source .venv/bin/activate       # Linux / macOS
```

### Step 2: Install the package

```bash
pip install --upgrade pip
pip install -e ".[frontend,dev]"
```

| Extra | What it installs |
| --- | --- |
| `frontend` | Reflex (web UI framework) |
| `dev` | pytest (test runner) |

### Step 3: Start the API server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API loads configuration from `configs/api.yaml` which points to:

- checkpoint → `experiments/latest/best.pt` (or `artifacts/best.pt` for Docker)
- model config → `configs/model.yaml`
- data config → `configs/data.yaml`
- inference config → `configs/inference.yaml`
- CV proxy config → `configs/cv_proxy.yaml`

### Step 4: Start the Reflex frontend

Open a **second terminal**, activate the same virtual environment, and run:

```bash
export EHC_API_BASE="http://localhost:8000"   # Linux / macOS
# $env:EHC_API_BASE="http://localhost:8000"   # Windows PowerShell
reflex run
```

Open `http://localhost:3000` in your browser to use the interactive UI.

### Step 5: Run a prediction from the CLI

You can also test a single prediction without the UI:

```bash
python -m inference.predict \
  --ckpt artifacts/best.pt \
  --left assets/sample_left.jpg \
  --right assets/sample_right.jpg \
  --age 55
```

---

## Option 2 — Docker Deployment

The included `Dockerfile` bundles the FastAPI backend and Reflex frontend into a single container that exposes port **7860**.

### Build the image

```bash
docker build -t eye-heart-connection .
```

### Run the container

```bash
docker run -p 7860:7860 eye-heart-connection
```

Once the container is up:

| What | URL |
| --- | --- |
| Reflex web app | `http://localhost:7860` |
| FastAPI predictions (internal) | `http://localhost:7860` (proxied through Reflex single-port mode) |
| API docs | `http://localhost:7860/docs` |

The startup script (`start_railway_reflex.sh`) automatically launches both the FastAPI backend on port 8000 and the Reflex frontend on port 7860 inside the container.

### Docker Compose (optional)

If you prefer Docker Compose, create a `docker-compose.yml`:

```yaml
version: "3.9"
services:
  app:
    build: .
    ports:
      - "7860:7860"
    environment:
      - PORT=7860
      - API_CONFIG_PATH=configs/api_railway.yaml
    restart: unless-stopped
```

Then run:

```bash
docker compose up --build
```

---

## Option 3 — Hugging Face Spaces

The project ships with dedicated Space configs (`configs/api_space.yaml` and `configs/inference_space.yaml`).

1. Create a new **Docker Space** on [huggingface.co/spaces](https://huggingface.co/spaces).
2. Push this repository as the Space's source code.
3. Set the environment variable `API_CONFIG_PATH=configs/api_space.yaml` if not using the default Dockerfile (which uses `api_railway.yaml`).
4. The Space will build and serve the app on port **7860**.

---

## Evaluation and Inference

### Evaluate a checkpoint

```bash
python -m evaluation.run \
  --ckpt experiments/latest/best.pt \
  --config configs/eval.yaml \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml
```

Evaluation can optionally tune thresholds and save:

- predictions CSV
- key metrics JSON and CSV
- full evaluation report JSON
- tuned thresholds JSON

### Run single-sample inference from the CLI

```bash
python -m inference.predict \
  --ckpt artifacts/best.pt \
  --left assets/sample_left.jpg \
  --right assets/sample_right.jpg \
  --age 55
```

---

## API Reference

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Serves the static API landing page |
| `GET` | `/health` | Basic health check |
| `POST` | `/predict` | Run multimodal inference on bilateral fundus images and age |
| `GET` | `/docs` | Interactive FastAPI schema docs |

### Request format

`POST /predict` expects multipart form fields:

- `left_image` — left eye fundus image file
- `right_image` — right eye fundus image file
- `age` — patient age (integer)

### Example request

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "left_image=@assets/sample_left.jpg" \
  -F "right_image=@assets/sample_right.jpg" \
  -F "age=55"
```

### Example response

```json
{
  "labels": {
    "N": 0, "D": 1, "G": 0, "C": 0,
    "A": 0, "H": 1, "M": 0, "O": 0
  },
  "probabilities": {
    "N": 0.14, "D": 0.81, "G": 0.08, "C": 0.12,
    "A": 0.19, "H": 0.76, "M": 0.10, "O": 0.07
  },
  "cv_summary": {
    "hypertension_proxy": 0.67,
    "diabetes_proxy": 0.80,
    "atherosclerotic_proxy": 0.38,
    "overall_cv_proxy": 0.64,
    "risk_band": "medium"
  }
}
```

---

## Configuration Reference

| File | Purpose |
| --- | --- |
| `configs/data.yaml` | Dataset paths, image normalization, splits, loader settings, metadata bounds |
| `configs/model.yaml` | Backbone choice, dropout settings, fusion head dimensions, freeze policy |
| `configs/train.yaml` | Device strategy, epochs, optimizer, scheduler, checkpoint monitor |
| `configs/eval.yaml` | Evaluation split, thresholds, prediction export settings |
| `configs/inference.yaml` | Runtime device, image size, threshold path, metadata stats path (local) |
| `configs/inference_railway.yaml` | Inference config for Railway / Docker deployment |
| `configs/inference_space.yaml` | Inference config for Hugging Face Spaces |
| `configs/api.yaml` | API host, port, checkpoint, and config wiring (local) |
| `configs/api_railway.yaml` | API config for Railway / Docker (uses `artifacts/best.pt`) |
| `configs/api_space.yaml` | API config for Hugging Face Spaces (port 7860) |
| `configs/cv_proxy.yaml` | Proxy score weights and risk-band thresholds |

---

## Frontend Features

The Reflex application includes:

- bilateral image upload panels
- sample-case loading from `assets/sample_cases/`
- age slider and numeric input
- progress and error states
- probability visualization with bar charts
- risk-band and cardiovascular proxy summaries
- explanatory text derived from the top predicted ophthalmic conditions

---

## Testing

Run the test suite with:

```bash
python -m pytest -q
```

The current tests cover:

- API smoke behavior
- predictor outputs
- dataset construction and data preparation
- model instantiation and forward pass expectations

---

## License

This project is distributed under the MIT license as declared in `pyproject.toml`.
