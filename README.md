---
title: Eye Heart Connection API
sdk: docker
app_port: 7860
---

# Eye-Heart Connection

FastAPI deployment package for multimodal retinal cardiovascular risk prediction.

## What This Space Serves

- `GET /`: lightweight upload UI (left image, right image, age)
- `GET /health`: health check
- `POST /predict`: returns per-label probabilities, binary labels, and CV proxy summary

## Runtime Artifacts (required)

Keep these files in `artifacts/`:

- `artifacts/best.pt`
- `artifacts/metadata_stats.json`
- `artifacts/thresholds.json`

## Main Model Setup

- Backbone: `EfficientNet-B4`
- Inputs: Left image + Right image + Age
- Output: 8-label multi-label prediction (`N,D,G,C,A,H,M,O`)

## Hugging Face Spaces (Docker)

This repository is configured for Docker Spaces.

- Docker entrypoint: `python -m api.gradio_app`
- API config used in container: `configs/api_space.yaml`

## Local Run

```bash
python -m pip install -e .
$env:API_CONFIG_PATH='configs/api_space.yaml'
uvicorn api.main:app --host 0.0.0.0 --port 7860
```

Open: `http://127.0.0.1:7860`

## Notes

- Large checkpoint files should be tracked with Git LFS (`.gitattributes` included).
- Training, evaluation, and dataset-heavy files are excluded from Docker upload for lean deployment.

## Gradio Frontend (Recommended Demo UI)

Run the visually rich Gradio frontend:

```bash
python -m api.gradio_app
```

Then open: `http://127.0.0.1:7860`

The Gradio page includes:
- Sidebar with project summary, tech stack, and library list
- Left/right fundus uploads + age input
- Probability plot and detailed prediction table
- CV summary output

