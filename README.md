
# Eye-Heart Connection

A multimodal deep learning system that predicts retinal diagnostic indicators from **left + right fundus images + patient age**, then derives an interpretable **cardiovascular risk proxy summary**.

This project is built for both:
- rigorous ML experimentation, and
- production-style serving (Docker + FastAPI + Gradio UI)

<img width="1892" height="911" alt="Screenshot 2026-03-18 224531" src="https://github.com/user-attachments/assets/cf9f5fc3-f092-4627-83b5-68a1750c5e0b" />

<img width="1691" height="885" alt="Screenshot 2026-03-18 224541" src="https://github.com/user-attachments/assets/12583bee-874d-44fb-b2c3-b3df575862c4" />


## Why This Project Matters

Retinal fundus images carry vascular and micro-structural patterns associated with systemic disease. This project explores how a bilateral-eye + metadata model can support early screening signals related to cardiovascular risk factors.

The goal is not to replace clinicians, but to build a practical and explainable ML pipeline that is:
- reproducible,
- deployable,
- and demo-ready for real-world health-tech workflows.

## What It Predicts

The model performs **8-label multi-label classification**:
- `N` Normal
- `D` Diabetes
- `G` Glaucoma
- `C` Cataract
- `A` AMD
- `H` Hypertension
- `M` Myopia
- `O` Other

From these outputs, it computes a **CV proxy summary** with:
- hypertension proxy,
- diabetes proxy,
- atherosclerotic proxy,
- overall risk band (`low` / `medium` / `high`).

## Model Highlights

- **Backbone**: EfficientNet-B4 (pretrained)
- **Inputs**: left eye image + right eye image + age
- **Metadata branch**: lightweight MLP
- **Fusion**: concatenation of image features + metadata features
- **Loss**: weighted BCEWithLogitsLoss
- **Optimizer**: AdamW
- **Scheduler**: Cosine Annealing
- **Transfer learning schedule**: freeze encoder for first 5 epochs, then fine-tune

## System Architecture

1. Left and right fundus images are independently encoded by a shared CNN encoder.
2. Age is normalized and passed through a small metadata MLP.
3. Features are fused in a dense head.
4. Output logits are converted to probabilities for 8 labels.
5. Thresholding produces final binary predictions.
6. A weighted post-processor derives CV proxy scores and risk band.

## Test Results (Current Checkpoint)

From `data/processed/eval_key_metrics.csv` and `data/processed/eval_report.json`:

### Global Threshold (`0.5`)
- Accuracy: **0.4474**
- Precision (macro): **0.5905**
- Recall (macro): **0.6462**
- F1 (macro): **0.6105**
- Macro AUROC: **0.8760**
- Macro PR-AUC: **0.6525**
- Label Accuracy: **0.8810**

### Tuned Thresholds (validation-driven)
- Accuracy: **0.3531**
- Precision (macro): **0.6225**
- Recall (macro): **0.6958**
- F1 (macro): **0.6410**
- Macro AUROC: **0.8760**
- Macro PR-AUC: **0.6525**
- Label Accuracy: **0.8734**

### Best Epoch Snapshot
- Best epoch: **12**
- Validation macro F1: **0.6668**
- Validation macro AUROC: **0.8931**

## Tech Stack

- **Modeling**: PyTorch, Torchvision, Albumentations
- **Data/Analysis**: Pandas, NumPy, scikit-learn, Matplotlib
- **Serving**: FastAPI, Uvicorn
- **Frontend**: Gradio
- **Packaging/Deploy**: Docker, Hugging Face Spaces

## Repository Structure

```text
api/                # FastAPI app + Gradio app + static UI assets
artifacts/          # Runtime model artifacts (best.pt, thresholds, metadata stats)
configs/            # Train/model/data/inference/api configs
datasets/           # Data prep and dataset pipeline code
evaluation/         # Metrics and evaluation scripts
inference/          # Predictor and CLI inference logic
models/             # Encoders and multimodal model definitions
training/           # Training loop and orchestration
utils/              # Shared configs, IO, logging, constants
notebooks/          # Showcase and experiment notebooks
```

## Quick Start (Local)

### 1) Create environment and install

```bash
python -m pip install -e .
```

### 2) Run Gradio UI (recommended demo)

```bash
python -m api.gradio_app
```

Open: `http://127.0.0.1:7860`

### 3) Run FastAPI directly (optional)

```bash
$env:API_CONFIG_PATH='configs/api_space.yaml'
uvicorn api.main:app --host 0.0.0.0 --port 7860
```

### 4) API endpoint

- `GET /health`
- `POST /predict` with multipart fields:
  - `left_image`
  - `right_image`
  - `age`

Example:

```bash
curl -X POST "http://127.0.0.1:7860/predict" \
  -F "left_image=@preprocessed_images/0_left.jpg" \
  -F "right_image=@preprocessed_images/0_right.jpg" \
  -F "age=69"
```

## Hugging Face Spaces Deployment (Docker)

This repo is configured as a Docker Space.

### Required runtime artifacts

Keep these files in `artifacts/`:
- `best.pt`
- `metadata_stats.json`
- `thresholds.json`

### Deploy steps

1. Create a Docker Space on Hugging Face.
2. Ensure Git LFS is enabled (for large checkpoint):

```bash
git lfs install
```

3. Commit and push to your Space remote.
4. Space builds the Docker image and serves app on port `7860`.

## Training and Evaluation (Reproducibility)

If you want to retrain:

```bash
python -m training.train --config configs/train.yaml --data-config configs/data.yaml --model-config configs/model.yaml
```

Evaluate a checkpoint:

```bash
python -m evaluation.run --config configs/eval.yaml --data-config configs/data.yaml --model-config configs/model.yaml --ckpt experiments/latest/best.pt
```

## Notebooks

- `notebooks/kaggle_train_eval.ipynb`: training/eval workflow for GPU environment
- `notebooks/hackathon_showcase.ipynb`: presentation-style visual walkthrough

## Artifacts and Experiment Tracking

The `experiments/` folder stores run-specific outputs such as:
- checkpoints (`best.pt`, `latest.pt`)
- epoch metrics (`metrics.csv`)
- post-train reports
- tensorboard logs

It is critical for model development and comparison, but not required in the final deployment image.

## Recruiter Notes

What this project demonstrates:
- end-to-end multimodal deep learning system design
- practical handling of class imbalance and threshold tuning
- clean modular Python architecture and config-driven experiments
- model-to-product path (inference layer + API + UI + Docker deployment)
- clear metric reporting for training, validation, and test phases

## Limitations

- Current outputs are predictive signals, not clinical diagnoses.
- Dataset composition and label quality directly affect generalization.
- External validation across demographic and device variations is needed.

## Roadmap

- better calibrated uncertainty reporting
- lightweight model distillation for faster inference
- expanded metadata features beyond age
- stronger explainability overlays (e.g., saliency/attention)
- prospective validation and clinician feedback loop

## Disclaimer

This repository is for research and educational use. It is **not** a medical device and must not be used for clinical decision-making without appropriate regulatory and clinical validation.
