<div align="center">

# 🔬 EYE ❤️ HEART CONNECTION

### *Multimodal AI for Cardiovascular Disease Prediction from Retinal Fundus Photography*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[![Deploy](https://img.shields.io/badge/🚀_Railway-Live-0B0D0E?style=flat-square)](https://railway.app)
[![Hugging Face](https://img.shields.io/badge/🤗_HF_Spaces-Deployed-FFD21E?style=flat-square)](https://huggingface.co)

<p align="center">
  <em>A production-grade, end-to-end deep learning system that fuses retinal fundus imagery with clinical tabular data to predict cardiovascular disease — demonstrating the remarkable connection between the eye and the heart.</em>
</p>

---

</div>

## 📌 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🧬 The Science](#-the-science)
- [🏗️ System Architecture](#️-system-architecture)
- [🤖 Model Architecture](#-model-architecture)
- [🚀 Key Features](#-key-features)
- [📂 Project Structure](#-project-structure)
- [⚙️ Installation & Setup](#️-installation--setup)
- [🎮 Quick Start](#-quick-start)
- [📊 Training Pipeline](#-training-pipeline)
- [🔍 Inference & Evaluation](#-inference--evaluation)
- [🌐 API & Deployment](#-api--deployment)
- [🧪 Testing](#-testing)
- [🔧 Tech Stack](#-tech-stack)
- [📈 Results & Performance](#-results--performance)
- [👤 Author](#-author)

---

## 🎯 Project Overview

**EYE ❤️ HEART CONNECTION** is a full-stack, production-ready AI system that leverages the clinically established correlation between retinal microvascular abnormalities and cardiovascular disease. The system ingests **retinal fundus photographs** alongside **patient clinical data** (demographics, blood pressure, lab values) through a multimodal deep learning architecture to predict cardiovascular risk with high accuracy.

> **Why this matters:** The retina is the only location in the human body where microvasculature can be directly observed non-invasively. Changes in retinal blood vessels have been shown to correlate with cardiovascular conditions including hypertension, coronary artery disease, and stroke risk. This project transforms that clinical insight into an automated, scalable AI prediction system.

### What Makes This Project Stand Out

| Aspect | Detail |
|---|---|
| **Multimodal Fusion** | Deep learning architecture that intelligently fuses vision (fundus images) and tabular (clinical) data |
| **Production Pipeline** | End-to-end MLOps — from data ingestion to deployed API with Docker containerization |
| **Config-Driven** | Hydra/YAML-based configuration system for reproducible experiments across environments |
| **Multi-Interface** | REST API (FastAPI), Gradio UI, and Reflex web app — serving diverse user needs |
| **Cloud Deployed** | Live deployment on Railway and Hugging Face Spaces |
| **Test Coverage** | Comprehensive unit and integration test suite |

---

## 🧬 The Science

### The Eye-Heart Connection

Retinal fundus photography reveals critical biomarkers for cardiovascular health:

- 🩸 **Vessel caliber changes** — Associated with hypertension and atherosclerosis
- 🔴 **Hemorrhages & microaneurysms** — Indicators of diabetic cardiovascular risk  
- 🌀 **Arteriovenous nicking** — Linked to chronic hypertensive damage
- 📐 **Vessel tortuosity** — Correlated with cardiovascular mortality

This system learns these patterns automatically through deep learning, augmented by structured clinical data, to produce robust cardiovascular risk predictions.

---

## 🏗️ System Architecture

<div align="center">
  <img src="mermaid-diagram-2026-03-30-191850.png" alt="System Architecture Diagram" width="100%">
</div>

---

## 🤖 Model Architecture

The core of the system is a **multimodal fusion network** that combines two parallel encoding streams:

### Image Encoder
- **Backbone options:** ResNet-18/34/50, EfficientNet-B0/B1/B2, ConvNeXt-Tiny, and more
- **Pretrained** on ImageNet with fine-tuning support (freeze/unfreeze backbone)
- **Global average pooling** → feature projection → embedding vector

### Tabular Encoder
- **Multi-layer perceptron** (MLP) with batch normalization and dropout
- Processes clinical features: demographics, vitals, lab values
- **Configurable** layer sizes and activation functions

### Fusion & Classification Head
- **Concatenation** of image and tabular embeddings
- **Fully connected classifier** with configurable hidden dimensions
- **Multi-class classification** output for cardiovascular disease prediction
- Supports **weighted loss** for class imbalance handling

```
Fundus Image ──→ [Image Encoder] ──→ Image Embedding (256-d) ──┐
                                                                ├──→ [Fusion MLP] ──→ CVD Prediction
Clinical Data ──→ [Tabular Encoder] ──→ Tabular Embedding (64-d) ┘
```

---

## 🚀 Key Features

### 🧠 Deep Learning & ML
- ✅ **Multimodal fusion architecture** (image + tabular data)
- ✅ **Multiple CNN backbones** (ResNet, EfficientNet, ConvNeXt families)
- ✅ **Transfer learning** with ImageNet pretrained weights
- ✅ **Advanced augmentations** (RandAugment, TrivialAugment, geometric/photometric transforms)
- ✅ **Class imbalance handling** (weighted cross-entropy loss)
- ✅ **Learning rate scheduling** (CosineAnnealing, ReduceLROnPlateau, StepLR)
- ✅ **Early stopping** with patience and checkpoint restoration
- ✅ **Mixed precision training** (AMP/GradScaler) for efficiency
- ✅ **K-Fold cross-validation** support for robust evaluation
- ✅ **Data quality validation** pipeline

### 🏭 Engineering & MLOps
- ✅ **Hydra configuration management** — composable, multi-environment configs
- ✅ **Docker containerization** for reproducible deployments
- ✅ **FastAPI REST API** with request/response schemas (Pydantic)
- ✅ **Gradio web interface** for interactive demos
- ✅ **Reflex full-stack web app** for production UI
- ✅ **Railway cloud deployment** with automated startup scripts
- ✅ **Hugging Face Spaces deployment** support
- ✅ **Comprehensive test suite** (pytest)
- ✅ **Structured logging** and seed management for reproducibility

---

## 📂 Project Structure

```
EYE_HEART_CONNECTION/
├── 📁 api/                          # REST API & Web Interfaces
│   ├── main.py                      # FastAPI application entry point
│   ├── gradio_app.py                # Gradio interactive demo
│   └── schemas.py                   # Pydantic request/response models
├── 📁 configs/                      # Hydra Configuration Files
│   ├── model.yaml                   # Model architecture config
│   ├── data.yaml                    # Dataset & preprocessing config
│   ├── train.yaml                   # Training hyperparameters
│   ├── eval.yaml                    # Evaluation settings
│   ├── inference.yaml               # Inference pipeline config
│   ├── api.yaml                     # API server config
│   ├── *_railway.yaml               # Railway deployment configs
│   └── *_space.yaml                 # HF Spaces deployment configs
├── 📁 datasets/                     # Data Processing & Loading
│   ├── fundus_multimodal_dataset.py # Core PyTorch Dataset
│   ├── data_module.py               # Lightning-style data module
│   ├── build_patient_df.py          # Patient record builder
│   ├── data_quality.py              # Data validation & cleaning
│   └── transforms.py                # Image augmentation pipeline
├── 📁 models/                       # Neural Network Architectures
│   ├── image_encoder.py             # CNN backbone encoder
│   └── multimodal_model.py          # Multimodal fusion model
├── 📁 training/                     # Training Infrastructure
│   ├── train.py                     # Training entry point
│   └── trainer.py                   # Trainer class with full loop
├── 📁 evaluation/                   # Model Evaluation
│   ├── metrics.py                   # Metric computation (AUC, F1, etc.)
│   └── run.py                       # Evaluation runner
├── 📁 inference/                    # Prediction Pipeline
│   ├── predict.py                   # Batch prediction script
│   ├── predictor.py                 # Predictor class
│   └── tabular/                     # Tabular-specific inference
├── 📁 reflex_app/                   # Reflex Web Application
│   └── reflex_app.py                # Production UI
├── 📁 tests/                        # Test Suite
│   ├── test_model.py                # Model architecture tests
│   ├── test_predictor.py            # Inference pipeline tests
│   ├── test_api.py                  # API endpoint tests
│   ├── test_dataset.py              # Dataset & data tests
│   ├── test_data_prep.py            # Data preparation tests
│   └── conftest.py                  # Shared test fixtures
├── 📁 utils/                        # Shared Utilities
│   ├── config.py                    # Configuration loading
│   ├── constants.py                 # Project-wide constants
│   ├── device.py                    # Device management (CPU/GPU/MPS)
│   ├── io.py                        # I/O helpers
│   ├── logging.py                   # Structured logging
│   └── seed.py                      # Reproducibility seed control
├── 📁 notebooks/                    # Jupyter Notebooks
│   ├── hackathon_showcase.ipynb     # Demo showcase notebook
│   └── kaggle_train_eval.ipynb      # Kaggle training notebook
├── 📁 data/                         # Data Storage
│   └── splits/                      # Train/val/test splits
├── Dockerfile                       # Container definition
├── railway.json                     # Railway deployment config
├── pyproject.toml                   # Project metadata & dependencies
└── rxconfig.py                      # Reflex framework config
```

---

## ⚙️ Installation & Setup

### Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** (recommended) or CPU
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/ayushsainime/EYE_HEART_CONNECTION.git
cd EYE_HEART_CONNECTION
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -e .
```

<details>
<summary>📦 Core Dependencies</summary>

| Package | Purpose |
|---|---|
| `torch` | Deep learning framework |
| `torchvision` | Pretrained models & vision utilities |
| `timm` | Additional model backbones |
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `gradio` | Interactive ML demo UI |
| `reflex` | Full-stack Python web framework |
| `hydra-core` | Configuration management |
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` | Metrics & preprocessing |
| `albumentations` | Image augmentation |
| `Pillow` | Image processing |
| `pydantic` | Data validation & schemas |
| `pytest` | Testing framework |

</details>

### 4. Docker Setup (Alternative)

```bash
# Build the container
docker build -t eye-heart-connection .

# Run the container
docker run -p 8000:8000 eye-heart-connection
```

---

## 🎮 Quick Start

### Launch the API Server

```bash
# Start FastAPI server
python -m api.main
```

The API will be available at `http://localhost:8000` with interactive docs at `/docs`.

### Launch the Gradio Demo

```bash
# Start Gradio web interface
python -m api.gradio_app
```

### Run a Prediction

```python
from inference.predictor import Predictor

predictor = Predictor(config_path="configs/inference.yaml")

result = predictor.predict(
    fundus_image_path="path/to/fundus_image.png",
    clinical_data={
        "age": 55,
        "sex": 1,
        "blood_pressure_systolic": 140,
        "blood_pressure_diastolic": 90,
        # ... additional clinical features
    }
)

print(f"Cardiovascular risk prediction: {result}")
```

---

## 📊 Training Pipeline

### Start Training

```bash
# Train with default configuration
python -m training.train

# Train with custom config override
python -m training.train model.backbone=resnet50 training.lr=1e-4 training.epochs=50
```

### Training Features

| Feature | Description |
|---|---|
| **Backbone Selection** | Swap CNN architectures via config (`resnet18`, `efficientnet_b0`, `convnext_tiny`, etc.) |
| **Mixed Precision** | Automatic AMP for faster training with lower memory |
| **LR Scheduling** | Cosine annealing, plateau-based, or step-wise decay |
| **Early Stopping** | Monitors validation loss with configurable patience |
| **Checkpointing** | Saves best model weights automatically |
| **Logging** | Structured logging with epoch-level metrics |
| **Reproducibility** | Seeded RNG across Python, NumPy, and PyTorch |

### Training Configuration Example

```yaml
# configs/train.yaml
model:
  backbone: resnet50
  pretrained: true
  freeze_backbone: false
  tabular_input_dim: 15
  fusion_hidden_dims: [128, 64]
  num_classes: 5

training:
  epochs: 100
  lr: 3.0e-4
  weight_decay: 1.0e-4
  batch_size: 32
  scheduler: cosine
  early_stopping_patience: 10
```

---

## 🔍 Inference & Evaluation

### Evaluate a Trained Model

```bash
# Run evaluation with metrics
python -m evaluation.run
```

### Batch Prediction

```bash
# Run inference on a dataset
python -m inference.predict
```

### Supported Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Overall classification accuracy |
| **AUC-ROC** | Area Under the Receiver Operating Characteristic Curve |
| **F1 Score** | Harmonic mean of precision and recall |
| **Precision / Recall** | Per-class and weighted averages |
| **Confusion Matrix** | Full classification breakdown |
| **Classification Report** | Comprehensive per-class metrics |

---

## 🌐 API & Deployment

### FastAPI Endpoints

The REST API exposes the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Single prediction with image + clinical data |
| `POST` | `/predict/batch` | Batch prediction on multiple samples |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/docs` | Interactive Swagger UI documentation |

### Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "fundus_image=@retinal_photo.png" \
  -F "clinical_data={\"age\": 55, \"bp_systolic\": 140}"
```

### Deployment Platforms

| Platform | Status | Config |
|---|---|---|
| **Railway** | ✅ Live | `railway.json`, `start_railway_reflex.sh` |
| **Hugging Face Spaces** | ✅ Ready | `configs/api_space.yaml` |
| **Docker** | ✅ Ready | `Dockerfile` |

---

## 🧪 Testing

The project includes a comprehensive test suite built with **pytest**:

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_model.py -v       # Model architecture tests
pytest tests/test_predictor.py -v   # Inference pipeline tests
pytest tests/test_api.py -v         # API endpoint tests
pytest tests/test_dataset.py -v     # Dataset & data loading tests
pytest tests/test_data_prep.py -v   # Data preparation tests
```

### Test Coverage

| Module | Tests |
|---|---|
| `models/` | Architecture instantiation, forward pass, output shapes |
| `inference/` | Predictor initialization, prediction pipeline |
| `api/` | Endpoint routing, request validation, response format |
| `datasets/` | Dataset creation, transform application, data loading |
| `data_prep/` | DataFrame building, data quality checks |

---

## 🔧 Tech Stack

<p align="center">

| Category | Technologies |
|---|---|
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) |
| **Deep Learning** | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![torchvision](https://img.shields.io/badge/torchvision-EE4C2C?style=flat-square) ![timm](https://img.shields.io/badge/timm-EE4C2C?style=flat-square) |
| **API** | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) ![Uvicorn](https://img.shields.io/badge/Uvicorn-009688?style=flat-square) |
| **UI** | ![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=flat-square) ![Reflex](https://img.shields.io/badge/Reflex-20232A?style=flat-square) |
| **Config** | ![Hydra](https://img.shields.io/badge/Hydra-5B5EA6?style=flat-square) ![YAML](https://img.shields.io/badge/YAML-CB171E?style=flat-square) |
| **Data** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) |
| **Deployment** | ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white) ![Railway](https://img.shields.io/badge/Railway-0B0D0E?style=flat-square) ![HuggingFace](https://img.shields.io/badge/🤗_HF-FFD21E?style=flat-square) |
| **Testing** | ![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white) |

</p>

---

## 📈 Results & Performance

The multimodal fusion approach demonstrates the power of combining retinal imaging with clinical data:

- 🎯 **Multimodal > Unimodal** — Fusion of image + tabular data outperforms either modality alone
- 🏥 **Clinically Relevant** — Predictions align with established cardiovascular risk markers
- ⚡ **Real-Time Inference** — Sub-second prediction latency via optimized API pipeline
- 🔄 **Reproducible** — Seeded experiments with config-driven hyperparameter management

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

## 👤 Author

### **Ayush Saini**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ayush-saini-30a4a0372/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ayushsainime)

*Building AI at the intersection of computer vision and healthcare.*

---

⭐ **If you found this project insightful, please consider giving it a star!** ⭐

</div>