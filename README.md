# Eye-Heart Connection

Simplified multimodal pipeline for retinal risk modeling.

## Default Training Setup

- Backbone: `EfficientNet-B4`
- Inputs: Left image + Right image + Age
- Loss: Weighted `BCEWithLogitsLoss`
- Optimizer: `AdamW`
- Scheduler: `Cosine Annealing`
- Freeze: first 5 epochs
- Fine-tune: unfreeze and continue training

## Setup

```bash
python -m pip install -e .[dev]
```

## Data Preparation

```bash
python -m datasets.build_patient_df --config configs/data.yaml
python -m datasets.data_quality --data-config configs/data.yaml
```

## Training

```bash
python -m training.train --config configs/train.yaml --data-config configs/data.yaml --model-config configs/model.yaml
```

## Evaluation

```bash
python -m evaluation.run --config configs/eval.yaml --data-config configs/data.yaml --model-config configs/model.yaml --ckpt experiments/latest/best.pt
```

Metrics reported include:
- accuracy
- precision
- recall
- f1
- AUROC / PR-AUC
- per-label metrics

## Inference

```bash
python -m inference.predict --ckpt experiments/latest/best.pt --left preprocessed_images/0_left.jpg --right preprocessed_images/0_right.jpg --age 69
```

## API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

`POST /predict` form fields:
- `left_image`
- `right_image`
- `age`
