# Eye-Heart Connection

Multimodal deep learning project for retinal fundus + metadata risk modeling.

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

Generic training entrypoint (choose any model config):

```bash
python -m training.train --config configs/train.yaml --data-config configs/data.yaml --model-config configs/model.yaml
```

VGG19 and ResNet50 convenience scripts:

```bash
python -m training.train_vgg19
python -m training.train_resnet50
```

## Evaluation

```bash
python -m evaluation.run --config configs/eval.yaml --data-config configs/data.yaml --model-config configs/model.yaml --ckpt experiments/latest/best.pt
```

Evaluation outputs include:
- accuracy
- precision
- recall
- f1
- AUROC / PR-AUC
- per-label report

Files:
- `data/processed/eval_predictions.csv`
- `data/processed/thresholds.json`
- `data/processed/eval_report.json`

## Inference

```bash
python -m inference.predict --ckpt experiments/latest/best.pt --left preprocessed_images/0_left.jpg --right preprocessed_images/0_right.jpg --age 69 --sex Female
```

## API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

`POST /predict` (multipart/form-data):
- `left_image`
- `right_image`
- `age`
- `sex`
