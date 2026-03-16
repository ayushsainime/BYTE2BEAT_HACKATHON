from __future__ import annotations

import os
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from api.main import app
from models.multimodal_model import MultimodalRiskModel
from utils.config import FreezePolicyConfig, ModelConfig
from utils.io import save_json
from tests.conftest import write_dummy_image


def test_api_smoke(tmp_path: Path) -> None:
    model_config = ModelConfig(
        backbone_name="efficientnet_b0",
        pretrained=False,
        num_labels=8,
        image_feature_dropout=0.2,
        metadata_hidden_dim=32,
        metadata_dropout=0.1,
        fusion_hidden_dims=[64, 32],
        fusion_dropout=0.2,
        freeze_policy=FreezePolicyConfig(enabled=True, freeze_encoder_epochs=1, unfreeze_last_n_stages=2, full_finetune_epoch=3),
    )
    model = MultimodalRiskModel(model_config)

    ckpt_path = tmp_path / "best.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    metadata_stats_path = tmp_path / "metadata_stats.json"
    thresholds_path = tmp_path / "thresholds.json"
    save_json(metadata_stats_path, {"age_mean": 60.0, "age_std": 10.0, "sex_mapping": {"Female": 0.0, "Male": 1.0}})
    save_json(thresholds_path, {k: 0.5 for k in ["N", "D", "G", "C", "A", "H", "M", "O"]})

    data_cfg = tmp_path / "data.yaml"
    model_cfg = tmp_path / "model.yaml"
    infer_cfg = tmp_path / "infer.yaml"
    cv_cfg = tmp_path / "cv.yaml"
    api_cfg = tmp_path / "api.yaml"

    data_cfg.write_text(
        """
labels: [N, D, G, C, A, H, M, O]
paths:
  data_root: .
  csv_path: full_df.csv
  images_dir: preprocessed_images
  processed_dir: data/processed
  splits_dir: data/splits
  metadata_stats_path: data/processed/metadata_stats.json
  thresholds_path: data/processed/thresholds.json
split:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  random_state: 42
  require_both_eyes: true
image:
  size: 64
  mean: [0.5, 0.5, 0.5]
  std: [0.2, 0.2, 0.2]
loader:
  num_workers: 0
  pin_memory: false
  persistent_workers: false
  train_batch_size: 2
  val_batch_size: 2
  test_batch_size: 2
metadata:
  age_min: 0.0
  age_max: 120.0
  sex_mapping:
    Female: 0.0
    Male: 1.0
""",
        encoding="utf-8",
    )
    model_cfg.write_text(
        """
backbone_name: efficientnet_b0
pretrained: false
num_labels: 8
image_feature_dropout: 0.2
metadata_hidden_dim: 32
metadata_dropout: 0.1
fusion_hidden_dims: [64, 32]
fusion_dropout: 0.2
freeze_policy:
  enabled: true
  freeze_encoder_epochs: 1
  unfreeze_last_n_stages: 2
  full_finetune_epoch: 3
""",
        encoding="utf-8",
    )
    infer_cfg.write_text(
        f"""
device: cpu
image_size: 64
global_threshold: 0.5
metadata_stats_path: {metadata_stats_path}
thresholds_path: {thresholds_path}
""",
        encoding="utf-8",
    )
    cv_cfg.write_text(
        """
weights:
  hypertension_proxy:
    H: 0.7
    D: 0.2
    A: 0.1
  diabetes_proxy:
    D: 0.8
    H: 0.2
  atherosclerotic_proxy:
    A: 0.6
    H: 0.2
    D: 0.2
overall_weights:
  hypertension_proxy: 0.4
  diabetes_proxy: 0.35
  atherosclerotic_proxy: 0.25
risk_bands:
  low_max: 0.33
  medium_max: 0.66
""",
        encoding="utf-8",
    )
    api_cfg.write_text(
        f"""
host: 0.0.0.0
port: 8000
reload: false
checkpoint_path: {ckpt_path}
model_config_path: {model_cfg}
data_config_path: {data_cfg}
inference_config_path: {infer_cfg}
cv_proxy_config_path: {cv_cfg}
""",
        encoding="utf-8",
    )

    os.environ["API_CONFIG_PATH"] = str(api_cfg)

    left = tmp_path / "left.jpg"
    right = tmp_path / "right.jpg"
    write_dummy_image(left)
    write_dummy_image(right)

    with TestClient(app) as client:
        with left.open("rb") as lfp, right.open("rb") as rfp:
            response = client.post(
                "/predict",
                data={"age": "64", "sex": "Female"},
                files={
                    "left_image": ("left.jpg", lfp, "image/jpeg"),
                    "right_image": ("right.jpg", rfp, "image/jpeg"),
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert "labels" in payload and "probabilities" in payload and "cv_summary" in payload
