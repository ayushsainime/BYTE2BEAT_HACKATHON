from __future__ import annotations

from pathlib import Path

import torch

from inference.predictor import Predictor
from models.multimodal_model import MultimodalRiskModel
from utils.config import (
    CVProxyConfig,
    DataConfig,
    DataPathsConfig,
    FreezePolicyConfig,
    ImageConfig,
    InferenceConfig,
    LoaderConfig,
    MetadataConfig,
    ModelConfig,
    SplitConfig,
)
from utils.io import save_json
from tests.conftest import write_dummy_image


def test_predictor_single_sample(tmp_path: Path) -> None:
    model_config = ModelConfig(
        backbone_name="efficientnet_b4",
        pretrained=False,
        num_labels=8,
        image_feature_dropout=0.2,
        metadata_hidden_dim=32,
        metadata_dropout=0.1,
        fusion_hidden_dims=[64, 32],
        fusion_dropout=0.2,
        freeze_policy=FreezePolicyConfig(enabled=True, freeze_encoder_epochs=5, unfreeze_last_n_stages=2, full_finetune_epoch=5),
    )

    data_config = DataConfig(
        labels=["N", "D", "G", "C", "A", "H", "M", "O"],
        paths=DataPathsConfig(
            data_root=tmp_path,
            csv_path=tmp_path / "full_df.csv",
            images_dir=tmp_path / "preprocessed_images",
            processed_dir=tmp_path / "data" / "processed",
            splits_dir=tmp_path / "data" / "splits",
            metadata_stats_path=tmp_path / "data" / "processed" / "metadata_stats.json",
            thresholds_path=tmp_path / "data" / "processed" / "thresholds.json",
        ),
        split=SplitConfig(0.7, 0.15, 0.15, 42, True),
        image=ImageConfig(size=64, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
        loader=LoaderConfig(0, False, False, 2, 2, 2),
        metadata=MetadataConfig(0.0, 120.0),
    )

    inference_config = InferenceConfig(
        device="cpu",
        image_size=64,
        global_threshold=0.5,
        metadata_stats_path=tmp_path / "metadata_stats.json",
        thresholds_path=tmp_path / "thresholds.json",
    )

    cv_proxy_config = CVProxyConfig(
        weights={
            "hypertension_proxy": {"H": 0.7, "D": 0.2, "A": 0.1},
            "diabetes_proxy": {"D": 0.8, "H": 0.2},
            "atherosclerotic_proxy": {"A": 0.6, "H": 0.2, "D": 0.2},
        },
        overall_weights={"hypertension_proxy": 0.4, "diabetes_proxy": 0.35, "atherosclerotic_proxy": 0.25},
        risk_bands={"low_max": 0.33, "medium_max": 0.66},
    )

    model = MultimodalRiskModel(model_config)
    ckpt_path = tmp_path / "best.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    save_json(inference_config.metadata_stats_path, {"age_mean": 60.0, "age_std": 10.0})
    save_json(inference_config.thresholds_path, {label: 0.5 for label in data_config.labels})

    left = tmp_path / "left.jpg"
    right = tmp_path / "right.jpg"
    write_dummy_image(left, 120)
    write_dummy_image(right, 130)

    predictor = Predictor(
        checkpoint_path=ckpt_path,
        model_config=model_config,
        data_config=data_config,
        inference_config=inference_config,
        cv_proxy_config=cv_proxy_config,
    )

    result = predictor.predict_single(left, right, age=65)
    assert set(result.labels.keys()) == set(data_config.labels)
    assert set(result.probabilities.keys()) == set(data_config.labels)
    assert "overall_cv_proxy" in result.cv_summary
