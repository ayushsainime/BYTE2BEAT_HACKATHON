from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataPathsConfig:
    data_root: Path
    csv_path: Path
    images_dir: Path
    processed_dir: Path
    splits_dir: Path
    metadata_stats_path: Path
    thresholds_path: Path


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_state: int
    require_both_eyes: bool
    stratify: bool = True
    stratify_min_count: int = 2


@dataclass(frozen=True)
class ImageConfig:
    size: int
    mean: list[float]
    std: list[float]


@dataclass(frozen=True)
class LoaderConfig:
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    train_batch_size: int
    val_batch_size: int
    test_batch_size: int


@dataclass(frozen=True)
class MetadataConfig:
    age_min: float
    age_max: float


@dataclass(frozen=True)
class DataConfig:
    labels: list[str]
    paths: DataPathsConfig
    split: SplitConfig
    image: ImageConfig
    loader: LoaderConfig
    metadata: MetadataConfig


@dataclass(frozen=True)
class FreezePolicyConfig:
    enabled: bool
    freeze_encoder_epochs: int
    unfreeze_last_n_stages: int
    full_finetune_epoch: int


@dataclass(frozen=True)
class ModelConfig:
    backbone_name: str
    pretrained: bool
    num_labels: int
    image_feature_dropout: float
    metadata_hidden_dim: int
    metadata_dropout: float
    fusion_hidden_dims: list[int]
    fusion_dropout: float
    freeze_policy: FreezePolicyConfig


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_root: Path
    log_every_n_steps: int


@dataclass(frozen=True)
class LossConfig:
    use_pos_weight: bool


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float


@dataclass(frozen=True)
class SchedulerConfig:
    name: str
    min_lr: float
    onecycle_pct_start: float


@dataclass(frozen=True)
class CheckpointConfig:
    monitor: str
    mode: str
    early_stopping_patience: int
    save_best_only: bool


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    device: str
    epochs: int
    mixed_precision: bool
    grad_clip_norm: float
    experiment: ExperimentConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    checkpoint: CheckpointConfig


@dataclass(frozen=True)
class EvalConfig:
    device: str
    split: str
    batch_size: int
    global_threshold: float
    tune_thresholds: bool
    save_predictions: bool
    predictions_file: Path
    thresholds_file: Path
    report_file: Path


@dataclass(frozen=True)
class InferenceConfig:
    device: str
    image_size: int
    global_threshold: float
    metadata_stats_path: Path
    thresholds_path: Path


@dataclass(frozen=True)
class APIConfig:
    host: str
    port: int
    reload: bool
    checkpoint_path: Path
    model_config_path: Path
    data_config_path: Path
    inference_config_path: Path
    cv_proxy_config_path: Path


@dataclass(frozen=True)
class CVProxyConfig:
    weights: dict[str, dict[str, float]]
    overall_weights: dict[str, float]
    risk_bands: dict[str, float]


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path)


def load_data_config(config_path: Path, project_root: Path | None = None) -> DataConfig:
    root = project_root or Path.cwd()
    raw = load_yaml(config_path)
    paths = raw["paths"]
    return DataConfig(
        labels=raw["labels"],
        paths=DataPathsConfig(
            data_root=_resolve(root, paths["data_root"]),
            csv_path=_resolve(root, paths["csv_path"]),
            images_dir=_resolve(root, paths["images_dir"]),
            processed_dir=_resolve(root, paths["processed_dir"]),
            splits_dir=_resolve(root, paths["splits_dir"]),
            metadata_stats_path=_resolve(root, paths["metadata_stats_path"]),
            thresholds_path=_resolve(root, paths["thresholds_path"]),
        ),
        split=SplitConfig(**raw["split"]),
        image=ImageConfig(**raw["image"]),
        loader=LoaderConfig(**raw["loader"]),
        metadata=MetadataConfig(**raw["metadata"]),
    )


def load_model_config(config_path: Path) -> ModelConfig:
    raw = load_yaml(config_path)
    return ModelConfig(
        backbone_name=raw["backbone_name"],
        pretrained=raw["pretrained"],
        num_labels=raw["num_labels"],
        image_feature_dropout=raw["image_feature_dropout"],
        metadata_hidden_dim=raw["metadata_hidden_dim"],
        metadata_dropout=raw["metadata_dropout"],
        fusion_hidden_dims=list(raw["fusion_hidden_dims"]),
        fusion_dropout=raw["fusion_dropout"],
        freeze_policy=FreezePolicyConfig(**raw["freeze_policy"]),
    )


def load_train_config(config_path: Path, project_root: Path | None = None) -> TrainConfig:
    root = project_root or Path.cwd()
    raw = load_yaml(config_path)
    return TrainConfig(
        seed=raw["seed"],
        device=raw["device"],
        epochs=raw["epochs"],
        mixed_precision=raw["mixed_precision"],
        grad_clip_norm=raw["grad_clip_norm"],
        experiment=ExperimentConfig(
            name=raw["experiment"]["name"],
            output_root=_resolve(root, raw["experiment"]["output_root"]),
            log_every_n_steps=raw["experiment"]["log_every_n_steps"],
        ),
        loss=LossConfig(**raw["loss"]),
        optimizer=OptimizerConfig(**raw["optimizer"]),
        scheduler=SchedulerConfig(**raw["scheduler"]),
        checkpoint=CheckpointConfig(**raw["checkpoint"]),
    )


def load_eval_config(config_path: Path, project_root: Path | None = None) -> EvalConfig:
    root = project_root or Path.cwd()
    raw = load_yaml(config_path)
    return EvalConfig(
        device=raw["device"],
        split=raw["split"],
        batch_size=raw["batch_size"],
        global_threshold=raw["global_threshold"],
        tune_thresholds=raw["tune_thresholds"],
        save_predictions=raw["save_predictions"],
        predictions_file=_resolve(root, raw["predictions_file"]),
        thresholds_file=_resolve(root, raw["thresholds_file"]),
        report_file=_resolve(root, raw.get("report_file", "data/processed/eval_report.json")),
    )


def load_inference_config(config_path: Path, project_root: Path | None = None) -> InferenceConfig:
    root = project_root or Path.cwd()
    raw = load_yaml(config_path)
    return InferenceConfig(
        device=raw["device"],
        image_size=raw["image_size"],
        global_threshold=raw["global_threshold"],
        metadata_stats_path=_resolve(root, raw["metadata_stats_path"]),
        thresholds_path=_resolve(root, raw["thresholds_path"]),
    )


def load_api_config(config_path: Path, project_root: Path | None = None) -> APIConfig:
    root = project_root or Path.cwd()
    raw = load_yaml(config_path)
    return APIConfig(
        host=raw["host"],
        port=raw["port"],
        reload=raw["reload"],
        checkpoint_path=_resolve(root, raw["checkpoint_path"]),
        model_config_path=_resolve(root, raw["model_config_path"]),
        data_config_path=_resolve(root, raw["data_config_path"]),
        inference_config_path=_resolve(root, raw["inference_config_path"]),
        cv_proxy_config_path=_resolve(root, raw["cv_proxy_config_path"]),
    )


def load_cv_proxy_config(config_path: Path) -> CVProxyConfig:
    raw = load_yaml(config_path)
    return CVProxyConfig(
        weights=raw["weights"],
        overall_weights=raw["overall_weights"],
        risk_bands=raw["risk_bands"],
    )







