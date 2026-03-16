from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import torch

from datasets.build_patient_df import run as prepare_data_run
from datasets.data_module import build_data_bundle
from models.multimodal_model import MultimodalRiskModel
from training.trainer import Trainer
from utils.config import load_data_config, load_model_config, load_train_config
from utils.device import resolve_device
from utils.io import ensure_dir, make_run_dir
from utils.logging import get_logger
from utils.seed import set_seed

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal retinal risk model")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"), help="Path to train config")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"), help="Path to data config")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"), help="Path to model config")
    return parser.parse_args()


def _prepare_data_if_needed(data_config_path: Path) -> None:
    data_config = load_data_config(data_config_path)
    required = [
        data_config.paths.splits_dir / "train.csv",
        data_config.paths.splits_dir / "val.csv",
        data_config.paths.splits_dir / "test.csv",
        data_config.paths.metadata_stats_path,
    ]
    if not all(path.exists() for path in required):
        LOGGER.info("Data splits or metadata stats missing, preparing dataset artifacts...")
        prepare_data_run(data_config)


def _update_latest_artifact(best_checkpoint: Path, run_dir: Path) -> None:
    latest_dir = run_dir.parent / "latest"
    ensure_dir(latest_dir)
    shutil.copy2(best_checkpoint, latest_dir / "best.pt")
    (latest_dir / "run_path.txt").write_text(str(run_dir), encoding="utf-8")


def _compute_pos_weight(train_df: pd.DataFrame, labels: list[str]) -> torch.Tensor:
    positives = torch.tensor([float(train_df[label].sum()) for label in labels], dtype=torch.float32)
    total = float(len(train_df))
    negatives = total - positives

    pos_weight = negatives / torch.clamp(positives, min=1.0)
    return torch.clamp(pos_weight, min=1.0, max=20.0)


def run_training(train_config_path: Path, data_config_path: Path, model_config_path: Path) -> Path:
    data_config = load_data_config(data_config_path)
    model_config = load_model_config(model_config_path)
    train_config = load_train_config(train_config_path)

    set_seed(train_config.seed)
    _prepare_data_if_needed(data_config_path)

    data_bundle = build_data_bundle(data_config)

    run_dir = make_run_dir(train_config.experiment.output_root, train_config.experiment.name)
    ensure_dir(run_dir)
    shutil.copy2(train_config_path, run_dir / "train_config.yaml")
    shutil.copy2(data_config_path, run_dir / "data_config.yaml")
    shutil.copy2(model_config_path, run_dir / "model_config.yaml")

    device = resolve_device(train_config.device)
    model = MultimodalRiskModel(model_config)

    pos_weight = None
    if train_config.loss.use_pos_weight:
        pos_weight = _compute_pos_weight(data_bundle.train_df, data_config.labels)

    trainer = Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=device,
        run_dir=run_dir,
        pos_weight=pos_weight,
    )

    best_checkpoint = trainer.fit(data_bundle.train_loader, data_bundle.val_loader)
    _update_latest_artifact(best_checkpoint, run_dir)

    LOGGER.info("Training complete. Best checkpoint: %s", best_checkpoint)
    return best_checkpoint


def main() -> None:
    args = parse_args()
    run_training(args.config, args.data_config, args.model_config)


if __name__ == "__main__":
    main()
