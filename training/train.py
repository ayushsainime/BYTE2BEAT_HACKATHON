from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from datasets.build_patient_df import run as prepare_data_run
from datasets.data_module import build_data_bundle
from evaluation.metrics import compute_multilabel_metrics
from models.multimodal_model import MultimodalRiskModel
from training.trainer import Trainer
from utils.config import load_data_config, load_model_config, load_train_config
from utils.device import resolve_device
from utils.io import ensure_dir, make_run_dir, save_json
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


@torch.no_grad()
def _collect_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for batch in data_loader:
        left_image = batch["left_image"].to(device, non_blocking=True)
        right_image = batch["right_image"].to(device, non_blocking=True)
        metadata = batch["metadata"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        logits = model(left_image, right_image, metadata)
        probs = torch.sigmoid(logits)

        all_targets.append(target.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_targets, axis=0), np.concatenate(all_probs, axis=0)


def _important_metric_block(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["macro_precision"]),
        "recall": float(metrics["macro_recall"]),
        "f1": float(metrics["macro_f1"]),
        "macro_auroc": float(metrics["macro_auroc"]),
        "macro_pr_auc": float(metrics["macro_pr_auc"]),
        "label_accuracy": float(metrics["label_accuracy"]),
    }


def _best_epoch_summary(metrics_csv_path: Path, monitor_key: str, mode: str) -> dict[str, Any]:
    df = pd.read_csv(metrics_csv_path)
    if df.empty:
        return {}

    if mode == "max":
        idx = int(df[monitor_key].idxmax())
    else:
        idx = int(df[monitor_key].idxmin())

    row = df.iloc[idx]
    return {
        "best_epoch": int(row["epoch"]),
        "monitor_key": monitor_key,
        "monitor_value": float(row[monitor_key]),
        "train": {
            "accuracy": float(row.get("train_accuracy", 0.0)),
            "precision": float(row.get("train_precision", 0.0)),
            "recall": float(row.get("train_recall", 0.0)),
            "f1": float(row.get("train_f1", 0.0)),
            "macro_auroc": float(row.get("train_macro_auroc", 0.0)),
            "macro_pr_auc": float(row.get("train_macro_pr_auc", 0.0)),
        },
        "val": {
            "accuracy": float(row.get("val_accuracy", 0.0)),
            "precision": float(row.get("val_precision", 0.0)),
            "recall": float(row.get("val_recall", 0.0)),
            "f1": float(row.get("val_f1", 0.0)),
            "macro_auroc": float(row.get("val_macro_auroc", 0.0)),
            "macro_pr_auc": float(row.get("val_macro_pr_auc", 0.0)),
        },
    }


def _save_post_train_report(
    run_dir: Path,
    monitor_key: str,
    monitor_mode: str,
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    best_checkpoint: Path,
) -> None:
    reports_dir = ensure_dir(run_dir / "reports")
    best_epoch = _best_epoch_summary(run_dir / "metrics.csv", monitor_key, monitor_mode)

    payload = {
        "best_checkpoint": str(best_checkpoint),
        "best_epoch_summary": best_epoch,
        "post_train_val_metrics": _important_metric_block(val_metrics),
        "post_train_test_metrics": _important_metric_block(test_metrics),
    }
    save_json(reports_dir / "post_train_metrics.json", payload)

    flat = {
        "best_epoch": best_epoch.get("best_epoch", -1),
        "monitor_key": best_epoch.get("monitor_key", monitor_key),
        "monitor_value": best_epoch.get("monitor_value", 0.0),
        **{f"val_post_{k}": v for k, v in _important_metric_block(val_metrics).items()},
        **{f"test_post_{k}": v for k, v in _important_metric_block(test_metrics).items()},
    }
    pd.DataFrame([flat]).to_csv(reports_dir / "post_train_metrics.csv", index=False)


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

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_true, val_prob = _collect_predictions(model, data_bundle.val_loader, device)
    test_true, test_prob = _collect_predictions(model, data_bundle.test_loader, device)

    val_metrics = compute_multilabel_metrics(val_true, val_prob, threshold=0.5, labels=data_config.labels)
    test_metrics = compute_multilabel_metrics(test_true, test_prob, threshold=0.5, labels=data_config.labels)

    _save_post_train_report(
        run_dir=run_dir,
        monitor_key=train_config.checkpoint.monitor,
        monitor_mode=train_config.checkpoint.mode,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        best_checkpoint=best_checkpoint,
    )

    LOGGER.info("Training complete. Best checkpoint: %s", best_checkpoint)
    LOGGER.info("Post-train val metrics: %s", _important_metric_block(val_metrics))
    LOGGER.info("Post-train test metrics: %s", _important_metric_block(test_metrics))
    return best_checkpoint


def main() -> None:
    args = parse_args()
    run_training(args.config, args.data_config, args.model_config)


if __name__ == "__main__":
    main()
