from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from datasets.data_module import build_data_bundle
from evaluation.metrics import compute_multilabel_metrics, tune_thresholds_for_f1
from models.multimodal_model import MultimodalRiskModel
from utils.config import load_data_config, load_eval_config, load_model_config
from utils.constants import LABELS
from utils.device import resolve_device
from utils.io import ensure_dir, save_json
from utils.logging import get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation for multimodal model")
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"), help="Eval config path")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"), help="Data config path")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"), help="Model config path")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path")
    return parser.parse_args()


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_patient_ids: list[str] = []

    for batch in data_loader:
        left_image = batch["left_image"].to(device, non_blocking=True)
        right_image = batch["right_image"].to(device, non_blocking=True)
        metadata = batch["metadata"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        logits = model(left_image, right_image, metadata)
        probs = torch.sigmoid(logits)

        all_targets.append(target.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_patient_ids.extend([str(x) for x in batch["patient_id"]])

    return np.concatenate(all_targets), np.concatenate(all_probs), all_patient_ids


def _threshold_array_from_dict(thresholds: dict[str, float]) -> np.ndarray:
    return np.array([thresholds[label] for label in LABELS], dtype=np.float32)


def _save_predictions(
    path: Path,
    patient_ids: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | np.ndarray,
) -> None:
    ensure_dir(path.parent)
    th_array = np.full(len(LABELS), threshold) if isinstance(threshold, float) else threshold
    y_pred = (y_prob >= th_array).astype(int)

    rows: list[dict[str, Any]] = []
    for idx, patient_id in enumerate(patient_ids):
        row: dict[str, Any] = {"patient_id": patient_id}
        for label_idx, label in enumerate(LABELS):
            row[f"true_{label}"] = int(y_true[idx, label_idx])
            row[f"prob_{label}"] = float(y_prob[idx, label_idx])
            row[f"pred_{label}"] = int(y_pred[idx, label_idx])
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)


def _summary(metrics: dict[str, Any]) -> str:
    return (
        f"acc={metrics['accuracy']:.4f} "
        f"prec={metrics['macro_precision']:.4f} "
        f"rec={metrics['macro_recall']:.4f} "
        f"f1={metrics['macro_f1']:.4f} "
        f"auc={metrics['macro_auroc']:.4f} "
        f"pr_auc={metrics['macro_pr_auc']:.4f}"
    )


def _ensure_json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        return value

    return convert(payload)


def main() -> None:
    args = parse_args()

    data_config = load_data_config(args.data_config)
    model_config = load_model_config(args.model_config)
    eval_config = load_eval_config(args.config)

    data_bundle = build_data_bundle(data_config)
    split_to_loader = {
        "train": data_bundle.train_loader,
        "val": data_bundle.val_loader,
        "test": data_bundle.test_loader,
    }
    if eval_config.split not in split_to_loader:
        raise ValueError(f"Unsupported split: {eval_config.split}")

    device = resolve_device(eval_config.device)

    model = MultimodalRiskModel(model_config).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_prob, patient_ids = collect_predictions(model, split_to_loader[eval_config.split], device)

    global_metrics = compute_multilabel_metrics(
        y_true,
        y_prob,
        threshold=eval_config.global_threshold,
        labels=LABELS,
    )
    LOGGER.info("Global threshold metrics | %s", _summary(global_metrics))

    threshold_used: float | np.ndarray = eval_config.global_threshold
    report_payload: dict[str, Any] = {
        "split": eval_config.split,
        "global_threshold": eval_config.global_threshold,
        "global_metrics": global_metrics,
    }

    if eval_config.tune_thresholds:
        tuned_thresholds = tune_thresholds_for_f1(y_true, y_prob, labels=LABELS)
        save_json(eval_config.thresholds_file, tuned_thresholds)
        threshold_used = _threshold_array_from_dict(tuned_thresholds)

        tuned_metrics = compute_multilabel_metrics(y_true, y_prob, threshold=threshold_used, labels=LABELS)
        LOGGER.info("Tuned threshold metrics  | %s", _summary(tuned_metrics))

        report_payload["tuned_thresholds"] = tuned_thresholds
        report_payload["tuned_metrics"] = tuned_metrics

    if eval_config.save_predictions:
        _save_predictions(eval_config.predictions_file, patient_ids, y_true, y_prob, threshold_used)
        LOGGER.info("Saved predictions to %s", eval_config.predictions_file)

    ensure_dir(eval_config.report_file.parent)
    save_json(eval_config.report_file, _ensure_json_safe(report_payload))
    LOGGER.info("Saved evaluation report to %s", eval_config.report_file)


if __name__ == "__main__":
    main()
