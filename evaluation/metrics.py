from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils.constants import LABELS


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def _safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def _confusion_stats(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    tn = float(((y_true == 0) & (y_pred == 0)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity


def _to_threshold_array(threshold: float | np.ndarray, num_labels: int) -> np.ndarray:
    if isinstance(threshold, float):
        return np.full(num_labels, threshold, dtype=np.float32)
    return np.asarray(threshold, dtype=np.float32)


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | np.ndarray = 0.5,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    labels = labels or LABELS
    thresholds = _to_threshold_array(threshold, len(labels))
    y_pred = (y_prob >= thresholds).astype(int)

    per_label: dict[str, dict[str, float]] = {}
    aucs: list[float] = []
    pr_aucs: list[float] = []

    for idx, label in enumerate(labels):
        y_true_i = y_true[:, idx]
        y_prob_i = y_prob[:, idx]
        y_pred_i = y_pred[:, idx]

        auc = _safe_roc_auc(y_true_i, y_prob_i)
        pr_auc = _safe_pr_auc(y_true_i, y_prob_i)
        accuracy = float(accuracy_score(y_true_i, y_pred_i))
        precision = float(precision_score(y_true_i, y_pred_i, zero_division=0))
        recall = float(recall_score(y_true_i, y_pred_i, zero_division=0))
        f1 = float(f1_score(y_true_i, y_pred_i, zero_division=0))
        sensitivity, specificity = _confusion_stats(y_true_i, y_pred_i)

        aucs.append(auc)
        pr_aucs.append(pr_auc)
        per_label[label] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": auc,
            "pr_auc": pr_auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

    macro_auroc = float(np.nanmean(np.array(aucs, dtype=np.float64)))
    macro_pr_auc = float(np.nanmean(np.array(pr_aucs, dtype=np.float64)))

    subset_accuracy = float(accuracy_score(y_true, y_pred))
    label_accuracy = float((y_true == y_pred).mean())

    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    micro_precision = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    micro_recall = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    micro_auroc = _safe_roc_auc(y_true.ravel(), y_prob.ravel())

    return {
        "accuracy": subset_accuracy,
        "label_accuracy": label_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_auroc": macro_auroc,
        "micro_auroc": micro_auroc,
        "macro_pr_auc": macro_pr_auc,
        "per_label": per_label,
    }


def tune_thresholds_for_f1(y_true: np.ndarray, y_prob: np.ndarray, labels: list[str] | None = None) -> dict[str, float]:
    labels = labels or LABELS
    thresholds: dict[str, float] = {}

    candidates = np.linspace(0.1, 0.9, 17)
    for idx, label in enumerate(labels):
        best_t = 0.5
        best_f1 = -1.0
        for threshold in candidates:
            preds = (y_prob[:, idx] >= threshold).astype(int)
            score = f1_score(y_true[:, idx], preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = float(threshold)
        thresholds[label] = best_t

    return thresholds
