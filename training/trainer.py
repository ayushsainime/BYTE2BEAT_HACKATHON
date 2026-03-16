from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from evaluation.metrics import compute_multilabel_metrics
from utils.config import ModelConfig, TrainConfig
from utils.constants import LABELS
from utils.logging import CSVMetricLogger, get_logger

LOGGER = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        train_config: TrainConfig,
        device: torch.device,
        run_dir: Path,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.train_config = train_config
        self.device = device
        self.run_dir = run_dir
        self.labels = LABELS

        self.model.to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = None
        self.scheduler_name = train_config.scheduler.name.lower()
        self.use_amp = train_config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device) if pos_weight is not None else None)

        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
        self.csv_logger = CSVMetricLogger(self.run_dir / "metrics.csv")
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_metric = -np.inf if train_config.checkpoint.mode == "max" else np.inf
        self.bad_epochs = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        optimizer_name = self.train_config.optimizer.name.lower()
        if optimizer_name != "adamw":
            raise ValueError(f"Unsupported optimizer: {self.train_config.optimizer.name}")
        return AdamW(
            self.model.parameters(),
            lr=self.train_config.optimizer.lr,
            weight_decay=self.train_config.optimizer.weight_decay,
        )

    def _build_scheduler(self, steps_per_epoch: int) -> torch.optim.lr_scheduler._LRScheduler | None:
        scheduler_name = self.scheduler_name
        if scheduler_name == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_config.epochs,
                eta_min=self.train_config.scheduler.min_lr,
            )
        if scheduler_name == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.train_config.optimizer.lr,
                epochs=self.train_config.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=self.train_config.scheduler.onecycle_pct_start,
            )
        if scheduler_name == "none":
            return None
        raise ValueError(f"Unsupported scheduler: {self.train_config.scheduler.name}")

    def _is_better(self, value: float) -> bool:
        if self.train_config.checkpoint.mode == "max":
            return value > self.best_metric
        return value < self.best_metric

    def _save_checkpoint(self, epoch: int, monitor_value: float, is_best: bool) -> None:
        payload: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "monitor_value": monitor_value,
            "model_config": asdict(self.model_config),
            "labels": self.labels,
        }
        if self.scheduler is not None:
            payload["scheduler_state_dict"] = self.scheduler.state_dict()

        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(payload, latest_path)
        if is_best:
            torch.save(payload, self.checkpoint_dir / "best.pt")

    def _run_epoch(self, data_loader: DataLoader, training: bool, epoch: int) -> tuple[float, np.ndarray, np.ndarray]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        all_targets: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []

        for step, batch in enumerate(data_loader, start=1):
            left_image = batch["left_image"].to(self.device, non_blocking=True)
            right_image = batch["right_image"].to(self.device, non_blocking=True)
            metadata = batch["metadata"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(left_image, right_image, metadata)
                    loss = self.criterion(logits, target)

            if training:
                self.scaler.scale(loss).backward()

                if self.train_config.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.scheduler is not None and self.scheduler_name == "onecycle":
                    self.scheduler.step()

                if step % self.train_config.experiment.log_every_n_steps == 0:
                    LOGGER.info("Epoch %s | Step %s | train_loss %.4f", epoch + 1, step, float(loss.item()))

            running_loss += float(loss.item())
            probs = torch.sigmoid(logits)
            all_targets.append(target.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

        y_true = np.concatenate(all_targets, axis=0)
        y_prob = np.concatenate(all_probs, axis=0)
        avg_loss = running_loss / max(len(data_loader), 1)
        return avg_loss, y_true, y_prob

    @staticmethod
    def _flatten_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
        return {
            f"{prefix}_accuracy": float(metrics["accuracy"]),
            f"{prefix}_precision": float(metrics["macro_precision"]),
            f"{prefix}_recall": float(metrics["macro_recall"]),
            f"{prefix}_f1": float(metrics["macro_f1"]),
            f"{prefix}_macro_auroc": float(metrics["macro_auroc"]),
            f"{prefix}_macro_pr_auc": float(metrics["macro_pr_auc"]),
            f"{prefix}_label_accuracy": float(metrics["label_accuracy"]),
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Path:
        self.scheduler = self._build_scheduler(steps_per_epoch=len(train_loader))

        for epoch in range(self.train_config.epochs):
            self.model.apply_freeze_policy(epoch, self.model_config.freeze_policy)

            train_loss, train_true, train_prob = self._run_epoch(train_loader, training=True, epoch=epoch)
            val_loss, val_true, val_prob = self._run_epoch(val_loader, training=False, epoch=epoch)

            train_metrics = compute_multilabel_metrics(train_true, train_prob, threshold=0.5, labels=self.labels)
            val_metrics = compute_multilabel_metrics(val_true, val_prob, threshold=0.5, labels=self.labels)

            if self.scheduler is not None and self.scheduler_name == "cosine":
                self.scheduler.step()

            lr = float(self.optimizer.param_groups[0]["lr"])
            row: dict[str, float | int] = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
            }
            row.update(self._flatten_metrics("train", train_metrics))
            row.update(self._flatten_metrics("val", val_metrics))

            monitor_key = self.train_config.checkpoint.monitor
            if monitor_key not in row:
                raise KeyError(f"Monitor metric '{monitor_key}' not found in epoch row. Available keys: {list(row.keys())}")
            monitor_value = float(row[monitor_key])

            self.csv_logger.log(row)
            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val_loss, epoch)
            self.writer.add_scalar("metrics/train_accuracy", float(train_metrics["accuracy"]), epoch)
            self.writer.add_scalar("metrics/val_accuracy", float(val_metrics["accuracy"]), epoch)
            self.writer.add_scalar("metrics/train_precision", float(train_metrics["macro_precision"]), epoch)
            self.writer.add_scalar("metrics/val_precision", float(val_metrics["macro_precision"]), epoch)
            self.writer.add_scalar("metrics/train_recall", float(train_metrics["macro_recall"]), epoch)
            self.writer.add_scalar("metrics/val_recall", float(val_metrics["macro_recall"]), epoch)
            self.writer.add_scalar("metrics/train_f1", float(train_metrics["macro_f1"]), epoch)
            self.writer.add_scalar("metrics/val_f1", float(val_metrics["macro_f1"]), epoch)
            self.writer.add_scalar("metrics/val_macro_auroc", float(val_metrics["macro_auroc"]), epoch)
            self.writer.add_scalar("lr", lr, epoch)

            is_best = self._is_better(monitor_value)
            if is_best:
                self.best_metric = monitor_value
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1

            self._save_checkpoint(epoch, monitor_value, is_best)

            LOGGER.info(
                (
                    "Epoch %s/%s | train_loss=%.4f val_loss=%.4f "
                    "val_acc=%.4f val_prec=%.4f val_rec=%.4f val_f1=%.4f val_auc=%.4f"
                ),
                epoch + 1,
                self.train_config.epochs,
                train_loss,
                val_loss,
                float(val_metrics["accuracy"]),
                float(val_metrics["macro_precision"]),
                float(val_metrics["macro_recall"]),
                float(val_metrics["macro_f1"]),
                float(val_metrics["macro_auroc"]),
            )

            if self.bad_epochs >= self.train_config.checkpoint.early_stopping_patience:
                LOGGER.info("Early stopping at epoch %s", epoch + 1)
                break

        self.writer.close()
        return self.checkpoint_dir / "best.pt"
