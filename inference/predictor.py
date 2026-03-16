from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from models.multimodal_model import MultimodalRiskModel
from utils.config import CVProxyConfig, DataConfig, InferenceConfig, ModelConfig
from utils.constants import LABELS
from utils.device import resolve_device
from utils.io import load_json


@dataclass(frozen=True)
class PredictionResult:
    labels: dict[str, int]
    probabilities: dict[str, float]
    cv_summary: dict[str, float | str]


class Predictor:
    def __init__(
        self,
        checkpoint_path: Path,
        model_config: ModelConfig,
        data_config: DataConfig,
        inference_config: InferenceConfig,
        cv_proxy_config: CVProxyConfig,
    ) -> None:
        self.device = resolve_device(inference_config.device)
        self.model = MultimodalRiskModel(model_config).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.transform = A.Compose(
            [
                A.Resize(inference_config.image_size, inference_config.image_size),
                A.Normalize(mean=data_config.image.mean, std=data_config.image.std),
                ToTensorV2(),
            ]
        )

        self.global_threshold = inference_config.global_threshold
        self.thresholds = self._load_thresholds(inference_config.thresholds_path)

        metadata_stats = load_json(inference_config.metadata_stats_path)
        self.age_mean = float(metadata_stats["age_mean"])
        self.age_std = float(metadata_stats["age_std"]) if float(metadata_stats["age_std"]) != 0 else 1.0
        self.sex_mapping = {str(k): float(v) for k, v in metadata_stats["sex_mapping"].items()}

        self.cv_proxy_config = cv_proxy_config

    def _load_thresholds(self, path: Path) -> dict[str, float]:
        if path.exists():
            raw = load_json(path)
            return {label: float(raw.get(label, self.global_threshold)) for label in LABELS}
        return {label: float(self.global_threshold) for label in LABELS}

    def _read_image(self, image_input: str | Path | bytes | np.ndarray) -> np.ndarray:
        if isinstance(image_input, np.ndarray):
            return image_input

        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input))
            if image is None:
                raise FileNotFoundError(f"Image not found or unreadable: {image_input}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if isinstance(image_input, bytes):
            encoded = np.frombuffer(image_input, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Unable to decode image bytes.")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def _encode_metadata(self, age: float, sex: str) -> torch.Tensor:
        age_norm = (float(age) - self.age_mean) / self.age_std
        sex_value = float(self.sex_mapping.get(str(sex), 0.0))
        return torch.tensor([[age_norm, sex_value]], dtype=torch.float32, device=self.device)

    def _cv_summary(self, probabilities: dict[str, float]) -> dict[str, float | str]:
        proxy_scores: dict[str, float] = {}
        for proxy_name, weight_map in self.cv_proxy_config.weights.items():
            numerator = 0.0
            denominator = 0.0
            for label, weight in weight_map.items():
                numerator += float(probabilities[label]) * float(weight)
                denominator += float(weight)
            proxy_scores[proxy_name] = numerator / denominator if denominator > 0 else 0.0

        overall_num = 0.0
        overall_den = 0.0
        for proxy_name, weight in self.cv_proxy_config.overall_weights.items():
            overall_num += proxy_scores.get(proxy_name, 0.0) * float(weight)
            overall_den += float(weight)
        overall_cv_proxy = overall_num / overall_den if overall_den > 0 else 0.0

        low_max = float(self.cv_proxy_config.risk_bands.get("low_max", 0.33))
        medium_max = float(self.cv_proxy_config.risk_bands.get("medium_max", 0.66))

        risk_band = "high"
        if overall_cv_proxy <= low_max:
            risk_band = "low"
        elif overall_cv_proxy <= medium_max:
            risk_band = "medium"

        return {
            **{k: float(v) for k, v in proxy_scores.items()},
            "overall_cv_proxy": float(overall_cv_proxy),
            "risk_band": risk_band,
        }

    @torch.no_grad()
    def predict_single(
        self,
        left_image: str | Path | bytes | np.ndarray,
        right_image: str | Path | bytes | np.ndarray,
        age: float,
        sex: str,
    ) -> PredictionResult:
        left_np = self._read_image(left_image)
        right_np = self._read_image(right_image)

        left_tensor = self.transform(image=left_np)["image"].unsqueeze(0).to(self.device)
        right_tensor = self.transform(image=right_np)["image"].unsqueeze(0).to(self.device)
        metadata_tensor = self._encode_metadata(age, sex)

        logits = self.model(left_tensor, right_tensor, metadata_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

        probabilities = {label: float(probs[idx]) for idx, label in enumerate(LABELS)}
        labels = {
            label: int(probabilities[label] >= self.thresholds.get(label, self.global_threshold))
            for label in LABELS
        }
        cv_summary = self._cv_summary(probabilities)

        return PredictionResult(labels=labels, probabilities=probabilities, cv_summary=cv_summary)

    def predict_batch(self, records: list[dict[str, Any]]) -> list[PredictionResult]:
        outputs: list[PredictionResult] = []
        for record in records:
            outputs.append(
                self.predict_single(
                    left_image=record["left_image"],
                    right_image=record["right_image"],
                    age=float(record["age"]),
                    sex=str(record["sex"]),
                )
            )
        return outputs
