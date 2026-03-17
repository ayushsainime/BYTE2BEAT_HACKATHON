from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.constants import LABELS


class FundusMultimodalDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Any,
        age_mean: float,
        age_std: float,
        include_target: bool = True,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.age_mean = age_mean
        self.age_std = age_std if age_std != 0 else 1.0
        self.include_target = include_target

    def __len__(self) -> int:
        return len(self.dataframe)

    def _read_image(self, image_path: str | Path) -> np.ndarray:
        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _encode_metadata(self, age: float) -> torch.Tensor:
        age_norm = (float(age) - self.age_mean) / self.age_std
        return torch.tensor([age_norm], dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]

        left_image = self._read_image(row["left_path"])
        right_image = self._read_image(row["right_path"])

        left_tensor = self.transform(image=left_image)["image"]
        right_tensor = self.transform(image=right_image)["image"]

        sample: dict[str, Any] = {
            "left_image": left_tensor,
            "right_image": right_tensor,
            "metadata": self._encode_metadata(row["age"]),
            "patient_id": str(row["patient_id"]),
        }

        if self.include_target:
            target = torch.tensor([float(row[label]) for label in LABELS], dtype=torch.float32)
            sample["target"] = target

        return sample
