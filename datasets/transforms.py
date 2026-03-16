from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.config import DataConfig


def get_train_transform(data_config: DataConfig) -> A.Compose:
    size = data_config.image.size
    return A.Compose(
        [
            A.Resize(size, size),
            # Fundus-specific photometric augmentations.
            A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.35),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.RandomGamma(gamma_limit=(85, 120), p=0.35),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=12, val_shift_limit=8, p=0.25),
            A.ShiftScaleRotate(
                shift_limit=0.03,
                scale_limit=0.08,
                rotate_limit=8,
                border_mode=0,
                p=0.45,
            ),
            A.GaussNoise(std_range=(0.01, 0.04), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.Normalize(mean=data_config.image.mean, std=data_config.image.std),
            ToTensorV2(),
        ]
    )


def get_eval_transform(data_config: DataConfig) -> A.Compose:
    size = data_config.image.size
    return A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=data_config.image.mean, std=data_config.image.std),
            ToTensorV2(),
        ]
    )
