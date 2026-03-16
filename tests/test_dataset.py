from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from datasets.fundus_multimodal_dataset import FundusMultimodalDataset
from datasets.transforms import get_eval_transform
from utils.config import (
    DataConfig,
    DataPathsConfig,
    ImageConfig,
    LoaderConfig,
    MetadataConfig,
    SplitConfig,
)
from tests.conftest import write_dummy_image


def make_data_config(tmp_path: Path) -> DataConfig:
    return DataConfig(
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
        metadata=MetadataConfig(0.0, 120.0, {"Female": 0.0, "Male": 1.0}),
    )


def test_dataset_output_contract(tmp_path: Path) -> None:
    config = make_data_config(tmp_path)

    left = config.paths.images_dir / "0_left.jpg"
    right = config.paths.images_dir / "0_right.jpg"
    write_dummy_image(left, value=120)
    write_dummy_image(right, value=140)

    df = pd.DataFrame(
        [
            {
                "patient_id": "0",
                "age": 65,
                "sex": "Female",
                "left_path": str(left),
                "right_path": str(right),
                "N": 1,
                "D": 0,
                "G": 0,
                "C": 0,
                "A": 0,
                "H": 0,
                "M": 0,
                "O": 0,
            }
        ]
    )

    dataset = FundusMultimodalDataset(
        dataframe=df,
        transform=get_eval_transform(config),
        age_mean=60.0,
        age_std=10.0,
        sex_mapping={"Female": 0.0, "Male": 1.0},
        include_target=True,
    )

    item = dataset[0]
    assert item["left_image"].shape == torch.Size([3, 64, 64])
    assert item["right_image"].shape == torch.Size([3, 64, 64])
    assert item["metadata"].shape == torch.Size([2])
    assert item["target"].shape == torch.Size([8])
    assert item["patient_id"] == "0"
