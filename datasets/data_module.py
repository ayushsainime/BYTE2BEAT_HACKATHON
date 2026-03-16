from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from torch.utils.data import DataLoader

from datasets.fundus_multimodal_dataset import FundusMultimodalDataset
from datasets.transforms import get_eval_transform, get_train_transform
from utils.config import DataConfig
from utils.io import load_json


@dataclass(frozen=True)
class DataBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    metadata_stats: dict[str, float | dict[str, float]]


def _read_split_csv(data_config: DataConfig, split_name: str) -> pd.DataFrame:
    path = data_config.paths.splits_dir / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing split file: {path}. Run `python -m datasets.build_patient_df --config configs/data.yaml` first."
        )
    return pd.read_csv(path)


def build_data_bundle(data_config: DataConfig) -> DataBundle:
    metadata_stats = load_json(data_config.paths.metadata_stats_path)

    train_df = _read_split_csv(data_config, "train")
    val_df = _read_split_csv(data_config, "val")
    test_df = _read_split_csv(data_config, "test")

    train_dataset = FundusMultimodalDataset(
        dataframe=train_df,
        transform=get_train_transform(data_config),
        age_mean=float(metadata_stats["age_mean"]),
        age_std=float(metadata_stats["age_std"]),
        sex_mapping=dict(metadata_stats["sex_mapping"]),
        include_target=True,
    )
    val_dataset = FundusMultimodalDataset(
        dataframe=val_df,
        transform=get_eval_transform(data_config),
        age_mean=float(metadata_stats["age_mean"]),
        age_std=float(metadata_stats["age_std"]),
        sex_mapping=dict(metadata_stats["sex_mapping"]),
        include_target=True,
    )
    test_dataset = FundusMultimodalDataset(
        dataframe=test_df,
        transform=get_eval_transform(data_config),
        age_mean=float(metadata_stats["age_mean"]),
        age_std=float(metadata_stats["age_std"]),
        sex_mapping=dict(metadata_stats["sex_mapping"]),
        include_target=True,
    )

    common_loader_kwargs = {
        "num_workers": data_config.loader.num_workers,
        "pin_memory": data_config.loader.pin_memory,
        "persistent_workers": data_config.loader.persistent_workers and data_config.loader.num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.loader.train_batch_size,
        shuffle=True,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.loader.val_batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config.loader.test_batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )

    return DataBundle(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        metadata_stats=metadata_stats,
    )
