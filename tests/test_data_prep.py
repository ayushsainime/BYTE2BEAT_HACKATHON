from __future__ import annotations

from pathlib import Path

import pandas as pd

from datasets.build_patient_df import build_patient_dataframe, split_patient_dataframe
from utils.config import (
    DataConfig,
    DataPathsConfig,
    ImageConfig,
    LoaderConfig,
    MetadataConfig,
    SplitConfig,
)
from tests.conftest import make_minimal_full_df, write_dummy_image


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
        split=SplitConfig(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
            require_both_eyes=True,
        ),
        image=ImageConfig(size=64, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
        loader=LoaderConfig(
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            train_batch_size=2,
            val_batch_size=2,
            test_batch_size=2,
        ),
        metadata=MetadataConfig(age_min=0.0, age_max=120.0),
    )


def test_build_patient_dataframe_filters_missing_eyes(tmp_path: Path) -> None:
    config = make_data_config(tmp_path)
    make_minimal_full_df(config.paths.csv_path)

    write_dummy_image(config.paths.images_dir / "0_left.jpg")
    write_dummy_image(config.paths.images_dir / "0_right.jpg")
    write_dummy_image(config.paths.images_dir / "1_left.jpg")
    # Intentionally missing 1_right.jpg

    patient_df = build_patient_dataframe(config)

    assert len(patient_df) == 1
    assert patient_df.iloc[0]["patient_id"] == "0"


def test_split_patient_dataframe_has_expected_keys(tmp_path: Path) -> None:
    config = make_data_config(tmp_path)
    make_minimal_full_df(config.paths.csv_path)

    write_dummy_image(config.paths.images_dir / "0_left.jpg")
    write_dummy_image(config.paths.images_dir / "0_right.jpg")
    write_dummy_image(config.paths.images_dir / "1_left.jpg")
    write_dummy_image(config.paths.images_dir / "1_right.jpg")

    patient_df = build_patient_dataframe(config)
    splits = split_patient_dataframe(patient_df, config)

    assert set(splits.keys()) == {"train", "val", "test"}
    assert sum(len(v) for v in splits.values()) == len(patient_df)


def test_stratified_split_preserves_label_prevalence(tmp_path: Path) -> None:
    config = make_data_config(tmp_path)

    rows: list[dict[str, int | float | str]] = []
    for idx in range(120):
        row: dict[str, int | float | str] = {
            "patient_id": str(idx),
            "age": float(40 + (idx % 35)),
            "sex": "Female" if idx % 2 == 0 else "Male",
            "left_path": "",
            "right_path": "",
            "N": 0,
            "D": 0,
            "G": 0,
            "C": 0,
            "A": 0,
            "H": 0,
            "M": 0,
            "O": 0,
        }

        if idx < 60:
            row["D"] = 1
        else:
            row["N"] = 1

        if idx % 20 == 0:
            row["H"] = 1
        if idx % 15 == 0:
            row["O"] = 1

        rows.append(row)

    patient_df = pd.DataFrame(rows)
    splits = split_patient_dataframe(patient_df, config)

    full_prevalence = patient_df[["D", "H", "O"]].mean()
    for split_name, split_df in splits.items():
        if split_df.empty:
            continue
        split_prevalence = split_df[["D", "H", "O"]].mean()
        for label in ["D", "H", "O"]:
            assert abs(float(split_prevalence[label]) - float(full_prevalence[label])) < 0.2, (
                f"{split_name} prevalence drift too high for {label}"
            )

