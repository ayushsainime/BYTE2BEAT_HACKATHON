from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.config import DataConfig, load_data_config
from utils.constants import LABELS
from utils.io import ensure_dir, save_json
from utils.logging import get_logger

LOGGER = get_logger(__name__)


def build_patient_dataframe(data_config: DataConfig) -> pd.DataFrame:
    df = pd.read_csv(data_config.paths.csv_path)

    if "ID" not in df.columns:
        raise ValueError("Column 'ID' not found in full_df.csv")

    grouped = df.groupby("ID", as_index=False).first()

    missing_labels = [label for label in LABELS if label not in grouped.columns]
    if missing_labels:
        raise ValueError(f"Missing label columns: {missing_labels}")

    patient_df = pd.DataFrame()
    patient_df["patient_id"] = grouped["ID"].astype(str)
    patient_df["age"] = grouped["Patient Age"].astype(float)
    patient_df["sex"] = grouped["Patient Sex"].astype(str)
    patient_df["left_image"] = grouped["Left-Fundus"].astype(str)
    patient_df["right_image"] = grouped["Right-Fundus"].astype(str)

    patient_df["left_path"] = patient_df["left_image"].map(lambda name: str((data_config.paths.images_dir / name).resolve()))
    patient_df["right_path"] = patient_df["right_image"].map(lambda name: str((data_config.paths.images_dir / name).resolve()))

    for label in LABELS:
        patient_df[label] = grouped[label].astype(int)

    patient_df["target"] = patient_df[LABELS].values.tolist()

    left_exists = patient_df["left_path"].map(lambda p: Path(p).exists())
    right_exists = patient_df["right_path"].map(lambda p: Path(p).exists())

    if data_config.split.require_both_eyes:
        pre_filter = len(patient_df)
        patient_df = patient_df[left_exists & right_exists].copy()
        dropped = pre_filter - len(patient_df)
        LOGGER.info("Dropped %s patients without both eyes.", dropped)

    patient_df = patient_df.reset_index(drop=True)
    return patient_df


def _primary_label(row: pd.Series, label_frequency: dict[str, int]) -> str:
    positives = [label for label in LABELS if int(row[label]) == 1]
    if not positives:
        return "NONE"
    return min(positives, key=lambda label: (label_frequency[label], label))


def _build_stratify_labels(patient_df: pd.DataFrame, min_count: int) -> pd.Series | None:
    label_df = patient_df[LABELS].astype(int)
    label_frequency = {label: int(label_df[label].sum()) for label in LABELS}

    primary_series = label_df.apply(lambda row: _primary_label(row, label_frequency), axis=1)
    positives_count_series = label_df.sum(axis=1).astype(int)

    # First pass: primary positive label + number of active labels.
    stratify_labels = primary_series + "_" + positives_count_series.astype(str)

    # Collapse rare combinations to primary label only.
    combo_counts = stratify_labels.value_counts()
    rare_combo_mask = stratify_labels.map(combo_counts) < min_count
    if rare_combo_mask.any():
        stratify_labels = stratify_labels.where(~rare_combo_mask, primary_series)

    # Collapse any remaining rare strata to a common bucket.
    strata_counts = stratify_labels.value_counts()
    rare_primary_mask = stratify_labels.map(strata_counts) < min_count
    if rare_primary_mask.any():
        stratify_labels = stratify_labels.where(~rare_primary_mask, "__OTHER__")

    final_counts = stratify_labels.value_counts()
    if final_counts.empty:
        return None
    if int(final_counts.min()) < min_count:
        return None
    if len(final_counts) < 2:
        return None

    return stratify_labels


def _split_with_optional_stratify(
    patient_df: pd.DataFrame,
    test_size: float,
    random_state: int,
    stratify: bool,
    stratify_min_count: int,
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify_labels = _build_stratify_labels(patient_df, stratify_min_count) if stratify else None

    try:
        return train_test_split(
            patient_df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify_labels,
        )
    except ValueError as exc:
        if stratify_labels is None:
            raise

        LOGGER.warning(
            "Stratified %s split failed (%s). Falling back to random split.",
            split_name,
            exc,
        )
        return train_test_split(
            patient_df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=None,
        )


def split_patient_dataframe(patient_df: pd.DataFrame, data_config: DataConfig) -> dict[str, pd.DataFrame]:
    train_ratio = data_config.split.train_ratio
    val_ratio = data_config.split.val_ratio
    test_ratio = data_config.split.test_ratio

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Small fixtures can break sklearn split constraints; handle these explicitly.
    if len(patient_df) < 3:
        shuffled = patient_df.sample(frac=1.0, random_state=data_config.split.random_state).reset_index(drop=True)
        if len(shuffled) == 1:
            train_df = shuffled.iloc[:1].copy()
            val_df = shuffled.iloc[0:0].copy()
            test_df = shuffled.iloc[0:0].copy()
        else:
            train_df = shuffled.iloc[:1].copy()
            val_df = shuffled.iloc[1:2].copy()
            test_df = shuffled.iloc[0:0].copy()
    else:
        train_df, temp_df = _split_with_optional_stratify(
            patient_df,
            test_size=(1.0 - train_ratio),
            random_state=data_config.split.random_state,
            stratify=data_config.split.stratify,
            stratify_min_count=data_config.split.stratify_min_count,
            split_name="train/temp",
        )

        if len(temp_df) < 2:
            val_df = temp_df.copy()
            test_df = temp_df.iloc[0:0].copy()
        else:
            rel_test_ratio = test_ratio / (val_ratio + test_ratio)
            val_df, test_df = _split_with_optional_stratify(
                temp_df,
                test_size=rel_test_ratio,
                random_state=data_config.split.random_state,
                stratify=data_config.split.stratify,
                stratify_min_count=data_config.split.stratify_min_count,
                split_name="val/test",
            )

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def compute_metadata_stats(train_df: pd.DataFrame, data_config: DataConfig) -> dict[str, float]:
    age_mean = float(train_df["age"].mean())
    age_std = float(train_df["age"].std(ddof=0))
    if age_std == 0.0:
        age_std = 1.0

    return {
        "age_mean": age_mean,
        "age_std": age_std,
    }


def save_outputs(
    patient_df: pd.DataFrame,
    split_dfs: dict[str, pd.DataFrame],
    metadata_stats: dict[str, float],
    data_config: DataConfig,
) -> None:
    ensure_dir(data_config.paths.processed_dir)
    ensure_dir(data_config.paths.splits_dir)

    patient_df.to_csv(data_config.paths.processed_dir / "patients.csv", index=False)

    for split_name, split_df in split_dfs.items():
        split_df.to_csv(data_config.paths.splits_dir / f"{split_name}.csv", index=False)

    save_json(data_config.paths.metadata_stats_path, metadata_stats)


def run(data_config: DataConfig) -> None:
    patient_df = build_patient_dataframe(data_config)
    split_dfs = split_patient_dataframe(patient_df, data_config)
    metadata_stats = compute_metadata_stats(split_dfs["train"], data_config)
    save_outputs(patient_df, split_dfs, metadata_stats, data_config)

    LOGGER.info("Prepared %s patients.", len(patient_df))
    for split_name, split_df in split_dfs.items():
        LOGGER.info("%s split size: %s", split_name, len(split_df))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build patient-level dataframe and splits.")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"), help="Path to data config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_config = load_data_config(args.config)
    run(data_config)


if __name__ == "__main__":
    main()



