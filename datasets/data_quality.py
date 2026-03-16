from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.config import load_data_config
from utils.constants import LABELS


def summarize_split(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    out: dict[str, float] = {"num_patients": float(len(df))}
    for label in LABELS:
        out[f"prevalence_{label}"] = float(df[label].mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data quality summary by split")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args()

    data_config = load_data_config(args.data_config)
    for split in ["train", "val", "test"]:
        split_path = data_config.paths.splits_dir / f"{split}.csv"
        if not split_path.exists():
            print(f"{split}: missing split file {split_path}")
            continue
        summary = summarize_split(split_path)
        print(f"[{split}] {summary}")


if __name__ == "__main__":
    main()
