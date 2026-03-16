from __future__ import annotations

from pathlib import Path

from training.train import run_training


if __name__ == "__main__":
    run_training(
        train_config_path=Path("configs/train.yaml"),
        data_config_path=Path("configs/data.yaml"),
        model_config_path=Path("configs/model_vgg19.yaml"),
    )
