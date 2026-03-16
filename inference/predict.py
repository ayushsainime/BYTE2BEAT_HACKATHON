from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from inference.predictor import Predictor
from utils.config import (
    load_cv_proxy_config,
    load_data_config,
    load_inference_config,
    load_model_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-sample inference")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--left", type=Path, required=True, help="Left eye image path")
    parser.add_argument("--right", type=Path, required=True, help="Right eye image path")
    parser.add_argument("--age", type=float, required=True, help="Patient age")
    parser.add_argument("--sex", type=str, required=True, help="Patient sex (Male/Female)")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--inference-config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--cv-proxy-config", type=Path, default=Path("configs/cv_proxy.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    predictor = Predictor(
        checkpoint_path=args.ckpt,
        model_config=load_model_config(args.model_config),
        data_config=load_data_config(args.data_config),
        inference_config=load_inference_config(args.inference_config),
        cv_proxy_config=load_cv_proxy_config(args.cv_proxy_config),
    )

    output = predictor.predict_single(
        left_image=args.left,
        right_image=args.right,
        age=args.age,
        sex=args.sex,
    )

    print(json.dumps(asdict(output), indent=2))


if __name__ == "__main__":
    main()
