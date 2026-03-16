from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def write_dummy_image(path: Path, value: int = 127, size: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((size, size, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def make_minimal_full_df(path: Path) -> pd.DataFrame:
    rows = [
        {
            "ID": 0,
            "Patient Age": 65,
            "Patient Sex": "Female",
            "Left-Fundus": "0_left.jpg",
            "Right-Fundus": "0_right.jpg",
            "Left-Diagnostic Keywords": "normal fundus",
            "Right-Diagnostic Keywords": "normal fundus",
            "N": 1,
            "D": 0,
            "G": 0,
            "C": 0,
            "A": 0,
            "H": 0,
            "M": 0,
            "O": 0,
        },
        {
            "ID": 1,
            "Patient Age": 58,
            "Patient Sex": "Male",
            "Left-Fundus": "1_left.jpg",
            "Right-Fundus": "1_right.jpg",
            "Left-Diagnostic Keywords": "hypertension",
            "Right-Diagnostic Keywords": "diabetes",
            "N": 0,
            "D": 1,
            "G": 0,
            "C": 0,
            "A": 0,
            "H": 1,
            "M": 0,
            "O": 0,
        },
        {
            "ID": 1,
            "Patient Age": 58,
            "Patient Sex": "Male",
            "Left-Fundus": "1_left.jpg",
            "Right-Fundus": "1_right.jpg",
            "Left-Diagnostic Keywords": "hypertension",
            "Right-Diagnostic Keywords": "diabetes",
            "N": 0,
            "D": 1,
            "G": 0,
            "C": 0,
            "A": 0,
            "H": 1,
            "M": 0,
            "O": 0,
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df
