from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from api.schemas import PredictResponse
from inference.predictor import Predictor
from utils.config import (
    load_api_config,
    load_cv_proxy_config,
    load_data_config,
    load_inference_config,
    load_model_config,
)
from utils.logging import get_logger

LOGGER = get_logger(__name__)
app = FastAPI(title="Multimodal Retinal CV Risk API", version="0.1.0")


@app.on_event("startup")
def startup_event() -> None:
    config_path = Path(os.getenv("API_CONFIG_PATH", "configs/api.yaml"))
    api_config = load_api_config(config_path)

    checkpoint_path = api_config.checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    app.state.predictor = Predictor(
        checkpoint_path=checkpoint_path,
        model_config=load_model_config(api_config.model_config_path),
        data_config=load_data_config(api_config.data_config_path),
        inference_config=load_inference_config(api_config.inference_config_path),
        cv_proxy_config=load_cv_proxy_config(api_config.cv_proxy_config_path),
    )
    LOGGER.info("Predictor loaded from %s", checkpoint_path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    left_image: UploadFile = File(...),
    right_image: UploadFile = File(...),
    age: float = Form(...),
    sex: str = Form(...),
) -> PredictResponse:
    predictor: Predictor = app.state.predictor

    try:
        left_bytes = await left_image.read()
        right_bytes = await right_image.read()
        output = predictor.predict_single(
            left_image=left_bytes,
            right_image=right_bytes,
            age=age,
            sex=sex,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return PredictResponse(
        labels=output.labels,
        probabilities=output.probabilities,
        cv_summary=output.cv_summary,
    )
