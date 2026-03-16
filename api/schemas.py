from __future__ import annotations

from pydantic import BaseModel, Field


class CVSummary(BaseModel):
    hypertension_proxy: float = Field(ge=0.0, le=1.0)
    diabetes_proxy: float = Field(ge=0.0, le=1.0)
    atherosclerotic_proxy: float = Field(ge=0.0, le=1.0)
    overall_cv_proxy: float = Field(ge=0.0, le=1.0)
    risk_band: str


class PredictResponse(BaseModel):
    labels: dict[str, int]
    probabilities: dict[str, float]
    cv_summary: CVSummary
