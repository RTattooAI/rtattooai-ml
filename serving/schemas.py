"""Modelos pydantic — forma EXACTA del contrato (API_CONTRACT.md, sección A)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str


class BBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class Match(BaseModel):
    catalog_id: str
    band: str
    similarity: float
    reference_image_url: Optional[str] = None


class Detection(BaseModel):
    bbox: BBox
    confidence: float
    verdict: Literal["match", "no_match"]
    matches: list[Match]


class IdentifyResponse(BaseModel):
    image_width: int
    image_height: int
    model_version: str
    processing_ms: int
    threshold: float
    margin: float
    detections: list[Detection]


class EnrollResponse(BaseModel):
    catalog_id: str
    band: str
