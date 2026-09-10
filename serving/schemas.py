"""Modelos pydantic — forma EXACTA del contrato (API_CONTRACT.md, sección A)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str


class InfoResponse(BaseModel):
    """Composicion del modelo en ejecucion (RF-16)."""
    model_version: str
    detector: str
    embedder: str
    head: Optional[str] = None
    embedding_dim: int
    threshold: float
    margin: float
    review_threshold: float
    # Aviso si la galeria no corresponde a la cabeza cargada; null si todo cuadra.
    gallery_warning: Optional[str] = None
    # Aviso si la galeria crecio muy por encima del tamano con el que se calibro.
    calibration_warning: Optional[str] = None
    calibrated_for_identities: int = 0
    det_conf: float
    det_iou: float
    gallery_size: int
    gallery_identities: int
    gallery_bands: int


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
    # "review": por encima del umbral de revision pero por debajo del firme.
    verdict: Literal["match", "review", "no_match"]
    matches: list[Match]


class IdentifyResponse(BaseModel):
    image_width: int
    image_height: int
    model_version: str
    processing_ms: int
    threshold: float
    margin: float
    review_threshold: float
    detections: list[Detection]


class EnrollResponse(BaseModel):
    catalog_id: str
    band: str
