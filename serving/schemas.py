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
    # Modo ensamble (opcionales: el modo clasico no los envia).
    mode: Optional[str] = None
    specialist: Optional[str] = None
    alpha: Optional[float] = None
    calibration_unknowns: Optional[int] = None
    # Linea conformal en la escala interna (puntaje fusionado normalizado).
    tau: Optional[float] = None
    model_fingerprint: Optional[str] = None


class BBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class Match(BaseModel):
    catalog_id: str
    band: str
    # Coseno (medio de los dos modelos en modo ensamble): lo que se muestra en pantalla.
    similarity: float
    reference_image_url: Optional[str] = None
    # Modo ensamble: puntaje normalizado con el que se decide y si pasa la linea conformal.
    score: Optional[float] = None
    candidate: Optional[bool] = None
    # Modo ensamble: coseno medio real de los dos modelos (similarity es el calibrado).
    cosine: Optional[float] = None


class Detection(BaseModel):
    bbox: BBox
    confidence: float
    # "review": por encima del umbral de revision pero por debajo del firme.
    verdict: Literal["match", "review", "no_match"]
    matches: list[Match]
    # Modo ensamble: si los dos codificadores eligen el mismo primer candidato.
    consensus: Optional[bool] = None
    # Modo ensamble: primer candidato de cada codificador por separado (trazabilidad).
    top1_generalista: Optional[str] = None
    top1_especialista: Optional[str] = None


class IdentifyResponse(BaseModel):
    image_width: int
    image_height: int
    model_version: str
    processing_ms: int
    threshold: float
    margin: float
    review_threshold: float
    mode: Optional[str] = None
    alpha: Optional[float] = None
    detections: list[Detection]


class EnrollResponse(BaseModel):
    catalog_id: str
    band: str


class GalleryEntry(BaseModel):
    catalog_id: str
    views: int
    # Huella SHA-256 de la imagen enrolada; null en vistas anteriores a que existiera.
    image_sha256: Optional[str] = None


class GalleryResponse(BaseModel):
    entries: list[GalleryEntry]
    # Grupos de códigos distintos que comparten exactamente un mismo vector.
    shared_vectors: list[list[str]]


class RemoveResponse(BaseModel):
    catalog_id: str
    removed_views: int
