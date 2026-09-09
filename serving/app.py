"""Servicio de inferencia (FastAPI) — interno, consumido por el backend ASP.NET Core.

Endpoints (API_CONTRACT.md, sección A):
  GET  /health
  GET  /info      (composición del modelo y de la galería — RF-16)
  POST /identify  (image, top_k opcional)
  POST /enroll    (image, band opcional — vacío = identidad sin banda asignada)

Arranca con:
    uvicorn serving.app:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from . import config
from .pipeline import TattooService
from .schemas import EnrollResponse, HealthResponse, IdentifyResponse, InfoResponse

app = FastAPI(title="RTattooAI — Servicio de inferencia", version="1.0.0")
service: TattooService | None = None


@app.on_event("startup")
def _load_service() -> None:
    global service
    service = TattooService()


def _read_image(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Imagen inválida: {exc}") from exc


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    return {"status": "ok", "model_version": config.MODEL_VERSION}


@app.get("/info", response_model=InfoResponse)
def info() -> dict:
    return service.info()


@app.post("/identify", response_model=IdentifyResponse)
async def identify(image: UploadFile = File(...), top_k: int = Form(config.DEFAULT_TOP_K)) -> dict:
    data = await image.read()
    img = _read_image(data)
    try:
        return service.identify(img, top_k=top_k)
    except OSError as exc:
        # Image.open() es perezoso: una imagen truncada/corrupta puede pasar la
        # validación inicial y fallar recién al decodificar los píxeles (convert/load).
        raise HTTPException(status_code=400, detail=f"Imagen inválida o corrupta: {exc}") from exc


@app.post("/enroll", response_model=EnrollResponse)
async def enroll(image: UploadFile = File(...), band: str = Form("")) -> dict:
    data = await image.read()
    img = _read_image(data)
    try:
        return service.enroll(img, band=band)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Imagen inválida o corrupta: {exc}") from exc
