"""Servicio de inferencia (FastAPI) — interno, consumido por el backend ASP.NET Core.

Endpoints (API_CONTRACT.md, secciones A e I):
  GET    /health
  GET    /info                  (composición del modelo y de la galería — RF-16)
  POST   /identify              (image, top_k opcional)
  POST   /enroll                (image, band opcional, catalog_id opcional)
  GET    /gallery               (códigos, vistas, huella de imagen y vectores compartidos)
  DELETE /gallery/{catalog_id}  (quita las vistas de un código)

Arranca con:
    uvicorn serving.app:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import hashlib
import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from . import config
from .gallery_store import CatalogIdOcupado
from .pipeline import TattooService
from .schemas import (EnrollResponse, GalleryResponse, HealthResponse, IdentifyResponse,
                      InfoResponse, RemoveResponse)

app = FastAPI(title="RTattooAI — Servicio de inferencia", version="1.0.0")
service: TattooService | None = None


@app.on_event("startup")
def _load_service() -> None:
    global service
    service = TattooService()
    if service.gallery_warning:
        # Se registra bien visible: es el fallo que de otro modo pasaria inadvertido
        # porque el servicio sigue respondiendo con normalidad.
        print("=" * 78, flush=True)
        print("AVISO DE GALERIA:", service.gallery_warning, flush=True)
        print("=" * 78, flush=True)


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
async def enroll(image: UploadFile = File(...), band: str = Form(""),
                 catalog_id: str = Form("")) -> dict:
    data = await image.read()
    img = _read_image(data)
    # La huella se calcula sobre los bytes recibidos, que son los mismos que el backend
    # guarda y resume en su ficha: así ambas partes pueden comprobar que hablan de la
    # misma imagen.
    huella = hashlib.sha256(data).hexdigest()
    try:
        return service.enroll(img, band=band, catalog_id=catalog_id.strip() or None, image_sha256=huella)
    except CatalogIdOcupado as exc:
        raise HTTPException(status_code=409,
                            detail=f"El código {exc} ya tiene vectores en la galería.") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Imagen inválida o corrupta: {exc}") from exc


@app.get("/gallery", response_model=GalleryResponse)
def gallery() -> dict:
    return {"entries": service.gallery.entries(), "shared_vectors": service.gallery.shared_vectors()}


@app.delete("/gallery/{catalog_id}", response_model=RemoveResponse)
def remove_from_gallery(catalog_id: str) -> dict:
    quitadas = service.remove(catalog_id)
    if quitadas == 0:
        raise HTTPException(status_code=404, detail=f"La galería no tiene vectores con el código {catalog_id}.")
    return {"catalog_id": catalog_id, "removed_views": quitadas}
