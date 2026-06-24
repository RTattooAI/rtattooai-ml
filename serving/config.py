"""Configuración del servicio de inferencia (variables de entorno + match_config.json)."""
from __future__ import annotations

import os
from pathlib import Path

from src.shared import paths
from src.shared.env import load_env
from src.embedding.identify import load_config as load_match_config

load_env()

MODEL_VERSION = "yolov8m + dinov2-vits14 + head-v1"

PORT = int(os.environ.get("PORT", "8000"))

DETECTOR_WEIGHTS = Path(os.environ.get("DETECTOR_WEIGHTS_PATH", str(paths.MODELS / "yolov8m_best.onnx")))
HEAD_PATH = Path(os.environ.get("HEAD_PATH", str(paths.MODELS / "embed_head.pt")))
GALLERY_LOCAL_PATH = Path(os.environ.get("GALLERY_LOCAL_PATH", str(paths.MODELS / "gallery_dinov2.npz")))

# Azure Blob (galería persistente, opcional). Si no hay connection string,
# se usa el archivo local de GALLERY_LOCAL_PATH como única fuente de verdad.
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
GALLERY_CONTAINER = os.environ.get("GALLERY_CONTAINER", "models")
GALLERY_BLOB_NAME = os.environ.get("GALLERY_BLOB_NAME", "gallery_dinov2.npz")

_match_cfg = load_match_config()
DET_CONF = float(os.environ.get("DET_CONF", _match_cfg["det_conf"]))
DET_IOU = float(os.environ.get("DET_IOU", _match_cfg["det_iou"]))
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", _match_cfg["match_threshold"]))
MATCH_MARGIN = float(os.environ.get("MATCH_MARGIN", _match_cfg["match_margin"]))

DEFAULT_TOP_K = 5
