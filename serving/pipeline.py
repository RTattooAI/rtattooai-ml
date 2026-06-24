"""Pipeline de inferencia: detección (YOLOv8m ONNX) + identificación (DINOv2 +
cabeza afinada + galería). Misma lógica que `src.embedding.identify.TattooIdentifier`
(coseno máximo por identidad, veredicto por umbral+margen), adaptada al esquema
de galería con catalog_id/band y reusando el mismo `TattooEmbedder`/`degrade`.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.embedding.embedder import TattooEmbedder
from src.embedding.degrade import degrade

from . import config
from .gallery_store import GalleryStore

DEGRADE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]  # mismo estándar que robustness_test.py


class TattooService:
    def __init__(self):
        from ultralytics import YOLO
        self.detector = YOLO(str(config.DETECTOR_WEIGHTS))
        self.embedder = TattooEmbedder(
            head_path=str(config.HEAD_PATH) if config.HEAD_PATH.exists() else None,
            device="cpu")
        self.gallery = GalleryStore(
            local_path=config.GALLERY_LOCAL_PATH,
            dim=self.embedder.dim,
            azure_connection_string=config.AZURE_STORAGE_CONNECTION_STRING,
            container=config.GALLERY_CONTAINER,
            blob_name=config.GALLERY_BLOB_NAME,
        )
        self.gallery.load()
        self.threshold = config.MATCH_THRESHOLD
        self.margin = config.MATCH_MARGIN

    def detect(self, image: Image.Image, conf: float = config.DET_CONF,
               iou: float = config.DET_IOU):
        """[(bbox_xyxy, crop_PIL, det_conf)] de los tatuajes detectados."""
        rgb = image.convert("RGB")
        res = self.detector.predict(rgb, conf=conf, iou=iou, imgsz=640, verbose=False,
                                    device="cpu")[0]
        out = []
        if res.boxes is not None:
            for b in res.boxes:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                if x2 > x1 and y2 > y1:
                    out.append(((x1, y1, x2, y2), rgb.crop((x1, y1, x2, y2)), float(b.conf[0])))
        return out

    def match_crop(self, crop: Image.Image, top_k: int):
        """Top-K de la galería para un recorte. Devuelve (matches, verdict)."""
        e = self.embedder.embed(crop)
        ranked = self.gallery.search(e)
        matches = ranked[:top_k]
        top1 = matches[0][2] if matches else -1.0
        top2 = matches[1][2] if len(matches) > 1 else -1.0
        is_match = bool(matches) and top1 >= self.threshold and (top1 - top2) >= self.margin
        verdict = "match" if is_match else "no_match"
        return matches, verdict

    def identify(self, image: Image.Image, top_k: int = config.DEFAULT_TOP_K) -> dict:
        t0 = time.perf_counter()
        width, height = image.size
        detections = []
        for (x1, y1, x2, y2), crop, det_conf in self.detect(image):
            matches, verdict = self.match_crop(crop, top_k)
            detections.append({
                "bbox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                "confidence": det_conf,
                "verdict": verdict,
                "matches": [
                    {"catalog_id": cid, "band": band, "similarity": sim, "reference_image_url": None}
                    for cid, band, sim in matches
                ],
            })
        processing_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "image_width": width,
            "image_height": height,
            "model_version": config.MODEL_VERSION,
            "processing_ms": processing_ms,
            "threshold": self.threshold,
            "margin": self.margin,
            "detections": detections,
        }

    def enroll(self, image: Image.Image, band: str) -> dict:
        """Genera vistas degradadas, embebe, agrega a la galería persistente y guarda."""
        rgb = image.convert("RGB")
        rng = np.random.RandomState()
        views = [rgb] + [degrade(rgb, level, rng) for level in DEGRADE_LEVELS if level > 0]
        embeddings = self.embedder.embed(views)  # (V, dim) L2-norm
        catalog_id = self.gallery.enroll_views(embeddings, band=band)
        return {"catalog_id": catalog_id, "band": band}
