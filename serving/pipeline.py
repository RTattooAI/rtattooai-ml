"""Pipeline de inferencia: detección (YOLOv8m ONNX) + identificación (DINOv2 +
cabeza afinada + galería). Misma lógica que `src.embedding.identify.TattooIdentifier`
(coseno máximo por identidad, veredicto por umbral+margen), adaptada al esquema
de galería con catalog_id/band y reusando el mismo `TattooEmbedder`/`degrade`.
"""
from __future__ import annotations

import hashlib
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
            device=config.INFER_DEVICE)
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
        self.review_threshold = config.MATCH_REVIEW_THRESHOLD
        self.gallery_warning = self._check_gallery_matches_head()

    def _check_gallery_matches_head(self) -> str | None:
        """La galeria y la cabeza tienen que ser la misma pareja.

        Los vectores de la galeria solo significan algo en el espacio que produce
        una cabeza concreta. Con otra cabeza el sistema no falla: sigue devolviendo
        un mejor candidato y un porcentaje, calculados sobre ruido. Un error que se
        presenta como respuesta normal es peor que uno que se ve, asi que se avisa
        en el arranque y en /info.
        """
        actual = self._head_fingerprint()
        esperada = getattr(self.gallery, "head_fingerprint", None)

        if self.gallery.embeddings.shape[0] == 0:
            return None
        if esperada is None:
            return ("La galeria no declara con que cabeza se construyo: no se puede "
                    "verificar que corresponda al modelo cargado. Reconstruyala con "
                    "src.embedding.rebuild_gallery.")
        if esperada != actual:
            return (f"La galeria se construyo con la cabeza {esperada} y la cargada es "
                    f"{actual}. Los resultados NO son validos: reconstruya la galeria "
                    f"con src.embedding.rebuild_gallery.")
        return None

    def _check_calibration_still_valid(self) -> str | None:
        """El umbral se calibro para un tamano de galeria concreto.

        Con mas identidades hay mas ocasiones de parecerse a algo por azar, de modo
        que el mismo umbral deja pasar mas impostores. El aviso es lo que evita que
        eso ocurra en silencio.
        """
        identidades = len(set(self.gallery.catalog_ids))
        limite = config.CALIBRATED_FOR_IDENTITIES * 2

        if identidades > limite:
            return (f"La galeria tiene {identidades} identidades y los umbrales se "
                    f"calibraron para {config.CALIBRATED_FOR_IDENTITIES}. Recalibre con "
                    f"src.embedding.enroll_recall_test: con el umbral actual el sistema "
                    f"esta aceptando mas impostores de los previstos.")
        return None

    @staticmethod
    def _head_fingerprint() -> str:
        ruta = config.HEAD_PATH
        if not ruta.exists():
            return "sin-cabeza"
        return hashlib.sha256(ruta.read_bytes()).hexdigest()[:16]

    def detect(self, image: Image.Image, conf: float = config.DET_CONF,
               iou: float = config.DET_IOU):
        """[(bbox_xyxy, crop_PIL, det_conf)] de los tatuajes detectados."""
        rgb = image.convert("RGB")
        res = self.detector.predict(rgb, conf=conf, iou=iou, imgsz=640, verbose=False,
                                    device=config.INFER_DEVICE)[0]
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

        # Tres niveles en vez de dos. La franja intermedia existe porque un tatuaje
        # gastado cae ahi con frecuencia: presentarlo como "sin coincidencia" pierde
        # informacion util, y presentarlo como coincidencia afirma algo que el
        # sistema no puede sostener. Se marca para que lo mire una persona.
        if matches and top1 >= self.threshold and (top1 - top2) >= self.margin:
            verdict = "match"
        elif matches and top1 >= self.review_threshold:
            verdict = "review"
        else:
            verdict = "no_match"

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
            "review_threshold": self.review_threshold,
            "detections": detections,
        }

    def info(self) -> dict:
        """Composicion exacta del modelo en ejecucion (RF-16).

        El backend la guarda junto a las metricas de cada version para que un
        resultado ya emitido siempre se pueda atribuir a la configuracion que lo
        produjo, incluso despues de reentrenar.
        """
        return {
            "model_version": config.MODEL_VERSION,
            "detector": config.DETECTOR_WEIGHTS.name,
            "embedder": self.embedder.model.default_cfg.get("architecture", "dinov2-vits14")
            if hasattr(self.embedder.model, "default_cfg") else "dinov2-vits14",
            "head": config.HEAD_PATH.name if self.embedder.head is not None else None,
            "embedding_dim": int(self.embedder.dim),
            "threshold": self.threshold,
            "margin": self.margin,
            "review_threshold": self.review_threshold,
            "gallery_warning": self.gallery_warning,
            "calibration_warning": self._check_calibration_still_valid(),
            "calibrated_for_identities": config.CALIBRATED_FOR_IDENTITIES,
            "det_conf": config.DET_CONF,
            "det_iou": config.DET_IOU,
            "gallery_size": int(self.gallery.embeddings.shape[0]),
            "gallery_identities": len(set(self.gallery.catalog_ids)),
            "gallery_bands": len({b for b in self.gallery.bands if b}),
        }

    def enroll(self, image: Image.Image, band: str) -> dict:
        """Genera vistas degradadas, embebe, agrega a la galería persistente y guarda."""
        rgb = image.convert("RGB")
        rng = np.random.RandomState()
        views = [rgb] + [degrade(rgb, level, rng) for level in DEGRADE_LEVELS if level > 0]
        embeddings = self.embedder.embed(views)  # (V, dim) L2-norm
        catalog_id = self.gallery.enroll_views(embeddings, band=band)
        return {"catalog_id": catalog_id, "band": band}
