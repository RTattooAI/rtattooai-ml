"""Pipeline de inferencia: detección (YOLOv8m) + identificación.

Dos modos (config.MODO):

* **ensamble** (por defecto): dos codificadores — generalista (DINOv2 + cabeza) y
  especialista (CNN) — combinados sin pesos a mano, verificación por consenso y
  semáforo conformal con tasa de desconocidos aceptados ≤ α. La lógica de decisión
  está en serving/ensamble.py.
* **clasico**: el modelo anterior (DINOv2 + cabeza, umbrales fijos 0.55/0.70), que
  queda disponible para volver atrás con MODO_IDENTIFICACION=clasico.
"""
from __future__ import annotations

import hashlib
import shutil
import threading
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.embedding.degrade import degrade

from . import config
from .ensamble import Calibracion, calibrar, decidir, indices_identidad, puntajes_por_modelo
from .gallery_store import GalleryStore

DEGRADE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]  # mismo estándar que robustness_test.py

# Giros que se enrolan además de las vistas degradadas. Medido con tatuajes reales:
# sube el acierto entre 4 y 7 puntos en todos los codificadores, sin reentrenar
# (informe de pruebas, sección 2.1). Una foto de teléfono puede llegar en cualquier
# orientación.
ROTACIONES = (90, 180, 270)


def vistas_de_enrolamiento(rgb: Image.Image, rng: np.random.RandomState, rotar: bool) -> list[Image.Image]:
    vistas = [rgb] + [degrade(rgb, level, rng) for level in DEGRADE_LEVELS if level > 0]
    if rotar:
        vistas += [rgb.rotate(a, expand=True) for a in ROTACIONES]
    return vistas


@dataclass(frozen=True)
class EstadoGaleria:
    """Instantánea inmutable de la galería y su calibración.

    Se reemplaza entera al enrolar o quitar, así una consulta concurrente nunca ve
    una galería nueva con una calibración vieja.
    """
    embeddings: np.ndarray
    codigos: np.ndarray
    ids: list[str]
    bandas: list[str]
    calibracion: Calibracion | None


class TattooService:
    def __init__(self):
        from ultralytics import YOLO
        self.detector = YOLO(str(config.DETECTOR_WEIGHTS))
        self.modo = config.MODO
        self._lock_estado = threading.Lock()
        if self.modo == "ensamble":
            self._init_ensamble()
        else:
            self._init_clasico()

    # ============================================================ inicialización
    def _init_clasico(self) -> None:
        from src.embedding.embedder import TattooEmbedder
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

    def _init_ensamble(self) -> None:
        from .codificadores import Codificadores
        self.codificadores = Codificadores(config.GENERALISTA_PATH, config.ESPECIALISTA_PATH,
                                           device=config.INFER_DEVICE)
        # Primera vez: la galería persistente se siembra con la base que trae la imagen.
        destino = config.GALERIA_ENSAMBLE_PATH
        if not destino.exists() and config.GALERIA_INICIAL_PATH.exists():
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config.GALERIA_INICIAL_PATH, destino)
        self.gallery = GalleryStore(
            local_path=destino,
            dim=self.codificadores.dim,
            azure_connection_string=config.AZURE_STORAGE_CONNECTION_STRING,
            container=config.GALLERY_CONTAINER,
            blob_name=config.GALERIA_ENSAMBLE_BLOB,
        )
        self.gallery.load()
        if self.gallery.embeddings.shape[0] == 0:
            self.gallery.head_fingerprint = self.codificadores.huella

        datos = np.load(config.CALIBRACION_PATH, allow_pickle=True)
        huella_cal = str(datos["huella"]) if "huella" in datos else None
        if huella_cal != self.codificadores.huella:
            # Sin esto el semáforo prometería una tasa de error calculada con otros
            # modelos: se prefiere no arrancar a arrancar mintiendo.
            raise RuntimeError(
                f"La calibración ({config.CALIBRACION_PATH.name}) se hizo con {huella_cal} y los "
                f"codificadores cargados son {self.codificadores.huella}. Regenere los artefactos "
                f"con python -m src.embedding.construir_ensamble.")
        self.desconocidos = datos["embeddings"].astype("float32")
        self.margin = 0.0
        self.gallery_warning = self._check_gallery_matches_head()
        self._refrescar_estado()

    def _refrescar_estado(self) -> None:
        """Recalcula índices y la línea conformal para la galería actual."""
        G = self.gallery.embeddings.copy()
        catalog = list(self.gallery.catalog_ids)
        codigos, ids = indices_identidad(catalog)
        banda_de: dict[str, str] = {}
        for cid, banda in zip(catalog, self.gallery.bands):
            banda_de.setdefault(cid, banda)
        cal = calibrar(self.desconocidos, G, catalog, self.codificadores.dims, config.ALFA)
        estado = EstadoGaleria(G, codigos, ids, [banda_de[c] for c in ids], cal)
        with self._lock_estado:
            self._estado = estado
        # Umbrales en la escala que se muestra (0.70 = línea conformal), que es la que
        # guardan el backend y la pantalla. La línea en la escala interna va en /info.
        from .ensamble import MOSTRADO_EN_TAU, MOSTRADO_REVISION
        self.threshold, self.review_threshold = MOSTRADO_EN_TAU, MOSTRADO_REVISION
        self.tau = cal.tau if cal else float("inf")

    # ============================================================ huellas y avisos
    def _huella_modelo(self) -> str:
        if self.modo == "ensamble":
            return self.codificadores.huella
        return self._head_fingerprint()

    def _check_gallery_matches_head(self) -> str | None:
        """La galeria y el modelo tienen que ser la misma pareja.

        Los vectores de la galeria solo significan algo en el espacio que produce
        un modelo concreto. Con otro el sistema no falla: sigue devolviendo un mejor
        candidato y un porcentaje, calculados sobre ruido. Un error que se presenta
        como respuesta normal es peor que uno que se ve, asi que se avisa en el
        arranque y en /info.
        """
        actual = self._huella_modelo()
        esperada = getattr(self.gallery, "head_fingerprint", None)

        if self.gallery.embeddings.shape[0] == 0:
            return None
        if esperada is None:
            return ("La galeria no declara con que modelo se construyo: no se puede "
                    "verificar que corresponda al modelo cargado. Reconstruyala.")
        if esperada != actual:
            return (f"La galeria se construyo con {esperada} y el modelo cargado es "
                    f"{actual}. Los resultados NO son validos: reconstruya la galeria.")
        return None

    def _check_calibration_still_valid(self) -> str | None:
        """Modo clasico: el umbral fijo se calibro para un tamano de galeria concreto.

        En modo ensamble no aplica: la linea conformal se recalcula sola cada vez que
        cambia la galeria.
        """
        if self.modo == "ensamble":
            cal = self._estado.calibracion
            if cal is not None and not np.isfinite(cal.tau):
                return (f"Con {cal.n_desconocidos} desconocidos de calibracion no se puede garantizar "
                        f"alfa={cal.alfa:g}: ningun tatuaje se acepta como coincidencia.")
            return None
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

    # ============================================================ detección
    def detect(self, image: Image.Image, conf: float = config.DET_CONF,
               iou: float = config.DET_IOU):
        """[(bbox_xyxy, crop_PIL, det_conf)] de los tatuajes detectados."""
        rgb = image.convert("RGB")
        res = self.detector.predict(rgb, conf=conf, iou=iou, imgsz=640, verbose=False,
                                    device=config.DETECTOR_DEVICE)[0]
        out = []
        if res.boxes is not None:
            for b in res.boxes:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                if x2 > x1 and y2 > y1:
                    out.append(((x1, y1, x2, y2), rgb.crop((x1, y1, x2, y2)), float(b.conf[0])))
        return out

    # ============================================================ identificación
    def match_crop(self, crop: Image.Image, top_k: int):
        """Modo clasico. Top-K de la galería para un recorte. Devuelve (matches, verdict)."""
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

    def _identificar_recortes_ensamble(self, recortes: list[Image.Image], top_k: int) -> list[dict]:
        """Modo ensamble: los recortes de una imagen se embeben juntos, en un lote."""
        estado = self._estado
        if not recortes:
            return []
        if estado.calibracion is None or len(estado.ids) == 0:
            return [{"verdict": "no_match", "consensus": None, "matches": []} for _ in recortes]
        Q = self.codificadores.embed(recortes)
        SA, SB = puntajes_por_modelo(Q, estado.embeddings, estado.codigos, len(estado.ids),
                                     self.codificadores.dims)
        salida = []
        for sa, sb in zip(SA, SB):
            d = decidir(sa, sb, estado.calibracion)
            matches = [{
                "catalog_id": estado.ids[k],
                "band": estado.bandas[k],
                "similarity": float(d.similitud[k]),
                "cosine": float(d.coseno[k]),
                "score": float(d.fusionado[k]),
                "candidate": bool(d.fusionado[k] >= estado.calibracion.tau),
                "reference_image_url": None,
            } for k in d.orden[:top_k]]
            salida.append({"verdict": d.verdict, "consensus": d.consenso, "matches": matches,
                           # Trazabilidad: lo que propuso cada codificador por separado.
                           "top1_generalista": estado.ids[int(np.argmax(sa))],
                           "top1_especialista": estado.ids[int(np.argmax(sb))]})
        return salida

    def identify(self, image: Image.Image, top_k: int = config.DEFAULT_TOP_K) -> dict:
        t0 = time.perf_counter()
        width, height = image.size
        detectados = self.detect(image)
        detections = []
        if self.modo == "ensamble":
            resultados = self._identificar_recortes_ensamble([c for _, c, _ in detectados], top_k)
            for ((x1, y1, x2, y2), _, det_conf), r in zip(detectados, resultados):
                detections.append({"bbox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                                   "confidence": det_conf, **r})
        else:
            for (x1, y1, x2, y2), crop, det_conf in detectados:
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
            "mode": self.modo,
            "alpha": config.ALFA if self.modo == "ensamble" else None,
            "detections": detections,
        }

    # ============================================================ info
    def info(self) -> dict:
        """Composicion exacta del modelo en ejecucion (RF-16).

        El backend la guarda junto a las metricas de cada version para que un
        resultado ya emitido siempre se pueda atribuir a la configuracion que lo
        produjo, incluso despues de reentrenar.
        """
        comun = {
            "model_version": config.MODEL_VERSION,
            "detector": config.DETECTOR_WEIGHTS.name,
            "threshold": self.threshold,
            "margin": self.margin,
            "review_threshold": self.review_threshold,
            "gallery_warning": self.gallery_warning,
            "calibration_warning": self._check_calibration_still_valid(),
            "det_conf": config.DET_CONF,
            "det_iou": config.DET_IOU,
            "gallery_size": int(self.gallery.embeddings.shape[0]),
            "gallery_identities": len(set(self.gallery.catalog_ids)),
            "gallery_bands": len({b for b in self.gallery.bands if b}),
            "mode": self.modo,
        }
        if self.modo == "ensamble":
            cal = self._estado.calibracion
            gen, esp = self.codificadores.nombres
            return {
                **comun,
                "embedder": "ensamble: dinov2-vits14 + cabeza | " + self.codificadores.especialista.nombre,
                "head": gen,
                "specialist": esp,
                "embedding_dim": int(self.codificadores.dim),
                "calibrated_for_identities": cal.n_identidades if cal else 0,
                "alpha": config.ALFA,
                "tau": float(cal.tau) if cal and np.isfinite(cal.tau) else None,
                "calibration_unknowns": cal.n_desconocidos if cal else 0,
                "model_fingerprint": self.codificadores.huella,
            }
        return {
            **comun,
            "embedder": self.embedder.model.default_cfg.get("architecture", "dinov2-vits14")
            if hasattr(self.embedder.model, "default_cfg") else "dinov2-vits14",
            "head": config.HEAD_PATH.name if self.embedder.head is not None else None,
            "embedding_dim": int(self.embedder.dim),
            "calibrated_for_identities": config.CALIBRATED_FOR_IDENTITIES,
        }

    # ============================================================ galería
    def enroll(self, image: Image.Image, band: str, catalog_id: str | None = None,
               image_sha256: str = "") -> dict:
        """Genera vistas, embebe, agrega a la galería persistente y guarda.

        `catalog_id` lo elige el backend, que es quien conoce los códigos del catálogo;
        si ya tiene vectores, la galería lanza `CatalogIdOcupado` sin tocar nada.
        """
        rgb = image.convert("RGB")
        rng = np.random.RandomState()
        ensamble = self.modo == "ensamble"
        views = vistas_de_enrolamiento(rgb, rng, rotar=ensamble)
        embeddings = (self.codificadores if ensamble else self.embedder).embed(views)
        elegido = self.gallery.enroll_views(embeddings, band=band, catalog_id=catalog_id,
                                            image_sha256=image_sha256)
        if ensamble:
            if not self.gallery.head_fingerprint:
                self.gallery.head_fingerprint = self.codificadores.huella
            self._refrescar_estado()
        return {"catalog_id": elegido, "band": band}

    def remove(self, catalog_id: str) -> int:
        quitadas = self.gallery.remove(catalog_id)
        if quitadas and self.modo == "ensamble":
            self._refrescar_estado()
        return quitadas
