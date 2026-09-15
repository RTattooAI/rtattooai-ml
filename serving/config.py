"""Configuración del servicio de inferencia (variables de entorno + match_config.json)."""
from __future__ import annotations

import os
from pathlib import Path

from src.shared import paths
from src.shared.env import load_env
from src.embedding.identify import load_config as load_match_config

load_env()

# Se sube al reentrenar la cabeza. La version viaja en cada consulta y queda
# grabada en el analisis: es lo que permite leer un resultado antiguo con las
# metricas que le corresponden y no con las del modelo cargado hoy (RF-16).
#
#   head-v1: cabeza entrenada con 2000 recortes, split completo (entrenaba tambien
#            con test, de modo que sus metricas salian optimistas).
#   head-v2: 6800 recortes del split train, 60 epocas, degradacion hasta 0.8.
# El detector se reentreno con negativos duros el 11/09/2026 (bitacora E-12): el
# archivo se llama igual, asi que si la version no lo dijera no habria forma de
# saber con cual se resolvio un analisis guardado.
MODEL_VERSION = os.environ.get(
    "MODEL_VERSION", "yolov8m-hardneg + dinov2-vits14 + head-v2")

PORT = int(os.environ.get("PORT", "8000"))

def _resolver_dispositivo() -> str:
    """Donde corre la CABEZA (torch). 'auto' usa la GPU si la hay.

    Estaba fijado a CPU en el codigo. Con la GPU disponible el embebedor DINOv2 a
    518 px pasa de aproximadamente un segundo a decenas de milisegundos, que es la
    diferencia entre que el analisis en vivo sirva o no.
    """
    pedido = os.environ.get("INFER_DEVICE", "auto").strip().lower()
    if pedido and pedido != "auto":
        return pedido
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def _resolver_dispositivo_detector(pesos) -> str:
    """Donde corre el DETECTOR (ONNX Runtime), que no tiene por que ser lo mismo.

    onnxruntime-gpu LISTA CUDAExecutionProvider aunque despues no pueda cargarlo.
    Paso el 11/09/2026: la version 1.30 exige CUDA 13 y la imagen trae la 12.8 que
    arrastra torch, de modo que el provider fallaba al cargarse, ORT se caia a CPU
    en silencio, ultralytics le pasaba tensores de GPU y cada peticion moria con
    "no data transfer registered". Un 500 en produccion por algo que la lista de
    providers daba por disponible.

    Tampoco basta con que crear la sesion no lance excepcion: pedir
    providers=["CUDAExecutionProvider"] y que el provider no cargue NO es un error
    para ORT, que registra el fallo en el log y devuelve una sesion que por dentro
    corre en CPU. La unica comprobacion que vale es preguntarle a la sesion ya
    creada que providers quedaron ACTIVOS.
    """
    pedido = os.environ.get("DETECTOR_DEVICE", "auto").strip().lower()
    if pedido and pedido != "auto":
        return pedido
    if _resolver_dispositivo() == "cpu":
        return "cpu"
    # Con pesos de PyTorch el detector corre sobre torch, igual que la cabeza: no hay
    # ONNX Runtime de por medio y vale el mismo dispositivo.
    if Path(str(pesos)).suffix.lower() == ".pt":
        return _resolver_dispositivo()
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return "cpu"
        opciones = ort.SessionOptions()
        opciones.log_severity_level = 3
        sesion = ort.InferenceSession(str(pesos), sess_options=opciones,
                                      providers=["CUDAExecutionProvider"])
        return "cuda:0" if "CUDAExecutionProvider" in sesion.get_providers() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


INFER_DEVICE = _resolver_dispositivo()

DETECTOR_WEIGHTS = Path(os.environ.get("DETECTOR_WEIGHTS_PATH", str(paths.MODELS / "yolov8m_best.onnx")))

# Se resuelve despues de DETECTOR_WEIGHTS porque la comprobacion carga esos pesos.
DETECTOR_DEVICE = _resolver_dispositivo_detector(DETECTOR_WEIGHTS)
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

# Zona intermedia: por debajo del umbral firme pero lo bastante alta como para que
# valga la pena que la mire una persona. El sistema no la afirma como coincidencia;
# la senala para revision. Sin esta franja, todo lo que no llega al umbral se
# presenta igual que un desconocido total, y se pierden los casos limite — que en
# un tatuaje gastado son justamente los mas frecuentes.
MATCH_REVIEW_THRESHOLD = float(
    os.environ.get("MATCH_REVIEW_THRESHOLD", _match_cfg.get("review_threshold", 0.55))
)

# Numero de identidades de la galeria con el que se calibraron los umbrales. Esta
# medido que el umbral necesario sube conforme crece el catalogo: al 5 % de falsos
# positivos hacen falta 0.665 con 50 identidades y 0.89 con 400. Si el catalogo
# crece mucho por encima de esta cifra y nadie recalibra, el sistema empieza a
# aceptar impostores sin que se note. Por eso se avisa en /info.
CALIBRATED_FOR_IDENTITIES = int(_match_cfg.get("calibrated_for_identities", 25))

DEFAULT_TOP_K = 5
