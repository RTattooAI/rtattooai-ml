"""Rutas centralizadas del proyecto RTattooAI-ML.

Todas las rutas se derivan de la raíz del repositorio (calculada a partir
de la ubicación de este archivo), de modo que los scripts funcionan sin
importar desde qué carpeta se ejecuten.
"""
from __future__ import annotations

from pathlib import Path

# src/shared/paths.py  ->  parents[0]=shared  parents[1]=src  parents[2]=raíz
ROOT = Path(__file__).resolve().parents[2]

# Datos
DATASETS = ROOT / "datasets"
RAW = DATASETS / "raw"               # cada fuente descargada sin tocar
UNIFIED = DATASETS / "unified"       # dataset fusionado final (formato YOLO)
REAL_TEST = DATASETS / "real_test"   # imágenes reales del contexto peruano

# Dataset unificado: estructura YOLO
UNIFIED_IMAGES = UNIFIED / "images"
UNIFIED_LABELS = UNIFIED / "labels"
UNIFIED_YAML = UNIFIED / "data.yaml"

# Modelos y reportes
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs"                 # salidas de entrenamiento de Ultralytics

# Config raíz del dataset
ROOT_DATA_YAML = ROOT / "data.yaml"

# Splits estándar
SPLITS = ("train", "val", "test")


def ensure_dirs(*paths: Path) -> None:
    """Crea los directorios indicados si no existen."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Diagnóstico rápido de rutas
    for name in (
        "ROOT", "DATASETS", "RAW", "UNIFIED", "UNIFIED_IMAGES",
        "UNIFIED_LABELS", "MODELS", "REPORTS", "REAL_TEST",
    ):
        print(f"{name:14s} = {globals()[name]}")
