"""Servicio de inferencia FastAPI (detección + identificación de tatuajes).

Se asegura que la raíz del repo esté en sys.path ANTES de que cualquier
submódulo de `serving` importe paquetes de `src` (TattooEmbedder, paths, etc.),
sin importar desde qué directorio se arranque uvicorn.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
