"""Los dos codificadores del ensamble, cargados juntos y con una sola huella.

Generalista: DINOv2 ViT-S/14 congelado + cabeza contrastiva.
Especialista: CNN (ResNet) afinada con tatuajes.

`embed` devuelve los dos vectores concatenados ``[generalista | especialista]``,
cada bloque L2-normalizado por su cuenta: es el formato de la galería del ensamble
(ver serving/ensamble.py).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def huella_archivo(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()[:16]


def huella_ensamble(generalista: Path, especialista: Path) -> str:
    """Identifica la pareja de pesos. Galería y calibración solo valen con la misma."""
    return f"gen:{huella_archivo(generalista)}+esp:{huella_archivo(especialista)}"


class Codificadores:
    def __init__(self, generalista: Path, especialista: Path, device: str | None = None):
        from src.embedding.backbones import CNNEmbedder
        from src.embedding.embedder import TattooEmbedder

        for ruta in (generalista, especialista):
            if not Path(ruta).exists():
                raise FileNotFoundError(
                    f"Falta {ruta}. Genere los artefactos del ensamble con "
                    f"python -m src.embedding.construir_ensamble.")
        self.generalista = TattooEmbedder(head_path=str(generalista), device=device)
        self.especialista = CNNEmbedder(especialista, device=device)
        self.dims = (int(self.generalista.dim), int(self.especialista.dim))
        self.dim = sum(self.dims)
        self.huella = huella_ensamble(generalista, especialista)
        self.nombres = (Path(generalista).name, Path(especialista).name)

    def embed(self, images) -> np.ndarray:
        single = isinstance(images, Image.Image)
        lista = [images] if single else list(images)
        a = np.atleast_2d(self.generalista.embed(lista))
        b = np.atleast_2d(self.especialista.embed(lista))
        z = np.concatenate([a, b], axis=1).astype("float32")
        return z[0] if single else z
