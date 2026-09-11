"""Codificadores alternativos, con la misma interfaz que `TattooEmbedder`.

Existen para poder medir el sistema actual contra otras opciones sin tocar el
arnés de evaluación: `enroll_recall_test` solo necesita un objeto con `.embed()`,
`.dim` y `.device`, así que cualquier codificador que cumpla eso entra en la
comparación con el mismo protocolo, la misma galería y las mismas semillas.

Se resuelven con una cadena corta:

    dinov2            → el sistema actual (DINOv2 congelado + cabeza afinada)
    dinov2:sin-head   → el backbone desnudo, para ver cuánto aporta la cabeza
    cnn:<ruta.pt>     → una ResNet-18 entrenada por nosotros (ver train_cnn.py)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.shared import paths
from src.embedding.embedder import TattooEmbedder, composite_on

MEDIA = np.array([0.485, 0.456, 0.406], dtype="float32")
DESV = np.array([0.229, 0.224, 0.225], dtype="float32")


class CNNEmbedder:
    """ResNet-18 entrenada con SupCon. Devuelve embeddings L2-normalizados."""

    def __init__(self, ckpt_path: str | Path, device=None):
        import torch

        from src.embedding.train_cnn import construir_red

        self.torch = torch
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        self.meta = ckpt.get("entrenamiento", {})
        self.tam = ckpt.get("tam", 224)
        self.dim = ckpt.get("out", 128)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Los pesos vienen del checkpoint; el argumento solo elige la arquitectura,
        # así que da igual con cuáles se inicializara.
        red = construir_red("ninguno", torch)
        red.load_state_dict(ckpt["state_dict"])
        self.red = red.eval().to(device)

        # Marcador para que el resto del arnés pueda informar qué se está midiendo.
        self.head = True
        self.nombre = f"resnet18({self.meta.get('pesos_iniciales', '?')})"

    def embed(self, images, bg=(255, 255, 255), chunk: int = 128) -> np.ndarray:
        single = isinstance(images, Image.Image)
        if single:
            images = [images]

        media = self.torch.tensor(MEDIA, device=self.device).view(1, 3, 1, 1)
        desv = self.torch.tensor(DESV, device=self.device).view(1, 3, 1, 1)

        salidas = []
        for i in range(0, len(images), chunk):
            trozo = images[i:i + chunk]
            arr = np.stack([
                np.asarray(composite_on(im, bg).resize((self.tam, self.tam), Image.BICUBIC))
                for im in trozo
            ]).astype("uint8")

            x = self.torch.from_numpy(arr).to(self.device)
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - media) / desv
            with self.torch.no_grad():
                z = self.red(x)
            z = self.torch.nn.functional.normalize(z.float(), dim=1)
            salidas.append(z.cpu().numpy().astype("float32"))

        arr = np.concatenate(salidas, axis=0)
        return arr[0] if single else arr


def resolver(spec: str | None):
    """Convierte la cadena del argumento --backbone en un codificador."""
    if not spec or spec == "dinov2":
        ruta = paths.MODELS / "embed_head.pt"
        return TattooEmbedder(head_path=str(ruta) if ruta.exists() else None)

    if spec == "dinov2:sin-head":
        return TattooEmbedder()

    if spec.startswith("cnn:"):
        ruta = Path(spec[4:])
        if not ruta.exists():
            raise SystemExit(f"No existe el checkpoint {ruta}. Entrénelo con train_cnn.py.")
        return CNNEmbedder(ruta)

    raise SystemExit(f"Codificador desconocido: {spec!r}. "
                     "Use dinov2, dinov2:sin-head o cnn:<ruta.pt>.")


def describir(emb) -> str:
    if isinstance(emb, CNNEmbedder):
        m = emb.meta
        return (f"{emb.nombre} dim={emb.dim} "
                f"({m.get('parametros', 0)/1e6:.1f} M parámetros, "
                f"{m.get('identidades', '?')} identidades, {m.get('epocas', '?')} épocas)")
    return (f"dinov2-vits14 dim={emb.dim} "
            f"cabeza={'sí' if emb.head is not None else 'no'}")
