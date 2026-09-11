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


class FusionEmbedder:
    """Dos codificadores a la vez: la similitud es la media ponderada de sus cosenos.

    Cada parte devuelve un vector unitario. Si se concatenan escalados por √w,

        [√w · a₁, √(1−w) · a₂] · [√w · b₁, √(1−w) · b₂] = w·cos₁ + (1−w)·cos₂

    y el resultado sigue siendo unitario, así que el resto del sistema — galería,
    umbral, índice — no tiene que saber que hay dos modelos detrás.

    Por qué puede valer la pena: un modelo generalista congelado y uno entrenado
    sobre el dominio fallan por motivos distintos. El primero no aprendió qué
    importa en un tatuaje; el segundo puede haber aprendido nuestro simulador de
    desgaste más que el desgaste. Donde uno se equivoca, el otro no tiene por qué.
    """

    def __init__(self, a, b, peso_a: float = 0.5):
        if not 0.0 < peso_a < 1.0:
            raise SystemExit(f"El peso de la fusión debe estar entre 0 y 1: {peso_a}")
        self.a, self.b, self.peso_a = a, b, peso_a
        self.dim = a.dim + b.dim
        self.device = getattr(a, "device", "cpu")
        self.head = True
        self.nombre = f"fusion({describir(a)} ×{peso_a:g} + {describir(b)} ×{1 - peso_a:g})"

    def embed(self, images, bg=(255, 255, 255)) -> np.ndarray:
        single = isinstance(images, Image.Image)
        ea = np.atleast_2d(self.a.embed(images, bg))
        eb = np.atleast_2d(self.b.embed(images, bg))
        z = np.concatenate([np.sqrt(self.peso_a) * ea,
                            np.sqrt(1.0 - self.peso_a) * eb], axis=1).astype("float32")
        return z[0] if single else z


def resolver(spec: str | None):
    """Convierte la cadena del argumento --backbone en un codificador.

    Fusión: ``fusion:<A>+<B>`` con pesos iguales, o ``fusion:<A>+<B>@0.3`` para dar
    0.3 a A y 0.7 a B. Por ejemplo ``fusion:dinov2+cnn:models/cnn_scratch.pt``.
    """
    if spec and spec.startswith("fusion:"):
        cuerpo = spec[len("fusion:"):]
        peso = 0.5
        if "@" in cuerpo:
            cuerpo, texto_peso = cuerpo.rsplit("@", 1)
            peso = float(texto_peso)
        if "+" not in cuerpo:
            raise SystemExit("La fusión necesita dos codificadores: fusion:<A>+<B>")
        spec_a, spec_b = cuerpo.split("+", 1)
        return FusionEmbedder(resolver(spec_a), resolver(spec_b), peso)

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
    if isinstance(emb, FusionEmbedder):
        return f"{emb.nombre} dim={emb.dim}"
    if isinstance(emb, CNNEmbedder):
        m = emb.meta
        return (f"{emb.nombre} dim={emb.dim} "
                f"({m.get('parametros', 0)/1e6:.1f} M parámetros, "
                f"{m.get('identidades', '?')} identidades, {m.get('epocas', '?')} épocas)")
    return (f"dinov2-vits14 dim={emb.dim} "
            f"cabeza={'sí' if emb.head is not None else 'no'}")
