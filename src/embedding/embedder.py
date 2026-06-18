"""Extractor de embeddings de tatuajes (Paso 0: DINOv2 preentrenado, sin afinar).

Convierte un recorte de tatuaje en un vector L2-normalizado. La similitud entre
dos tatuajes es el coseno = producto punto de sus embeddings.

No requiere entrenamiento: DINOv2 (entrenado de forma auto-supervisada sobre
millones de imágenes) ya produce representaciones visuales muy fuertes. Esto
permite probar el enfoque de "huella + galería" antes de invertir en datos
sintéticos o fine-tuning.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

DEFAULT_MODEL = "vit_small_patch14_dinov2.lvd142m"  # ViT-S/14, embedding 384-d


def composite_on(img: Image.Image, bg=(255, 255, 255)) -> Image.Image:
    """Compone una imagen (posible RGBA con transparencia) sobre un fondo sólido."""
    img = img.convert("RGBA")
    canvas = Image.new("RGBA", img.size, (*bg, 255))
    canvas.alpha_composite(img)
    return canvas.convert("RGB")


class TattooEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | int | None = None):
        import torch
        import timm
        if device is None:
            device = 0 if torch.cuda.is_available() else "cpu"
        self.device = device
        self.torch = torch
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval().to(device)
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**cfg)
        self.dim = self.model.num_features

    def embed(self, images, bg=(255, 255, 255)) -> np.ndarray:
        """Embebe una imagen PIL o una lista. Devuelve array (dim,) o (N, dim) L2-norm."""
        single = isinstance(images, Image.Image)
        if single:
            images = [images]
        tensors = [self.transform(composite_on(im, bg)) for im in images]
        batch = self.torch.stack(tensors).to(self.device)
        with self.torch.no_grad():
            feats = self.model(batch)
        feats = self.torch.nn.functional.normalize(feats, dim=1)
        arr = feats.cpu().numpy().astype("float32")
        return arr[0] if single else arr
