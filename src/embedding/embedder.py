"""Extractor de embeddings de tatuajes.

Base: DINOv2 preentrenado (congelado) — representaciones visuales fuertes sin
entrenar. Opcionalmente, una CABEZA (MLP) afinada de forma contrastiva sobre los
features de DINOv2 para mejorar invarianza a degradación/condiciones
(ver finetune_head.py). El embedding final es L2-normalizado (similitud = coseno).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

DEFAULT_MODEL = "vit_small_patch14_dinov2.lvd142m"  # ViT-S/14, features 384-d
HEAD_HIDDEN = 256
HEAD_OUT = 128


def composite_on(img: Image.Image, bg=(255, 255, 255)) -> Image.Image:
    """Compone una imagen (posible RGBA) sobre un fondo sólido."""
    img = img.convert("RGBA")
    canvas = Image.new("RGBA", img.size, (*bg, 255))
    canvas.alpha_composite(img)
    return canvas.convert("RGB")


def build_head(in_dim: int, hidden: int = HEAD_HIDDEN, out: int = HEAD_OUT):
    """MLP de proyección (misma arquitectura en entrenamiento e inferencia)."""
    import torch.nn as nn
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out))


class TattooEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device=None,
                 head_path: str | None = None):
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
        self.backbone_dim = self.model.num_features
        self.head = None
        self.dim = self.backbone_dim
        if head_path:
            from pathlib import Path
            if Path(head_path).exists():
                ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
                self.head = build_head(self.backbone_dim, ckpt.get("hidden", HEAD_HIDDEN),
                                       ckpt.get("out", HEAD_OUT))
                self.head.load_state_dict(ckpt["state_dict"])
                self.head.eval().to(device)
                self.dim = ckpt.get("out", HEAD_OUT)

    def backbone_features(self, images, bg=(255, 255, 255), chunk: int = 64):
        """Features CRUDOS de DINOv2 (sin cabeza, sin normalizar). Por lotes.

        Devuelve un tensor (N, backbone_dim) en CPU (para cachear/entrenar la cabeza).
        """
        if isinstance(images, Image.Image):
            images = [images]
        outs = []
        for i in range(0, len(images), chunk):
            batch_imgs = images[i:i + chunk]
            tensors = [self.transform(composite_on(im, bg)) for im in batch_imgs]
            batch = self.torch.stack(tensors).to(self.device)
            with self.torch.no_grad():
                outs.append(self.model(batch).cpu())
        return self.torch.cat(outs, dim=0)

    def embed(self, images, bg=(255, 255, 255), chunk: int = 64) -> np.ndarray:
        """Embedding final (cabeza si existe) L2-norm. Acepta PIL o lista."""
        single = isinstance(images, Image.Image)
        feats = self.backbone_features(images, bg, chunk).to(self.device)
        if self.head is not None:
            with self.torch.no_grad():
                feats = self.head(feats)
        feats = self.torch.nn.functional.normalize(feats, dim=1)
        arr = feats.cpu().numpy().astype("float32")
        return arr[0] if single else arr
