"""Degradación realista de tatuajes (decoloración, borrado parcial, desenfoque,
envejecimiento de tinta, oclusión).

Sirve para dos cosas:
  - Medir la ROBUSTEZ del embedding (¿reconoce un tatuaje borrado como el mismo?).
  - Generar variaciones degradadas para enriquecer galería / fine-tuning.

Cada función recibe una imagen PIL RGB y un `amount` 0..1 (0 = sin cambio).
`degrade(img, level, rng)` combina varias para simular un tatuaje real gastado.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

SKIN = (198, 150, 120)   # tono de piel neutro para rellenos de borrado/oclusión


def _to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB")).astype(np.float32)


def _to_img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


def fade(img: Image.Image, amount: float) -> Image.Image:
    """Decoloración: la tinta pierde contraste y se aclara hacia la piel."""
    img = ImageEnhance.Contrast(img).enhance(1 - 0.75 * amount)
    img = ImageEnhance.Brightness(img).enhance(1 + 0.20 * amount)
    return img


def blur(img: Image.Image, amount: float) -> Image.Image:
    """La tinta migra con los años → bordes difusos."""
    return img.filter(ImageFilter.GaussianBlur(radius=3.0 * amount))


def age_colors(img: Image.Image, amount: float) -> Image.Image:
    """Tinta vieja: se dessatura y vira a azul/verde."""
    img = ImageEnhance.Color(img).enhance(1 - 0.6 * amount)
    arr = _to_np(img)
    arr[..., 0] -= 18 * amount   # menos rojo
    arr[..., 2] += 12 * amount   # más azul
    return _to_img(arr)


def partial_erase(img: Image.Image, amount: float, rng: np.random.RandomState,
                  skin=SKIN) -> Image.Image:
    """Borrado parcial (láser/cicatriz): zonas del tatuaje desaparecen a piel."""
    arr = _to_np(img)
    h, w = arr.shape[:2]
    # ruido de baja frecuencia -> manchas suaves
    small = rng.rand(max(2, h // 24), max(2, w // 24)).astype(np.float32)
    mask = np.asarray(Image.fromarray((small * 255).astype("uint8")).resize(
        (w, h), Image.BICUBIC)).astype(np.float32) / 255.0
    thr = np.quantile(mask, 1 - 0.6 * amount)   # más amount => más se borra
    erase = (mask >= thr)[..., None].astype(np.float32)
    skin_arr = np.array(skin, np.float32)[None, None, :]
    skin_field = skin_arr + rng.normal(0, 6, arr.shape)
    return _to_img(arr * (1 - erase) + skin_field * erase)


def occlude(img: Image.Image, amount: float, rng: np.random.RandomState,
            skin=SKIN) -> Image.Image:
    """Oclusión: parte cubierta (vello, ropa, otro tatuaje)."""
    if amount <= 0:
        return img
    arr = _to_np(img)
    h, w = arr.shape[:2]
    n = 1 + int(rng.rand() < amount)
    for _ in range(n):
        bw, bh = int(w * (0.2 + 0.4 * amount)), int(h * (0.2 + 0.4 * amount))
        x0 = rng.randint(0, max(1, w - bw)); y0 = rng.randint(0, max(1, h - bh))
        dark = rng.rand() < 0.5
        fill = np.array((30, 25, 22) if dark else skin, np.float32)
        arr[y0:y0 + bh, x0:x0 + bw] = fill + rng.normal(0, 5, (bh, bw, 3))
    return _to_img(arr)


def degrade(img: Image.Image, level: float, rng: np.random.RandomState) -> Image.Image:
    """Combina degradaciones escaladas por `level` (0=limpio, 1=muy gastado)."""
    out = fade(img, level)
    out = age_colors(out, level)
    out = blur(out, 0.7 * level)
    if level > 0:
        out = partial_erase(out, 0.7 * level, rng)
        if rng.rand() < level:
            out = occlude(out, level, rng)
    return out
