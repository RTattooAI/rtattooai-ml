"""Construye la galería de embeddings (multi-vista) para la identificación.

Cada identidad (tatuaje de banda) puede tener VARIAS vistas: el diseño limpio +
renders realistas de Blender + fotos reales. En la identificación, el puntaje de
una identidad es el coseno MÁXIMO entre sus vistas → sube el match verdadero.

Fuentes admitidas (se pueden combinar):
  - carpeta PLANA: cada archivo = una identidad (nombre = stem).  (p.ej. datasets/gang_tattoos)
  - carpeta con SUBCARPETAS: cada subcarpeta = una identidad, con muchas vistas.
    (p.ej. la salida de Blender: OUTPUT/crops/<tattoo_id>/*.png)

Uso:
    python -m src.embedding.build_gallery
    python -m src.embedding.build_gallery --sources datasets/gang_tattoos OUTPUT/crops
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.shared.seeds import set_seed
from src.embedding.embedder import TattooEmbedder

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
GALLERY_NPZ = paths.MODELS / "gallery_dinov2.npz"


def clean_name(stem: str) -> str:
    return stem.replace("-removebg-preview", "")


def collect(source: Path) -> list[tuple[str, Path]]:
    """Devuelve [(identidad, ruta)]. Subcarpetas => identidad; plano => stem."""
    items = []
    subdirs = [d for d in source.iterdir() if d.is_dir()] if source.exists() else []
    if subdirs:
        for d in subdirs:
            for p in d.iterdir():
                if p.suffix.lower() in IMG_EXTS:
                    items.append((d.name, p))
    else:
        for p in sorted(source.glob("*")):
            if p.suffix.lower() in IMG_EXTS:
                items.append((clean_name(p.stem), p))
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description="Construye la galería de embeddings (multi-vista)")
    ap.add_argument("--sources", nargs="+", default=[str(paths.DATASETS / "gang_tattoos")])
    ap.add_argument("--out", default=str(GALLERY_NPZ))
    args = ap.parse_args()

    set_seed()
    items = []
    for s in args.sources:
        found = collect(Path(s))
        print(f"  {s}: {len(found)} imágenes")
        items += found
    if not items:
        print("ERROR: no se hallaron imágenes en las fuentes.")
        return 1

    print(f"Embebiendo {len(items)} vistas con DINOv2…")
    emb = TattooEmbedder()
    names = [ident for ident, _ in items]
    imgs = [Image.open(p) for _, p in items]
    vecs = emb.embed(imgs, bg=(255, 255, 255))   # (N, dim) L2-norm

    paths.ensure_dirs(paths.MODELS)
    np.savez(args.out, embeddings=vecs.astype("float32"), names=np.array(names))

    per_id = Counter(names)
    print(f"\nGalería: {len(per_id)} identidades, {len(items)} vistas en total.")
    print(f"  vistas por identidad: min {min(per_id.values())}, "
          f"máx {max(per_id.values())}, media {len(items)/len(per_id):.1f}")
    print(f"Guardada en: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
