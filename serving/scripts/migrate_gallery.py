"""Migración one-shot de la galería: `names` (identidad) -> `catalog_id` + `band`.

La galería original (`build_gallery.py`) sólo guarda `embeddings` + `names`
(nombre del diseño, p.ej. "Ak47"). El contrato de la API de identificación
exige por coincidencia un `catalog_id` (tat_00001...) y un `band` (banda
asociada). Como no existe un mapeo real nombre->banda para las identidades
preexistentes, se usa el nombre de la identidad como `band` (no se inventan
bandas) y se asigna un `catalog_id` secuencial por identidad única.

Idempotente: si el npz ya tiene `catalog_ids`, no hace nada (a menos que se
pase --force). Crea un respaldo `<nombre>.backup.npz` antes de sobrescribir.

Uso:
    python -m serving.scripts.migrate_gallery
    python -m serving.scripts.migrate_gallery --gallery models/gallery_dinov2.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths

GALLERY_NPZ = paths.MODELS / "gallery_dinov2.npz"


def migrate(gallery_path: Path, force: bool = False) -> None:
    data = np.load(str(gallery_path), allow_pickle=True)
    if "catalog_ids" in data and not force:
        print(f"Ya migrada ({gallery_path}): tiene 'catalog_ids'. Usa --force para regenerar.")
        return

    embeddings = data["embeddings"].astype("float32")
    names = [str(n) for n in data["names"]]

    unique_names = sorted(set(names))
    name_to_catalog_id = {name: f"tat_{i+1:05d}" for i, name in enumerate(unique_names)}

    catalog_ids = np.array([name_to_catalog_id[n] for n in names])
    bands = np.array(names)  # band = nombre de la identidad (sin mapeo real disponible)

    backup_path = gallery_path.with_suffix(".backup.npz")
    if not backup_path.exists():
        np.savez(str(backup_path), embeddings=embeddings, names=np.array(names))
        print(f"Respaldo creado: {backup_path}")

    np.savez(str(gallery_path), embeddings=embeddings, names=np.array(names),
             catalog_ids=catalog_ids, bands=bands)

    print(f"Migrada: {gallery_path}")
    print(f"  {len(unique_names)} identidades -> catalog_id tat_00001..tat_{len(unique_names):05d}")
    for name in unique_names:
        print(f"    {name_to_catalog_id[name]}  band={name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Migra la galería a esquema catalog_id + band")
    ap.add_argument("--gallery", default=str(GALLERY_NPZ))
    ap.add_argument("--force", action="store_true", help="regenerar aunque ya esté migrada")
    args = ap.parse_args()

    gallery_path = Path(args.gallery)
    if not gallery_path.exists():
        print(f"ERROR: no existe {gallery_path}")
        return 1

    migrate(gallery_path, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
