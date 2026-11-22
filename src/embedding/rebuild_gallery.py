"""Reconstruye la galería de producción con la cabeza actual, conservando los códigos.

**Por qué existe.** La galería guarda vectores, no imágenes. Esos vectores solo
tienen sentido en el espacio que produce una cabeza concreta: al reentrenar la
cabeza, la galería anterior deja de significar nada. Y el fallo es silencioso —
el sistema sigue devolviendo un "mejor candidato" y un porcentaje, solo que
calculados contra ruido. Es la peor clase de fallo posible en una herramienta de
apoyo pericial: sigue respondiendo con aplomo y responde cualquier cosa.

Este script recalcula los vectores desde las imágenes de origen y **conserva el
mapeo `catalog_id → banda`** de la galería anterior, porque los análisis ya
emitidos guardan esos códigos en sus coincidencias y renumerarlos rompería la
trazabilidad hacia atrás.

Además estampa una huella de la cabeza en el archivo. `GalleryStore` la compara al
arrancar y se niega a operar con una pareja incompatible, en vez de responder con
números sin sentido.

Uso:
    python -m src.embedding.rebuild_gallery
    python -m src.embedding.rebuild_gallery --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.shared.seeds import set_seed
from src.embedding.embedder import TattooEmbedder

GALLERY_NPZ = paths.MODELS / "gallery_dinov2.npz"
HEAD_PATH = paths.MODELS / "embed_head.pt"
DISENOS = paths.DATASETS / "gang_tattoos"
BLENDER = paths.DATASETS / "output" / "gang_tattoos_out" / "crops"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def huella_cabeza(ruta: Path) -> str:
    """SHA-256 del archivo de la cabeza, recortado. Identifica la pareja cabeza/galería."""
    if not ruta.exists():
        return "sin-cabeza"
    return hashlib.sha256(ruta.read_bytes()).hexdigest()[:16]


def nombre_limpio(stem: str) -> str:
    return stem.replace("-removebg-preview", "")


def vistas_por_identidad() -> dict[str, list[Path]]:
    """Las mismas fuentes con las que se construyó la galería original."""
    vistas: dict[str, list[Path]] = {}

    for p in sorted(DISENOS.glob("*")):
        if p.suffix.lower() in IMG_EXTS:
            vistas.setdefault(nombre_limpio(p.stem), []).append(p)

    if BLENDER.exists():
        for carpeta in sorted(BLENDER.iterdir()):
            if carpeta.is_dir():
                for p in sorted(carpeta.iterdir()):
                    if p.suffix.lower() in IMG_EXTS:
                        vistas.setdefault(carpeta.name, []).append(p)

    return vistas


def main() -> int:
    ap = argparse.ArgumentParser(description="Reconstruye la galería con la cabeza actual")
    ap.add_argument("--head", default=str(HEAD_PATH))
    ap.add_argument("--gallery", default=str(GALLERY_NPZ))
    ap.add_argument("--dry-run", action="store_true", help="informar sin escribir nada")
    args = ap.parse_args()

    set_seed()
    ruta_galeria = Path(args.gallery)
    ruta_cabeza = Path(args.head)

    if not ruta_galeria.exists():
        print(f"ERROR: no existe {ruta_galeria}. Use build_gallery para crearla desde cero.")
        return 1

    anterior = np.load(str(ruta_galeria), allow_pickle=True)
    ids_previos = [str(x) for x in anterior["catalog_ids"]]
    bandas_previas = [str(x) for x in anterior["bands"]]
    nombres_previos = [str(x) for x in anterior["names"]]

    # nombre de diseño -> (catalog_id, banda), tal como estaba
    mapeo: dict[str, tuple[str, str]] = {}
    for nombre, cid, banda in zip(nombres_previos, ids_previos, bandas_previas):
        mapeo.setdefault(nombre, (cid, banda))

    print(f"Galería anterior: {anterior['embeddings'].shape[0]} vistas, "
          f"{len(set(ids_previos))} identidades, dim {anterior['embeddings'].shape[1]}")
    print(f"Huella de la cabeza anterior: {str(anterior['head_fingerprint']) if 'head_fingerprint' in anterior else '(no consta)'}")

    vistas = vistas_por_identidad()
    conocidas = {n: v for n, v in vistas.items() if n in mapeo}
    huerfanas = sorted(set(mapeo) - set(vistas))

    print(f"\nFuentes encontradas: {len(vistas)} identidades, "
          f"{sum(len(v) for v in vistas.values())} vistas")
    print(f"  con código previo: {len(conocidas)}")
    if huerfanas:
        # Sin imagen de origen no hay forma de recalcular el vector: esas identidades
        # se perderían. Mejor abortar que publicar una galería incompleta en silencio.
        print(f"\nERROR: {len(huerfanas)} identidades de la galería no tienen imagen de origen:")
        for n in huerfanas[:10]:
            print(f"    {n} ({mapeo[n][0]})")
        print("  Reconstruir ahora las dejaría fuera. Recupere las imágenes o use build_gallery.")
        return 1

    if args.dry_run:
        print("\n(dry-run: no se escribe nada)")
        return 0

    emb = TattooEmbedder(head_path=str(ruta_cabeza) if ruta_cabeza.exists() else None)
    print(f"\nEmbedder: dim={emb.dim}, cabeza={'sí' if emb.head is not None else 'no'}")

    imagenes, ids, bandas, nombres = [], [], [], []
    for nombre, rutas in sorted(conocidas.items()):
        cid, banda = mapeo[nombre]
        for ruta in rutas:
            imagenes.append(Image.open(ruta))
            ids.append(cid)
            bandas.append(banda)
            nombres.append(nombre)

    print(f"Embebiendo {len(imagenes)} vistas…")
    vectores = emb.embed(imagenes, bg=(255, 255, 255)).astype("float32")

    respaldo = ruta_galeria.with_suffix(f".{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.backup.npz")
    shutil.copy2(ruta_galeria, respaldo)
    print(f"Respaldo de la anterior: {respaldo.name}")

    np.savez(
        str(ruta_galeria),
        embeddings=vectores,
        catalog_ids=np.array(ids),
        bands=np.array(bandas),
        names=np.array(nombres),
        head_fingerprint=np.array(huella_cabeza(ruta_cabeza)),
        built_at=np.array(datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )

    por_id = Counter(ids)
    print(f"\nGalería nueva: {len(por_id)} identidades, {len(imagenes)} vistas, dim {emb.dim}")
    print(f"  vistas por identidad: min {min(por_id.values())}, máx {max(por_id.values())}")
    print(f"  huella de la cabeza: {huella_cabeza(ruta_cabeza)}")
    print(f"Guardada en: {ruta_galeria}")
    print("\nLos tatuajes enrolados desde la aplicación NO están aquí: hay que volver a")
    print("enrolarlos desde sus imágenes, que sí conserva el backend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
