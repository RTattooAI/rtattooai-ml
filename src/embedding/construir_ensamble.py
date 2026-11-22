"""Genera los artefactos que el servicio necesita para el modo ensamble.

En models/ensamble/:
  generalista.pt               cabeza contrastiva sobre DINOv2 (copia del checkpoint elegido)
  especialista.pt              CNN afinada con tatuajes (copia del checkpoint elegido)
  galeria_inicial.npz          galería base: los mismos códigos y bandas que la galería
                               clásica; por diseño, el original, 3 giros y vistas
                               sintéticas generadas por procesamiento de imágenes
  calibracion_desconocidos.npz vectores de tatuajes que NO están en la galería, con los
                               que el servicio calcula la normalización y la línea
                               conformal cada vez que cambia la galería
  manifiesto.json              de dónde salió cada cosa

Galería y calibración llevan la huella de la pareja de pesos: el servicio se niega a
usarlas con otros.

Los tatuajes enrolados desde la aplicación no están en la galería base: se recuperan
con «Reparar galería» en la pantalla de Modelos, que los vuelve a enrolar desde las
imágenes que conserva el backend.

Uso:
    python -m src.embedding.construir_ensamble \\
        --generalista models/candidatos/head_k10.pt --especialista models/candidatos/cnn_r18_ref.pt
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.shared.seeds import set_seed

DESTINO = paths.MODELS / "ensamble"
GALERIA_CLASICA = paths.MODELS / "gallery_dinov2.npz"
ROTACIONES = (90, 180, 270)


TONOS_PIEL = [(241, 194, 167), (224, 172, 138), (198, 134, 100), (141, 85, 58)]


def sobre_piel(diseno: Image.Image, rng: np.random.RandomState) -> Image.Image:
    """El diseño (PNG con transparencia) sobre un fondo de piel con textura leve."""
    from PIL import ImageFilter
    im = diseno.convert("RGBA")
    w, h = im.size
    m = int(0.15 * max(w, h))
    tono = np.array(TONOS_PIEL[rng.randint(len(TONOS_PIEL))], dtype="float32")
    ruido = rng.normal(0, 6, (h + 2 * m, w + 2 * m, 1)).astype("float32")
    fondo = Image.fromarray(np.clip(tono + ruido, 0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(2))
    fondo = fondo.convert("RGBA")
    tinta = im.copy()
    alfa = np.asarray(tinta.split()[-1], dtype="float32") * rng.uniform(0.75, 1.0)
    tinta.putalpha(Image.fromarray(alfa.astype("uint8")))
    fondo.alpha_composite(tinta, (m, m))
    return fondo.convert("RGB")


def vistas_sinteticas(diseno: Image.Image, n: int, semilla: int) -> list[Image.Image]:
    """Vistas sintéticas por PROCESAMIENTO DE IMÁGENES (sin Blender): el diseño sobre
    piel con perspectiva, giro, encuadre, luz, color, desenfoque, baja resolución y
    compresión JPEG; las mismas transformaciones con que se entrenan los codificadores."""
    from src.embedding.finetune_head import augment_real
    rng = np.random.RandomState(semilla)
    return [augment_real(sobre_piel(diseno, rng), rng, max_degrade=0.5) for _ in range(n)]


def galeria_procesada(codificadores, n_sinteticas: int) -> dict:
    """Galería base sin Blender: por diseño, el original, 3 giros y n vistas sintéticas."""
    from src.embedding.rebuild_gallery import DISENOS, IMG_EXTS, nombre_limpio
    anterior = np.load(GALERIA_CLASICA, allow_pickle=True)
    mapeo: dict[str, tuple[str, str]] = {}
    for nombre, cid, banda in zip(anterior["names"], anterior["catalog_ids"], anterior["bands"]):
        mapeo.setdefault(str(nombre), (str(cid), str(banda)))
    imagenes, ids, bandas, nombres = [], [], [], []
    for k, ruta in enumerate(sorted(p for p in DISENOS.glob("*") if p.suffix.lower() in IMG_EXTS)):
        nombre = nombre_limpio(ruta.stem)
        if nombre not in mapeo:
            continue
        cid, banda = mapeo[nombre]
        diseno = Image.open(ruta)
        fondo = Image.new("RGBA", diseno.size, (255, 255, 255, 255))
        fondo.alpha_composite(diseno.convert("RGBA"))
        base = fondo.convert("RGB")
        vistas = ([base] + [base.rotate(a, expand=True, fillcolor=(255, 255, 255)) for a in ROTACIONES]
                  + vistas_sinteticas(diseno, n_sinteticas, semilla=1000 + k))
        for v in vistas:
            imagenes.append(v)
            ids.append(cid)
            bandas.append(banda)
            nombres.append(nombre)
    print(f"  galería base (procesamiento de imágenes): {len(set(ids))} identidades, {len(imagenes)} vistas")
    return {"embeddings": codificadores.embed(imagenes), "catalog_ids": np.array(ids),
            "bands": np.array(bandas), "names": np.array(nombres),
            "image_sha256": np.array([""] * len(ids))}


def galeria_base(codificadores) -> dict:
    """Variante anterior: diseño + renders de Blender. Se conserva solo para comparar."""
    from src.embedding.rebuild_gallery import vistas_por_identidad
    anterior = np.load(GALERIA_CLASICA, allow_pickle=True)
    mapeo: dict[str, tuple[str, str]] = {}
    for nombre, cid, banda in zip(anterior["names"], anterior["catalog_ids"], anterior["bands"]):
        mapeo.setdefault(str(nombre), (str(cid), str(banda)))
    fuentes = vistas_por_identidad()
    faltan = sorted(set(mapeo) - set(fuentes))
    if faltan:
        print(f"  Aviso: {len(faltan)} identidades de la galería clásica no tienen imagen de origen "
              f"(enroladas desde la app); se recuperan con «Reparar galería»: {faltan[:5]}")
    imagenes, ids, bandas, nombres = [], [], [], []
    for nombre, rutas in sorted(fuentes.items()):
        if nombre not in mapeo:
            continue
        cid, banda = mapeo[nombre]
        for ruta in rutas:
            im = Image.open(ruta).convert("RGBA")
            fondo = Image.new("RGBA", im.size, (255, 255, 255, 255))
            fondo.alpha_composite(im)
            base = fondo.convert("RGB")
            for v in [base] + [base.rotate(a, expand=True, fillcolor=(255, 255, 255)) for a in ROTACIONES]:
                imagenes.append(v)
                ids.append(cid)
                bandas.append(banda)
                nombres.append(nombre)
    print(f"  galería base: {len(set(ids))} identidades, {len(imagenes)} vistas (con giros)")
    return {"embeddings": codificadores.embed(imagenes), "catalog_ids": np.array(ids),
            "bands": np.array(bandas), "names": np.array(nombres),
            "image_sha256": np.array([""] * len(ids))}


def desconocidos(maximo: int) -> list[Image.Image]:
    """Tatuajes de fuentes públicas que no están en la galería ni en el entrenamiento."""
    from src.embedding import experimento_ab as ab
    cache = paths.REPORTS / "experimentos" / "phash_recortes_train.npy"
    if cache.exists():
        excluir = np.load(cache)
    else:
        excluir = np.array([ab.phash(Image.open(f).convert("RGB"))
                            for f in sorted((paths.DATASETS / "real_crops" / "train").glob("*.png"))],
                           dtype=np.uint64)
        np.save(cache, excluir)
    return ab.desconocidos_ampliados(excluir, np.random.RandomState(7), maximo)


def main() -> int:
    ap = argparse.ArgumentParser(description="Artefactos del modo ensamble")
    ap.add_argument("--generalista", required=True)
    ap.add_argument("--especialista", required=True)
    ap.add_argument("--desconocidos", type=int, default=4000)
    ap.add_argument("--version", default=None, help="etiqueta de versión que mostrará /info")
    ap.add_argument("--vistas-sinteticas", type=int, default=16,
                    help="vistas por diseño generadas por procesamiento de imágenes")
    ap.add_argument("--con-blender", action="store_true",
                    help="usar los renders de Blender antiguos en lugar del procesamiento de imágenes")
    args = ap.parse_args()
    set_seed()

    DESTINO.mkdir(parents=True, exist_ok=True)
    if (DESTINO / "generalista.pt").exists():
        respaldo = DESTINO.parent / f"ensamble.{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.backup"
        shutil.copytree(DESTINO, respaldo)
        print(f"Respaldo de los artefactos anteriores: {respaldo.name}")
    shutil.copy2(args.generalista, DESTINO / "generalista.pt")
    shutil.copy2(args.especialista, DESTINO / "especialista.pt")

    from serving.codificadores import Codificadores
    cod = Codificadores(DESTINO / "generalista.pt", DESTINO / "especialista.pt")
    print(f"Codificadores: dims {cod.dims} · huella {cod.huella}")

    print("Galería base…")
    g = galeria_base(cod) if args.con_blender else galeria_procesada(cod, args.vistas_sinteticas)
    np.savez(DESTINO / "galeria_inicial.npz", **g, head_fingerprint=np.array(cod.huella),
             built_at=np.array(datetime.now(timezone.utc).isoformat(timespec="seconds")))

    print("Desconocidos de calibración…")
    imgs = desconocidos(args.desconocidos)
    E = cod.embed(imgs)
    np.savez(DESTINO / "calibracion_desconocidos.npz", embeddings=E, huella=np.array(cod.huella),
             fuentes=np.array("DeMSI completo + tattoodet/playground fuera del esquema de versiones; "
                              "sin copias de recortes de entrenamiento (pHash en 8 orientaciones)"))
    print(f"  {len(imgs)} desconocidos")

    version = args.version or (f"yolov8m-hardneg-pt + ensamble[{Path(args.generalista).stem} | "
                               f"{Path(args.especialista).stem}] + consenso + conformal")
    (DESTINO / "manifiesto.json").write_text(json.dumps({
        "version": version,
        "generalista_origen": str(args.generalista),
        "especialista_origen": str(args.especialista),
        "huella": cod.huella,
        "galeria_identidades": int(len(set(g["catalog_ids"].tolist()))),
        "galeria_vistas": int(len(g["catalog_ids"])),
        "desconocidos_calibracion": len(imgs),
        "creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Listo: {DESTINO}")
    print("Si ya existe models/galeria_ensamble.npz de una versión anterior, bórrela (o use "
          "«Reparar galería») para que el servicio arranque desde la nueva galería base.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
