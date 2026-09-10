"""Revisa las fotografías descargadas y las separa según lo que vea el detector.

Dos salidas, y las dos sirven:

  - `sin_deteccion/`: el detector no encuentra nada. Son los negativos que hacían
    falta para CP015 (estado `no_detection`) y para calibrar el umbral.
  - `con_deteccion/`: el detector dibujó al menos una caja. Si la fotografía
    realmente no tiene tatuajes, cada caja es un **falso positivo del detector**, y
    esa tasa es un dato que hasta ahora no se podía reportar.

Genera además hojas de contactos con las cajas dibujadas. **Hay que mirarlas**: el
filtro por título de la descarga es aproximado, y solo mirando se descarta lo que no
corresponde (una foto que sí tiene un tatuaje, un encuadre inservible, contenido
que no debería estar en el conjunto).

Uso:
    python -m src.detection.screen_negatives
    python -m src.detection.screen_negatives --conf 0.15
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.embedding.identify import DETECTOR, load_config

BASE = paths.DATASETS / "negativos_personas"
PENDIENTES = BASE / "pendientes"
SIN_DETECCION = BASE / "sin_deteccion"
CON_DETECCION = BASE / "con_deteccion"
CONTACTOS = BASE / "contactos"

CELDA = 320
COLUMNAS = 5
FILAS_POR_HOJA = 4


def hoja_de_contactos(imagenes: list[tuple[Path, list]], destino: Path, titulo: str) -> list[Path]:
    """Rejillas con las cajas dibujadas, para revisarlas de un vistazo."""
    destino.mkdir(parents=True, exist_ok=True)
    por_hoja = COLUMNAS * FILAS_POR_HOJA
    salidas = []

    for n in range(0, len(imagenes), por_hoja):
        lote = imagenes[n : n + por_hoja]
        filas = (len(lote) + COLUMNAS - 1) // COLUMNAS
        hoja = Image.new("RGB", (COLUMNAS * CELDA, filas * CELDA), (24, 28, 38))

        for i, (ruta, cajas) in enumerate(lote):
            try:
                img = Image.open(ruta).convert("RGB")
            except Exception:  # noqa: BLE001
                continue

            escala = min(CELDA / img.width, CELDA / img.height)
            mini = img.resize((max(1, int(img.width * escala)), max(1, int(img.height * escala))))

            dibujo = ImageDraw.Draw(mini)
            for (x1, y1, x2, y2), conf in cajas:
                dibujo.rectangle(
                    [x1 * escala, y1 * escala, x2 * escala, y2 * escala],
                    outline=(239, 68, 68), width=3)
                dibujo.text((x1 * escala + 3, y1 * escala + 3), f"{conf:.2f}", fill=(255, 220, 220))

            cx = (i % COLUMNAS) * CELDA + (CELDA - mini.width) // 2
            cy = (i // COLUMNAS) * CELDA + (CELDA - mini.height) // 2
            hoja.paste(mini, (cx, cy))

        ruta_hoja = destino / f"{titulo}_{n // por_hoja + 1:02d}.jpg"
        hoja.save(ruta_hoja, "JPEG", quality=88)
        salidas.append(ruta_hoja)

    return salidas


def main() -> int:
    ap = argparse.ArgumentParser(description="Separa los negativos según lo que detecte YOLO")
    ap.add_argument("--conf", type=float, default=None, help="confianza mínima (por defecto, la del sistema)")
    ap.add_argument("--origen", default=str(PENDIENTES))
    args = ap.parse_args()

    origen = Path(args.origen)
    imagenes = sorted(p for p in origen.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not imagenes:
        print(f"No hay imágenes en {origen}. Ejecute antes download_negatives.")
        return 1

    cfg = load_config()
    conf = args.conf if args.conf is not None else cfg["det_conf"]

    from ultralytics import YOLO

    detector = YOLO(str(DETECTOR))
    print(f"{len(imagenes)} imágenes, confianza mínima {conf}\n")

    sin, con = [], []
    for ruta in imagenes:
        try:
            img = Image.open(ruta).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            print(f"  ilegible, se descarta: {ruta.name} ({exc})")
            continue

        r = detector.predict(img, conf=conf, iou=cfg["det_iou"], imgsz=640, verbose=False)[0]
        cajas = []
        for b in (r.boxes or []):
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            if x2 > x1 and y2 > y1:
                cajas.append(((x1, y1, x2, y2), float(b.conf[0])))

        (con if cajas else sin).append((ruta, cajas))

    for carpeta in (SIN_DETECCION, CON_DETECCION):
        carpeta.mkdir(parents=True, exist_ok=True)

    for ruta, _ in sin:
        shutil.copy2(ruta, SIN_DETECCION / ruta.name)
    for ruta, _ in con:
        shutil.copy2(ruta, CON_DETECCION / ruta.name)

    hojas_sin = hoja_de_contactos(sin, CONTACTOS, "sin_deteccion")
    hojas_con = hoja_de_contactos(con, CONTACTOS, "con_deteccion")

    total = len(sin) + len(con)
    print(f"Sin detección: {len(sin)} ({len(sin)/total*100:.0f} %)  ->  {SIN_DETECCION}")
    print(f"Con detección: {len(con)} ({len(con)/total*100:.0f} %)  ->  {CON_DETECCION}")
    print(f"  cajas totales en esas {len(con)}: {sum(len(c) for _, c in con)}")

    resumen = {
        "confianza_minima": conf,
        "total": total,
        "sin_deteccion": len(sin),
        "con_deteccion": len(con),
        "cajas_totales": sum(len(c) for _, c in con),
        "con_deteccion_detalle": [
            {"archivo": r.name, "cajas": [{"conf": round(c, 3)} for _, c in cs]}
            for r, cs in con
        ],
    }
    (BASE / "resumen_deteccion.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nHojas de contactos: {len(hojas_sin)} sin detección, {len(hojas_con)} con detección")
    print(f"  en {CONTACTOS}")
    print("\nMÍRELAS antes de dar el conjunto por bueno: si una fotografía de")
    print("'con_deteccion' sí tiene un tatuaje, no es un falso positivo del detector;")
    print("y si una de 'sin_deteccion' tiene uno, es un falso NEGATIVO y tampoco sirve")
    print("como negativo de calibración.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
