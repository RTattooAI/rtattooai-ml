"""¿Qué confianza mínima debe exigir el detector?

El sistema usa `det_conf = 0.15`, heredado del paso sintético. Sobre fotografías
reales de personas **sin tatuajes**, con ese valor el detector dibuja al menos una
caja en la mitad de las imágenes: se confunde sobre todo con uñas decoradas y con
telas estampadas. Cada una de esas cajas llega al perito como un tatuaje detectado.

Subir el umbral quita falsos positivos y cuesta detecciones verdaderas. Este script
mide las dos cosas a la vez para poder elegir con criterio:

  - sobre `datasets/negativos_personas` (personas sin tatuajes, revisadas a ojo):
    qué proporción de imágenes produce alguna caja — todas son errores;
  - sobre el conjunto de validación etiquetado: qué proporción de tatuajes reales
    se sigue encontrando.

Uso:
    python -m src.detection.sweep_conf
    python -m src.detection.sweep_conf --val-max 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.embedding.identify import DETECTOR, load_config

NEGATIVOS = paths.DATASETS / "negativos_personas"
VAL_IMAGENES = paths.DATASETS / "unified" / "images" / "val"
VAL_ETIQUETAS = paths.DATASETS / "unified" / "labels" / "val"

CONFIANZAS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]

IOU_ACIERTO = 0.5


def cajas_reales(etiqueta: Path, ancho: int, alto: int) -> list[tuple[float, float, float, float]]:
    """Lee un archivo YOLO (clase cx cy w h normalizados) y devuelve xyxy en píxeles."""
    if not etiqueta.exists():
        return []

    salida = []
    for linea in etiqueta.read_text(encoding="utf-8").splitlines():
        partes = linea.split()
        if len(partes) < 5:
            continue
        _, cx, cy, w, h = (float(p) for p in partes[:5])
        salida.append(((cx - w / 2) * ancho, (cy - h / 2) * alto,
                       (cx + w / 2) * ancho, (cy + h / 2) * alto))
    return salida


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def main() -> int:
    ap = argparse.ArgumentParser(description="Barrido de la confianza mínima del detector")
    ap.add_argument("--val-max", type=int, default=300, help="cuántas imágenes de validación usar")
    ap.add_argument("--json", default="reports/detector_conf_sweep.json")
    ap.add_argument("--pesos", default=None,
                    help="pesos alternativos del detector. Sirve para comparar un "
                         "reentrenamiento con el desplegado sobre los mismos negativos.")
    args = ap.parse_args()

    cfg = load_config()
    negativos = sorted(
        p for carpeta in ("sin_deteccion", "con_deteccion")
        for p in (NEGATIVOS / carpeta).glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not negativos:
        print("No hay negativos revisados. Ejecute download_negatives y screen_negatives.")
        return 1

    validacion = sorted(p for p in VAL_IMAGENES.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    validacion = validacion[: args.val_max]

    print(f"Negativos revisados (personas sin tatuajes): {len(negativos)}")
    print(f"Validación etiquetada: {len(validacion)} imágenes\n")

    from ultralytics import YOLO

    pesos = Path(args.pesos) if args.pesos else DETECTOR
    if not pesos.exists():
        print(f"No existen los pesos {pesos}.")
        return 1
    print(f"Detector: {pesos}")
    detector = YOLO(str(pesos))

    # Se predice UNA vez con la confianza más baja y luego se filtra: repetir la
    # inferencia por cada umbral multiplicaría el tiempo sin cambiar el resultado.
    minima = min(CONFIANZAS)

    def predecir(rutas):
        salida = []
        for ruta in rutas:
            try:
                img = Image.open(ruta).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            r = detector.predict(img, conf=minima, iou=cfg["det_iou"], imgsz=640, verbose=False)[0]
            cajas = []
            for b in (r.boxes or []):
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                if x2 > x1 and y2 > y1:
                    cajas.append(((x1, y1, x2, y2), float(b.conf[0])))
            salida.append((ruta, img.size, cajas))
        return salida

    print("Prediciendo sobre los negativos…")
    pred_neg = predecir(negativos)
    print("Prediciendo sobre la validación…")
    pred_val = predecir(validacion)

    verdaderas = {}
    total_reales = 0
    for ruta, (ancho, alto), _ in pred_val:
        cajas = cajas_reales(VAL_ETIQUETAS / f"{ruta.stem}.txt", ancho, alto)
        verdaderas[ruta] = cajas
        total_reales += len(cajas)

    print(f"  tatuajes etiquetados en la validación: {total_reales}\n")
    print("conf   imágenes limpias   cajas falsas   tatuajes encontrados")

    filas = []
    for conf in CONFIANZAS:
        con_caja = sum(1 for _, _, cajas in pred_neg if any(c >= conf for _, c in cajas))
        cajas_falsas = sum(1 for _, _, cajas in pred_neg for _, c in cajas if c >= conf)
        limpias = (len(pred_neg) - con_caja) / len(pred_neg)

        encontrados = 0
        for ruta, _, cajas in pred_val:
            propuestas = [caja for caja, c in cajas if c >= conf]
            for real in verdaderas[ruta]:
                if any(iou(real, p) >= IOU_ACIERTO for p in propuestas):
                    encontrados += 1

        recall = encontrados / total_reales if total_reales else 0.0
        filas.append({"conf": conf, "imagenes_limpias": round(limpias, 4),
                      "cajas_falsas": cajas_falsas, "recall": round(recall, 4)})
        print(f"{conf:.2f}   {limpias*100:14.1f} %   {cajas_falsas:12d}   {recall*100:17.1f} %")

    destino = Path(args.json)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({
        "negativos": len(pred_neg),
        "validacion": len(pred_val),
        "tatuajes_etiquetados": total_reales,
        "iou_acierto": IOU_ACIERTO,
        "conf_actual": cfg["det_conf"],
        "pesos": str(pesos),
        "filas": filas,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nEscrito: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
