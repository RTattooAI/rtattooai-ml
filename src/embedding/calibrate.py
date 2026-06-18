"""Calibra el ESTÁNDAR de coincidencia (umbral + margen) con datos REALES.

Le das dos carpetas de fotos reales:
  --positives : tatuajes que SÍ están en la galería (deben aceptarse).
                Si las organizas en subcarpetas con el nombre de la identidad
                (igual que en la galería), además se verifica el acierto de id.
  --negatives : tatuajes que NO están en la galería (deben rechazarse).

Barre umbral×margen, elige el que maximiza F1 de la decisión "aceptar", y
escribe match_config.json. Así el sistema funciona sin andar moviendo sliders.

Uso:
    python -m src.embedding.calibrate --positives datasets/real_pos --negatives datasets/real_neg
    python -m src.embedding.calibrate ... --detect    # si son fotos completas (recorta con YOLO)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.embedding.embedder import TattooEmbedder
from src.embedding.identify import GALLERY_NPZ, DETECTOR, CONFIG_PATH, load_config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_labeled(folder: Path):
    """Devuelve [(ruta, identidad_o_None)]. Subcarpeta => identidad."""
    items = []
    if not folder.exists():
        return items
    for p in folder.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            ident = p.parent.name if p.parent != folder else None
            items.append((p, ident))
    return items


def best_two(gallery, names, emb_vec):
    sims = gallery @ emb_vec
    order = np.argsort(-sims)
    t1 = float(sims[order[0]])
    t2 = float(sims[order[1]]) if len(order) > 1 else -1.0
    return t1, t2, names[order[0]]


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibra umbral/margen con datos reales")
    ap.add_argument("--positives", required=True)
    ap.add_argument("--negatives", required=True)
    ap.add_argument("--detect", action="store_true", help="recortar con YOLO (fotos completas)")
    ap.add_argument("--write", action="store_true", help="escribir match_config.json")
    args = ap.parse_args()

    if not GALLERY_NPZ.exists():
        print(f"ERROR: falta la galería {GALLERY_NPZ} (corre step0_baseline).")
        return 1
    data = np.load(str(GALLERY_NPZ), allow_pickle=True)
    gallery = data["embeddings"].astype("float32")
    names = [str(n) for n in data["names"]]
    emb = TattooEmbedder()
    detector = None
    if args.detect:
        from ultralytics import YOLO
        detector = YOLO(str(DETECTOR))

    def crops_of(path: Path):
        im = Image.open(path).convert("RGB")
        if not detector:
            return [im]
        r = detector.predict(im, conf=0.15, imgsz=640, verbose=False)[0]
        out = []
        for b in (r.boxes or []):
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            if x2 > x1 and y2 > y1:
                out.append(im.crop((x1, y1, x2, y2)))
        return out or [im]

    def score(items):
        rows = []  # (top1, gap, correct_or_None)
        for path, ident in items:
            for c in crops_of(path):
                t1, t2, pred = best_two(gallery, names, emb.embed(c))
                correct = None if ident is None else (pred == ident)
                rows.append((t1, t1 - t2, correct))
        return rows

    pos = score(list_labeled(Path(args.positives)))
    neg = score(list_labeled(Path(args.negatives)))
    if not pos or not neg:
        print(f"ERROR: positivos={len(pos)} negativos={len(neg)} (revisa las carpetas).")
        return 1
    print(f"Positivos: {len(pos)} | Negativos: {len(neg)}")

    best = None
    for thr in np.arange(0.30, 0.86, 0.01):
        for mar in (0.0, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15):
            tp = sum(1 for t1, g, c in pos if t1 >= thr and g >= mar and (c is not False))
            fn = len(pos) - tp
            fp = sum(1 for t1, g, c in neg if t1 >= thr and g >= mar)
            tn = len(neg) - fp
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if best is None or f1 > best["f1"]:
                best = {"threshold": round(float(thr), 3), "margin": round(float(mar), 3),
                        "f1": round(f1, 3), "precision": round(prec, 3),
                        "recall": round(rec, 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    print("\nMejor operación encontrada:")
    for k in ("threshold", "margin", "f1", "precision", "recall", "tp", "fp", "fn", "tn"):
        print(f"  {k}: {best[k]}")

    if args.write:
        cfg = load_config()
        cfg["match_threshold"] = best["threshold"]
        cfg["match_margin"] = best["margin"]
        cfg["_note"] = (f"Calibrado con datos reales (F1={best['f1']}, "
                        f"P={best['precision']}, R={best['recall']}).")
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nEscrito: {CONFIG_PATH}")
    else:
        print("\n(usa --write para guardar en match_config.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
