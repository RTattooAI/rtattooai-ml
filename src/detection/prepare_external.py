"""Convierte un conjunto EXTERNO con anotaciones a un dataset YOLO de evaluación.

Pensado para evaluar el modelo ya entrenado sobre datos que NO se usaron en el
entrenamiento (p. ej. DeMSI, para comparar de forma justa con el baseline).

Soporta el formato de anotaciones de **DeMSI** (FER/Zagreb):
    nombre.jpg [(x1,y1) (x2,y2) ...] [(x1,y1) ...]
donde cada grupo entre corchetes es un polígono en coordenadas de PÍXEL
absolutas. Cada polígono se convierte a bounding box (min/max) y se normaliza.

Genera datasets/<name>/{images,labels}/<split>/ + data.yaml (1 clase: tatuaje),
listo para `python -m src.detection.evaluate --data datasets/<name>/data.yaml --splits <split>`.

Uso:
    python -m src.detection.prepare_external \
        --images datasets/raw/demsi --annotations datasets/raw/demsi/tattoo_annotations.txt \
        --name demsi_eval --split val
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.shared.seeds import set_seed

POINT_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
GROUP_RE = re.compile(r"\[(.*?)\]")


def parse_demsi_line(line: str):
    """Devuelve (filename, [polígonos]) donde cada polígono es [(x,y), ...]."""
    line = line.strip()
    if not line:
        return None, []
    fname = line.split()[0]
    rest = line[len(fname):]
    polys = []
    for grp in GROUP_RE.findall(rest):
        pts = [(float(x), float(y)) for x, y in POINT_RE.findall(grp)]
        if len(pts) >= 3:               # un polígono necesita ≥3 puntos
            polys.append(pts)
    return fname, polys


def poly_to_yolo(pts, w: int, h: int):
    """Polígono en píxeles -> (cx,cy,bw,bh) normalizado YOLO, o None si inválido."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = max(min(xs), 0), min(max(xs), w)
    ymin, ymax = max(min(ys), 0), min(max(ys), h)
    bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
    cx, cy = (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h
    if bw <= 0 or bh <= 0:
        return None
    return cx, cy, bw, bh


def main() -> int:
    ap = argparse.ArgumentParser(description="Construye dataset YOLO de eval desde anotaciones externas")
    ap.add_argument("--images", required=True, help="carpeta con las imágenes")
    ap.add_argument("--annotations", required=True, help="archivo de anotaciones (formato DeMSI)")
    ap.add_argument("--name", default="external_eval", help="nombre del dataset de salida")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="split bajo el que se evaluará (val por defecto)")
    args = ap.parse_args()

    set_seed()
    from PIL import Image

    images_dir = Path(args.images)
    ann_file = Path(args.annotations)
    if not ann_file.exists():
        print(f"ERROR: no existe {ann_file}")
        return 1

    out_root = paths.DATASETS / args.name
    img_out = out_root / "images" / args.split
    lbl_out = out_root / "labels" / args.split
    if out_root.exists():
        shutil.rmtree(out_root)
    img_out.mkdir(parents=True)
    lbl_out.mkdir(parents=True)

    n_img = n_box = n_missing = n_bad = 0
    for line in ann_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        fname, polys = parse_demsi_line(line)
        if not fname:
            continue
        src_img = images_dir / fname
        if not src_img.exists():
            n_missing += 1
            continue
        try:
            with Image.open(src_img) as im:
                w, h = im.size
        except Exception:  # noqa: BLE001
            n_bad += 1
            continue
        lines = []
        for poly in polys:
            bb = poly_to_yolo(poly, w, h)
            if bb:
                lines.append("0 " + " ".join(f"{v:.6f}" for v in bb))
        # copia imagen + escribe label (vacío si no hay polígonos válidos = negativo)
        shutil.copy2(src_img, img_out / fname)
        (lbl_out / (Path(fname).stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_img += 1
        n_box += len(lines)

    yaml_text = (
        f"# Dataset de EVALUACIÓN externa generado por prepare_external.py\n"
        f"path: {out_root.as_posix()}\n"
        f"{args.split}: images/{args.split}\n"
        f"# (train/val/test apuntan al mismo split para que model.val funcione)\n"
        f"train: images/{args.split}\n"
        f"val: images/{args.split}\n"
        f"test: images/{args.split}\n\n"
        f"nc: 1\nnames:\n  0: tatuaje\n"
    )
    (out_root / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print(f"Dataset externo '{args.name}' creado en: {out_root}")
    print(f"  imágenes: {n_img} | cajas: {n_box} | sin imagen: {n_missing} | corruptas: {n_bad}")
    print(f"  data.yaml: {out_root / 'data.yaml'}  (split='{args.split}')")
    print(f"\nEvaluar con:\n  python -m src.detection.evaluate "
          f"--data {(out_root / 'data.yaml').as_posix()} --splits {args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
