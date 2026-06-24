"""Recorta los tatuajes reales del dataset unificado usando las cajas GT.

Genera un pool de recortes reales de tatuajes (miles, muy diversos) a partir de
las etiquetas YOLO existentes — sin re-inferir, usando las cajas verdaderas.
Sirve como:
  - diversidad de identidades para el fine-tuning contrastivo del embedding,
  - distractores no-banda para el rechazo open-set.

Salida: datasets/real_crops/<split>/<imagen>_b<i>.png  (gitignored)

Uso:
    python -m src.embedding.crop_dataset
    python -m src.embedding.crop_dataset --limit 2500 --margin 0.08
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.shared.seeds import set_seed

OUT = paths.DATASETS / "real_crops"
MIN_SIDE = 32   # descarta recortes diminutos


def crop_split(split: str, margin: float, limit: int | None) -> int:
    img_dir = paths.UNIFIED_IMAGES / split
    lbl_dir = paths.UNIFIED_LABELS / split
    if not img_dir.exists():
        return 0
    out_dir = OUT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    imgs = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for img_path in imgs:
        if limit and n >= limit:
            break
        lbl = lbl_dir / (img_path.stem + ".txt")
        if not lbl.exists():
            continue
        lines = [l for l in lbl.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        try:
            im = Image.open(img_path).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        W, H = im.size
        for i, line in enumerate(lines):
            p = line.split()
            if len(p) != 5:
                continue
            _, cx, cy, w, h = (float(x) for x in p)
            mw, mh = w * margin, h * margin
            x1 = int(max(0, (cx - w / 2 - mw) * W)); x2 = int(min(W, (cx + w / 2 + mw) * W))
            y1 = int(max(0, (cy - h / 2 - mh) * H)); y2 = int(min(H, (cy + h / 2 + mh) * H))
            if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
                continue
            im.crop((x1, y1, x2, y2)).save(out_dir / f"{img_path.stem}_b{i}.png")
            n += 1
            if limit and n >= limit:
                break
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Recorta tatuajes reales del dataset unificado")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--margin", type=float, default=0.06, help="margen alrededor de la caja")
    ap.add_argument("--limit", type=int, default=None, help="máx recortes por split")
    args = ap.parse_args()

    set_seed()
    if not paths.UNIFIED_IMAGES.exists():
        print(f"ERROR: no existe {paths.UNIFIED_IMAGES}. Corre prepare_dataset.py.")
        return 1
    total = 0
    for split in args.splits:
        c = crop_split(split, args.margin, args.limit)
        print(f"  {split}: {c} recortes")
        total += c
    print(f"\nTotal: {total} recortes reales en {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
