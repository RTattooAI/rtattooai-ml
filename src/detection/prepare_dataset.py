"""Unifica todas las fuentes descargadas en un único dataset YOLO de 1 clase.

Hace:
  1. Recorre datasets/raw/<fuente>/ (export yolov8 de Roboflow: */images, */labels).
  2. Normaliza TODAS las anotaciones a la clase 0 ('tatuaje').
  3. Convierte polígonos/máscaras de segmentación a bounding boxes (min/max).
  4. Deduplica imágenes casi idénticas entre fuentes con hash perceptual
     (imagehash.phash + LSH por bandas para que sea rápido y por umbral).
  5. Re-divide en train/val/test 80/10/10 con semilla fija.
  6. Escribe datasets/unified/ (images/labels por split) + data.yaml.
  7. Imprime y guarda en reports/ la tabla de procedencia (fuente, imágenes,
     cajas, tarea, licencia, URL).

Uso:
    python -m src.detection.prepare_dataset
    python -m src.detection.prepare_dataset --dup-threshold 6 --val 0.1 --test 0.1
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.shared.seeds import set_seed, SEED
from src.shared.metrics import write_markdown_table, write_csv

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Procedencia/licencia por fuente (espejo de download_data.SOURCES, para el
# caso de que se ejecute prepare sin volver a importar el registro).
PROVENANCE = {
    "lance_nelson": ("Roboflow Universe — Lance Nelson (ARTAT)",
                     "https://universe.roboflow.com/lance-nelson-gtmdt/artat",
                     "Roboflow Universe (verificar)"),
    "xisketos": ("Roboflow Universe — Xisketos (Detección de Tatuajes)",
                 "https://universe.roboflow.com/xisketos/deteccion-de-tatuajes",
                 "Roboflow Universe (verificar)"),
    "by_tatoo": ("Roboflow Universe — by tatoo (Tattoo Detection)",
                 "https://universe.roboflow.com/tatoo-9syny/tattoo-detection-qia5k",
                 "Roboflow Universe (verificar)"),
    "code_with_heady": ("Roboflow Universe — Code with Heady (Tattoo Detection-Updated)",
                        "https://universe.roboflow.com/code-with-heady-5wdhf/tattoo-detection-updated",
                        "Roboflow Universe (verificar)"),
    "playground_tattootest": ("Roboflow Universe — Playground (TattooTest, segmentación)",
                              "https://universe.roboflow.com/playground-xxvvy/tattootest-utzal",
                              "Roboflow Universe (verificar)"),
    "pocevice_tattoo": ("Roboflow Universe — G. Pocevičė (Tattoo localization & recognition)",
                        "https://universe.roboflow.com/gabijapocevicemajor-7dvfl/tattoo-localization-and-recognition",
                        "Roboflow Universe (verificar)"),
    "tattoodet": ("Roboflow Universe — TattooDet",
                  "https://universe.roboflow.com/tattoodet/tattoo-detection-cbtkn",
                  "Roboflow Universe (verificar)"),
    "image_analysis_region": ("Roboflow Universe — Image Analysis (region detection)",
                              "https://universe.roboflow.com/image-analysis-5i4lg/tattoo-region-detection",
                              "Roboflow Universe (verificar)"),
    "demsi": ("DeMSI — FER, U. de Zagreb (ImageNet, polígonos)",
              "https://www.fer.unizg.hr/demsi/databases_and_code/tattoo_dataset",
              "FER/U. Zagreb (uso académico — verificar)"),
}


# ---------------------------------------------------------------- labels
def yolo_line_to_bbox(line: str) -> tuple[float, float, float, float] | None:
    """Convierte una línea YOLO (bbox o polígono) a (cx, cy, w, h) clase 0.

    - 5 tokens  -> 'cls cx cy w h'  (detección)
    - >5 tokens -> 'cls x1 y1 x2 y2 ...' (polígono de segmentación) => min/max
    Devuelve None si la línea es inválida.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        coords = [float(x) for x in parts[1:]]
    except ValueError:
        return None

    if len(coords) == 4:
        cx, cy, w, h = coords
    else:
        if len(coords) % 2 != 0 or len(coords) < 6:
            return None
        xs = coords[0::2]
        ys = coords[1::2]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin

    # clamp a [0,1]
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    if w <= 0 or h <= 0:
        return None
    return cx, cy, w, h


def read_label(label_path: Path) -> list[tuple[float, float, float, float]]:
    """Lee un .txt YOLO y devuelve la lista de bboxes (todas clase 0)."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        bb = yolo_line_to_bbox(line)
        if bb:
            boxes.append(bb)
    return boxes


# ---------------------------------------------------------------- dedup (LSH)
class PerceptualDedup:
    """Deduplicación por hash perceptual con LSH de bandas (rápido + umbral)."""

    def __init__(self, threshold: int = 6, bands: int = 8):
        self.threshold = threshold
        self.bands = bands           # 8 bandas de 8 bits sobre un phash de 64 bits
        self.kept_hashes: list[int] = []
        self.buckets: dict[tuple[int, int], list[int]] = defaultdict(list)

    @staticmethod
    def _hamming(a: int, b: int) -> int:
        return bin(a ^ b).count("1")

    def _band_keys(self, val: int):
        return [(i, (val >> (8 * i)) & 0xFF) for i in range(self.bands)]

    def is_duplicate(self, val: int) -> bool:
        """Devuelve True si `val` es casi idéntico a alguno ya conservado."""
        seen = set()
        for bk in self._band_keys(val):
            for idx in self.buckets.get(bk, ()):  # candidatos del mismo bucket
                if idx in seen:
                    continue
                seen.add(idx)
                if self._hamming(val, self.kept_hashes[idx]) <= self.threshold:
                    return True
        return False

    def add(self, val: int) -> None:
        idx = len(self.kept_hashes)
        self.kept_hashes.append(val)
        for bk in self._band_keys(val):
            self.buckets[bk].append(idx)


def phash_int(img_path: Path) -> int | None:
    import imagehash
    from PIL import Image
    try:
        with Image.open(img_path) as im:
            h = imagehash.phash(im.convert("RGB"), hash_size=8)
        return int(str(h), 16)
    except Exception:  # noqa: BLE001  (imagen corrupta -> se ignora)
        return None


# ---------------------------------------------------------------- discovery
def discover_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    """Encuentra pares (imagen, label) en una fuente con layout */images/*.

    Si no hay carpetas 'images'/'labels' (p. ej. DeMSI manual), busca
    imágenes con un .txt hermano del mismo nombre.
    """
    pairs = []
    img_dirs = [p for p in source_dir.rglob("images") if p.is_dir()]
    if img_dirs:
        for idir in img_dirs:
            ldir = idir.parent / "labels"
            for img in idir.iterdir():
                if img.suffix.lower() in IMG_EXTS:
                    pairs.append((img, ldir / (img.stem + ".txt")))
    else:
        # layout plano: imagen.jpg + imagen.txt
        for img in source_dir.rglob("*"):
            if img.suffix.lower() in IMG_EXTS:
                pairs.append((img, img.with_suffix(".txt")))
    return pairs


# ---------------------------------------------------------------- main build
def build(dup_threshold: int, val_frac: float, test_frac: float,
          include_negatives: bool) -> None:
    set_seed()
    rng = random.Random(SEED)

    if not paths.RAW.exists() or not any(paths.RAW.iterdir()):
        print(f"No hay fuentes en {paths.RAW}. Ejecuta antes download_data.py.")
        sys.exit(1)

    source_dirs = sorted([p for p in paths.RAW.iterdir() if p.is_dir()])
    print(f"Fuentes detectadas: {[p.name for p in source_dirs]}")

    dedup = PerceptualDedup(threshold=dup_threshold)
    records = []                      # (src, img_path, boxes)
    per_source = defaultdict(lambda: {"images": 0, "boxes": 0, "raw_images": 0, "dups": 0})
    total_raw = 0
    total_dups = 0
    total_unhashable = 0

    for sdir in source_dirs:
        src = sdir.name
        pairs = discover_pairs(sdir)
        per_source[src]["raw_images"] = len(pairs)
        total_raw += len(pairs)
        print(f"  - {src}: {len(pairs)} imágenes crudas")
        for img_path, label_path in pairs:
            boxes = read_label(label_path)
            if not boxes and not include_negatives:
                continue
            h = phash_int(img_path)
            if h is None:
                total_unhashable += 1
                continue
            if dedup.is_duplicate(h):
                per_source[src]["dups"] += 1
                total_dups += 1
                continue
            dedup.add(h)
            records.append((src, img_path, boxes))
            per_source[src]["images"] += 1
            per_source[src]["boxes"] += len(boxes)

    if not records:
        print("No se reunió ninguna imagen utilizable. Aborta.")
        sys.exit(1)

    # ---- split 80/10/10 a nivel de imagen (tras dedup, sin fuga) ----
    rng.shuffle(records)
    n = len(records)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = records[:n_test]
    val = records[n_test:n_test + n_val]
    train = records[n_test + n_val:]
    split_map = {"train": train, "val": val, "test": test}

    # ---- escribir dataset unificado ----
    if paths.UNIFIED.exists():
        shutil.rmtree(paths.UNIFIED)
    for split in paths.SPLITS:
        paths.ensure_dirs(paths.UNIFIED_IMAGES / split, paths.UNIFIED_LABELS / split)

    counter = 0
    for split, recs in split_map.items():
        for src, img_path, boxes in recs:
            counter += 1
            stem = f"{src}__{counter:06d}"
            dst_img = paths.UNIFIED_IMAGES / split / (stem + img_path.suffix.lower())
            dst_lbl = paths.UNIFIED_LABELS / split / (stem + ".txt")
            shutil.copy2(img_path, dst_img)
            lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for (cx, cy, w, h) in boxes]
            dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    # ---- data.yaml del dataset unificado (rutas absolutas) ----
    yaml_text = (
        f"# Generado por prepare_dataset.py (seed={SEED})\n"
        f"path: {paths.UNIFIED.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"nc: 1\n"
        f"names:\n  0: tatuaje\n"
    )
    paths.UNIFIED_YAML.write_text(yaml_text, encoding="utf-8")

    # ---- tabla de procedencia / resumen ----
    rows = []
    for src in sorted(per_source):
        name, url, lic = PROVENANCE.get(src, (src, "-", "desconocida"))
        s = per_source[src]
        rows.append({
            "fuente": name,
            "key": src,
            "imagenes": s["images"],
            "cajas": s["boxes"],
            "duplicados_descartados": s["dups"],
            "licencia": lic,
            "url": url,
        })
    totals = {
        "fuente": "TOTAL (tras dedup)",
        "key": "",
        "imagenes": sum(r["imagenes"] for r in rows),
        "cajas": sum(r["cajas"] for r in rows),
        "duplicados_descartados": total_dups,
        "licencia": "",
        "url": "",
    }
    rows.append(totals)

    headers = ["fuente", "key", "imagenes", "cajas", "duplicados_descartados", "licencia", "url"]
    paths.ensure_dirs(paths.REPORTS)
    md = write_markdown_table(rows, headers, paths.REPORTS / "dataset_summary.md")
    write_csv(rows, headers, paths.REPORTS / "dataset_summary.csv")

    # nota de splits añadida al markdown
    split_note = (
        f"\n**Splits (seed={SEED}):** train={len(train)}, val={len(val)}, test={len(test)} "
        f"(total {n}).\n"
        f"**Imágenes crudas vistas:** {total_raw} | "
        f"**duplicados descartados:** {total_dups} | "
        f"**no procesables:** {total_unhashable}.\n"
    )
    (paths.REPORTS / "dataset_summary.md").write_text(md + split_note, encoding="utf-8")

    print("\n" + md + split_note)
    print(f"Dataset unificado en: {paths.UNIFIED}")
    print(f"data.yaml: {paths.UNIFIED_YAML}")
    print(f"Resumen: {paths.REPORTS / 'dataset_summary.md'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Unifica fuentes a dataset YOLO 1 clase")
    ap.add_argument("--dup-threshold", type=int, default=6,
                    help="umbral de hamming para considerar duplicado perceptual")
    ap.add_argument("--val", type=float, default=0.1, help="fracción de validación")
    ap.add_argument("--test", type=float, default=0.1, help="fracción de test")
    ap.add_argument("--no-negatives", action="store_true",
                    help="descarta imágenes sin cajas (por defecto se conservan)")
    args = ap.parse_args()
    build(args.dup_threshold, args.val, args.test, include_negatives=not args.no_negatives)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
