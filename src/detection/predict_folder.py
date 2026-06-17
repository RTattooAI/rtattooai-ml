"""Corre el detector entrenado sobre TODAS las imágenes de una carpeta.

Uso cualitativo (NO calcula mAP: para eso hacen falta anotaciones de verdad).
Sirve para ver el comportamiento del modelo en un conjunto externo —p. ej.
DeMSI o las fotos reales peruanas— y medir la "brecha sim-to-real".

Genera en reports/external_<nombre>/:
  - detections.csv         (por imagen: nº de cajas, confianza máx.)
  - summary.md             (estadísticas: % con detección, confianza media…)
  - predictions_grid.png   (muestra visual de predicciones)

Uso:
    python -m src.detection.predict_folder --source datasets/raw/demsi --name demsi
    python -m src.detection.predict_folder --source datasets/real_test/images --name real_test --conf 0.25
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.shared.seeds import set_seed
from src.shared.metrics import write_csv

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS)


def main() -> int:
    ap = argparse.ArgumentParser(description="Inferencia cualitativa sobre una carpeta")
    ap.add_argument("--source", required=True, help="carpeta (o imagen) de entrada")
    ap.add_argument("--name", default=None, help="nombre del reporte (por defecto, el de la carpeta)")
    ap.add_argument("--weights", default=str(paths.MODELS / "yolov8m_best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--grid", type=int, default=20, help="nº de imágenes en la grilla")
    ap.add_argument("--save-annotated", action="store_true",
                    help="guarda TODAS las imágenes anotadas (ocupa disco)")
    args = ap.parse_args()

    set_seed()
    source = Path(args.source)
    name = args.name or source.name
    weights = Path(args.weights)
    if not weights.exists():
        print(f"ERROR: no existe {weights}. Entrena/exporta primero.")
        return 1
    images = list_images(source)
    if not images:
        print(f"ERROR: no se hallaron imágenes en {source}")
        return 1
    print(f"Imágenes: {len(images)} | modelo: {weights.name} | conf={args.conf} iou={args.iou}")

    try:
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    from ultralytics import YOLO
    model = YOLO(str(weights))

    out_dir = paths.REPORTS / f"external_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = out_dir / "annotated"
    if args.save_annotated:
        ann_dir.mkdir(exist_ok=True)

    # ---- pasada 1: estadísticas (por lotes; libera VRAM entre lotes) ----
    # Se procesa en chunks con imgsz fijo para no agotar memoria en GPUs chicas.
    try:
        import torch
    except ImportError:
        torch = None
    rows = []
    CHUNK = 16
    # se pasan RUTAS de archivo: Ultralytics las lee con cv2 (BGR correcto)
    for start in range(0, len(images), CHUNK):
        chunk = images[start:start + CHUNK]
        res_list = model.predict(source=[str(p) for p in chunk], conf=args.conf,
                                 iou=args.iou, imgsz=640, device=device, verbose=False)
        for img_path, res in zip(chunk, res_list):
            confs = [float(b.conf[0]) for b in (res.boxes or [])]
            rows.append({
                "imagen": img_path.name,
                "n_detecciones": len(confs),
                "confianza_max": round(max(confs), 4) if confs else 0.0,
                "confianza_media": round(sum(confs) / len(confs), 4) if confs else 0.0,
            })
            if args.save_annotated:
                from PIL import Image
                Image.fromarray(res.plot()[:, :, ::-1]).save(ann_dir / img_path.name)
        if torch is not None and device != "cpu":
            torch.cuda.empty_cache()

    # ---- estadísticas agregadas ----
    n = len(rows)
    with_det = [r for r in rows if r["n_detecciones"] > 0]
    pct = 100 * len(with_det) / n if n else 0
    total_boxes = sum(r["n_detecciones"] for r in rows)
    maxconfs = [r["confianza_max"] for r in with_det]
    summary = (
        f"# Inferencia externa — {name}\n\n"
        f"> Reporte **cualitativo** (sin anotaciones de verdad → **no** es mAP).\n"
        f"> Modelo: `{weights.name}`, conf={args.conf}, iou={args.iou}.\n\n"
        f"- Imágenes procesadas: **{n}**\n"
        f"- Imágenes con ≥1 detección: **{len(with_det)} ({pct:.1f}%)**\n"
        f"- Cajas detectadas en total: **{total_boxes}**\n"
    )
    if maxconfs:
        summary += (
            f"- Confianza máx. por imagen (en las que detecta): "
            f"media={st.mean(maxconfs):.3f}, mediana={st.median(maxconfs):.3f}, "
            f"mín={min(maxconfs):.3f}, máx={max(maxconfs):.3f}\n"
        )
    summary += (
        f"\n*Nota:* sin etiquetas de referencia no se puede saber cuántas de "
        f"estas detecciones son correctas (precisión) ni cuántos tatuajes se "
        f"escaparon (recall). Para métricas reales se necesitan las anotaciones.\n"
    )

    write_csv(rows, ["imagen", "n_detecciones", "confianza_max", "confianza_media"],
              out_dir / "detections.csv")
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")

    # ---- pasada 2: grilla visual (muestra repartida por confianza) ----
    if args.grid > 0:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = sorted(range(n), key=lambda i: rows[i]["confianza_max"], reverse=True)
        k = min(args.grid, n)
        idxs = [order[int(round(i * (n - 1) / max(k - 1, 1)))] for i in range(k)]
        sample = [images[i] for i in idxs]
        res2 = model.predict(source=[str(p) for p in sample], conf=args.conf,
                             iou=args.iou, imgsz=640, device=device, verbose=False)
        cols = 4
        rws = (len(res2) + cols - 1) // cols
        fig, axes = plt.subplots(rws, cols, figsize=(4 * cols, 4 * rws))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, p, r in zip(axes, sample, res2):
            ax.imshow(r.plot()[:, :, ::-1])
            nd = 0 if r.boxes is None else len(r.boxes)
            ax.set_title(f"{nd} det", fontsize=9)
            ax.axis("off")
        for ax in axes[len(res2):]:
            ax.axis("off")
        fig.suptitle(f"Predicciones — {name} ({weights.stem})")
        fig.tight_layout()
        fig.savefig(out_dir / "predictions_grid.png", dpi=110)
        plt.close(fig)

    print("\n" + summary)
    print(f"Reportes en: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
