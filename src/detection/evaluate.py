"""Evaluación de los modelos YOLOv8 entrenados y generación de reportes.

Reporta sobre val y test: mAP@0.5, mAP@0.5:0.95, precisión y recall.
Genera en reports/:
  - metrics.md / metrics.csv      (tabla comparativa de modelos x split)
  - *_PR_curve.png                (curva precisión-recall)
  - *_confusion_matrix.png        (matriz de confusión)
  - predictions_<modelo>.png      (grilla de predicciones de ejemplo)
Compara explícitamente contra:
  - baseline mAP@0.5 = 0.79 (Pocevičė et al., 2024, YOLOv5/DeMSI)
  - meta tesis: mAP@0.5 > 0.85 y precisión >= 0.80

Uso:
    python -m src.detection.evaluate
    python -m src.detection.evaluate --models models/yolov8m_best.pt
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.shared.seeds import set_seed
from src.shared.metrics import (
    write_markdown_table, write_csv, goal_verdict,
    BASELINE_MAP50, GOAL_MAP50, GOAL_PRECISION,
)


def find_models(explicit: list[str] | None) -> list[Path]:
    if explicit:
        return [Path(m) for m in explicit]
    if paths.MODELS.exists():
        found = sorted(paths.MODELS.glob("*_best.pt"))
        if found:
            return found
    # fallback: pesos en runs/detect/*/weights/best.pt
    return sorted(paths.RUNS.glob("detect/*/weights/best.pt"))


def evaluate_model(weights: Path, data_yaml: Path, split: str, device: int):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    name = weights.stem.replace("_best", "")
    run_name = f"{name}_{split}"
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        project=str(paths.RUNS / "val"),
        name=run_name,
        exist_ok=True,
        plots=True,
        device=device,
        verbose=False,
    )
    row = {
        "modelo": name,
        "split": split,
        "mAP@0.5": round(float(metrics.box.map50), 4),
        "mAP@0.5:0.95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }
    # copiar figuras al reports/
    run_dir = paths.RUNS / "val" / run_name
    paths.ensure_dirs(paths.REPORTS)
    for pattern, out in (("*PR_curve.png", f"{run_name}_PR_curve.png"),
                         ("*confusion_matrix.png", f"{run_name}_confusion_matrix.png")):
        for f in run_dir.glob(pattern):
            shutil.copy2(f, paths.REPORTS / out)
            break
    return name, row


def prediction_grid(weights: Path, n: int, device: int) -> None:
    """Genera una grilla de predicciones sobre imágenes de test."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    test_imgs = sorted((paths.UNIFIED_IMAGES / "test").glob("*"))
    test_imgs = [p for p in test_imgs if p.suffix.lower() in {".jpg", ".jpeg", ".png"}][:n]
    if not test_imgs:
        print("Sin imágenes de test para la grilla de predicciones.")
        return
    model = YOLO(str(weights))
    results = model.predict(source=[str(p) for p in test_imgs], conf=0.25,
                            device=device, verbose=False)
    cols = min(3, len(results))
    rows = (len(results) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, res in zip(axes, results):
        im = res.plot()[:, :, ::-1]  # BGR -> RGB
        ax.imshow(im)
        ax.axis("off")
    for ax in axes[len(results):]:
        ax.axis("off")
    fig.suptitle(f"Predicciones — {weights.stem}")
    fig.tight_layout()
    out = paths.REPORTS / f"predictions_{weights.stem.replace('_best','')}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Grilla de predicciones: {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluación YOLOv8 (tatuajes)")
    ap.add_argument("--models", nargs="*", help="rutas a pesos .pt (best)")
    ap.add_argument("--data", default=str(paths.UNIFIED_YAML))
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--grid", type=int, default=6, help="nº de imágenes en la grilla")
    args = ap.parse_args()

    set_seed()
    try:
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"ERROR: no existe {data_yaml}. Ejecuta prepare_dataset.py y train.py.")
        return 1

    models = find_models(args.models)
    if not models:
        print("ERROR: no se encontraron pesos entrenados (models/*_best.pt).")
        return 1
    print(f"Modelos a evaluar: {[m.name for m in models]}")

    rows = []
    primary_name = models[0].stem.replace("_best", "")
    primary_test_row = None
    for w in models:
        for split in args.splits:
            name, row = evaluate_model(w, data_yaml, split, device)
            rows.append(row)
            if name == primary_name and split == "test":
                primary_test_row = row

    headers = ["modelo", "split", "mAP@0.5", "mAP@0.5:0.95", "precision", "recall"]
    md = write_markdown_table(rows, headers, paths.REPORTS / "metrics.md")
    write_csv(rows, headers, paths.REPORTS / "metrics.csv")

    # veredicto vs baseline/meta (sobre el modelo definitivo en test)
    verdict = ""
    if primary_test_row:
        verdict = (
            f"\n## Comparación contra baseline y meta\n\n"
            f"- **Baseline** (Pocevičė et al., 2024, YOLOv5/DeMSI): mAP@0.5 = {BASELINE_MAP50}\n"
            f"- **Meta tesis**: mAP@0.5 > {GOAL_MAP50} y precisión >= {GOAL_PRECISION}\n\n"
            f"**Modelo definitivo `{primary_name}` (split test):**\n\n"
            f"```\n{goal_verdict(primary_test_row['mAP@0.5'], primary_test_row['precision'])}\n```\n"
        )
    (paths.REPORTS / "metrics.md").write_text(md + verdict, encoding="utf-8")

    print("\n" + md + verdict)

    # grilla de predicciones del modelo definitivo
    prediction_grid(models[0], args.grid, device)

    print(f"\nReportes en: {paths.REPORTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
