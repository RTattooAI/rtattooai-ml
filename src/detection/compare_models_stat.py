"""Test estadístico pareado: yolov8m vs yolov8s sobre el MISMO test set.

evaluate.py solo reporta métricas agregadas (mAP/P/R) de Ultralytics, que no
alcanzan para un test pareado. Aquí se corre inferencia con ambos modelos
sobre las mismas 701 imágenes de test, se calcula por imagen precisión/
recall/F1 (matching greedy por IoU≥0.5 contra el ground truth) y se compara
la diferencia con un Wilcoxon signed-rank (test no paramétrico para 2
muestras pareadas — Friedman aplicaría con 3+ modelos, no es el caso aquí).

Uso:
    python -m src.detection.compare_models_stat
    python -m src.detection.compare_models_stat --conf 0.25 --iou-match 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Lee un .txt YOLO (class xc yc w h, normalizado) -> array (N,4) xyxy en píxeles."""
    if not label_path.exists():
        return np.zeros((0, 4))
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, w, h = (float(v) for v in parts[:5])
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return np.array(boxes) if boxes else np.zeros((0, 4))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU pareado entre dos conjuntos de cajas xyxy. Devuelve (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1, iy1 = np.maximum(ax1, bx1), np.maximum(ay1, by1)
    ix2, iy2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)
    iw, ih = np.clip(ix2 - ix1, 0, None), np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def per_image_scores(pred_boxes: np.ndarray, pred_conf: np.ndarray, gt_boxes: np.ndarray,
                      iou_thr: float) -> dict:
    """TP/FP/FN por imagen (greedy, predicciones ordenadas por confianza desc)."""
    order = np.argsort(-pred_conf)
    pred_boxes = pred_boxes[order]
    matched_gt = np.zeros(len(gt_boxes), dtype=bool)
    tp = 0
    ious_tp = []
    ious = iou_matrix(pred_boxes, gt_boxes)
    for i in range(len(pred_boxes)):
        if len(gt_boxes) == 0:
            break
        row = ious[i].copy()
        row[matched_gt] = -1.0
        j = int(np.argmax(row))
        if row[j] >= iou_thr:
            matched_gt[j] = True
            tp += 1
            ious_tp.append(float(row[j]))
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if len(gt_boxes) == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_iou = float(np.mean(ious_tp)) if ious_tp else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": f1, "mean_iou": mean_iou}


def run_model(weights: Path, images: list[Path], labels_dir: Path, conf: float, iou_nms: float,
              iou_match: float, imgsz: int = 640, batch: int = 16) -> list[dict]:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    rows = []
    for i in range(0, len(images), batch):
        chunk = images[i:i + batch]
        results = model.predict(source=[str(p) for p in chunk], conf=conf, iou=iou_nms,
                                imgsz=imgsz, verbose=False)
        for p, res in zip(chunk, results):
            h, w = res.orig_shape
            gt = load_gt_boxes(labels_dir / f"{p.stem}.txt", w, h)
            if res.boxes is not None and len(res.boxes) > 0:
                pb = res.boxes.xyxy.cpu().numpy()
                pc = res.boxes.conf.cpu().numpy()
            else:
                pb, pc = np.zeros((0, 4)), np.zeros((0,))
            rows.append({"image": p.stem, **per_image_scores(pb, pc, gt, iou_match)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Test estadístico pareado yolov8m vs yolov8s")
    ap.add_argument("--model-a", default=str(paths.MODELS / "yolov8m_best.pt"))
    ap.add_argument("--model-b", default=str(paths.MODELS / "yolov8s_best.pt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-nms", type=float, default=0.45)
    ap.add_argument("--iou-match", type=float, default=0.5)
    args = ap.parse_args()

    images_dir = paths.UNIFIED_IMAGES / args.split
    labels_dir = paths.UNIFIED_LABELS / args.split
    images = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        print(f"ERROR: sin imágenes en {images_dir}")
        return 1
    print(f"Evaluando {len(images)} imágenes de '{args.split}' con ambos modelos "
          f"(conf={args.conf}, iou_nms={args.iou_nms}, iou_match={args.iou_match})…")

    name_a, name_b = Path(args.model_a).stem.replace("_best", ""), Path(args.model_b).stem.replace("_best", "")
    rows_a = run_model(Path(args.model_a), images, labels_dir, args.conf, args.iou_nms, args.iou_match)
    rows_b = run_model(Path(args.model_b), images, labels_dir, args.conf, args.iou_nms, args.iou_match)

    by_a = {r["image"]: r for r in rows_a}
    by_b = {r["image"]: r for r in rows_b}
    common = [s for s in by_a if s in by_b]
    f1_a = np.array([by_a[s]["f1"] for s in common])
    f1_b = np.array([by_b[s]["f1"] for s in common])
    iou_a = np.array([by_a[s]["mean_iou"] for s in common])
    iou_b = np.array([by_b[s]["mean_iou"] for s in common])

    from scipy.stats import wilcoxon

    def wilcoxon_safe(x, y):
        diff = x - y
        if np.allclose(diff, 0):
            return None
        try:
            stat, p = wilcoxon(x, y)
            return float(stat), float(p)
        except ValueError:
            return None

    res_f1 = wilcoxon_safe(f1_a, f1_b)
    res_iou = wilcoxon_safe(iou_a, iou_b)

    paths.ensure_dirs(paths.REPORTS)
    lines = [
        f"# Test estadístico pareado — {name_a} vs {name_b} (split `{args.split}`)\n",
        f"- Imágenes comparadas (mismo set, ambos modelos): **{len(common)}**.",
        f"- Operación: conf={args.conf}, IoU-NMS={args.iou_nms}, IoU-match (TP)={args.iou_match}.",
        "- Test: **Wilcoxon signed-rank** (2 muestras pareadas; Friedman no aplica con solo 2 modelos).\n",
        "## Resumen por imagen",
        "| Métrica | " + name_a + " (media ± sd) | " + name_b + " (media ± sd) |",
        "| --- | ---: | ---: |",
        f"| F1 @ IoU≥{args.iou_match} | {f1_a.mean():.4f} ± {f1_a.std():.4f} | "
        f"{f1_b.mean():.4f} ± {f1_b.std():.4f} |",
        f"| IoU medio (TPs) | {iou_a.mean():.4f} ± {iou_a.std():.4f} | "
        f"{iou_b.mean():.4f} ± {iou_b.std():.4f} |\n",
        "## Wilcoxon signed-rank",
    ]
    if res_f1:
        lines.append(f"- **F1 por imagen**: W = {res_f1[0]:.1f}, p = {res_f1[1]:.4f} → "
                     + ("diferencia significativa (α=0.05)" if res_f1[1] < 0.05
                        else "**sin diferencia significativa** (α=0.05)"))
    else:
        lines.append("- F1 por imagen: todas las diferencias son 0 (no aplica el test).")
    if res_iou:
        lines.append(f"- **IoU medio por imagen**: W = {res_iou[0]:.1f}, p = {res_iou[1]:.4f} → "
                     + ("diferencia significativa (α=0.05)" if res_iou[1] < 0.05
                        else "**sin diferencia significativa** (α=0.05)"))
    else:
        lines.append("- IoU medio por imagen: todas las diferencias son 0 (no aplica el test).")
    report = "\n".join(lines) + "\n"
    out_path = paths.REPORTS / f"stat_test_{name_a}_vs_{name_b}.md"
    out_path.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Reporte: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
