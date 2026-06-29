"""Métricas extendidas de identificación: Top-k, CMC, mAP de retrieval y FAR/FRR.

Completa lo que pide el paper y que step0_baseline/gallery_ablation no reportan:
  - Top-1 / Top-5 / Top-10 y curva CMC completa (rank 1..N identidades).
  - mAP de retrieval (cada consulta tiene exactamente 1 identidad relevante:
    AP = 1/rank de la identidad correcta).
  - FAR/FRR en función del umbral ρ (margen fijo = el calibrado), con el
    punto de operación actual marcado, más el EER (equal error rate).

Usa la GALERÍA DE PRODUCCIÓN actual (models/gallery_dinov2.npz: 25 identidades
× 17 vistas, embedding con la cabeza afinada) — el mismo artefacto que usa
serving/. Las consultas son SINTÉTICAS (variaciones 2D de step0_baseline) y
los distractores son tatuajes NO-banda recortados con YOLO del test set: por
tanto estas cifras son orientativas, no la validación final con fotos reales.

Uso:
    python -m src.embedding.identification_metrics
    python -m src.embedding.identification_metrics --queries-per 20 --distractors 150
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
from src.shared.seeds import set_seed, SEED
from src.embedding.embedder import TattooEmbedder
from src.embedding.step0_baseline import make_query, crop_distractors
from src.embedding.identify import GALLERY_NPZ, HEAD_PATH, DETECTOR, load_config

GANG_DIR = paths.DATASETS / "gang_tattoos"


def per_identity_ranking(gallery: np.ndarray, names: list[str], qvec: np.ndarray):
    """Similitud MÁXIMA por identidad (multi-vista), orden descendente.

    Devuelve [(identidad, score)] de mayor a menor.
    """
    sims = gallery @ qvec
    best: dict[str, float] = {}
    for n, s in zip(names, sims):
        s = float(s)
        if n not in best or s > best[n]:
            best[n] = s
    return sorted(best.items(), key=lambda kv: -kv[1])


def main() -> int:
    ap = argparse.ArgumentParser(description="Top-k / CMC / mAP / FAR-FRR de identificación")
    ap.add_argument("--gallery", default=str(GALLERY_NPZ))
    ap.add_argument("--designs", default=str(GANG_DIR), help="diseños limpios para generar consultas")
    ap.add_argument("--queries-per", type=int, default=20)
    ap.add_argument("--distractors", type=int, default=150)
    ap.add_argument("--weights", default=str(DETECTOR))
    args = ap.parse_args()

    set_seed()
    rng = np.random.RandomState(SEED)

    gallery_path = Path(args.gallery)
    if not gallery_path.exists():
        print(f"ERROR: no existe la galería {gallery_path}.")
        return 1
    data = np.load(str(gallery_path), allow_pickle=True)
    gallery = data["embeddings"].astype("float32")
    names = [str(n) for n in data["names"]]
    identities = sorted(set(names))
    n_ident = len(identities)
    print(f"Galería: {n_ident} identidades, {len(names)} vistas en total.")

    cfg = load_config()
    margin = cfg["match_margin"]
    op_threshold = cfg["match_threshold"]

    emb = TattooEmbedder(head_path=str(HEAD_PATH) if HEAD_PATH.exists() else None)

    gfiles = sorted(p for p in Path(args.designs).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not gfiles:
        print(f"ERROR: sin diseños en {args.designs}")
        return 1
    design_names = [p.stem.replace("-removebg-preview", "") for p in gfiles]
    missing = set(design_names) - set(identities)
    if missing:
        print(f"AVISO: diseños sin identidad en la galería (se omiten): {missing}")

    # ---- consultas genuinas: rank de la identidad correcta por consulta ----
    ranks: list[int] = []
    for design, src_img in zip(design_names, (Image.open(p) for p in gfiles)):
        if design not in identities:
            continue
        for _ in range(args.queries_per):
            q = emb.embed(make_query(src_img, rng))
            ranked = per_identity_ranking(gallery, names, q)
            rank = next(i for i, (ident, _) in enumerate(ranked, start=1) if ident == design)
            ranks.append(rank)
    ranks = np.array(ranks)
    n_q = len(ranks)
    print(f"Consultas genuinas evaluadas: {n_q}")

    top1 = float(np.mean(ranks <= 1))
    top5 = float(np.mean(ranks <= 5))
    top10 = float(np.mean(ranks <= 10))
    retrieval_map = float(np.mean(1.0 / ranks))

    cmc = [float(np.mean(ranks <= k)) for k in range(1, n_ident + 1)]

    # ---- accept/reject (top1 vs umbral+margen) para genuinas y distractores ----
    def decision_scores(qvec):
        ranked = per_identity_ranking(gallery, names, qvec)
        t1 = ranked[0][1]
        t2 = ranked[1][1] if len(ranked) > 1 else -1.0
        return t1, t1 - t2

    genuine_scores = []  # (top1, gap)
    for design, src_img in zip(design_names, (Image.open(p) for p in gfiles)):
        if design not in identities:
            continue
        for _ in range(args.queries_per):
            q = emb.embed(make_query(src_img, rng))
            genuine_scores.append(decision_scores(q))
    genuine_scores = np.array(genuine_scores)

    print(f"Recortando ~{args.distractors} distractores no-banda con YOLO…")
    distractors = crop_distractors(Path(args.weights), args.distractors, rng)
    distractor_scores = []
    if distractors:
        for img in distractors:
            q = emb.embed(img)
            distractor_scores.append(decision_scores(q))
    distractor_scores = np.array(distractor_scores)
    print(f"Distractores evaluados: {len(distractor_scores)}")

    thresholds = np.round(np.arange(0.30, 0.96, 0.01), 3)
    far, frr = [], []
    for thr in thresholds:
        accepted_g = (genuine_scores[:, 0] >= thr) & (genuine_scores[:, 1] >= margin)
        frr.append(float(1.0 - accepted_g.mean()))
        if len(distractor_scores):
            accepted_d = (distractor_scores[:, 0] >= thr) & (distractor_scores[:, 1] >= margin)
            far.append(float(accepted_d.mean()))
        else:
            far.append(float("nan"))
    far, frr = np.array(far), np.array(frr)

    eer_idx = int(np.argmin(np.abs(far - frr)))
    eer_threshold = float(thresholds[eer_idx])
    eer_value = float((far[eer_idx] + frr[eer_idx]) / 2)

    op_idx = int(np.argmin(np.abs(thresholds - op_threshold)))
    far_at_op = float(far[op_idx])
    frr_at_op = float(frr[op_idx])

    # ---- reporte ----
    paths.ensure_dirs(paths.REPORTS)
    lines = [
        "# Métricas extendidas de identificación — Top-k, CMC, mAP, FAR/FRR\n",
        "> Galería de PRODUCCIÓN actual (`models/gallery_dinov2.npz`): "
        f"**{n_ident}** identidades × ~17 vistas, embedding DINOv2 + cabeza afinada.\n"
        "> Consultas: variaciones sintéticas 2D por diseño (rotación/color/fondo/ruido) — "
        "**orientativo**, no sustituye la validación con fotos reales "
        "(ver `reports/real_photo_validation.md`).\n",
        "## Top-k y mAP de retrieval",
        f"- Consultas evaluadas: **{n_q}** ({args.queries_per} por diseño × {len(design_names)} diseños).",
        f"- **Top-1: {100*top1:.1f}%** | **Top-5: {100*top5:.1f}%** | **Top-10: {100*top10:.1f}%**",
        f"- **mAP de retrieval: {retrieval_map:.3f}** (AP por consulta = 1/rank de la identidad correcta).\n",
        "## Curva CMC (rank-k cumulative match)",
        "| k | tasa acumulada |",
        "| ---: | ---: |",
    ]
    for k in (1, 2, 3, 5, 10, 15, 20, n_ident):
        if k <= n_ident:
            lines.append(f"| {k} | {100*cmc[k-1]:.1f}% |")
    lines += [
        "\n## FAR / FRR en función del umbral ρ (margen fijo = "
        f"{margin})",
        f"- Punto de operación actual: **ρ = {op_threshold}** → "
        f"FAR ≈ **{100*far_at_op:.1f}%**, FRR ≈ **{100*frr_at_op:.1f}%**.",
        f"- EER (equal error rate): **{100*eer_value:.1f}%** en ρ ≈ **{eer_threshold}**.\n",
        "| ρ | FAR | FRR |",
        "| ---: | ---: | ---: |",
    ]
    for thr, f_, r_ in zip(thresholds, far, frr):
        if round(float(thr) * 100) % 5 == 0:  # cada 0.05 para no inflar la tabla
            lines.append(f"| {thr:.2f} | {100*f_:.1f}% | {100*r_:.1f}% |")
    report = "\n".join(lines) + "\n"
    (paths.REPORTS / "identification_topk_cmc_far_frr.md").write_text(report, encoding="utf-8")

    # JSON con los valores crudos (para la Tabla II / Fig. 3 del paper)
    out = {
        "n_identidades": n_ident, "n_consultas": n_q,
        "top1": top1, "top5": top5, "top10": top10, "retrieval_map": retrieval_map,
        "cmc": cmc, "match_margin": margin, "op_threshold": op_threshold,
        "far_at_op": far_at_op, "frr_at_op": frr_at_op,
        "eer_threshold": eer_threshold, "eer_value": eer_value,
        "thresholds": thresholds.tolist(), "far": far.tolist(), "frr": frr.tolist(),
    }
    (paths.REPORTS / "identification_topk_cmc_far_frr.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, n_ident + 1), [100 * c for c in cmc], "o-", ms=3)
        ax.set_xlabel("rank k"); ax.set_ylabel("tasa acumulada (%)")
        ax.set_title("Curva CMC — identificación por embeddings")
        ax.set_ylim(0, 101); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(paths.REPORTS / "cmc_curve.png", dpi=110)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(thresholds, 100 * far, label="FAR (no-banda aceptada)")
        ax.plot(thresholds, 100 * frr, label="FRR (genuina rechazada)")
        ax.axvline(op_threshold, color="k", ls="--", lw=1, label=f"ρ operación = {op_threshold}")
        ax.scatter([eer_threshold], [100 * eer_value], color="red", zorder=5,
                   label=f"EER ≈ {100*eer_value:.1f}% @ ρ={eer_threshold}")
        ax.set_xlabel("umbral ρ"); ax.set_ylabel("tasa (%)")
        ax.set_title("FAR / FRR vs. umbral ρ (margen fijo)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(paths.REPORTS / "far_frr_sweep.png", dpi=110)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"(aviso: sin gráficos: {e})")

    print("\n" + report)
    print(f"Reportes: {paths.REPORTS / 'identification_topk_cmc_far_frr.md'} "
          f"(+ .json, cmc_curve.png, far_frr_sweep.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
