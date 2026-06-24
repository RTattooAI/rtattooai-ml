"""Ablación: ¿cuánto aporta la galería MULTI-VISTA (Blender) vs la de solo-diseño?

Compara dos galerías sobre el MISMO conjunto de consultas + distractores:
  - Galería A (baseline): solo el diseño limpio por tatuaje.
  - Galería B (enriquecida): diseño + vistas realistas (crops de Blender / reales).

Reporta para cada una: top-1, separación genuino vs no-banda, y umbral sugerido;
y el DELTA (B − A). Es la evidencia para la tesis de que el sintético mejora.

Uso (lo ideal: consultas = fotos reales held-out, distintas de la galería):
    python -m src.embedding.compare_galleries \
        --gallery-a datasets/gang_tattoos \
        --gallery-b datasets/gang_tattoos OUTPUT/crops \
        --queries datasets/real_queries        # subcarpeta = identidad
Si no hay fotos reales aún, usa consultas SINTÉTICAS (solo orientativo, no para
la conclusión final de la tesis):
    python -m src.embedding.compare_galleries --synthetic-queries 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.shared.seeds import set_seed, SEED
from src.embedding.embedder import TattooEmbedder
from src.embedding.build_gallery import collect, IMG_EXTS
from src.embedding.step0_baseline import make_query, crop_distractors


def gallery_embeddings(sources, emb):
    items = []
    for s in sources:
        items += collect(Path(s))
    names = [n for n, _ in items]
    vecs = emb.embed([Image.open(p) for _, p in items], bg=(255, 255, 255))
    return vecs.astype("float32"), names


def per_identity_best(vecs, names, qvec):
    sims = vecs @ qvec
    best = {}
    for n, s in zip(names, sims):
        s = float(s)
        if n not in best or s > best[n]:
            best[n] = s
    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    return ranked  # [(identidad, score)] desc


def evaluate(vecs, names, query_embs, query_idents, distractor_embs):
    top1 = total = 0
    genuine = []
    for qe, ident in zip(query_embs, query_idents):
        ranked = per_identity_best(vecs, names, qe)
        total += 1
        if ranked and ranked[0][0] == ident:
            top1 += 1
        genuine.append(ranked[0][1] if ranked else 0.0)
    distractor = [per_identity_best(vecs, names, de)[0][1] for de in distractor_embs]
    g, d = np.array(genuine), np.array(distractor) if distractor else np.array([])
    thr = float((g.mean() + d.mean()) / 2) if d.size else float(g.mean() - 0.1)
    return {
        "top1": top1 / total if total else 0.0,
        "genuine_mean": float(g.mean()), "genuine_min": float(g.min()),
        "distractor_mean": float(d.mean()) if d.size else None,
        "gap": float(g.mean() - d.mean()) if d.size else None,
        "threshold": thr, "genuine": g, "distractor": d,
    }


def build_synthetic_queries(emb, k, rng):
    """Consultas sintéticas: variaciones 2D de los diseños limpios (orientativo)."""
    gfiles = sorted(p for p in (paths.DATASETS / "gang_tattoos").glob("*")
                    if p.suffix.lower() in IMG_EXTS)
    imgs, idents = [], []
    for p in gfiles:
        for _ in range(k):
            imgs.append(make_query(Image.open(p), rng))
            idents.append(p.stem.replace("-removebg-preview", ""))
    return emb.embed(imgs), idents


def load_real_queries(folder: Path, emb):
    items = collect(folder)   # subcarpeta = identidad
    if not items:
        return None, None
    idents = [n for n, _ in items]
    embs = emb.embed([Image.open(p) for _, p in items])
    return embs, idents


def main() -> int:
    ap = argparse.ArgumentParser(description="Ablación de galería: solo-diseño vs multi-vista")
    ap.add_argument("--gallery-a", nargs="+", default=[str(paths.DATASETS / "gang_tattoos")])
    ap.add_argument("--gallery-b", nargs="+",
                    default=[str(paths.DATASETS / "gang_tattoos"), "OUTPUT/crops"])
    ap.add_argument("--queries", default=None, help="carpeta de consultas reales (subcarpeta=identidad)")
    ap.add_argument("--synthetic-queries", type=int, default=0,
                    help="si no hay reales: nº de variaciones sintéticas por tatuaje")
    ap.add_argument("--distractors", type=int, default=60)
    ap.add_argument("--head", default=None, help="ruta a la cabeza afinada (opcional)")
    args = ap.parse_args()

    set_seed()
    rng = np.random.RandomState(SEED)
    emb = TattooEmbedder(head_path=args.head)

    # consultas
    if args.queries:
        q_embs, q_idents = load_real_queries(Path(args.queries), emb)
        qsrc = f"reales ({args.queries})"
    elif args.synthetic_queries > 0:
        q_embs, q_idents = build_synthetic_queries(emb, args.synthetic_queries, rng)
        qsrc = f"sintéticas ({args.synthetic_queries}/tatuaje) — orientativo"
    else:
        print("ERROR: da --queries CARPETA o --synthetic-queries N")
        return 1
    if q_embs is None or len(q_embs) == 0:
        print("ERROR: sin consultas.")
        return 1

    # distractores no-banda (open-set)
    distractors = crop_distractors(paths.MODELS / "yolov8m_best.pt", args.distractors, rng)
    d_embs = emb.embed(distractors) if distractors else []

    # galerías
    va, na = gallery_embeddings(args.gallery_a, emb)
    vb, nb = gallery_embeddings(args.gallery_b, emb)
    ra = evaluate(va, na, q_embs, q_idents, d_embs)
    rb = evaluate(vb, nb, q_embs, q_idents, d_embs)

    def fmt(r):
        dm = "-" if r["distractor_mean"] is None else f"{r['distractor_mean']:.3f}"
        gp = "-" if r["gap"] is None else f"{r['gap']:+.3f}"
        return (f"top1={100*r['top1']:.1f}% | genuino μ={r['genuine_mean']:.3f} "
                f"(min {r['genuine_min']:.3f}) | no-banda μ={dm} "
                f"| brecha={gp} | umbral≈{r['threshold']:.3f}")

    a_views = len(na); b_views = len(nb)
    lines = [
        "# Ablación de galería — solo-diseño (A) vs multi-vista (B)\n",
        f"- Consultas: **{len(q_idents)}** ({qsrc}); distractores: {len(d_embs)}.",
        f"- Galería A: {a_views} vistas | Galería B: {b_views} vistas.\n",
        f"**A (baseline):** {fmt(ra)}",
        f"**B (multi-vista):** {fmt(rb)}\n",
        f"**Δ top-1:** {100*(rb['top1']-ra['top1']):+.1f} pts | "
        f"**Δ genuino:** {rb['genuine_mean']-ra['genuine_mean']:+.3f}"
        + ("" if ra['gap'] is None else f" | **Δ brecha:** {(rb['gap']-ra['gap']):+.3f}"),
    ]
    if args.synthetic_queries > 0 and not args.queries:
        lines.append("\n> ⚠️ Consultas SINTÉTICAS: sirve para ver mecánica/tendencia, "
                     "no como conclusión final. Repetir con **fotos reales** para la tesis.")
    report = "\n".join(lines) + "\n"
    paths.ensure_dirs(paths.REPORTS)
    (paths.REPORTS / "gallery_ablation.md").write_text(report, encoding="utf-8")

    # gráfico comparativo
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
        for ax, r, ttl in ((axes[0], ra, f"A: solo-diseño ({a_views})"),
                           (axes[1], rb, f"B: multi-vista ({b_views})")):
            ax.hist(r["genuine"], bins=18, alpha=0.6, label="genuino")
            if r["distractor"].size:
                ax.hist(r["distractor"], bins=18, alpha=0.6, label="no-banda")
            ax.axvline(r["threshold"], color="k", ls="--", lw=1)
            ax.set_title(f"{ttl}\ntop1={100*r['top1']:.0f}%"); ax.legend()
        fig.suptitle("Separación banda vs no-banda: galería A vs B")
        fig.tight_layout()
        fig.savefig(paths.REPORTS / "gallery_ablation.png", dpi=110)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"(aviso: sin gráfico: {e})")

    print("\n" + report)
    print(f"Reporte: {paths.REPORTS / 'gallery_ablation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
