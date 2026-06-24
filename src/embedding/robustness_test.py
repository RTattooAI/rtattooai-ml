"""Robustez del embedding a la DEGRADACIÓN del tatuaje.

Pregunta de la tesis: ¿hasta qué punto un tatuaje puede estar borrado/decolorado
y el sistema lo sigue reconociendo como el mismo?

Método: galería = los 25 diseños limpios (sobre piel). Para cada tatuaje se
generan versiones cada vez más degradadas (decoloración + borrado parcial +
desenfoque + envejecimiento + oclusión) y se mide, por nivel:
  - coseno consulta-degradada ↔ su propio diseño limpio,
  - top-1 (¿sigue siendo el más parecido entre los 25?).

Uso:
    python -m src.embedding.robustness_test
    python -m src.embedding.robustness_test --reps 5
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
from src.embedding.embedder import TattooEmbedder, composite_on
from src.embedding.degrade import degrade, SKIN

LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]


def main() -> int:
    ap = argparse.ArgumentParser(description="Robustez del embedding a la degradación")
    ap.add_argument("--gallery", default=str(paths.DATASETS / "gang_tattoos"))
    ap.add_argument("--reps", type=int, default=3, help="versiones degradadas por nivel")
    ap.add_argument("--head", default=None, help="ruta a la cabeza afinada (opcional)")
    args = ap.parse_args()

    set_seed()
    rng = np.random.RandomState(SEED)

    gfiles = sorted(p for p in Path(args.gallery).glob("*")
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not gfiles:
        print(f"ERROR: sin tatuajes en {args.gallery}")
        return 1
    names = [p.stem.replace("-removebg-preview", "") for p in gfiles]
    print(f"Galería: {len(gfiles)} tatuajes. Cargando DINOv2…"
          + (" + cabeza" if args.head else ""))
    emb = TattooEmbedder(head_path=args.head)

    # galería limpia sobre piel
    clean = [composite_on(Image.open(p), SKIN) for p in gfiles]
    gallery = emb.embed(clean, bg=SKIN)   # (G, dim)

    rows = []
    for level in LEVELS:
        cos_correct, top1, total = [], 0, 0
        for gi, base in enumerate(clean):
            for _ in range(args.reps):
                q = emb.embed(degrade(base, level, rng))
                sims = gallery @ q
                cos_correct.append(float(sims[gi]))
                if int(np.argmax(sims)) == gi:
                    top1 += 1
                total += 1
        rows.append({"nivel": level, "coseno_correcto": float(np.mean(cos_correct)),
                     "top1": top1 / total})
        print(f"  nivel {level:.1f}: coseno↔limpio={np.mean(cos_correct):.3f} | "
              f"top1={100*top1/total:.1f}%")

    # reporte
    paths.ensure_dirs(paths.REPORTS)
    lines = ["# Robustez del embedding a la degradación del tatuaje\n",
             f"- Galería: {len(gfiles)} tatuajes; {args.reps} versiones por nivel.",
             "- Degradación: decoloración + borrado parcial + desenfoque + "
             "envejecimiento + oclusión (nivel 0 = limpio).\n",
             "| Nivel degradación | Coseno ↔ diseño limpio | Top-1 |",
             "| --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['nivel']:.1f} | {r['coseno_correcto']:.3f} | "
                     f"{100*r['top1']:.1f}% |")
    # nivel de degradación que aún mantiene top-1 alto
    ok = [r["nivel"] for r in rows if r["top1"] >= 0.8]
    lines.append(f"\n**Tolerancia:** el sistema mantiene top-1 ≥ 80% hasta un nivel "
                 f"de degradación de **{max(ok) if ok else 0.0:.1f}**.")
    (paths.REPORTS / "robustness_degradation.md").write_text("\n".join(lines) + "\n",
                                                             encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lv = [r["nivel"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(lv, [r["coseno_correcto"] for r in rows], "o-", color="tab:blue")
        ax1.set_xlabel("nivel de degradación"); ax1.set_ylabel("coseno ↔ limpio", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(lv, [100 * r["top1"] for r in rows], "s--", color="tab:green")
        ax2.set_ylabel("top-1 (%)", color="tab:green"); ax2.set_ylim(0, 100)
        plt.title("Robustez a la degradación del tatuaje")
        fig.tight_layout()
        fig.savefig(paths.REPORTS / "robustness_degradation.png", dpi=110)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"(aviso: sin gráfico: {e})")

    print(f"\nReporte: {paths.REPORTS / 'robustness_degradation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
