"""Paso 0 — Prueba de concepto de la identificación por embeddings (sin entrenar).

Responde 3 preguntas con lo que YA tenemos (25 tatuajes de banda + el YOLO):
  1) ¿DINOv2 da embeddings INVARIANTES a condiciones? -> se generan variaciones
     2D (rotación, color, fondo de piel, desenfoque, ruido) de cada tatuaje y se
     mide si la "huella" recupera el tatuaje correcto de la galería (top-1/top-5).
  2) ¿SEPARA identidades distintas? -> brecha de coseno correcto vs mejor-incorrecto.
  3) ¿Permite RECHAZO open-set? -> compara contra tatuajes NO-banda (recortados de
     datasets/unified con el YOLO): su mejor coseno debería ser MENOR que el de
     una variación genuina, para poder fijar un umbral de "no conocido".

Uso:
    python -m src.embedding.step0_baseline
    python -m src.embedding.step0_baseline --queries-per 8 --distractors 60

NO entrena nada. Si los resultados son buenos, gran parte del esfuerzo en Blender
y fine-tuning se simplifica (basta una buena galería).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.shared.seeds import set_seed, SEED
from src.embedding.embedder import TattooEmbedder, composite_on

GANG_DIR = paths.DATASETS / "gang_tattoos"
SKIN_COLORS = [(241, 194, 167), (224, 172, 138), (198, 134, 100),
               (172, 112, 80), (141, 85, 58), (210, 160, 130)]


def make_query(rgba: Image.Image, rng: np.random.RandomState) -> Image.Image:
    """Simula una foto real del tatuaje: sobre piel, rotado, con color/blur/ruido."""
    S = int(rng.choice([224, 320, 448]))
    skin = SKIN_COLORS[rng.randint(len(SKIN_COLORS))]
    canvas = Image.new("RGBA", (S, S), (*skin, 255))

    tat = rgba.convert("RGBA")
    w, h = tat.size
    frac = rng.uniform(0.45, 0.9)
    scale = (S * frac) / max(w, h)
    tat = tat.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    tat = tat.rotate(rng.uniform(-25, 25), expand=True, resample=Image.BICUBIC)

    px = rng.randint(0, max(1, S - tat.size[0]))
    py = rng.randint(0, max(1, S - tat.size[1]))
    canvas.alpha_composite(tat, (px, py))
    img = canvas.convert("RGB")

    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.7, 1.3))
    if rng.rand() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.6)))
    arr = np.array(img).astype(np.int16)
    arr += rng.normal(0, rng.uniform(3, 12), arr.shape).astype(np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def crop_distractors(weights: Path, n: int, rng: np.random.RandomState) -> list[Image.Image]:
    """Recorta tatuajes reales (NO banda) de datasets/unified con el YOLO -> open-set."""
    src = paths.UNIFIED_IMAGES / "test"
    imgs = sorted(p for p in src.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not imgs or not weights.exists():
        return []
    rng.shuffle(imgs)
    imgs = imgs[: n * 2]  # de más, por si alguna no tiene detección
    from ultralytics import YOLO
    model = YOLO(str(weights))
    crops = []
    for i in range(0, len(imgs), 16):
        chunk = imgs[i:i + 16]
        res = model.predict(source=[str(p) for p in chunk], conf=0.25, imgsz=640,
                            verbose=False)
        for p, r in zip(chunk, res):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            b = max(r.boxes, key=lambda bb: float(bb.conf[0]))
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            im = Image.open(p).convert("RGB")
            if x2 > x1 and y2 > y1:
                crops.append(im.crop((x1, y1, x2, y2)))
            if len(crops) >= n:
                return crops
    return crops


def main() -> int:
    ap = argparse.ArgumentParser(description="Paso 0: baseline de embeddings (DINOv2)")
    ap.add_argument("--gallery", default=str(GANG_DIR))
    ap.add_argument("--weights", default=str(paths.MODELS / "yolov8m_best.pt"))
    ap.add_argument("--queries-per", type=int, default=8)
    ap.add_argument("--distractors", type=int, default=60)
    args = ap.parse_args()

    set_seed()
    rng = np.random.RandomState(SEED)

    gdir = Path(args.gallery)
    gfiles = sorted(p for p in gdir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not gfiles:
        print(f"ERROR: no hay tatuajes en {gdir}")
        return 1
    names = [p.stem.replace("-removebg-preview", "") for p in gfiles]
    print(f"Galería: {len(gfiles)} tatuajes de banda. Cargando DINOv2…")

    emb = TattooEmbedder()
    originals = [Image.open(p) for p in gfiles]
    gallery = emb.embed(originals, bg=(255, 255, 255))   # (G, dim) sobre blanco

    # ---- 1+2: recuperación con variaciones ----
    top1 = top5 = total = 0
    gap_correct, gap_wrong, genuine_best = [], [], []
    for gi, src_img in enumerate(originals):
        for _ in range(args.queries_per):
            q = emb.embed(make_query(src_img, rng))
            sims = gallery @ q                  # coseno (todo L2-norm)
            order = np.argsort(-sims)
            total += 1
            if order[0] == gi:
                top1 += 1
            if gi in order[:5]:
                top5 += 1
            gap_correct.append(float(sims[gi]))
            wrong = float(max(sims[j] for j in range(len(sims)) if j != gi))
            gap_wrong.append(wrong)
            genuine_best.append(float(sims[order[0]]))

    # ---- 3: open-set con distractores no-banda ----
    print(f"Recortando ~{args.distractors} tatuajes NO-banda con el YOLO…")
    distractors = crop_distractors(Path(args.weights), args.distractors, rng)
    distractor_best = []
    if distractors:
        de = emb.embed(distractors)
        for q in de:
            distractor_best.append(float(np.max(gallery @ q)))

    # ---- reporte ----
    paths.ensure_dirs(paths.REPORTS)
    gc, gw = np.array(gap_correct), np.array(gap_wrong)
    gb, db = np.array(genuine_best), np.array(distractor_best)
    lines = [
        "# Paso 0 — Baseline de identificación por embeddings (DINOv2, sin entrenar)\n",
        f"- Galería: **{len(gfiles)}** tatuajes de banda | variaciones por tatuaje: "
        f"**{args.queries_per}** (total {total} consultas).",
        f"- Embedding: DINOv2 ViT-S/14 ({emb.dim}-d), similitud coseno.\n",
        "## 1-2) Recuperación (¿reconoce el tatuaje pese a las condiciones?)",
        f"- **Top-1: {100*top1/total:.1f}%** | **Top-5: {100*top5/total:.1f}%**",
        f"- Coseno consulta↔correcto: media **{gc.mean():.3f}** (min {gc.min():.3f})",
        f"- Coseno consulta↔mejor-incorrecto: media **{gw.mean():.3f}**",
        f"- Brecha media correcto−incorrecto: **{(gc-gw).mean():+.3f}** "
        f"(positiva = separa identidades)\n",
        "## 3) Rechazo open-set (¿distingue banda de no-banda?)",
    ]
    if distractor_best:
        # umbral sugerido: punto medio entre distractores y consultas genuinas
        thr = (db.mean() + gb.mean()) / 2
        tp = float((gb >= thr).mean()); fp = float((db >= thr).mean())
        lines += [
            f"- Distractores no-banda: {len(distractor_best)} (recortados del test).",
            f"- Mejor-coseno consulta genuina: media **{gb.mean():.3f}**",
            f"- Mejor-coseno distractor no-banda: media **{db.mean():.3f}**",
            f"- Umbral sugerido ≈ **{thr:.3f}** → aceptaría {100*tp:.0f}% de genuinas "
            f"y {100*fp:.0f}% de no-banda (falsos).",
        ]
    else:
        lines.append("- (no se pudieron recortar distractores; revisa el YOLO/test)")
    report = "\n".join(lines) + "\n"
    (paths.REPORTS / "embedding_step0.md").write_text(report, encoding="utf-8")

    # histograma de separación
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.hist(gb, bins=20, alpha=0.6, label="consulta genuina (mejor coseno)")
        if distractor_best:
            plt.hist(db, bins=20, alpha=0.6, label="no-banda (mejor coseno)")
        plt.xlabel("mejor similitud coseno con la galería"); plt.ylabel("frecuencia")
        plt.legend(); plt.title("Separación banda vs no-banda (Paso 0)")
        plt.tight_layout()
        plt.savefig(paths.REPORTS / "embedding_step0_separacion.png", dpi=110)
        plt.close()
    except Exception as e:  # noqa: BLE001
        print(f"(aviso: no se pudo graficar: {e})")

    # guardar galería (gitignored)
    paths.ensure_dirs(paths.MODELS)
    np.savez(paths.MODELS / "gallery_dinov2.npz", embeddings=gallery, names=np.array(names))

    print("\n" + report)
    print(f"Reporte: {paths.REPORTS / 'embedding_step0.md'}")
    print(f"Galería guardada: {paths.MODELS / 'gallery_dinov2.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
