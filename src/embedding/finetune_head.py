"""Fine-tuning CONTRASTIVO de una cabeza sobre DINOv2 (congelado).

Idea: DINOv2 ya es fuerte; entrenamos solo una cabeza MLP que proyecta sus
features a un espacio más invariante a degradación/condiciones, SIN tocar el
backbone (no se sobreajusta ni se daña su generalidad).

Datos (cada "fuente" = una identidad; sus vistas aumentadas/degradadas = positivos):
  - ~2000 recortes REALES (datasets/real_crops): diversidad de identidades.
  - 25 tatuajes de banda + sus vistas Blender (datasets/output/.../crops).
Negativos = el resto del batch. Pérdida: Supervised Contrastive (SupCon).
Los features de DINOv2 se CACHEAN (backbone congelado) → entrenamiento rápido.

Salida: models/embed_head.pt  (la usa TattooEmbedder con head_path=...)

Uso:
    python -m src.embedding.finetune_head
    python -m src.embedding.finetune_head --real 2000 --epochs 40
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
from src.embedding.embedder import TattooEmbedder, build_head, composite_on, HEAD_HIDDEN, HEAD_OUT
from src.embedding.degrade import degrade, SKIN

GANG_DIR = paths.DATASETS / "gang_tattoos"
BLENDER_CROPS = paths.DATASETS / "output" / "gang_tattoos_out" / "crops"
REAL_CROPS = paths.DATASETS / "real_crops"
HEAD_OUT_PATH = paths.MODELS / "embed_head.pt"
SKIN_PALETTE = [(241, 194, 167), (224, 172, 138), (198, 134, 100), (141, 85, 58)]


def augment_rgb(img: Image.Image, rng: np.random.RandomState) -> Image.Image:
    img = img.convert("RGB")
    img = img.rotate(rng.uniform(-15, 15), expand=False, resample=Image.BICUBIC,
                     fillcolor=(180, 140, 110))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
    if rng.rand() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    if rng.rand() < 0.5:
        img = degrade(img, rng.uniform(0.1, 0.5), rng)
    return img


def gang_views(design_path: Path, blender_dir: Path, rng) -> list[Image.Image]:
    name = design_path.stem.replace("-removebg-preview", "")
    design = Image.open(design_path)
    views = [composite_on(design, SKIN_PALETTE[rng.randint(len(SKIN_PALETTE))]) for _ in range(3)]
    bdir = blender_dir / name
    if bdir.exists():
        for p in sorted(bdir.glob("*.png")):
            views.append(Image.open(p).convert("RGB"))
    # un par degradadas para enseñar invarianza al desgaste
    base = views[0]
    views += [degrade(base, rng.uniform(0.2, 0.6), rng) for _ in range(2)]
    return views


def build_sources(n_real: int, k_real: int, rng):
    """Devuelve (lista_imagenes, labels_int) con vistas por identidad."""
    imgs, labels = [], []
    lid = 0
    # gang
    for d in sorted(GANG_DIR.glob("*.png")):
        for v in gang_views(d, BLENDER_CROPS, rng):
            imgs.append(v); labels.append(lid)
        lid += 1
    n_gang = lid
    # reales
    crops = [p for p in REAL_CROPS.rglob("*.png")]
    rng.shuffle(crops)
    for cp in crops[:n_real]:
        try:
            base = Image.open(cp).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        for _ in range(k_real):
            imgs.append(augment_rgb(base, rng)); labels.append(lid)
        lid += 1
    print(f"  identidades: {lid} ({n_gang} gang + {lid - n_gang} reales) | vistas: {len(imgs)}")
    return imgs, np.array(labels)


def supcon_loss(z, labels, torch, temp=0.1):
    z = torch.nn.functional.normalize(z, dim=1)
    sim = z @ z.t() / temp
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(len(z), device=z.device)
    exp = torch.exp(sim) * (1 - eye)
    log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-9)
    pos = (labels[:, None] == labels[None, :]).float() * (1 - eye)
    mlpp = (pos * log_prob).sum(1) / pos.sum(1).clamp(min=1)
    valid = pos.sum(1) > 0
    return -(mlpp[valid].mean())


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tuning contrastivo de la cabeza")
    ap.add_argument("--real", type=int, default=2000, help="nº de recortes reales")
    ap.add_argument("--k-real", type=int, default=3, help="vistas por recorte real")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--ids-per-batch", type=int, default=48)
    ap.add_argument("--views-per-id", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    set_seed()
    import torch
    rng = np.random.RandomState(SEED)

    print("Construyendo fuentes y vistas…")
    imgs, labels = build_sources(args.real, args.k_real, rng)
    print("Cacheando features de DINOv2 (congelado)…")
    emb = TattooEmbedder()
    feats = emb.backbone_features(imgs)            # (M, 384) en CPU
    labels_t = torch.tensor(labels)

    # índices por identidad (solo las con ≥2 vistas)
    by_id = {}
    for i, l in enumerate(labels):
        by_id.setdefault(int(l), []).append(i)
    ids = [k for k, v in by_id.items() if len(v) >= 2]
    print(f"  identidades entrenables (≥2 vistas): {len(ids)}")

    device = emb.device
    head = build_head(emb.backbone_dim).to(device).train()
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)
    feats_dev = feats.to(device)

    steps = max(20, len(ids) // args.ids_per_batch)
    print(f"Entrenando: {args.epochs} épocas × {steps} pasos…")
    for ep in range(args.epochs):
        rng.shuffle(ids)
        tot = 0.0
        for s in range(steps):
            chosen = ids[(s * args.ids_per_batch) % len(ids):][:args.ids_per_batch]
            if len(chosen) < 2:
                chosen = rng.choice(ids, size=args.ids_per_batch, replace=False).tolist()
            idx = []
            for cid in chosen:
                pool = by_id[cid]
                take = pool if len(pool) <= args.views_per_id else \
                    list(rng.choice(pool, size=args.views_per_id, replace=False))
                idx += take
            bl = labels_t[idx].to(device)
            z = head(feats_dev[idx])
            loss = supcon_loss(z, bl, torch)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  época {ep+1:3d}/{args.epochs}  loss={tot/steps:.4f}")

    paths.ensure_dirs(paths.MODELS)
    torch.save({"state_dict": head.state_dict(), "hidden": HEAD_HIDDEN,
                "out": HEAD_OUT, "in_dim": emb.backbone_dim}, HEAD_OUT_PATH)
    print(f"\nCabeza guardada: {HEAD_OUT_PATH}")
    print("Evaluar: python -m src.embedding.robustness_test --head models/embed_head.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
