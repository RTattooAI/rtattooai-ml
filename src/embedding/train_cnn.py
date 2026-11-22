"""¿Y si entrenamos una CNN desde cero, en lugar de usar DINOv2 congelado?

El sistema usa hoy un ViT preentrenado (DINOv2 ViT-S/14) al que solo se le afina
una cabeza MLP. La pregunta razonable es si una red convolucional entrenada con
nuestros propios datos daría mejor resultado — y de paso si el mérito del sistema
es nuestro o del modelo que descargamos.

La pregunta mezcla dos variables distintas, y este script las separa:

                    | preentrenado        | desde cero
    ----------------|---------------------|---------------------
    CNN (ResNet-18) | --pesos imagenet    | --pesos ninguno
    ViT (DINOv2)    | el sistema actual   | inviable con 6.8k imagenes

Para que la comparación signifique algo, los tres brazos comparten TODO menos el
codificador: las mismas identidades de `real_crops/train`, las mismas vistas
aumentadas (`build_sources`, misma semilla), la misma pérdida SupCon con la misma
temperatura, y después la misma evaluación con `enroll_recall_test`.

Diferencia inevitable: aquí se entrena la red entera, no una cabeza sobre features
congelados, así que no se pueden cachear. Las vistas se generan una vez y se
guardan en RAM como uint8; el aumento no cambia entre épocas, igual que en
`finetune_head.py`, para que ambos brazos vean literalmente las mismas imágenes.

Uso:
    python -m src.embedding.train_cnn --pesos ninguno  --salida models/cnn_scratch.pt
    python -m src.embedding.train_cnn --pesos imagenet --salida models/cnn_imagenet.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.shared.seeds import set_seed, SEED
from src.embedding.finetune_head import build_sources, build_sources_manifest, mascara_ignorar, supcon_loss

# Resolución de entrada. DINOv2 ViT-S/14 trabaja a 518 px; una ResNet-18 a 224 px
# es su tamaño nativo y el que usa toda la literatura de la que salen sus pesos.
# Subirla no la mejora y multiplica el coste.
TAM = 224
PROJ_OUT = 128          # misma dimensión de salida que la cabeza sobre DINOv2
PROJ_HIDDEN = 256

# Estadísticas de ImageNet. Se usan en los dos brazos: para el preentrenado son
# las que corresponden a sus pesos, y para el de cero son solo un centrado, que
# no le da ninguna ventaja.
MEDIA = np.array([0.485, 0.456, 0.406], dtype="float32")
DESV = np.array([0.229, 0.224, 0.225], dtype="float32")


def construir_red(pesos: str, torch, arquitectura: str = "resnet18"):
    """ResNet-18 con una cabeza de proyección, igual de profunda que la de DINOv2."""
    import torch.nn as nn
    from torchvision import models

    if arquitectura == "resnet50":
        red = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pesos == "imagenet" else None)
    elif pesos == "imagenet":
        red = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        red = models.resnet18(weights=None)

    dim = red.fc.in_features           # 512
    red.fc = nn.Sequential(
        nn.Linear(dim, PROJ_HIDDEN), nn.GELU(), nn.Linear(PROJ_HIDDEN, PROJ_OUT)
    )
    return red


def a_tensor_uint8(imgs, tam: int = TAM) -> np.ndarray:
    """Pasa la lista de PIL a un único array (N, tam, tam, 3) uint8 en RAM."""
    from PIL import Image

    salida = np.zeros((len(imgs), tam, tam, 3), dtype="uint8")
    for i, im in enumerate(imgs):
        salida[i] = np.asarray(im.convert("RGB").resize((tam, tam), Image.BICUBIC))
    return salida


def main() -> int:
    ap = argparse.ArgumentParser(description="CNN entrenada con SupCon sobre los mismos datos")
    ap.add_argument("--pesos", choices=["ninguno", "imagenet"], default="ninguno",
                    help="'ninguno' = desde cero; 'imagenet' = preentrenada")
    ap.add_argument("--salida", default=None, help="ruta del checkpoint")
    ap.add_argument("--real", type=int, default=6800)
    ap.add_argument("--k-real", type=int, default=4)
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "todo"])
    ap.add_argument("--max-degrade", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--ids-per-batch", type=int, default=64)
    ap.add_argument("--views-per-id", type=int, default=4)
    ap.add_argument("--arquitectura", choices=["resnet18", "resnet50"], default="resnet18")
    ap.add_argument("--manifiesto", default=None,
                    help="identidades agrupadas (agrupar_duplicados); activa la aumentación realista")
    ap.add_argument("--lr", type=float, default=None,
                    help="por defecto 1e-3 desde cero, 1e-4 al afinar preentrenada")
    args = ap.parse_args()

    salida = Path(args.salida) if args.salida else paths.MODELS / f"cnn_{args.pesos}.pt"
    # Afinar unos pesos ya buenos con el paso grande del entrenamiento desde cero
    # los destruye en las primeras épocas; es el error clásico del fine-tuning.
    lr = args.lr if args.lr is not None else (1e-3 if args.pesos == "ninguno" else 1e-4)

    set_seed()
    import torch

    rng = np.random.RandomState(SEED)

    print(f"=== ResNet-18 · pesos={args.pesos} · lr={lr:g} ===")
    print("Construyendo fuentes y vistas (mismas que la cabeza sobre DINOv2)…")
    ignorar = {}
    if args.manifiesto:
        imgs, labels, ignorar = build_sources_manifest(Path(args.manifiesto), args.k_real, rng, args.max_degrade)
    else:
        imgs, labels = build_sources(args.real, args.k_real, rng, args.split, args.max_degrade)

    print(f"Pasando {len(imgs)} vistas a memoria como uint8 {TAM}×{TAM}…")
    datos = a_tensor_uint8(imgs)
    del imgs
    print(f"  {datos.nbytes / 1e9:.2f} GB en RAM")

    by_id: dict[int, list[int]] = {}
    for i, l in enumerate(labels):
        by_id.setdefault(int(l), []).append(i)
    ids = [k for k, v in by_id.items() if len(v) >= 2]
    print(f"  identidades entrenables (≥2 vistas): {len(ids)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    red = construir_red(args.pesos, torch, args.arquitectura).to(device).train()
    n_par = sum(p.numel() for p in red.parameters())
    print(f"  parámetros entrenables: {n_par/1e6:.1f} M   (la cabeza sobre DINOv2: 0.13 M)")

    opt = torch.optim.AdamW(red.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(20, len(ids) // args.ids_per_batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps)
    escala = torch.amp.GradScaler(device)

    media = torch.tensor(MEDIA, device=device).view(1, 3, 1, 1)
    desv = torch.tensor(DESV, device=device).view(1, 3, 1, 1)

    def lote_gpu(indices):
        """uint8 (B,H,W,3) en RAM → float normalizado (B,3,H,W) en GPU."""
        x = torch.from_numpy(datos[indices]).to(device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        return (x - media) / desv

    print(f"Entrenando: {args.epochs} épocas × {steps} pasos "
          f"({args.ids_per_batch} identidades × {args.views_per_id} vistas por lote)…")
    t0 = time.time()
    historia = []

    for ep in range(args.epochs):
        rng.shuffle(ids)
        tot = 0.0
        for s in range(steps):
            elegidos = ids[s * args.ids_per_batch:(s + 1) * args.ids_per_batch]
            if len(elegidos) < 2:
                elegidos = rng.choice(ids, size=args.ids_per_batch, replace=False).tolist()
            idx = []
            for cid in elegidos:
                pool = by_id[cid]
                idx += (pool if len(pool) <= args.views_per_id
                        else list(rng.choice(pool, size=args.views_per_id, replace=False)))
            idx = np.array(idx)
            bl = torch.tensor(labels[idx], device=device)

            with torch.autocast(device_type=device, dtype=torch.float16):
                z = red(lote_gpu(idx))
                loss = supcon_loss(z.float(), bl, torch,
                                   ignorar=mascara_ignorar(bl, ignorar, torch) if ignorar else None)

            opt.zero_grad(set_to_none=True)
            escala.scale(loss).backward()
            escala.step(opt)
            escala.update()
            sched.step()
            tot += float(loss.detach())

        medio = tot / steps
        historia.append(round(medio, 4))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  época {ep+1:3d}/{args.epochs}  loss={medio:.4f}  "
                  f"({time.time()-t0:.0f}s)")

    paths.ensure_dirs(paths.MODELS)
    torch.save({
        "arquitectura": args.arquitectura,
        "state_dict": red.state_dict(),
        "out": PROJ_OUT,
        "hidden": PROJ_HIDDEN,
        "tam": TAM,
        "entrenamiento": {
            "pesos_iniciales": args.pesos, "split": args.split,
            "recortes_reales": args.real, "vistas_por_recorte": args.k_real,
            "epocas": args.epochs, "ids_por_lote": args.ids_per_batch,
            "degradacion_maxima": args.max_degrade, "identidades": len(ids),
            "lr": lr, "parametros": n_par, "manifiesto": args.manifiesto, "segundos": round(time.time() - t0),
            "loss_por_epoca": historia,
        },
    }, salida)
    print(f"\nGuardado: {salida}  ({time.time()-t0:.0f}s de entrenamiento)")
    print(f"Evaluar: python -m src.embedding.enroll_recall_test --backbone cnn:{salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
