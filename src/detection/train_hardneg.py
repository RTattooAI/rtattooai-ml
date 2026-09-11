"""Reentrena el detector con NEGATIVOS DUROS, para que solo reconozca tatuajes.

El problema, medido en `sweep_conf` sobre 140 fotografías de personas sin tatuajes:
el detector no aprendió a buscar tatuajes, aprendió a buscar **marcas de tinta sobre
superficie clara**. Da igual que la superficie sea piel. Sus peores errores son una
nota escrita a mano (confianza 0.91), una alfombra (0.83) y uñas decoradas (0.78).

Eso no se arregla subiendo el umbral: subirlo solo cambia falsos positivos por falsos
negativos, porque el detector está igual de seguro de la nota que de un tatuaje. Se
arregla enseñándole esos casos **como fondo**: imágenes sin ninguna caja etiquetada.
YOLO las usa para penalizar justamente lo que aquí sobra.

**Separación entre entrenar y medir.** Las 140 fotografías del banco de pruebas NO se
usan aquí. El corpus de entrenamiento se descargó aparte, de categorías distintas
(escritura a mano, caligrafía, pizarras, cuadernos, alfombras, arena, grafiti, henna).
Así, la mejora que se mida después sobre las 140 es generalización real y no memoria:
si bajan también los falsos positivos de uñas y logotipos —clases que no entrenó—,
el aprendizaje fue del concepto y no de las fotos.

Es el mismo cuidado que obligó a corregir `finetune_head.py`: entrenar con aquello
sobre lo que luego se reporta produce cifras que no significan nada.

Uso:
    python -m src.detection.train_hardneg --preparar
    python -m src.detection.train_hardneg --epochs 40
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.shared.seeds import set_seed, SEED

EXTENSIONES = {".jpg", ".jpeg", ".png"}

ORIGEN_DUROS = paths.DATASETS / "negativos_duros"
IMAGENES_DUROS = ORIGEN_DUROS / "images" / "train"
ETIQUETAS_DUROS = ORIGEN_DUROS / "labels" / "train"

YAML_SALIDA = paths.DATASETS / "unified_hardneg.yaml"
PESOS_BASE = paths.MODELS / "yolov8m_best.pt"


def preparar_estructura(origen: Path) -> int:
    """Coloca los negativos donde YOLO los lee como fondo (imagen sin etiquetas)."""
    IMAGENES_DUROS.mkdir(parents=True, exist_ok=True)
    # La carpeta de etiquetas existe pero queda vacía: para YOLO, una imagen sin su
    # archivo de etiquetas es una imagen de fondo, que es exactamente lo que son.
    ETIQUETAS_DUROS.mkdir(parents=True, exist_ok=True)

    copiadas = 0
    for ruta in sorted(origen.rglob("*")):
        if ruta.suffix.lower() not in EXTENSIONES:
            continue
        destino = IMAGENES_DUROS / ruta.name
        if not destino.exists():
            shutil.copy2(ruta, destino)
        copiadas += 1
    return copiadas


def escribir_yaml() -> Path:
    """Dataset = el de siempre para entrenar y validar, más los negativos al entrenar.

    Se genera aquí en vez de reutilizar `datasets/unified/data.yaml` porque ese
    archivo conserva la ruta absoluta de antes de mover el repositorio.
    """
    raiz = paths.DATASETS.resolve().as_posix()
    YAML_SALIDA.write_text(
        "# Generado por train_hardneg.py\n"
        "# La validacion es la de siempre, sin negativos: si se anadieran, el mAP\n"
        "# subiria solo porque hay mas imagenes faciles y no seria comparable.\n"
        f"path: {raiz}\n"
        "train:\n"
        "  - unified/images/train\n"
        "  - negativos_duros/images/train\n"
        "val: unified/images/val\n"
        "test: unified/images/test\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        "  0: tatuaje\n",
        encoding="utf-8",
    )
    return YAML_SALIDA


def main() -> int:
    ap = argparse.ArgumentParser(description="Reentrenamiento con negativos duros")
    ap.add_argument("--origen", default=str(ORIGEN_DUROS / "revisados"),
                    help="carpeta con los negativos ya revisados a ojo")
    ap.add_argument("--preparar", action="store_true",
                    help="solo copiar y escribir el yaml, sin entrenar")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8, help="8 cabe en 8 GB con yolov8m")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr0", type=float, default=0.002,
                    help="bajo: se parte de pesos ya buenos, no de cero")
    ap.add_argument("--nombre", default="yolov8m_hardneg")
    args = ap.parse_args()

    origen = Path(args.origen)
    if not origen.exists():
        print(f"ERROR: no existe {origen}.")
        print("Descargue con download_negatives.py y revise con screen_negatives.py.")
        return 1

    copiadas = preparar_estructura(origen)
    yaml = escribir_yaml()
    total_train = len(list((paths.DATASETS / "unified" / "images" / "train").glob("*")))

    print(f"Negativos duros como fondo: {copiadas}")
    print(f"Imágenes con tatuajes:      {total_train}")
    print(f"Proporción de fondo:        {copiadas / max(1, copiadas + total_train) * 100:.1f} %")
    print(f"Dataset: {yaml}")

    if args.preparar:
        return 0

    if not PESOS_BASE.exists():
        print(f"ERROR: no están los pesos de partida en {PESOS_BASE}.")
        return 1

    import torch
    if not torch.cuda.is_available():
        print("BLOQUEO: no hay GPU CUDA. Política del proyecto: no entrenar en CPU.")
        return 3

    set_seed()
    from ultralytics import YOLO

    print(f"\nPartiendo de {PESOS_BASE.name} — se afina, no se entrena de cero.")
    modelo = YOLO(str(PESOS_BASE))
    modelo.train(
        data=str(yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=10,
        seed=SEED,
        flipud=0.0,          # un tatuaje boca abajo es poco natural
        project=str(paths.ROOT / "runs" / "detect"),
        name=args.nombre,
        exist_ok=True,
        amp=True,
    )

    print("\nMedir el efecto (sobre el banco de pruebas, que no entrenó):")
    print("  python -m src.detection.sweep_conf --val-max 500 \\")
    print(f"      --pesos runs/detect/{args.nombre}/weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
