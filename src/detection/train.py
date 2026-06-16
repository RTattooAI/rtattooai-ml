"""Entrenamiento YOLOv8 para detección de tatuajes (1 clase).

- Modelo definitivo: yolov8m (mejor precisión).
- Baseline ligero:   yolov8s (para la tabla comparativa).
- Transfer learning desde pesos COCO, imgsz=640, epochs=150, patience=30,
  batch auto, semilla fija, flipud=0 (un tatuaje boca abajo es poco natural).

REQUIERE GPU. Si no hay CUDA, NO entrena en CPU: se detiene y avisa.

Uso:
    python -m src.detection.train                 # entrena yolov8m y yolov8s
    python -m src.detection.train --models yolov8m.pt
    python -m src.detection.train --epochs 150 --batch 16
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.shared.seeds import set_seed, SEED


def require_gpu() -> int:
    """Verifica que haya GPU CUDA. Si no, aborta con mensaje claro."""
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch no está instalado. Instala requirements.txt.")
        sys.exit(3)
    if not torch.cuda.is_available():
        print("=" * 64)
        print("BLOQUEO: no se detecta GPU CUDA disponible.")
        print("Política del proyecto: NO entrenar en CPU (sería inviable).")
        print("Opciones recomendadas:")
        print("  - Google Colab (GPU T4 gratuita) con notebooks/")
        print("  - VM de Azure con GPU (NC/NV series)")
        print("=" * 64)
        sys.exit(3)
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU detectada: {name} ({vram:.1f} GB VRAM)")
    return 0


def _last_epoch(results_csv: Path) -> int:
    """Última época registrada en results.csv (0 si no existe/ilegible)."""
    if not results_csv.exists():
        return 0
    try:
        lines = [l for l in results_csv.read_text().splitlines() if l.strip()]
        return int(lines[-1].split(",")[0]) if len(lines) > 1 else 0
    except Exception:  # noqa: BLE001
        return 0


def train_one(model_name: str, data_yaml: Path, epochs: int, imgsz: int,
              patience: int, batch, device: int) -> Path:
    """Entrena (o reanuda) un modelo y devuelve best.pt copiado a models/.

    Restartable: si existe runs/detect/<name>/weights/last.pt y el run no ha
    terminado, reanuda desde la última época (útil tras un corte/suspensión).
    """
    from ultralytics import YOLO

    run_name = Path(model_name).stem  # 'yolov8m'
    run_dir = paths.RUNS / "detect" / run_name
    last_pt = run_dir / "weights" / "last.pt"
    done_epoch = _last_epoch(run_dir / "results.csv")

    if last_pt.exists() and done_epoch >= epochs:
        print(f"\n===== {run_name}: ya completo ({done_epoch}/{epochs} épocas), se omite =====")
    elif last_pt.exists() and done_epoch > 0:
        print(f"\n===== Reanudando {run_name} desde época {done_epoch}/{epochs} =====")
        model = YOLO(str(last_pt))
        model.train(resume=True)   # reanuda con los args guardados del run
    else:
        print(f"\n===== Entrenando {run_name} (desde cero) =====")
        model = YOLO(model_name)
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            patience=patience,
            batch=batch,
            seed=SEED,
            deterministic=True,
            flipud=0.0,            # sin volteo vertical
            # resto de augmentations: valores por defecto de Ultralytics
            project=str(paths.RUNS / "detect"),
            name=run_name,
            exist_ok=True,
            device=device,
            verbose=True,
        )
    best = paths.RUNS / "detect" / run_name / "weights" / "best.pt"
    paths.ensure_dirs(paths.MODELS)
    dst = paths.MODELS / f"{run_name}_best.pt"
    if best.exists():
        shutil.copy2(best, dst)
        print(f"Pesos copiados a: {dst}")
    else:
        print(f"AVISO: no se encontró {best}")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description="Entrenamiento YOLOv8 (tatuajes)")
    ap.add_argument("--models", nargs="+", default=["yolov8m.pt", "yolov8s.pt"],
                    help="modelos a entrenar (el 1º es el definitivo)")
    ap.add_argument("--data", default=str(paths.UNIFIED_YAML),
                    help="ruta al data.yaml del dataset unificado")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", default=-1,
                    help="-1 = auto (recomendado); o un entero, p. ej. 16")
    args = ap.parse_args()

    set_seed()
    require_gpu()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"ERROR: no existe {data_yaml}. Ejecuta antes prepare_dataset.py.")
        sys.exit(1)

    try:
        batch = int(args.batch)
    except (TypeError, ValueError):
        batch = -1

    for m in args.models:
        train_one(m, data_yaml, args.epochs, args.imgsz, args.patience, batch, device=0)

    print("\nEntrenamiento completo. Siguiente: evaluate.py y export_onnx.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
