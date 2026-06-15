"""Exporta el modelo YOLOv8 entrenado a ONNX (para ONNX Runtime / ML.NET).

- Exporta models/<modelo>_best.pt -> models/<modelo>_best.onnx
- opset configurable (por defecto 12, compatible con ML.NET / ONNX Runtime).
- Entrada fija 1x3x640x640 (NCHW, float32, RGB normalizado 0-1) para que el
  backend .NET la consuma sin lógica de shapes dinámicos.
- Verifica el .onnx cargándolo con onnxruntime y corriendo un tensor dummy.

Uso:
    python -m src.detection.export_onnx
    python -m src.detection.export_onnx --weights models/yolov8m_best.pt --opset 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.shared.seeds import set_seed


def export(weights: Path, opset: int, imgsz: int, dynamic: bool, simplify: bool) -> Path:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    out = model.export(
        format="onnx",
        opset=opset,
        imgsz=imgsz,
        dynamic=dynamic,
        simplify=simplify,
    )
    out = Path(out)
    print(f"ONNX exportado: {out}")
    return out


def verify(onnx_path: Path, imgsz: int) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("AVISO: onnxruntime/numpy no instalados; se omite la verificación.")
        return
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    outs = sess.get_outputs()
    dummy = np.random.rand(1, 3, imgsz, imgsz).astype("float32")
    res = sess.run(None, {inp.name: dummy})
    print("\n--- Verificación ONNX Runtime ---")
    print(f"  Entrada : {inp.name}  shape={inp.shape}  dtype={inp.type}")
    for o, r in zip(outs, res):
        print(f"  Salida  : {o.name}  shape={o.shape}  -> run ok, out={getattr(r, 'shape', None)}")
    print("  Inferencia dummy OK.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Exportación a ONNX (tatuajes)")
    ap.add_argument("--weights", default=str(paths.MODELS / "yolov8m_best.pt"),
                    help="pesos .pt a exportar (modelo definitivo)")
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--dynamic", action="store_true",
                    help="batch/size dinámico (por defecto fijo para ML.NET)")
    ap.add_argument("--no-simplify", action="store_true")
    args = ap.parse_args()

    set_seed()
    weights = Path(args.weights)
    if not weights.exists():
        print(f"ERROR: no existe {weights}. Entrena primero (train.py).")
        return 1

    onnx_path = export(weights, args.opset, args.imgsz,
                       dynamic=args.dynamic, simplify=not args.no_simplify)
    verify(onnx_path, args.imgsz)

    print("\nResumen para el backend .NET:")
    print(f"  - Archivo  : {onnx_path}")
    print(f"  - Opset    : {args.opset}")
    print(f"  - Entrada  : 1x3x{args.imgsz}x{args.imgsz} (NCHW, float32, RGB 0-1)")
    print("  - Salida   : detecciones YOLOv8 [1, 5, 8400] (x,y,w,h,score) en 1 clase")
    print("  - Postproceso: aplicar NMS (conf>=0.25, iou>=0.45) en el backend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
