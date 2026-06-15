# Verificación del entorno (toolchain)

> Este documento registra que el **entorno de ejecución** quedó verificado.
> **No contiene métricas del proyecto**: el entrenamiento real está pendiente
> de los datos (ver bloqueo de `ROBOFLOW_API_KEY` en el README). No se ha
> fabricado ninguna métrica de detección de tatuajes.

**Fecha:** 2026-06-14

## Hardware / software

| Componente | Valor |
|------------|-------|
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop GPU (8 GB VRAM) |
| Driver / CUDA | 596.49 / CUDA 13.2 (driver) |
| OS | Windows 11 |
| Python | 3.14.3 |
| PyTorch | 2.11.0+cu128 (CUDA build 12.8) |
| torchvision | 0.26.0+cu128 |
| Ultralytics | 8.4.67 |
| ONNX / ONNX Runtime | 1.21.0 / 1.26.0 |
| roboflow | 1.3.10 |

`torch.cuda.is_available()` → **True**; operación matricial en GPU verificada.

## Smoke test end-to-end (datos sintéticos, NO del proyecto)

Para validar que el pipeline completo funciona en esta máquina se ejecutó un
*smoke test* con imágenes **sintéticas** en un directorio temporal (luego
eliminado). Resultado:

- Entrenamiento `yolov8n` 2 épocas en GPU (CUDA:0) — **OK**
- Validación (`model.val`) — **OK** (mAP de juguete, sin significado real)
- Export a ONNX (opset 12, entrada `images [1,3,160,160]`,
  salida `output0 [1,5,N]`) — **OK**
- Carga y inferencia con ONNX Runtime — **OK**

Esto confirma que `train.py`, `evaluate.py` y `export_onnx.py` corren de
principio a fin en este entorno. Los nombres de tensores ONNX verificados
(`images` / `output0`) coinciden con lo documentado para el backend .NET.

## Nota Windows + Python 3.14

El DataLoader de Ultralytics usa multiprocesamiento; en Windows requiere el
guard `if __name__ == "__main__":`. Todos los scripts del repo lo incluyen y
deben ejecutarse como módulo (`python -m src.detection.<script>`).
