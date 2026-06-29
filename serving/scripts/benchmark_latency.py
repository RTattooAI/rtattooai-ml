"""Benchmark local de latencia del pipeline completo (detección + identificación).

Mide `processing_ms` (ya instrumentado en `serving.pipeline.TattooService.identify`)
sobre una muestra de imágenes de test, en CPU. Corre dos escenarios:
  - "full": todos los hilos disponibles de esta máquina (cota optimista).
  - "1vcpu": hilos de torch/onnxruntime limitados a 1 (aproxima un Container
    App de Azure con 1 vCPU, antes de tener el despliegue real para medir).

Uso:
    python -m serving.scripts.benchmark_latency --n 40
"""
from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image

from src.shared import paths


def run_scenario(n: int, threads: int | None) -> dict:
    if threads:
        import os
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        import torch
        torch.set_num_threads(threads)

    from serving.pipeline import TattooService

    t0 = time.perf_counter()
    service = TattooService()
    cold_start_ms = (time.perf_counter() - t0) * 1000

    images_dir = paths.UNIFIED_IMAGES / "test"
    images = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:n]

    times = []
    for p in images:
        img = Image.open(p)
        result = service.identify(img)
        times.append(result["processing_ms"])
        # warm-up: descarta la primera (carga de caches internas)
    warm = times[1:] if len(times) > 1 else times

    return {
        "n": len(warm),
        "cold_start_ms": round(cold_start_ms, 1),
        "first_request_ms": times[0] if times else None,
        "mean_ms": round(statistics.mean(warm), 1),
        "median_ms": round(statistics.median(warm), 1),
        "p90_ms": round(sorted(warm)[int(0.9 * len(warm))], 1) if warm else None,
        "p95_ms": round(sorted(warm)[int(0.95 * len(warm))], 1) if warm else None,
        "max_ms": round(max(warm), 1) if warm else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark de latencia del pipeline de serving")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    print(f"CPU local: {platform.processor()}")
    print("\n== Escenario 'full' (todos los hilos disponibles) ==")
    full = run_scenario(args.n, threads=None)
    print(full)

    print("\n== Escenario '1vcpu' (hilos limitados a 1, proxy de 1 vCPU Azure) ==")
    one = run_scenario(args.n, threads=1)
    print(one)

    paths.ensure_dirs(paths.REPORTS)
    lines = [
        "# Benchmark de latencia local (serving/) — proxy mientras no hay despliegue real\n",
        f"> Medido en hardware local ({platform.processor()}), NO en el Container App de "
        "Azure final. Sirve como cota orientativa hasta poder medir en el entorno real "
        "de despliegue; re-medir con esta misma herramienta una vez desplegado.\n",
        "| Escenario | N | cold start (ms) | media (ms) | mediana (ms) | p90 (ms) | p95 (ms) | máx (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| full (multi-hilo) | {full['n']} | {full['cold_start_ms']} | {full['mean_ms']} | "
        f"{full['median_ms']} | {full['p90_ms']} | {full['p95_ms']} | {full['max_ms']} |",
        f"| 1 vCPU (proxy) | {one['n']} | {one['cold_start_ms']} | {one['mean_ms']} | "
        f"{one['median_ms']} | {one['p90_ms']} | {one['p95_ms']} | {one['max_ms']} |",
        "\n**Nota:** `cold_start_ms` es el costo de cargar YOLOv8m ONNX + DINOv2 + galería "
        "una sola vez (al iniciar el contenedor); no se paga por request.",
    ]
    report = "\n".join(lines) + "\n"
    (paths.REPORTS / "serving_latency_local.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
