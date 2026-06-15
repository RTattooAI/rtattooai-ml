"""Descarga las fuentes públicas de detección de tatuajes.

Fuentes:
  * Roboflow Universe (varias) -> requiere ROBOFLOW_API_KEY en el entorno.
  * DeMSI (FER, U. de Zagreb) -> descarga MANUAL (login al repositorio FER);
    el script no puede automatizarla, solo deja instrucciones y la integra
    si ya fue colocada en datasets/raw/demsi/.

Cada fuente se descarga SIN modificar dentro de datasets/raw/<key>/.
La unificación a formato YOLO de 1 clase la hace prepare_dataset.py.

Uso:
    python -m src.detection.download_data
    python -m src.detection.download_data --only lance_nelson xisketos
    python -m src.detection.download_data --list

NOTA de slugs: los slugs de workspace/project se obtuvieron de
universe.roboflow.com (junio 2026). Si una descarga falla por slug/version,
edita SOURCES abajo (la URL de cada proyecto está en el campo "url").
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Permite ejecutar como script suelto (python src/detection/download_data.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.shared.seeds import set_seed

# ------------------------------------------------------------------
# Registro de fuentes Roboflow Universe.
#   key:        identificador interno (carpeta en datasets/raw/)
#   workspace:  slug del workspace en Roboflow
#   project:    slug del proyecto
#   version:    nº de versión a descargar (None = auto: la más alta)
#   fmt:        formato de export ("yolov8" = detección; los proyectos de
#               segmentación exportan polígonos que prepare_dataset convierte)
#   task:       "detect" | "segment" (informativo)
#   approx:     nº aproximado de imágenes (a verificar)
#   url:        página pública del dataset (para procedencia/licencia)
#   license:    licencia declarada (VERIFICAR en la página; Roboflow varía)
# ------------------------------------------------------------------
SOURCES: list[dict] = [
    {
        "key": "lance_nelson",
        "workspace": "lance-nelson-gtmdt",
        "project": "artat",
        "version": None,
        "fmt": "yolov8",
        "task": "detect",
        "approx": 1330,
        "url": "https://universe.roboflow.com/lance-nelson-gtmdt/artat",
        "license": "Roboflow Universe (VERIFICAR licencia en la página)",
    },
    {
        "key": "xisketos",
        "workspace": "xisketos",
        "project": "deteccion-de-tatuajes",
        "version": None,
        "fmt": "yolov8",
        "task": "detect",
        "approx": 1330,
        "url": "https://universe.roboflow.com/xisketos/deteccion-de-tatuajes",
        "license": "Roboflow Universe (VERIFICAR licencia en la página)",
    },
    {
        "key": "by_tatoo",
        "workspace": "tatoo-9syny",
        "project": "tattoo-detection-qia5k",
        "version": None,
        "fmt": "yolov8",
        "task": "detect",
        "approx": 603,
        "url": "https://universe.roboflow.com/tatoo-9syny/tattoo-detection-qia5k",
        "license": "Roboflow Universe (VERIFICAR licencia en la página)",
    },
    {
        "key": "code_with_heady",
        "workspace": "code-with-heady-5wdhf",
        "project": "tattoo-detection-updated",
        "version": None,
        "fmt": "yolov8",
        "task": "detect",
        "approx": 561,
        "url": "https://universe.roboflow.com/code-with-heady-5wdhf/tattoo-detection-updated",
        "license": "Roboflow Universe (VERIFICAR licencia en la página)",
    },
    {
        "key": "playground_tattootest",
        "workspace": "playground-xxvvy",
        "project": "tattootest-utzal",
        "version": None,
        "fmt": "yolov8",          # proyecto de segmentación -> exporta polígonos
        "task": "segment",
        "approx": 5190,
        "url": "https://universe.roboflow.com/playground-xxvvy/tattootest-utzal",
        "license": "Roboflow Universe (VERIFICAR licencia en la página)",
    },
    # ---- Fuentes adicionales encontradas en Universe (opcionales) ----
    # La de Gabija Pocevičė es la autora de la referencia citada (DeMSI/YOLOv5).
    {
        "key": "pocevice_tattoo",
        "workspace": "gabijapocevicemajor-7dvfl",
        "project": "tattoo-localization-and-recognition",
        "version": None,
        "fmt": "yolov8",
        "task": "detect",
        "approx": None,
        "url": "https://universe.roboflow.com/gabijapocevicemajor-7dvfl/tattoo-localization-and-recognition",
        "license": "Roboflow Universe (VERIFICAR licencia en la página)",
        "optional": True,
    },
    {
        "key": "tattoodet",
        "workspace": "tattoodet",
        "project": "tattoo-detection-cbtkn",
        "version": None,
        "fmt": "yolov8",
        "task": "detect",
        "approx": None,
        "url": "https://universe.roboflow.com/tattoodet/tattoo-detection-cbtkn",
        "license": "Roboflow Universe (VERIFICAR licencia en la página)",
        "optional": True,
    },
    {
        "key": "image_analysis_region",
        "workspace": "image-analysis-5i4lg",
        "project": "tattoo-region-detection",
        "version": None,
        "fmt": "yolov8",
        "task": "detect",
        "approx": None,
        "url": "https://universe.roboflow.com/image-analysis-5i4lg/tattoo-region-detection",
        "license": "Roboflow Universe (VERIFICAR licencia en la página)",
        "optional": True,
    },
]


def _resolve_version(project, preferred):
    """Devuelve un objeto Version: el indicado o el de número más alto."""
    if preferred is not None:
        return project.version(preferred)
    try:
        versions = project.versions()
        if versions:
            def vnum(v):
                try:
                    return int(str(getattr(v, "version", "0")).split("/")[-1])
                except Exception:
                    return 0
            return sorted(versions, key=vnum, reverse=True)[0]
    except Exception as e:  # noqa: BLE001
        print(f"   ! versions() falló ({e}); intento version(1)")
    return project.version(1)


def download_roboflow(sources: list[dict], api_key: str) -> list[dict]:
    """Descarga cada fuente Roboflow. Devuelve la lista con su estado."""
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    paths.ensure_dirs(paths.RAW)
    results = []
    for s in sources:
        dest = paths.RAW / s["key"]
        rec = {**s, "dest": str(dest), "status": "pendiente", "n_images": None}
        print(f"\n>> {s['key']}  ({s['workspace']}/{s['project']})")
        if dest.exists() and any(dest.rglob("*.jpg")):
            n = sum(1 for _ in dest.rglob("*.jpg")) + sum(1 for _ in dest.rglob("*.png"))
            rec.update(status="ya_existe", n_images=n)
            print(f"   ya existe ({n} imágenes); se omite.")
            results.append(rec)
            continue
        try:
            project = rf.workspace(s["workspace"]).project(s["project"])
            version = _resolve_version(project, s["version"])
            version.download(s["fmt"], location=str(dest), overwrite=True)
            n = sum(1 for _ in dest.rglob("*.jpg")) + sum(1 for _ in dest.rglob("*.png"))
            rec.update(status="ok", n_images=n,
                       version=str(getattr(version, "version", s["version"])))
            print(f"   OK -> {dest}  ({n} imágenes)")
        except Exception as e:  # noqa: BLE001
            rec.update(status=f"FALLÓ: {type(e).__name__}: {e}")
            print(f"   FALLÓ: {e}")
        results.append(rec)
    return results


DEMSI_DIR = paths.RAW / "demsi"
DEMSI_NOTE = """\
================= DeMSI (descarga MANUAL) =================
El dataset DeMSI (FER, U. de Zagreb, ~890 imágenes de ImageNet anotadas como
polígonos) NO tiene enlace de descarga público: está detrás del login del
repositorio del proyecto DeMSI en FER. No es automatizable desde este script.

Para incluirlo:
  1. Solicita/obtén acceso en:
     https://www.fer.unizg.hr/demsi/databases_and_code/tattoo_dataset
  2. Coloca las imágenes + anotaciones (polígonos) en:
     datasets/raw/demsi/
  3. Vuelve a correr prepare_dataset.py: detectará la carpeta y convertirá
     los polígonos a bounding boxes automáticamente.
Referencia: Pocevičė et al., 2024 (YOLOv5 sobre DeMSI, mAP@0.5 = 0.79).
==========================================================
"""


def check_demsi() -> dict:
    rec = {"key": "demsi", "url": "https://www.fer.unizg.hr/demsi/databases_and_code/tattoo_dataset",
           "license": "FER / U. Zagreb (uso académico — VERIFICAR términos)",
           "task": "segment", "approx": 890}
    if DEMSI_DIR.exists() and any(DEMSI_DIR.rglob("*")):
        n = sum(1 for ext in ("*.jpg", "*.jpeg", "*.png") for _ in DEMSI_DIR.rglob(ext))
        rec.update(status="ya_existe", n_images=n, dest=str(DEMSI_DIR))
        print(f"\n>> demsi: encontrada carpeta manual ({n} imágenes).")
    else:
        rec.update(status="MANUAL_REQUERIDA", n_images=None, dest=str(DEMSI_DIR))
        print("\n" + DEMSI_NOTE)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="Descarga de fuentes de tatuajes")
    ap.add_argument("--only", nargs="*", help="solo estas keys")
    ap.add_argument("--list", action="store_true", help="lista las fuentes y sale")
    ap.add_argument("--skip-optional", action="store_true",
                    help="omite las fuentes marcadas como opcionales")
    args = ap.parse_args()

    set_seed()

    if args.list:
        for s in SOURCES:
            opt = " [opcional]" if s.get("optional") else ""
            print(f"  {s['key']:22s} {s['workspace']}/{s['project']}  ~{s['approx']}{opt}")
        print(f"  {'demsi':22s} (descarga MANUAL)")
        return 0

    sources = SOURCES
    if args.only:
        sources = [s for s in SOURCES if s["key"] in set(args.only)]
    elif args.skip_optional:
        sources = [s for s in SOURCES if not s.get("optional")]

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    results = []
    if not api_key:
        print("=" * 64)
        print("BLOQUEO: falta ROBOFLOW_API_KEY en el entorno.")
        print("No se pueden descargar las fuentes de Roboflow Universe.")
        print("Configúrala (PowerShell, persistente para el usuario):")
        print('  setx ROBOFLOW_API_KEY "tu_clave_aqui"')
        print("Abre una terminal nueva y reintenta. (No la commitees.)")
        print("=" * 64)
    else:
        results = download_roboflow(sources, api_key)

    # DeMSI siempre se chequea (no bloquea)
    results.append(check_demsi())

    # Manifiesto
    paths.ensure_dirs(paths.REPORTS)
    manifest = paths.REPORTS / "download_manifest.json"
    manifest.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nManifiesto guardado en: {manifest}")

    ok = [r for r in results if r.get("status") in ("ok", "ya_existe")]
    print(f"Fuentes con datos: {len(ok)} / {len(results)}")
    if not api_key:
        return 2  # código de bloqueo
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
