"""Separa un conjunto de negativos según si aparece o no una persona.

Para qué. Un negativo se usa como **imagen de fondo**: se le dice al detector que
ahí no hay ningún tatuaje. Si una de esas imágenes sí tiene uno, se le está
enseñando lo contrario de lo que se quiere, y el error es peor que el que se
intenta corregir: en vez de sobrar detecciones, faltarían.

Un tatuaje está sobre piel. Una fotografía **sin ninguna persona** — un muro con
grafiti, un cuaderno, una pizarra — no puede contener uno, así que es segura como
fondo sin más revisión. Solo hace falta mirar con cuidado las que sí tienen gente.

Revisar 245 imágenes en una hoja de contactos es poco fiable: un tatuaje pequeño o
medio tapado se escapa. Esto convierte el problema en uno acotado — mirar bien
treinta fotografías en vez de mirar mal doscientas cuarenta.

Se usa YOLOv8n con pesos COCO, que trae la clase «person». Es una herramienta de
cribado, no forma parte del sistema: no interviene en ninguna medición ni se
despliega.

Uso:
    python -m src.detection.split_by_person --origen datasets/negativos_duros
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths

EXTENSIONES = {".jpg", ".jpeg", ".png"}

# Bajo a proposito: para decidir «hay que mirar esta», conviene equivocarse por
# exceso. Una imagen sin personas clasificada como «con personas» solo cuesta una
# mirada; al reves, cuesta un negativo mal etiquetado.
CONF_PERSONA = 0.25

# Fraccion del area de la imagen por debajo de la cual la persona sale tan pequena
# que no se le distinguiria un tatuaje aunque lo tuviera.
AREA_MINIMA = 0.01


def main() -> int:
    ap = argparse.ArgumentParser(description="Separa negativos por presencia de personas")
    ap.add_argument("--origen", default=str(paths.DATASETS / "negativos_duros"))
    ap.add_argument("--conf", type=float, default=CONF_PERSONA)
    args = ap.parse_args()

    base = Path(args.origen)
    fuentes = [base / "sin_deteccion", base / "con_deteccion"]
    imagenes = sorted(p for f in fuentes if f.exists()
                      for p in f.glob("*") if p.suffix.lower() in EXTENSIONES)
    if not imagenes:
        print(f"No hay imágenes en {base}. Ejecute antes screen_negatives.")
        return 1

    from PIL import Image
    from ultralytics import YOLO

    print(f"{len(imagenes)} imágenes. Cribando con YOLOv8n (COCO, clase 'person')…")
    detector = YOLO("yolov8n.pt")

    con_persona: list[dict] = []
    sin_persona: list[str] = []

    for ruta in imagenes:
        try:
            img = Image.open(ruta).convert("RGB")
        except Exception:  # noqa: BLE001
            print(f"  ilegible, se descarta: {ruta.name}")
            continue

        ancho, alto = img.size
        area_total = ancho * alto
        r = detector.predict(img, conf=args.conf, imgsz=640, classes=[0], verbose=False)[0]

        personas = []
        for b in (r.boxes or []):
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1) / area_total
            if area >= AREA_MINIMA:
                personas.append({"area": round(area, 4), "conf": round(float(b.conf[0]), 3)})

        if personas:
            con_persona.append({
                "archivo": ruta.name,
                "carpeta": ruta.parent.name,
                "personas": len(personas),
                "area_mayor": max(p["area"] for p in personas),
            })
        else:
            sin_persona.append(ruta.name)

    destino = base / "con_personas"
    destino.mkdir(parents=True, exist_ok=True)
    for item in con_persona:
        origen_ruta = base / item["carpeta"] / item["archivo"]
        copia = destino / item["archivo"]
        if not copia.exists():
            shutil.copy2(origen_ruta, copia)

    resumen = {
        "total": len(imagenes),
        "sin_personas": len(sin_persona),
        "con_personas": len(con_persona),
        "conf_persona": args.conf,
        "area_minima": AREA_MINIMA,
        "criterio": "Una fotografia sin personas no puede contener un tatuaje, de modo "
                    "que es segura como imagen de fondo sin revision adicional. Las que "
                    "tienen personas se revisan una a una a resolucion completa.",
        "a_revisar": sorted(con_persona, key=lambda x: -x["area_mayor"]),
    }
    (base / "cribado_personas.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSin personas (seguras como fondo): {len(sin_persona)}")
    print(f"Con personas (hay que mirarlas):   {len(con_persona)}  ->  {destino}")
    print(f"\nDetalle: {base / 'cribado_personas.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
