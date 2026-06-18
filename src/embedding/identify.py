"""Identificación end-to-end: detecta tatuajes (YOLO), los recorta y los compara
contra la galería de bandas por similitud coseno (embeddings DINOv2).

NO es un clasificador "banda/no-banda": devuelve el top-K más parecido de la
galería con su puntaje, y un veredicto por **umbral** (posible coincidencia vs
"ninguna conocida"). Siempre como apoyo: requiere validación humana.

Uso (consulta por línea de comandos):
    python -m src.embedding.identify --image ruta/a/foto.jpg --topk 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.embedding.embedder import TattooEmbedder

GANG_DIR = paths.DATASETS / "gang_tattoos"
GALLERY_NPZ = paths.MODELS / "gallery_dinov2.npz"
DETECTOR = paths.MODELS / "yolov8m_best.pt"
DEFAULT_THRESHOLD = 0.45   # calibrado en el Paso 0 (separa banda ~0.71 de no-banda ~0.28)


class TattooIdentifier:
    def __init__(self, detector_weights: Path = DETECTOR, gallery_path: Path = GALLERY_NPZ,
                 threshold: float = DEFAULT_THRESHOLD, embedder: TattooEmbedder | None = None):
        from ultralytics import YOLO
        self.detector = YOLO(str(detector_weights))
        data = np.load(str(gallery_path), allow_pickle=True)
        self.gallery = data["embeddings"].astype("float32")      # (G, dim) L2-norm
        self.names = [str(n) for n in data["names"]]
        self.embedder = embedder or TattooEmbedder()
        self.threshold = threshold

    def detect_crops(self, image: Image.Image, conf=0.25, iou=0.45):
        """Devuelve [(xyxy, crop_PIL, conf_det)] de los tatuajes detectados."""
        res = self.detector.predict(image.convert("RGB"), conf=conf, iou=iou,
                                    imgsz=640, verbose=False)[0]
        out = []
        if res.boxes is not None:
            for b in res.boxes:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                if x2 > x1 and y2 > y1:
                    crop = image.convert("RGB").crop((x1, y1, x2, y2))
                    out.append(((x1, y1, x2, y2), crop, float(b.conf[0])))
        return out

    def match(self, crop: Image.Image, topk=5):
        """Top-K de la galería para un recorte. Devuelve (matches, es_coincidencia)."""
        e = self.embedder.embed(crop)
        sims = self.gallery @ e
        order = np.argsort(-sims)[:topk]
        matches = [(self.names[i], float(sims[i])) for i in order]
        is_match = bool(matches and matches[0][1] >= self.threshold)
        return matches, is_match

    def identify(self, image: Image.Image, conf=0.25, iou=0.45, topk=5):
        """Flujo completo sobre una imagen. Lista de detecciones con su top-K."""
        results = []
        for box, crop, dconf in self.detect_crops(image, conf, iou):
            matches, is_match = self.match(crop, topk)
            results.append({"box": box, "crop": crop, "det_conf": dconf,
                            "matches": matches, "is_match": is_match})
        return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Consulta: identificar tatuaje(s) en una foto")
    ap.add_argument("--image", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = ap.parse_args()

    if not GALLERY_NPZ.exists():
        print(f"ERROR: no existe la galería {GALLERY_NPZ}. Corre step0_baseline.py primero.")
        return 1
    img = Image.open(args.image)
    ident = TattooIdentifier(threshold=args.threshold)
    results = ident.identify(img, topk=args.topk)
    if not results:
        print("No se detectaron tatuajes en la imagen.")
        return 0
    for i, r in enumerate(results, 1):
        print(f"\nTatuaje #{i}  (det conf={r['det_conf']:.2f}, caja={r['box']})")
        verdict = (f"POSIBLE coincidencia: {r['matches'][0][0]}"
                   if r["is_match"] else "Ninguna banda conocida (bajo umbral)")
        print(f"  Veredicto: {verdict}")
        for name, score in r["matches"]:
            print(f"    {score:.3f}  {name}")
    print("\n(Apoyo preliminar — requiere validación humana.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
