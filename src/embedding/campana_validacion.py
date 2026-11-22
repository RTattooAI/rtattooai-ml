"""Prepara el material del experimento con usuarios (OE4): catálogo de campaña, consultas y clave.

Toma identidades reales que no se usaron para entrenar (fuentes by_tatoo, tattoodet y
playground_tattootest) y distractores de otras fuentes. Simula la campaña con el mismo código
que producción (serving.pipeline en modo ensamble) sobre una copia de la galería base, para
quedarse solo con fotos donde el detector encuentra exactamente un tatuaje y conocer de
antemano el resultado del sistema. La selección es aleatoria entre las elegibles: no se
eligen fotos por su resultado.

Uso:
    python -m src.embedding.campana_validacion --salida <carpeta>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.embedding import experimento_ab as ab

N_CATALOGO, N_DISTRACTORES = 20, 10
TIPOS = ["ar", "i", "a", "b", "c", "r"]  # se rota el tipo preferido para variar las consultas
NOMBRE_TIPO = {"ar": "perspectiva + giro", "i": "encuadre / iluminación", "a": "perspectiva",
               "b": "desenfoque", "c": "recorte", "r": "giro"}


def grupos_con_rutas(train_h):
    grupos: dict[tuple, dict] = defaultdict(dict)
    for fuente in ab.FUENTES_ID:
        for img in sorted((paths.RAW / fuente).glob("*/images/*")):
            m = ab.NOMBRE.match(img.name)
            if not m:
                continue
            clave = (m[1], m[2])
            variante = (m[3] or "orig") + (m[4] or "")
            if variante in grupos[clave]:
                continue
            lbl = img.parent.parent / "labels" / (img.stem + ".txt")
            if len(ab.recortar(img, lbl)) != 1:
                continue
            grupos[clave][variante] = img
    salida = []
    for clave, vers in sorted(grupos.items()):
        # Mismas identidades que la prueba final del OE3 (≥ 3 versiones): sus recortes se
        # excluyeron del entrenamiento de los codificadores.
        if len(vers) < 3:
            continue
        orden = sorted(vers, key=lambda v: (v != "orig", v))  # se enrola la original si existe (como en OE3)
        salida.append({"id": f"{clave[0]}_{clave[1]}", "enrol": orden[0],
                       "versiones": {k: str(v) for k, v in vers.items()}})
    return salida


def distractores(train_h, rng, n):
    imgs = [p for f in ab.FUENTES_DISTRACTOR for p in sorted((paths.RAW / f).glob("*/images/*"))]
    vistos, unicas = set(), []
    for p in imgs:
        base = p.name.split(".rf.")[0]
        if base not in vistos:
            vistos.add(base); unicas.append(p)
    rng.shuffle(unicas)
    out = []
    for p in unicas:
        if len(ab.recortar(p, p.parent.parent / "labels" / (p.stem + ".txt"))) != 1:
            continue
        if ab.vista_en(ab.phash(Image.open(p).convert("RGB")), train_h):
            continue
        out.append(p)
        if len(out) >= n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", required=True)
    ap.add_argument("--semilla", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.salida)
    rng = np.random.RandomState(a.semilla)

    tmp = Path(tempfile.mkdtemp())
    shutil.copy(paths.MODELS / "ensamble" / "galeria_inicial.npz", tmp / "galeria.npz")
    os.environ["GALERIA_ENSAMBLE_PATH"] = str(tmp / "galeria.npz")
    os.environ.pop("AZURE_STORAGE_CONNECTION_STRING", None)
    from serving.pipeline import TattooService
    svc = TattooService()

    train_h = ab.hashes_entrenamiento()
    ids = grupos_con_rutas(train_h)
    print("identidades elegibles (prueba final del OE3):", len(ids))
    rng.shuffle(ids)

    def una_deteccion(ruta):
        d = svc.detect(Image.open(ruta))
        return d[0] if len(d) == 1 else None

    elegidas = []
    for it in ids:
        det_e = una_deteccion(it["versiones"][it["enrol"]])
        if det_e is None:
            continue
        otras = [v for v in it["versiones"] if v != it["enrol"]]
        pref = TIPOS[len(elegidas) % len(TIPOS)]
        otras.sort(key=lambda v: (v.rstrip("0123456789") != pref, rng.rand()))
        consulta = next((v for v in otras if una_deteccion(it["versiones"][v]) is not None), None)
        if consulta is None:
            continue
        it.update(crop=det_e[1], consulta=consulta)
        elegidas.append(it)
        if len(elegidas) == N_CATALOGO:
            break
    print("catálogo de campaña:", len(elegidas))

    dist = [p for p in distractores(train_h, rng, 60) if una_deteccion(p) is not None][:N_DISTRACTORES]
    print("distractores:", len(dist))

    (out / "enrolar").mkdir(parents=True, exist_ok=True)
    (out / "consultas_todas").mkdir(exist_ok=True)
    bandas = ["Banda de prueba A", "Banda de prueba B", "Banda de prueba C", "Banda de prueba D"]
    cat = []
    for k, it in enumerate(elegidas, 1):
        cod = f"C{k:02d}"
        banda = bandas[(k - 1) % 4]
        foto = it["versiones"][it["enrol"]]
        shutil.copy(foto, out / "enrolar" / f"{cod}_foto_completa{Path(foto).suffix}")
        it["crop"].save(out / "enrolar" / f"{cod}_recorte.png")
        r = svc.enroll(it["crop"], band=banda, catalog_id=cod)
        cat.append({"codigo": cod, "banda": banda, "id_fuente": it["id"], "foto": foto})

    consultas = []
    for it, c in zip(elegidas, cat):
        consultas.append({"verdad": c["codigo"], "banda": c["banda"], "ruta": it["versiones"][it["consulta"]],
                          "version": NOMBRE_TIPO.get(it["consulta"].rstrip("0123456789"), it["consulta"])})
    for p in dist:
        consultas.append({"verdad": "fuera", "banda": "—", "ruta": str(p), "version": "distractor"})

    for q in consultas:
        res = svc.identify(Image.open(q["ruta"]))
        d = res["detections"]
        q["detecciones"] = len(d)
        if d:
            q["semaforo"] = d[0]["verdict"]
            q["top1"] = d[0]["matches"][0]["catalog_id"] if d[0]["matches"] else None
            q["similitud_top1"] = round(d[0]["matches"][0]["similarity"], 3) if d[0]["matches"] else None
        q["acierto_sistema"] = ((q["verdad"] == "fuera" and q.get("semaforo") == "no_match") or
                                (q["verdad"] != "fuera" and q.get("semaforo") != "no_match" and q.get("top1") == q["verdad"]))

    # Conjuntos A y B equivalentes: 10 del catálogo + 5 fuera cada uno; los 10 primeros de cada
    # conjunto contienen 7 + 3 para poder recortar la sesión sin desbalancear.
    dentro = [q for q in consultas if q["verdad"] != "fuera"]
    fuera = [q for q in consultas if q["verdad"] == "fuera"]
    conj = {"A": (dentro[0::2], fuera[0::2]), "B": (dentro[1::2], fuera[1::2])}
    final = []
    for nombre, (di, fu) in conj.items():
        orden = di[:7] + fu[:3]
        rng.shuffle(orden)
        resto = di[7:] + fu[3:]
        rng.shuffle(resto)
        for k, q in enumerate(orden + resto, 1):
            q["foto"] = f"{nombre}{k:02d}"
            q["conjunto"] = nombre
            dst = out / "consultas_todas" / f"{q['foto']}{Path(q['ruta']).suffix}"
            shutil.copy(q["ruta"], dst)
            final.append(q)

    resumen = {
        "semilla": a.semilla, "catalogo_campana": cat, "consultas": final,
        "acierto_sistema": round(float(np.mean([q["acierto_sistema"] for q in final])), 3),
        "semaforo": {s: sum(q.get("semaforo") == s for q in final) for s in ("match", "review", "no_match")},
        "modelo": svc.info().get("model_version"),
    }
    (out / "clave.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: resumen[k] for k in ("acierto_sistema", "semaforo", "modelo")}, ensure_ascii=False))
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
