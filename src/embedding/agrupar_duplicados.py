"""Agrupa los recortes de entrenamiento que son el mismo tatuaje.

Por qué. Al entrenar la cabeza y la CNN, cada recorte era una identidad. Pero los
datasets traen varias versiones de la misma foto (giradas, volteadas, re-subidas),
y la pérdida contrastiva las separaba a propósito: aprendía lo contrario de lo
que necesitamos. Revisando muestras a vista, cerca del 30 % de los recortes tiene
otra versión en el mismo split.

Cómo. Dos opiniones independientes:

  1. **Semántica (DINOv2 sin cabeza):** similitud coseno de los embeddings.
  2. **Píxeles (orientación libre):** correlación de la imagen en gris a 32×32,
     tomando la mejor de las 8 orientaciones (4 giros × volteo). Detecta «misma
     foto girada» sin depender de ningún modelo entrenado.

Reglas, ajustadas con 96 pares revisados a vista por el equipo
(reports/experimentos/duplicados_etiquetados.json):

  * mismo tatuaje  → se fusionan en una identidad (positivos entre sí);
  * dudoso         → no se usan como negativos entre sí (se enmascaran);
  * distinto       → negativos normales.

Además se excluyen del entrenamiento los recortes que coinciden con cualquier
imagen usada en experimento_ab, para que la evaluación no tenga fuga.

Uso:
    python -m src.embedding.agrupar_duplicados --etiquetar   # recupera los 96 pares revisados
    python -m src.embedding.agrupar_duplicados               # ajusta reglas y escribe el manifiesto
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths

TRAIN = paths.DATASETS / "real_crops" / "train"
SALIDA = paths.REPORTS / "experimentos"
CACHE = SALIDA / "emb_train_dinov2.npz"
ETIQUETAS = SALIDA / "duplicados_etiquetados.json"
MANIFIESTO = paths.DATASETS / "real_crops" / "identidades_train.json"

# Revisión a vista del 16/10/2026 (números = posición en cada hoja de 24 pares).
DISTINTOS = {
    "0.95-1.00": set(),
    "0.92-0.95": {3, 8, 12},
    "0.90-0.92": {1, 2, 4, 6, 8, 21, 22, 23},   # el 15 es el mismo tatuaje en otra foto
    "0.85-0.90": set(range(1, 25)) - {8, 15, 18},
}
MAX_GRUPO = 12
PIX_FUERTE = 0.85
BANDAS = [(0.95, 1.01), (0.92, 0.95), (0.90, 0.92), (0.85, 0.90)]


def archivos() -> list[Path]:
    return sorted(TRAIN.glob("*.png"))


def embeddings(fs: list[Path]) -> np.ndarray:
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if list(z["nombres"]) == [f.name for f in fs]:
            return z["E"]
    from src.embedding.backbones import resolver
    emb = resolver("dinov2:sin-head")
    E = np.concatenate([emb.embed([Image.open(f).convert("RGB") for f in fs[i:i + 512]])
                        for i in range(0, len(fs), 512)])
    np.savez_compressed(CACHE, E=E, nombres=np.array([f.name for f in fs]))
    return E


# ----------------------------------------------------------------- opinión de píxeles
def miniatura(path: Path) -> np.ndarray:
    g = Image.open(path).convert("L").filter(ImageFilter.GaussianBlur(1))
    return np.asarray(g.resize((32, 32), Image.BILINEAR), dtype="float32")


def orientaciones(a: np.ndarray) -> list[np.ndarray]:
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out += [r, r[:, ::-1]]
    return out


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float((a * b).mean())


def pixeles(pa: Path, pb: Path, cache: dict) -> float:
    for p in (pa, pb):
        if p not in cache:
            cache[p] = miniatura(p)
    return max(ncc(o, cache[pb]) for o in orientaciones(cache[pa]))


# ----------------------------------------------------------------- pares revisados
def recuperar_pares(fs, E) -> list[dict]:
    """Reproduce el muestreo de las hojas enviadas (misma semilla y orden)."""
    S = E @ E.T
    np.fill_diagonal(S, 0)
    iu = np.triu_indices(len(fs), 1)
    s = S[iu]
    rng = np.random.RandomState(0)
    pares = []
    for lo, hi in BANDAS:
        clave = f"{lo:.2f}-{min(hi, 1):.2f}"
        idx = np.where((s >= lo) & (s < hi))[0]
        pick = rng.choice(idx, min(24, len(idx)), replace=False)
        for n, k in enumerate(pick, 1):
            a, b = int(iu[0][k]), int(iu[1][k])
            pares.append({"hoja": clave, "n": n, "a": fs[a].name, "b": fs[b].name, "cos": round(float(s[k]), 4),
                          "mismo": n not in DISTINTOS[clave]})
    return pares


# ----------------------------------------------------------------- exclusión por experimento
def imagenes_experimento() -> list[Image.Image]:
    from src.embedding import experimento_ab as ab
    rng = np.random.RandomState(42)
    train_h = ab.hashes_entrenamiento()
    ids = ab.identidades(3, train_h)
    out = [i["enrol"] for i in ids] + [im for i in ids for _, im in i["consultas"]]
    out += ab.recortes_de(ab.FUENTES_DISTRACTOR, 300, rng)
    out += ab.recortes_demsi(600, rng)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Agrupa duplicados del split train")
    ap.add_argument("--etiquetar", action="store_true", help="solo recupera y guarda los pares revisados")
    args = ap.parse_args()
    fs = archivos()
    E = embeddings(fs)
    cache: dict = {}

    if args.etiquetar or not ETIQUETAS.exists():
        pares = recuperar_pares(fs, E)
        ETIQUETAS.write_text(json.dumps(pares, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Guardados {len(pares)} pares revisados en {ETIQUETAS}")
        print("  cos de los primeros de cada hoja:", [p["cos"] for p in pares[::24]])
        if args.etiquetar:
            return 0

    pares = json.loads(ETIQUETAS.read_text(encoding="utf-8"))
    for p in pares:
        p["pix"] = pixeles(TRAIN / p["a"], TRAIN / p["b"], cache)
    print("\nPares revisados (cos, píxeles, mismo):")
    for hoja in dict.fromkeys(p["hoja"] for p in pares):
        g = [p for p in pares if p["hoja"] == hoja]
        m = [p["pix"] for p in g if p["mismo"]]
        d = [p["pix"] for p in g if not p["mismo"]]
        print(f"  {hoja}: mismos={len(m)} pix∈[{min(m, default=0):.2f},{max(m, default=0):.2f}] | "
              f"distintos={len(d)} pix∈[{min(d, default=0):.2f},{max(d, default=0):.2f}]")

    # reglas: elegir el umbral de píxeles que no fusione ningún par distinto
    dist_pix = [p["pix"] for p in pares if not p["mismo"] and p["cos"] >= 0.85]
    umbral_pix = round(max(dist_pix) + 0.02, 3) if dist_pix else 0.8
    def decide(c, px):
        if c >= 0.95 or (c >= 0.85 and px >= umbral_pix):
            return "mismo"
        if c >= 0.88:
            return "dudoso"
        return "distinto"
    ok = sum(decide(p["cos"], p["pix"]) == "mismo" for p in pares if p["mismo"])
    falsos = sum(decide(p["cos"], p["pix"]) == "mismo" for p in pares if not p["mismo"])
    print(f"\nRegla: mismo si cos ≥ 0.95, o cos ≥ 0.85 y píxeles ≥ {umbral_pix}; dudoso si cos ≥ 0.88")
    print(f"  en los pares revisados: {ok}/{sum(p['mismo'] for p in pares)} mismos detectados, "
          f"{falsos}/{sum(not p['mismo'] for p in pares)} distintos fusionados por error")

    # aplicar a todos los pares candidatos
    S = E @ E.T
    np.fill_diagonal(S, 0)
    ii, jj = np.where(np.triu(S >= 0.85, 1))
    padre = list(range(len(fs)))

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    dudosos, aristas = [], []
    for a, b in zip(ii, jj):
        c = float(S[a, b])
        px = pixeles(fs[a], fs[b], cache) if c < 0.95 else 1.0
        d = decide(c, px)
        if d == "mismo":
            aristas.append((int(a), int(b), c, px))
            padre[raiz(a)] = raiz(b)
        elif d == "dudoso":
            dudosos.append((int(a), int(b)))
    grupo = np.array([raiz(i) for i in range(len(fs))])

    # Cadenas: textos en fondo claro se parecen en píxeles y se encadenan en grupos
    # enormes de frases distintas. Un grupo grande se rehace solo con aristas fuertes
    # y sus aristas débiles pasan a «dudoso» (no se usan como negativos).
    tam = np.bincount(grupo, minlength=len(fs))
    grandes = set(np.where(tam > MAX_GRUPO)[0])
    if grandes:
        padre = list(range(len(fs)))
        for a, b, c, px in aristas:
            if grupo[a] in grandes:
                if c >= 0.95 or px >= PIX_FUERTE:
                    padre[raiz(a)] = raiz(b)
                else:
                    dudosos.append((a, b))
            else:
                padre[raiz(a)] = raiz(b)
        print(f"Grupos de más de {MAX_GRUPO} recortes rehechos con aristas fuertes: {len(grandes)}")
        grupo = np.array([raiz(i) for i in range(len(fs))])

    # exclusión de lo que se evalúa
    from src.embedding.backbones import resolver
    emb = resolver("dinov2:sin-head")
    X = emb.embed(imagenes_experimento())
    fuga = (E @ X.T).max(1) >= 0.85
    print(f"\nExcluidos por coincidir con el experimento: {int(fuga.sum())} recortes")

    keep = ~fuga
    _, gid = np.unique(grupo[keep], return_inverse=True)
    nombres = [f.name for f, k in zip(fs, keep) if k]
    reidx = -np.ones(len(fs), dtype=int)
    reidx[np.where(keep)[0]] = np.arange(keep.sum())
    dud = [(int(reidx[a]), int(reidx[b])) for a, b in dudosos if keep[a] and keep[b]]
    tam = np.bincount(gid)
    print(f"Identidades: {len(tam)} a partir de {keep.sum()} recortes · grupos con ≥2: {(tam >= 2).sum()} "
          f"({tam[tam >= 2].sum()} recortes) · pares dudosos enmascarados: {len(dud)}")
    MANIFIESTO.write_text(json.dumps({"recortes": nombres, "identidad": gid.tolist(), "dudosos": dud,
                                      "regla": {"cos_seguro": 0.95, "cos_min": 0.85, "pixeles": umbral_pix,
                                                "cos_dudoso": 0.88}}, ensure_ascii=False), encoding="utf-8")
    print(f"Manifiesto: {MANIFIESTO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
