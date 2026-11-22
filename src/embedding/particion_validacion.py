"""Separa un conjunto de VALIDACIÓN del manifiesto de entrenamiento.

Por qué. La prueba con tatuajes reales (experimento_ab) es la prueba final: si se
usara también para elegir entre modelos, sus cifras dejarían de ser una medida
honesta. Para elegir hace falta otro conjunto, que ningún candidato haya visto.

Qué se separa del manifiesto (identidades_train.json):
  * GRUPOS: identidades con varias fotos reales (versiones del mismo tatuaje, o el
    mismo símbolo en personas distintas). De cada uno se enrola una foto y se
    consulta con las demás.
  * DESCONOCIDOS: identidades de una sola foto, que no se enrolan y el sistema
    debe rechazar.

Salida:
  datasets/real_crops/identidades_train_sinval.json  (para entrenar candidatos)
  datasets/real_crops/validacion.json                 (para elegir entre ellos)

Uso:
    python -m src.embedding.particion_validacion --grupos 150 --desconocidos 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths

MANIFIESTO = paths.DATASETS / "real_crops" / "identidades_train.json"
SALIDA_TRAIN = paths.DATASETS / "real_crops" / "identidades_train_sinval.json"
SALIDA_VAL = paths.DATASETS / "real_crops" / "validacion.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Partición de validación para elegir modelos")
    ap.add_argument("--grupos", type=int, default=150)
    ap.add_argument("--desconocidos", type=int, default=500)
    ap.add_argument("--semilla", type=int, default=2026)
    args = ap.parse_args()

    m = json.loads(MANIFIESTO.read_text(encoding="utf-8"))
    ident = np.array(m["identidad"])
    tam = np.bincount(ident)
    rng = np.random.RandomState(args.semilla)

    multi = np.where(tam >= 2)[0]
    solo = np.where(tam == 1)[0]
    # Los pares dudosos tocan identidades que podrían ser el mismo tatuaje que otra:
    # no sirven como desconocidos limpios ni como grupos de validación.
    tocadas = {ident[a] for a, b in m["dudosos"]} | {ident[b] for a, b in m["dudosos"]}
    multi = np.array([g for g in multi if g not in tocadas])
    solo = np.array([g for g in solo if g not in tocadas])
    val_grupos = set(rng.choice(multi, min(args.grupos, len(multi)), replace=False).tolist())
    val_desc = set(rng.choice(solo, min(args.desconocidos, len(solo)), replace=False).tolist())
    fuera = val_grupos | val_desc

    grupos: dict[int, list[str]] = {}
    desconocidos: list[str] = []
    keep = []
    for i, (nombre, g) in enumerate(zip(m["recortes"], m["identidad"])):
        if g in val_grupos:
            grupos.setdefault(int(g), []).append(nombre)
        elif g in val_desc:
            desconocidos.append(nombre)
        else:
            keep.append(i)

    reidx = {old: new for new, old in enumerate(keep)}
    _, nuevas = np.unique([m["identidad"][i] for i in keep], return_inverse=True)
    dudosos = [(reidx[a], reidx[b]) for a, b in m["dudosos"] if a in reidx and b in reidx]
    SALIDA_TRAIN.write_text(json.dumps({"recortes": [m["recortes"][i] for i in keep],
                                        "identidad": nuevas.tolist(), "dudosos": dudosos,
                                        "regla": m.get("regla"), "validacion_excluida": True},
                                       ensure_ascii=False), encoding="utf-8")
    SALIDA_VAL.write_text(json.dumps({"grupos": list(grupos.values()), "desconocidos": desconocidos,
                                      "semilla": args.semilla}, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"Validación: {len(grupos)} grupos ({sum(len(v) for v in grupos.values())} recortes), "
          f"{len(desconocidos)} desconocidos")
    print(f"Entrenamiento: {len(keep)} recortes, {nuevas.max() + 1} identidades")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
