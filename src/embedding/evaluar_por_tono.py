"""Reconocimiento por tono de piel (auditoría de sesgo del TB2 / OE4).

Requisito: clasificar a mano las fotos de datasets/tono_piel/sin_clasificar moviéndolas a
claro / medio / oscuro (escala Monk 1–3 / 4–7 / 8–10) o no_se_ve. El nombre del archivo es el
código del tatuaje enrolado en la prueba final (evaluar_ensamble), de modo que todas sus
versiones de consulta heredan su tono.

Usa el mismo código de decisión que producción (serving.ensamble) y promedia sobre
particiones de los desconocidos en calibración y prueba. Reporta, por grupo, el
reconocimiento (identidad correcta y sobre la línea) con su intervalo de Wilson al 95 %, y la
brecha entre el mejor y el peor grupo.

Uso:
    python -m src.embedding.evaluar_por_tono
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.embedding import experimento_ab as ab
from src.embedding.evaluar_ensamble import datos
from serving import config
from serving.ensamble import calibrar, decidir, indices_identidad, puntajes_por_modelo

CARPETA = paths.DATASETS / "tono_piel"
GRUPOS = ("claro", "medio", "oscuro")
SALIDA = paths.REPORTS / "experimentos" / "evaluacion_final" / "por_tono.json"


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - m), min(1.0, c + m))


def main() -> int:
    tono = {}
    for g in GRUPOS + ("no_se_ve",):
        for f in (CARPETA / g).glob("*.jpg"):
            tono[f.stem] = g
    faltan = len(list((CARPETA / "sin_clasificar").glob("*.jpg")))
    if faltan:
        print(f"Aviso: quedan {faltan} fotos sin clasificar; se excluyen.")

    galeria, gal_id, consultas, verdad, _, desconocidos, n_ident = datos(4000)
    ids = [i["id"] for i in ab.identidades(3, ab.hashes_entrenamiento())]
    assert len(ids) == n_ident
    grupo_q = np.array([tono.get(ids[v], "sin_clasificar") for v in verdad])

    from serving.codificadores import Codificadores
    cod = Codificadores(config.GENERALISTA_PATH, config.ESPECIALISTA_PATH)
    G, Q, D = cod.embed(galeria), cod.embed(consultas), cod.embed(desconocidos)
    codigos, lista = indices_identidad(gal_id)
    SA, SB = puntajes_por_modelo(Q, G, codigos, len(lista), cod.dims)

    rng = np.random.RandomState(0)
    acum = np.zeros(len(verdad))
    reps = 100
    for _ in range(reps):
        md = rng.rand(len(D)) < 0.5
        cal = calibrar(D[md], G, gal_id, cod.dims, config.ALFA)
        for i, (a, b, v) in enumerate(zip(SA, SB, verdad)):
            d = decidir(a, b, cal)
            acum[i] += float(d.verdict != "no_match" and int(d.orden[0]) == v)
    reconoce = acum / reps

    res = {"alfa": config.ALFA, "grupos": {}}
    for g in GRUPOS:
        m = grupo_q == g
        n = int(m.sum())
        k = float(reconoce[m].sum())
        lo, hi = wilson(k, n)
        res["grupos"][g] = {"tatuajes": len({ids[v] for v in verdad[m]}), "consultas": n,
                            "reconoce": round(k / n, 4) if n else None, "ic95": [round(lo, 4), round(hi, 4)]}
        print(f"{g:7s} tatuajes={res['grupos'][g]['tatuajes']:3d} consultas={n:4d} "
              f"reconoce={(k / n if n else float('nan')):.3f} IC95=[{lo:.3f}, {hi:.3f}]")
    validos = [v["reconoce"] for v in res["grupos"].values() if v["reconoce"] is not None]
    if len(validos) >= 2:
        res["brecha_pp"] = round(100 * (max(validos) - min(validos)), 1)
        print(f"Brecha entre el mejor y el peor grupo: {res['brecha_pp']} pp")
    SALIDA.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
