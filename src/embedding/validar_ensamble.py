"""Elige la pareja de codificadores con el conjunto de VALIDACIÓN (no con la prueba final).

Protocolo, idéntico al servicio:
  * cada grupo de validación enrola su primera foto con el mismo procedimiento que
    /enroll (original + 4 degradadas + 3 giros) y se consulta con las demás;
  * los desconocidos de validación nunca se enrolan;
  * 200 particiones: la mitad de los desconocidos calibra (serving.ensamble.calibrar)
    y la otra mitad, junto con la mitad de los grupos, mide;
  * la decisión es serving.ensamble.decidir: el mismo código que producción.

Uso:
    python -m src.embedding.validar_ensamble --generalistas models/candidatos/head_*.pt \\
        --especialistas models/candidatos/cnn_*.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.embedding.degrade import degrade
from serving.ensamble import calibrar, decidir, indices_identidad, puntajes_por_modelo

VAL = paths.DATASETS / "real_crops" / "validacion.json"
CROPS = paths.DATASETS / "real_crops" / "train"
SALIDA = paths.REPORTS / "experimentos" / "candidatos"
NIVELES = [0.2, 0.4, 0.6, 0.8]
GIROS = (90, 180, 270)


def datos():
    v = json.loads(VAL.read_text(encoding="utf-8"))
    rng = np.random.RandomState(11)
    galeria, gal_id, consultas, verdad = [], [], [], []
    for k, grupo in enumerate(v["grupos"]):
        base = Image.open(CROPS / grupo[0]).convert("RGB")
        for vista in [base] + [degrade(base, n, rng) for n in NIVELES] + [base.rotate(a, expand=True) for a in GIROS]:
            galeria.append(vista)
            gal_id.append(f"g{k:03d}")
        for nombre in grupo[1:]:
            consultas.append(Image.open(CROPS / nombre).convert("RGB"))
            verdad.append(k)
    desconocidos = [Image.open(CROPS / n).convert("RGB") for n in v["desconocidos"]]
    return galeria, gal_id, consultas, np.array(verdad), desconocidos


def embeber(spec: str, imgs_por_parte: dict) -> dict:
    from src.embedding.backbones import resolver
    emb = resolver(spec)
    out = {k: emb.embed(v) for k, v in imgs_por_parte.items()}
    del emb
    import torch
    torch.cuda.empty_cache()
    return out


def evaluar(EA: dict, EB: dict, gal_id, verdad, alfas=(0.01, 0.05), reps=200) -> dict:
    dims = (EA["galeria"].shape[1], EB["galeria"].shape[1])
    G = np.concatenate([EA["galeria"], EB["galeria"]], 1)
    Q = np.concatenate([EA["consultas"], EB["consultas"]], 1)
    D = np.concatenate([EA["desconocidos"], EB["desconocidos"]], 1)
    codigos, ids = indices_identidad(gal_id)
    SA, SB = puntajes_por_modelo(Q, G, codigos, len(ids), dims)
    DA, DB = puntajes_por_modelo(D, G, codigos, len(ids), dims)
    top1 = float(((SA + SB).argmax(1) == verdad).mean())
    res = {"top1_suma_coseno": top1}
    rng = np.random.RandomState(0)
    grupos = np.unique(verdad)
    for alfa in alfas:
        filas = []
        for _ in range(reps):
            prueba_g = set(rng.choice(grupos, len(grupos) // 2, replace=False))
            mq = np.array([g in prueba_g for g in verdad])
            md = rng.rand(len(D)) < 0.5
            cal = calibrar(D[md], G, gal_id, dims, alfa)
            dq = [decidir(a, b, cal) for a, b in zip(SA[mq], SB[mq])]
            dd = [decidir(a, b, cal) for a, b in zip(DA[~md], DB[~md])]
            ok = np.array([int(d.orden[0]) == v for d, v in zip(dq, verdad[mq])])
            ver = np.array([d.verdict for d in dq])
            vd = np.array([d.verdict for d in dd])
            filas.append({"reconoce": float(((ver != "no_match") & ok).mean()),
                          "verde_ok": float(((ver == "match") & ok).mean()),
                          "verde_mal": float(((ver == "match") & ~ok).mean()),
                          "amarillo": float((ver == "review").mean()),
                          "rojo": float((ver == "no_match").mean()),
                          "far": float((vd != "no_match").mean()),
                          "far_verde": float((vd == "match").mean())})
        res[f"alfa={alfa}"] = {k: round(float(np.mean([f[k] for f in filas])), 4) for k in filas[0]}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Validación de parejas de codificadores")
    ap.add_argument("--generalistas", nargs="+", required=True)
    ap.add_argument("--especialistas", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    SALIDA.mkdir(parents=True, exist_ok=True)

    galeria, gal_id, consultas, verdad, desconocidos = datos()
    print(f"Validación: {len(set(gal_id))} identidades enroladas ({len(galeria)} vistas), "
          f"{len(consultas)} consultas, {len(desconocidos)} desconocidos", flush=True)
    partes = {"galeria": galeria, "consultas": consultas, "desconocidos": desconocidos}
    emb = {}
    for ruta in args.generalistas:
        emb[ruta] = embeber(f"dinov2:head={ruta}", partes)
    for ruta in args.especialistas:
        emb[ruta] = embeber(f"cnn:{ruta}", partes)

    resultados = {}
    for g in args.generalistas:
        for e in args.especialistas:
            clave = f"{Path(g).stem} + {Path(e).stem}"
            resultados[clave] = r = evaluar(emb[g], emb[e], gal_id, verdad, reps=args.reps)
            a1, a5 = r["alfa=0.01"], r["alfa=0.05"]
            print(f"{clave:40s} top1={r['top1_suma_coseno']:.3f} | α1%: reconoce={a1['reconoce']:.3f} "
                  f"verde={a1['verde_ok']:.3f} mal={a1['verde_mal']:.3f} far={a1['far']:.4f} | α5%: "
                  f"reconoce={a5['reconoce']:.3f} verde={a5['verde_ok']:.3f} mal={a5['verde_mal']:.3f} "
                  f"far={a5['far']:.4f}", flush=True)
    # Criterio fijado antes de mirar: reconocimiento medio en α = 1 % y 5 %.
    mejor = max(resultados, key=lambda k: (resultados[k]["alfa=0.01"]["reconoce"]
                                           + resultados[k]["alfa=0.05"]["reconoce"]) / 2)
    print(f"\nMejor pareja (reconocimiento medio α 1 % y 5 %): {mejor}")
    (SALIDA / "validacion.json").write_text(json.dumps({"resultados": resultados, "mejor": mejor},
                                                       indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
