"""Evaluación FINAL del modo ensamble con la prueba de tatuajes reales (experimento_ab).

Usa el código de producción: serving.codificadores para los vectores y
serving.ensamble para calibrar y decidir. Lo que se mide es lo que corre.

Datos (los de experimento_ab --rotaciones --desconocidos-ampliados):
  85 tatuajes reales enrolados como /enroll (original + 4 degradadas + 3 giros),
  sus otras versiones como consultas, 300 distractores enrolados y ~2 200
  desconocidos de fuentes públicas.

Se compara con la producción actual: DINOv2 + cabeza v2 y umbrales fijos 0.55/0.70.

Uso:
    python -m src.embedding.evaluar_ensamble
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.embedding import experimento_ab as ab
from src.embedding.degrade import degrade
from serving import config
from serving.ensamble import calibrar, decidir, indices_identidad, puntajes_por_modelo

SALIDA = paths.REPORTS / "experimentos" / "evaluacion_final"
GIROS = (90, 180, 270)


def datos(n_desconocidos: int):
    rng = np.random.RandomState(42)
    train_h = ab.hashes_entrenamiento()
    idents = ab.identidades(3, train_h)
    distr = ab.recortes_de(ab.FUENTES_DISTRACTOR, 300, rng)
    tr = np.load(paths.REPORTS / "experimentos" / "phash_recortes_train.npy")
    gal = [i["enrol"] for i in idents] + [im for i in idents for _, im in i["consultas"]] + distr
    gal_h = np.array([ab.phash(im) for im in gal], dtype=np.uint64)
    desconocidos = ab.desconocidos_ampliados(np.concatenate([tr, gal_h]), rng, n_desconocidos)
    erng = np.random.RandomState(7)
    galeria, gal_id = [], []
    for k, im in enumerate([i["enrol"] for i in idents] + distr):
        for v in [im] + [degrade(im, n, erng) for n in ab.ENROLL_LEVELS] + [im.rotate(a, expand=True) for a in GIROS]:
            galeria.append(v)
            gal_id.append(f"id{k:04d}")
    consultas, verdad, variante = [], [], []
    for k, i in enumerate(idents):
        for v, im in i["consultas"]:
            consultas.append(im)
            verdad.append(k)
            variante.append(re.sub(r"\d+$", "", v))
    return galeria, gal_id, consultas, np.array(verdad), np.array(variante), desconocidos, len(idents)


def semaforo_fijo(cos_q, cos_d, verdad):
    top, sim = cos_q.argmax(1), cos_q.max(1)
    ver = np.where(sim >= 0.70, "match", np.where(sim >= 0.55, "review", "no_match"))
    vd = np.where(cos_d.max(1) >= 0.70, "match", np.where(cos_d.max(1) >= 0.55, "review", "no_match"))
    ok = top == verdad
    return resumen(ver, ok, vd)


def resumen(ver, ok, vd) -> dict:
    return {"reconoce": float(((ver != "no_match") & ok).mean()), "verde_ok": float(((ver == "match") & ok).mean()),
            "verde_mal": float(((ver == "match") & ~ok).mean()), "amarillo": float((ver == "review").mean()),
            "rojo": float((ver == "no_match").mean()), "far": float((vd != "no_match").mean()),
            "far_verde": float((vd == "match").mean()),
            "precision_verde": float(((ver == "match") & ok).sum() / max(1, (ver == "match").sum()))}


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluación final del modo ensamble")
    ap.add_argument("--desconocidos", type=int, default=4000)
    ap.add_argument("--reps", type=int, default=300)
    args = ap.parse_args()
    SALIDA.mkdir(parents=True, exist_ok=True)

    galeria, gal_id, consultas, verdad, variante, desconocidos, n_ident = datos(args.desconocidos)
    print(f"{n_ident} tatuajes enrolados + {len(set(gal_id)) - n_ident} distractores ({len(galeria)} vistas), "
          f"{len(consultas)} consultas, {len(desconocidos)} desconocidos", flush=True)

    from serving.codificadores import Codificadores
    cod = Codificadores(config.GENERALISTA_PATH, config.ESPECIALISTA_PATH)
    G, Q, D = cod.embed(galeria), cod.embed(consultas), cod.embed(desconocidos)
    codigos, ids = indices_identidad(gal_id)
    SA, SB = puntajes_por_modelo(Q, G, codigos, len(ids), cod.dims)
    DA, DB = puntajes_por_modelo(D, G, codigos, len(ids), cod.dims)

    from src.embedding.backbones import resolver
    base = resolver("dinov2")            # producción actual: models/embed_head.pt (head-v2)
    gb, qb, db = base.embed(galeria), base.embed(consultas), base.embed(desconocidos)
    cos_q = ab.puntajes(qb @ gb.T, codigos, len(ids))
    cos_d = ab.puntajes(db @ gb.T, codigos, len(ids))

    res = {"datos": {"tatuajes_enrolados": n_ident, "identidades_galeria": len(ids), "consultas": len(consultas),
                     "desconocidos": len(desconocidos), "huella_ensamble": cod.huella},
           "top1": {"produccion_actual": float((cos_q.argmax(1) == verdad).mean()),
                    "generalista": float((SA.argmax(1) == verdad).mean()),
                    "especialista": float((SB.argmax(1) == verdad).mean())},
           "produccion_actual_fijo": semaforo_fijo(cos_q, cos_d, verdad)}

    rng = np.random.RandomState(0)
    grupos = np.unique(verdad)
    for alfa in (0.01, 0.05):
        filas, por_var = [], {v: [] for v in np.unique(variante)}
        for _ in range(args.reps):
            prueba = set(rng.choice(grupos, len(grupos) // 2, replace=False))
            mq = np.array([g in prueba for g in verdad])
            md = rng.rand(len(D)) < 0.5
            cal = calibrar(D[md], G, gal_id, cod.dims, alfa)
            dq = [decidir(a, b, cal) for a, b in zip(SA[mq], SB[mq])]
            dd = [decidir(a, b, cal) for a, b in zip(DA[~md], DB[~md])]
            ok = np.array([int(d.orden[0]) == v for d, v in zip(dq, verdad[mq])])
            ver = np.array([d.verdict for d in dq])
            filas.append(resumen(ver, ok, np.array([d.verdict for d in dd])) | {"tau": cal.tau})
            for v in por_var:
                m = variante[mq] == v
                if m.any():
                    por_var[v].append(float(((ver[m] != "no_match") & ok[m]).mean()))
        r = {k: round(float(np.mean([f[k] for f in filas])), 4) for k in filas[0]}
        r["far_p95"] = round(float(np.quantile([f["far"] for f in filas], 0.95)), 4)
        r["reconoce_por_variante"] = {v: round(float(np.mean(x)), 3) for v, x in por_var.items()}
        res[f"ensamble_alfa={alfa}"] = r
        fused_top1 = float(np.mean([int(decidir(a, b, cal).orden[0]) == v for a, b, v in zip(SA, SB, verdad)]))
        res["top1"]["ensamble"] = fused_top1

    # ablación (α = 5 %): cada codificador solo frente al ensamble, y el ensamble sin consenso
    def semaforo_cv(GA_, QA_, DA_, GB_, QB_, DB_, alfa=0.05, reps=100, consenso=True):
        dims_ = (GA_.shape[1], GB_.shape[1])
        Gx = np.concatenate([GA_, GB_], 1)
        a_, b_ = puntajes_por_modelo(np.concatenate([QA_, QB_], 1), Gx, codigos, len(ids), dims_)
        da_, db_ = puntajes_por_modelo(np.concatenate([DA_, DB_], 1), Gx, codigos, len(ids), dims_)
        r_ = np.random.RandomState(5)
        filas_ = []
        for _ in range(reps):
            prueba_ = set(r_.choice(grupos, len(grupos) // 2, replace=False))
            mq_ = np.array([g in prueba_ for g in verdad])
            md_ = r_.rand(len(D)) < 0.5
            cal_ = calibrar(np.concatenate([DA_, DB_], 1)[md_], Gx, gal_id, dims_, alfa)
            dq_ = [decidir(x, y, cal_) for x, y in zip(a_[mq_], b_[mq_])]
            dd_ = [decidir(x, y, cal_) for x, y in zip(da_[~md_], db_[~md_])]
            def ver_(d):
                if consenso:
                    return d.verdict
                return "no_match" if d.candidatos == 0 else ("match" if d.candidatos == 1 else "review")
            ok_ = np.array([int(d.orden[0]) == v for d, v in zip(dq_, verdad[mq_])])
            filas_.append(resumen(np.array([ver_(d) for d in dq_]), ok_, np.array([ver_(d) for d in dd_])))
        return {k: round(float(np.mean([f[k] for f in filas_])), 4) for k in filas_[0]}
    da, dbb = cod.dims
    GA, GB, QA, QB, DA_, DB_ = G[:, :da], G[:, da:], Q[:, :da], Q[:, da:], D[:, :da], D[:, da:]
    res["ablacion_alfa5"] = {
        "solo_generalista": semaforo_cv(GA, QA, DA_, GA, QA, DA_),
        "solo_especialista": semaforo_cv(GB, QB, DB_, GB, QB, DB_),
        "ensamble_sin_consenso": semaforo_cv(GA, QA, DA_, GB, QB, DB_, consenso=False),
        "ensamble_con_consenso": semaforo_cv(GA, QA, DA_, GB, QB, DB_),
    }
    print("ablación", json.dumps(res["ablacion_alfa5"], ensure_ascii=False), flush=True)

    # tamaño de galería: la línea fija se degrada, la conformal no
    res["tamano_galeria"] = {}
    n_extra = len(ids) - n_ident
    for extra in (0, 100, 200, n_extra):
        keep = [i for i, g in enumerate(gal_id) if int(g[2:]) < n_ident + extra]
        sub_ids = [gal_id[i] for i in keep]
        c2, i2 = indices_identidad(sub_ids)
        cq = ab.puntajes(qb @ gb[keep].T, c2, len(i2)); cd = ab.puntajes(db @ gb[keep].T, c2, len(i2))
        a2, b2 = puntajes_por_modelo(D, G[keep], c2, len(i2), cod.dims)
        fila = {"fijo_far": semaforo_fijo(cq, cd, verdad)["far"]}
        for alfa in (0.01, 0.05):
            fars = []
            for rep in range(30):
                md = np.random.RandomState(100 + rep).rand(len(D)) < 0.5
                cal = calibrar(D[md], G[keep], sub_ids, cod.dims, alfa)
                fars.append(float(np.mean([decidir(x, y, cal).verdict != "no_match"
                                           for x, y in zip(a2[~md], b2[~md])])))
            fila[f"conformal_far_alfa{alfa:g}"] = round(float(np.mean(fars)), 4)
        res["tamano_galeria"][str(len(i2))] = fila

    (SALIDA / "resultados.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
