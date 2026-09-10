"""¿Reconoce el sistema un tatuaje que él mismo enroló, cuando vuelve a verlo degradado?

Es la pregunta operativa del sistema en uso real: un perito enrola un tatuaje desde
la aplicación y semanas después fotografía a otra persona con el mismo tatuaje, peor
iluminado, más gastado, en otro ángulo. ¿Lo encuentra?

El experimento reproduce ese ciclo exacto:

  1. Se toman N recortes reales que NO están en la galería base.
  2. Cada uno se enrola **con el mismo procedimiento que usa el servicio**:
     la vista original más las degradaciones de DEGRADE_LEVELS.
  3. Se consulta con versiones degradadas del mismo tatuaje, generadas con un
     **flujo de azar distinto** al del enrolamiento. Esto es lo que hace honesta la
     medición: con la misma semilla, la consulta reproduciría una vista ya enrolada
     y la similitud saldría artificialmente alta.
  4. Se consulta además con recortes de otras identidades (impostores), que deben
     rechazarse.
  5. Se barre umbral × margen y se reporta la operación que maximiza F1.

La galería base (los tatuajes de banda) se incluye como distractor, para que la
búsqueda no sea artificialmente fácil.

Uso:
    python -m src.embedding.enroll_recall_test
    python -m src.embedding.enroll_recall_test --identidades 300 --head models/embed_head.pt
    python -m src.embedding.enroll_recall_test --json reports/enroll_recall.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared import paths
from src.shared.seeds import set_seed, SEED
from src.embedding.embedder import TattooEmbedder
from src.embedding.degrade import degrade
from src.embedding.score_rules import COHORTE, auc, barrer, puntajes, ranking_por_identidad

# Mismos niveles que usa serving/pipeline.py al enrolar.
ENROLL_LEVELS = [0.2, 0.4, 0.6, 0.8]

# Niveles con los que se consulta después. 0.0 mide el caso trivial (misma foto);
# los demás miden desgaste creciente.
PROBE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]

# Tasa de falsos positivos admitida al elegir el punto de operación. En una
# herramienta de apoyo policial, señalar a quien no corresponde cuesta mucho más
# que no señalar a quien sí: el falso negativo solo obliga a revisar a mano.
FPR_OBJETIVO = 0.05

REAL_CROPS_TEST = paths.DATASETS / "real_crops" / "test"
GALLERY_NPZ = paths.MODELS / "gallery_dinov2.npz"


def vistas_de_enrolamiento(img: Image.Image, rng: np.random.RandomState) -> list[Image.Image]:
    """Idéntico a TattooService.enroll: original + una vista por nivel."""
    return [img] + [degrade(img, nivel, rng) for nivel in ENROLL_LEVELS]


def consulta_degradada(img: Image.Image, nivel: float, rng: np.random.RandomState) -> Image.Image:
    """Consulta realista: degradación más un cambio de encuadre y exposición.

    Una foto tomada otro día no solo está más gastada: está tomada desde otro
    ángulo y con otra luz. Sin esto, el experimento mediría solo la invarianza a
    la degradación sintética, que es la mitad del problema.
    """
    salida = img
    if nivel > 0:
        salida = degrade(salida, nivel, rng)

    salida = salida.rotate(rng.uniform(-12, 12), expand=False, resample=Image.BICUBIC,
                           fillcolor=(190, 150, 120))
    salida = ImageEnhance.Brightness(salida).enhance(rng.uniform(0.85, 1.15))
    salida = ImageEnhance.Contrast(salida).enhance(rng.uniform(0.9, 1.1))
    return salida


def cargar_galeria_base() -> tuple[np.ndarray, list[str]]:
    if not GALLERY_NPZ.exists():
        return np.zeros((0, 0), dtype="float32"), []
    data = np.load(str(GALLERY_NPZ), allow_pickle=True)
    return data["embeddings"].astype("float32"), [str(c) for c in data["catalog_ids"]]


def main() -> int:
    ap = argparse.ArgumentParser(description="Recuperación de un tatuaje enrolado y luego degradado")
    ap.add_argument("--identidades", type=int, default=200, help="cuántos tatuajes enrolar")
    ap.add_argument("--impostores", type=int, default=200, help="cuántas consultas de tatuajes no enrolados")
    ap.add_argument("--head", default=None, help="cabeza afinada; por defecto la de models/")
    ap.add_argument("--sin-head", action="store_true", help="medir el backbone desnudo, para comparar")
    ap.add_argument("--sin-galeria-base", action="store_true",
                    help="no usar la galeria de produccion como distractor. Obligatorio al "
                         "medir una cabeza distinta de la que la construyo: sus vectores "
                         "viven en otro espacio y serian ruido, no distractores")
    ap.add_argument("--json", default=None, help="ruta donde volcar el resultado")
    args = ap.parse_args()

    set_seed()

    # Los recortes se agrupan por fotografia de origen ("by_tatoo__006406_b0",
    # "..._b1" salen de la misma imagen). Si se enrolara un recorte y se usara otro
    # de la misma fotografia como impostor, compartirian piel, luz y fondo: se
    # contarian como falsos positivos cosas que en realidad son el mismo caso.
    todos = sorted(REAL_CROPS_TEST.rglob("*.png"))
    por_origen: dict[str, list[Path]] = {}
    for ruta in todos:
        origen = re.sub(r"_b\d+$", "", ruta.stem)
        por_origen.setdefault(origen, []).append(ruta)

    origenes = sorted(por_origen)
    if len(origenes) < args.identidades + args.impostores:
        print(f"ERROR: hacen falta {args.identidades + args.impostores} fotografias "
              f"distintas y hay {len(origenes)}.")
        return 1

    rng_seleccion = np.random.RandomState(SEED)
    rng_seleccion.shuffle(origenes)
    # Un solo recorte por origen: los demas se descartan.
    a_enrolar = [por_origen[o][0] for o in origenes[: args.identidades]]
    impostores = [por_origen[o][0]
                  for o in origenes[args.identidades : args.identidades + args.impostores]]
    print(f"{len(todos)} recortes en {len(origenes)} fotografias de origen; "
          f"{len(a_enrolar)} a enrolar, {len(impostores)} impostores")

    head = None if args.sin_head else (args.head or str(paths.MODELS / "embed_head.pt"))
    if head and not Path(head).exists():
        head = None
    emb = TattooEmbedder(head_path=head)
    print(f"Embedder: dim={emb.dim}, cabeza={'sí' if emb.head is not None else 'no'}, device={emb.device}")

    # --- 1. Enrolar -----------------------------------------------------------
    # Flujo de azar propio del enrolamiento.
    rng_enrol = np.random.RandomState(SEED + 1)
    imagenes, ids = [], []
    originales = []

    for i, ruta in enumerate(a_enrolar):
        try:
            base = Image.open(ruta).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        originales.append((f"id_{i:04d}", base))
        for vista in vistas_de_enrolamiento(base, rng_enrol):
            imagenes.append(vista)
            ids.append(f"id_{i:04d}")

    print(f"Enrolando {len(originales)} identidades ({len(imagenes)} vistas)…")
    vectores = emb.embed(imagenes)

    base_vec, base_ids = (np.zeros((0, 0), dtype="float32"), []) if args.sin_galeria_base else cargar_galeria_base()
    if base_vec.shape[0] and base_vec.shape[1] == vectores.shape[1]:
        galeria = np.concatenate([vectores, base_vec], axis=0)
        galeria_ids = ids + base_ids
        print(f"  + galería base como distractor: {base_vec.shape[0]} vistas, "
              f"{len(set(base_ids))} identidades")
    else:
        galeria = vectores
        galeria_ids = ids
        if base_vec.shape[0]:
            print("  (galería base descartada: su dimensión no coincide con la cabeza actual)")

    # --- 2. Consultas genuinas ------------------------------------------------
    # Flujo de azar DISTINTO al del enrolamiento: si fuera el mismo, la consulta
    # reproduciría una vista ya enrolada y la similitud saldría inflada.
    rng_consulta = np.random.RandomState(SEED + 999)
    genuinas: list[dict] = []

    for nivel in PROBE_LEVELS:
        lote = [consulta_degradada(img, nivel, rng_consulta) for _, img in originales]
        vecs = emb.embed(lote)
        for (ident, _), v in zip(originales, vecs):
            p = puntajes(ranking_por_identidad(galeria, galeria_ids, v))
            p["nivel"] = nivel
            p["acierto"] = p["id"] == ident
            genuinas.append(p)

    # --- 3. Consultas impostoras ---------------------------------------------
    impostoras: list[dict] = []
    lote_imp = []
    for ruta in impostores:
        try:
            lote_imp.append(Image.open(ruta).convert("RGB"))
        except Exception:  # noqa: BLE001
            continue
    if lote_imp:
        vecs = emb.embed(lote_imp)
        for v in vecs:
            impostoras.append(puntajes(ranking_por_identidad(galeria, galeria_ids, v)))

    # --- 4. Resultados por nivel ---------------------------------------------
    print()
    print("Recuperación por nivel de degradación de la consulta")
    print("nivel  top1-acierto   similitud media   margen medio    z-score")
    por_nivel = {}
    for nivel in PROBE_LEVELS:
        filas = [f for f in genuinas if f["nivel"] == nivel]
        acierto = sum(1 for f in filas if f["acierto"]) / len(filas)
        sim = float(np.mean([f["top1"] for f in filas]))
        mar = float(np.mean([f["margen"] for f in filas]))
        z = float(np.mean([f["z"] for f in filas]))
        por_nivel[nivel] = {"top1": round(acierto, 4), "similitud_media": round(sim, 4),
                            "margen_medio": round(mar, 4), "z_medio": round(z, 4)}
        print(f"{nivel:5.1f}  {acierto*100:11.1f} %  {sim:15.4f}  {mar:13.4f}  {z:9.2f}")

    if impostoras:
        sim_imp = [x["top1"] for x in impostoras]
        z_imp = [x["z"] for x in impostoras]
        print()
        print(f"Impostores: coseno media {np.mean(sim_imp):.4f}, p95 {np.quantile(sim_imp, 0.95):.4f}, "
              f"máx {max(sim_imp):.4f}")
        print(f"            z-score media {np.mean(z_imp):.2f}, p95 {np.quantile(z_imp, 0.95):.2f}, "
              f"máx {max(z_imp):.2f}")

    # --- 5. Comparación de reglas de decisión --------------------------------
    # Tres formas de decidir sobre el mismo ranking. El AUC las compara sin
    # depender del umbral que se elija para cada una.
    print()
    print(f"Regla de decisión            AUC     operación al {FPR_OBJETIVO*100:.0f} % de falsos positivos")
    reglas = {
        "coseno del top-1": ("top1", np.arange(0.30, 0.99, 0.005)),
        "margen top1-top2": ("margen", np.arange(0.0, 0.60, 0.005)),
        f"z-score (cohorte {COHORTE})": ("z", np.arange(0.0, 12.0, 0.05)),
    }

    comparacion = {}
    for etiqueta, (campo, valores) in reglas.items():
        area = auc(genuinas, impostoras, campo)
        punto = barrer(genuinas, impostoras, campo, valores, fpr_objetivo=FPR_OBJETIVO)
        comparacion[campo] = {"auc": round(area, 4), "al_5_por_ciento_fp": punto}
        print(f"{etiqueta:28s} {area:.4f}   umbral {punto['umbral']:>6}  "
              f"recall {punto['recall']*100:5.1f} %  FP {punto['fp']:3d} ({punto['fpr']*100:.1f} %)")

    # Regla combinada: coseno y margen a la vez, que es la forma del sistema actual.
    # Se busca en la rejilla el punto que más recall da sin pasar de la tasa de
    # falsos positivos admitida.
    mejor_combinada = None
    for thr in np.arange(0.50, 0.96, 0.01):
        for delta in (0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20):
            tp = sum(1 for g in genuinas if g["top1"] >= thr and g["margen"] >= delta and g["acierto"])
            fp = sum(1 for i in impostoras if i["top1"] >= thr and i["margen"] >= delta)
            fpr = fp / len(impostoras) if impostoras else 0.0
            if fpr > FPR_OBJETIVO:
                continue
            recall = tp / len(genuinas)
            if mejor_combinada is None or recall > mejor_combinada["recall"]:
                mejor_combinada = {"threshold": round(float(thr), 3), "margin": round(float(delta), 3),
                                   "recall": round(recall, 4), "fp": fp, "fpr": round(fpr, 4)}

    if mejor_combinada:
        print(f"{'coseno + margen':28s} {'—':>6}   ρ {mejor_combinada['threshold']:>5} "
              f"δ {mejor_combinada['margin']:>5}  recall {mejor_combinada['recall']*100:5.1f} %  "
              f"FP {mejor_combinada['fp']:3d} ({mejor_combinada['fpr']*100:.1f} %)")

        # El recall promediado esconde lo que importa: a poco desgaste el sistema
        # acierta casi siempre y a desgaste extremo casi nunca. Un solo número
        # mezcla ambas cosas y no sirve para decidir si la herramienta es usable.
        print()
        print(f"Recall por nivel de desgaste en ese punto de operación (ρ={mejor_combinada['threshold']}, "
              f"δ={mejor_combinada['margin']})")
        recall_por_nivel = {}
        for nivel in PROBE_LEVELS:
            filas = [g for g in genuinas if g["nivel"] == nivel]
            aceptadas = sum(1 for g in filas
                            if g["top1"] >= mejor_combinada["threshold"]
                            and g["margen"] >= mejor_combinada["margin"] and g["acierto"])
            r = aceptadas / len(filas)
            recall_por_nivel[str(nivel)] = round(r, 4)
            print(f"  nivel {nivel:.1f}: {r*100:5.1f} %")
        mejor_combinada["recall_por_nivel"] = recall_por_nivel

    # Tabla completa: es lo que permite elegir el punto de operación con criterio
    # en vez de aceptar el que salga de maximizar una métrica agregada.
    print()
    print("Umbral   FP     FPR    recall por nivel de desgaste (0.0 / 0.2 / 0.4 / 0.6 / 0.8)")
    tabla = []
    for thr in [0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90]:
        fp = sum(1 for i in impostoras if i["top1"] >= thr)
        fpr = fp / len(impostoras) if impostoras else 0.0
        recalls = []
        for nivel in PROBE_LEVELS:
            filas = [g for g in genuinas if g["nivel"] == nivel]
            recalls.append(sum(1 for g in filas if g["top1"] >= thr and g["acierto"]) / len(filas))
        tabla.append({"umbral": thr, "fp": fp, "fpr": round(fpr, 4),
                      "recall_por_nivel": [round(r, 4) for r in recalls]})
        detalle = "  ".join(f"{r*100:5.1f}" for r in recalls)
        print(f"  {thr:.2f}  {fp:4d}  {fpr*100:5.1f} %   {detalle}")

    # Operación actual, para comparar contra lo que hay desplegado.
    thr_actual, mar_actual = 0.55, 0.05
    tp = sum(1 for g in genuinas if g["top1"] >= thr_actual and g["margen"] >= mar_actual and g["acierto"])
    fp = sum(1 for i in impostoras if i["top1"] >= thr_actual and i["margen"] >= mar_actual)
    prec_a = tp / (tp + fp) if (tp + fp) else 0.0
    rec_a = tp / len(genuinas)
    actual = {"threshold": thr_actual, "margin": mar_actual,
              "precision": round(prec_a, 4), "recall": round(rec_a, 4),
              "fp": fp, "fpr": round(fp / len(impostoras), 4) if impostoras else None}

    print()
    print(f"Operación desplegada (ρ={thr_actual}, δ={mar_actual}): "
          f"recall {actual['recall']*100:.1f} %, falsos positivos {fp} de {len(impostoras)} "
          f"({actual['fpr']*100:.1f} %)")

    resultado = {
        "identidades_enroladas": len(originales),
        "impostores": len(impostoras),
        "cabeza": emb.head is not None,
        "dim": int(emb.dim),
        "por_nivel": {str(k): v for k, v in por_nivel.items()},
        "impostores_similitud": {
            "media": round(float(np.mean([x["top1"] for x in impostoras])), 4) if impostoras else None,
            "p95": round(float(np.quantile([x["top1"] for x in impostoras], 0.95)), 4) if impostoras else None,
            "max": round(float(max(x["top1"] for x in impostoras)), 4) if impostoras else None,
        },
        "operacion_desplegada": actual,
        "reglas": comparacion,
        "operacion_recomendada": mejor_combinada,
        "tabla_umbrales": tabla,
    }

    if args.json:
        destino = Path(args.json)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nEscrito: {destino}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
