"""Reglas de decisión sobre el ranking de la galería.

El sistema comparaba con el coseno crudo del mejor candidato: `top1 >= rho` y
`top1 - top2 >= delta`. El problema de esa regla es que el valor absoluto del
coseno no es comparable entre consultas. La cabeza afinada proyecta a un espacio
donde **todo se parece bastante a todo**: en la medición sobre datos reales, los
impostores promedian 0.62 y llegan a 0.93, solapándose con los genuinos. Con un
umbral fijo, o se rechazan coincidencias buenas o se aceptan impostores.

La alternativa es una técnica estándar en biometría: **normalizar el puntaje
contra la propia cohorte de la consulta**. Si una fotografía se parece 0.90 al
mejor candidato pero también 0.88 a los veinte siguientes, ese 0.90 no significa
nada: la imagen simplemente cae en una zona densa del espacio. Si se parece 0.75
al mejor y 0.45 al resto, ese 0.75 sí es una coincidencia.

    z = (top1 - media(cohorte)) / desviación(cohorte)

donde la cohorte son los mejores candidatos **excluyendo la identidad ganadora**.
El resultado es una medida de cuánto destaca el ganador sobre su propio contexto,
que sí es comparable entre consultas y estable cuando la galería crece.
"""
from __future__ import annotations

import numpy as np

# Tamaño de la cohorte: suficientes candidatos para estimar media y desviación,
# pocos como para que sigan siendo "vecinos" de la consulta y no la galería entera.
COHORTE = 20


def ranking_por_identidad(galeria: np.ndarray, ids: list[str], consulta: np.ndarray) -> list[tuple[str, float]]:
    """Similitud máxima por identidad, de mayor a menor. Igual que GalleryStore.search."""
    sims = galeria @ consulta
    mejor: dict[str, float] = {}
    for cid, s in zip(ids, sims):
        s = float(s)
        if cid not in mejor or s > mejor[cid]:
            mejor[cid] = s
    return sorted(mejor.items(), key=lambda kv: -kv[1])


def puntajes(ranking: list[tuple[str, float]], cohorte: int = COHORTE) -> dict:
    """Las tres magnitudes con las que se puede decidir, a partir del mismo ranking."""
    if not ranking:
        return {"id": None, "top1": -1.0, "margen": 0.0, "z": 0.0}

    top1_id, top1 = ranking[0]
    resto = np.array([s for _, s in ranking[1 : 1 + cohorte]], dtype="float32")

    margen = float(top1 - ranking[1][1]) if len(ranking) > 1 else 1.0

    if resto.size >= 2:
        media = float(resto.mean())
        desv = float(resto.std())
        # Una cohorte sin dispersión (galería diminuta o idéntica) haría explotar la
        # división; en ese caso el z-score no aporta y se neutraliza.
        z = (top1 - media) / desv if desv > 1e-6 else 0.0
    else:
        z = 0.0

    return {"id": top1_id, "top1": float(top1), "margen": margen, "z": float(z)}


def barrer(genuinos: list[dict], impostores: list[dict], campo: str,
           valores: np.ndarray, fpr_objetivo: float | None = None) -> dict:
    """Recorre umbrales de una magnitud y devuelve el punto de operación elegido.

    Con `fpr_objetivo` se elige el umbral más permisivo que respeta esa tasa de
    falsos positivos. Es lo que corresponde a esta herramienta: maximizar F1 elige
    el punto que más aciertos totales da, y en un sistema de apoyo policial un
    falso positivo — señalar a alguien que no corresponde — cuesta mucho más que
    un falso negativo, que solo obliga a revisar a mano.
    """
    filas = []
    for umbral in valores:
        tp = sum(1 for g in genuinos if g[campo] >= umbral and g["acierto"])
        fn = len(genuinos) - tp
        fp = sum(1 for i in impostores if i[campo] >= umbral)
        fpr = fp / len(impostores) if impostores else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        filas.append({"umbral": round(float(umbral), 4), "f1": round(f1, 4),
                      "precision": round(prec, 4), "recall": round(rec, 4),
                      "fp": fp, "fpr": round(fpr, 4)})

    if fpr_objetivo is None:
        return max(filas, key=lambda f: f["f1"])

    admisibles = [f for f in filas if f["fpr"] <= fpr_objetivo]
    if not admisibles:
        # Ni el umbral más estricto alcanza la tasa pedida: se devuelve el mejor
        # disponible en vez de fingir que se cumplió.
        return min(filas, key=lambda f: f["fpr"])
    return max(admisibles, key=lambda f: f["recall"])


def auc(genuinos: list[dict], impostores: list[dict], campo: str) -> float:
    """Área bajo la curva ROC por el estadístico de Mann-Whitney.

    Es la forma de comparar dos reglas de decisión sin depender de qué umbral se
    elija para cada una.
    """
    g = np.array([x[campo] for x in genuinos if x["acierto"]], dtype="float64")
    i = np.array([x[campo] for x in impostores], dtype="float64")
    if g.size == 0 or i.size == 0:
        return 0.0

    todos = np.concatenate([g, i])
    orden = todos.argsort()
    rangos = np.empty_like(orden, dtype="float64")
    rangos[orden] = np.arange(1, len(todos) + 1)

    # Empates: se reparte el rango medio, o el AUC saldría sesgado.
    valores, inicio, cuenta = np.unique(todos, return_index=True, return_counts=True)
    for ini, n in zip(inicio, cuenta):
        if n > 1:
            iguales = np.flatnonzero(todos == todos[ini])
            rangos[iguales] = rangos[iguales].mean()

    suma_g = rangos[: g.size].sum()
    return float((suma_g - g.size * (g.size + 1) / 2) / (g.size * i.size))
