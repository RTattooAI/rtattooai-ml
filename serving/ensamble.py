"""Ensamble de codificadores con verificación por consenso y semáforo conformal.

Es la lógica de decisión del servicio, separada de los modelos para poder probarla
sin GPU y para que la evaluación (src.embedding.evaluar_ensamble) use exactamente
el mismo código que producción.

Vectores. Cada vista de la galería guarda los dos vectores concatenados:
``[generalista | especialista]``, cada bloque L2-normalizado por su cuenta. Así la
galería sigue siendo una sola matriz y GalleryStore no necesita saber que hay dos
modelos detrás.

Puntaje de una identidad (por modelo): el coseno máximo entre sus vistas.

Normalización. Cada modelo tiene su propia escala de cosenos. Se lleva a la misma
escala con la media y la desviación de sus puntajes contra tatuajes DESCONOCIDOS
(el conjunto de calibración), y se promedian los dos con el mismo peso. No hay
pesos elegidos a mano: se probó elegirlos y el «óptimo» saltaba de 0.35 a 0.95
sin mejorar nada (informe de reentrenamiento, sección 3.1).

Semáforo conformal. La línea τ es el cuantil ⌈(n+1)(1−α)⌉ del mejor puntaje de
cada desconocido de calibración. Garantía: la probabilidad de que un desconocido
supere τ es como máximo α. La línea se recalcula cada vez que cambia la galería,
porque con más identidades hay más ocasiones de parecerse a algo por azar.

Veredicto:
  * ningún candidato ≥ τ                                   → no_match (rojo)
  * un candidato ≥ τ y los dos modelos eligen ese mismo    → match    (verde)
  * en cualquier otro caso                                 → review   (amarillo)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EPS = 1e-6


# ----------------------------------------------------------------- puntajes
def indices_identidad(catalog_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Código entero por vista y la lista de identidades en orden de aparición."""
    orden: dict[str, int] = {}
    codigos = np.array([orden.setdefault(c, len(orden)) for c in catalog_ids], dtype=np.int64)
    return codigos, list(orden)


def maximo_por_identidad(S: np.ndarray, codigos: np.ndarray, n_id: int) -> np.ndarray:
    """(n, vistas) → (n, identidades): el máximo de cada identidad entre sus vistas."""
    if S.shape[1] == 0:
        return np.zeros((S.shape[0], 0), dtype="float32")
    orden = np.argsort(codigos, kind="stable")
    cortes = np.flatnonzero(np.r_[True, np.diff(codigos[orden]) != 0])
    agrupado = np.maximum.reduceat(S[:, orden], cortes, axis=1)
    out = np.full((S.shape[0], n_id), -1.0, dtype="float32")
    out[:, codigos[orden][cortes]] = agrupado
    return out


def puntajes_por_modelo(Q: np.ndarray, G: np.ndarray, codigos: np.ndarray, n_id: int,
                        dims: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Similitud por identidad de cada consulta, por separado para cada modelo."""
    Q = np.atleast_2d(Q)
    d = dims[0]
    sa = maximo_por_identidad(Q[:, :d] @ G[:, :d].T, codigos, n_id)
    sb = maximo_por_identidad(Q[:, d:] @ G[:, d:].T, codigos, n_id)
    return sa, sb


def cuantil_conformal(x: np.ndarray, nivel: float) -> float:
    """Cuantil de muestra finita: el valor de orden ⌈(n+1)·nivel⌉."""
    n = len(x)
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * nivel)
    if k > n:
        return math.inf          # con tan pocos datos no se puede garantizar ese nivel
    return float(np.sort(x)[k - 1])


# ----------------------------------------------------------------- calibración
@dataclass(frozen=True)
class Calibracion:
    media_a: float
    desv_a: float
    media_b: float
    desv_b: float
    tau: float
    alfa: float
    n_desconocidos: int
    n_identidades: int

    def fusionar(self, sa: np.ndarray, sb: np.ndarray) -> np.ndarray:
        return ((sa - self.media_a) / self.desv_a + (sb - self.media_b) / self.desv_b) / 2


def calibrar(desconocidos: np.ndarray, G: np.ndarray, catalog_ids: list[str],
             dims: tuple[int, int], alfa: float) -> Calibracion | None:
    """Normalización y línea τ a partir de los desconocidos y la galería ACTUAL."""
    codigos, ids = indices_identidad(catalog_ids)
    if not ids or len(desconocidos) == 0:
        return None
    sa, sb = puntajes_por_modelo(desconocidos, G, codigos, len(ids), dims)
    ma, mb = sa.max(1), sb.max(1)
    cal = Calibracion(float(ma.mean()), float(ma.std()) + EPS, float(mb.mean()), float(mb.std()) + EPS,
                      tau=0.0, alfa=alfa, n_desconocidos=len(desconocidos), n_identidades=len(ids))
    tau = cuantil_conformal(cal.fusionar(sa, sb).max(1), 1 - alfa)
    return Calibracion(**{**cal.__dict__, "tau": tau})


# ----------------------------------------------------------------- escala que se muestra
# La pantalla (frontend) pinta el verde solo con similitud ≥ 0.70 y etiqueta cada
# candidato con los cortes 0.70 / 0.55. Para que lo que se ve diga lo mismo que lo
# que el modelo decide, el porcentaje que se muestra es un puntaje CALIBRADO:
#   0.70 = exactamente la línea conformal τ (pasarla es ser candidato),
#   0.55 = medio punto normalizado por debajo de la línea,
# y crece de forma monótona con el puntaje fusionado. El coseno real viaja aparte.
MOSTRADO_EN_TAU = 0.70
MOSTRADO_REVISION = 0.55
DISTANCIA_REVISION = 0.5


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


PENDIENTE = (_logit(MOSTRADO_EN_TAU) - _logit(MOSTRADO_REVISION)) / DISTANCIA_REVISION


def similitud_mostrada(fusionado: np.ndarray, tau: float) -> np.ndarray:
    if not np.isfinite(tau):
        return np.zeros_like(fusionado, dtype="float32")
    z = PENDIENTE * (fusionado - tau) + _logit(MOSTRADO_EN_TAU)
    return (1 / (1 + np.exp(-z))).astype("float32")


# ----------------------------------------------------------------- decisión
@dataclass(frozen=True)
class Decision:
    verdict: str                      # match | review | no_match
    consenso: bool                    # los dos modelos eligen el mismo primer candidato
    candidatos: int                   # identidades sobre la línea τ
    orden: np.ndarray                 # identidades ordenadas por puntaje fusionado
    fusionado: np.ndarray             # puntaje fusionado por identidad (con él se decide)
    similitud: np.ndarray             # puntaje calibrado para mostrar (0.70 = línea τ)
    coseno: np.ndarray                # coseno medio de los dos modelos


def decidir(sa: np.ndarray, sb: np.ndarray, cal: Calibracion) -> Decision:
    """Veredicto para UNA consulta, dados sus puntajes por identidad."""
    f = cal.fusionar(sa, sb)
    orden = np.argsort(-f, kind="stable")
    candidatos = int((f >= cal.tau).sum())
    consenso = bool(len(f) and int(np.argmax(sa)) == int(np.argmax(sb)) == int(orden[0]))
    if candidatos == 0:
        verdict = "no_match"
    elif candidatos == 1 and consenso:
        verdict = "match"
    else:
        verdict = "review"
    return Decision(verdict, consenso, candidatos, orden, f, similitud_mostrada(f, cal.tau), (sa + sb) / 2)


def decidir_lote(SA: np.ndarray, SB: np.ndarray, cal: Calibracion) -> list[Decision]:
    return [decidir(a, b, cal) for a, b in zip(SA, SB)]
