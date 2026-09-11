"""Distorsiones que NINGÚN modelo vio al entrenar.

Por qué hacen falta. Todos los codificadores se entrenan con `augment_rgb` +
`degrade()`, y `enroll_recall_test` consulta con `degrade()` también. Si un modelo
con millones de parámetros aprende a deshacer **nuestro simulador** en vez de
volverse robusto de verdad, la medición no lo nota: premia justo eso.

La prueba honesta es consultar con alteraciones de otra naturaleza. Estas se
eligieron porque son las que produce un teléfono fotografiando a una persona y
porque ninguna aparece en el entrenamiento:

    compresión JPEG fuerte    bloques 8×8 (el entrenamiento no comprime)
    perspectiva               ángulo fuera del plano (el entrenamiento solo rota en el plano)
    desenfoque de movimiento  direccional (el entrenamiento solo usa gaussiano, isótropo)
    baja resolución           tomado de lejos, reducir y volver a ampliar
    dominante de color        balance de blancos: luz de sodio, fluorescente
    ruido de sensor           poca luz

En cada consulta se aplican tres de las seis, elegidas al azar, con intensidad
proporcional al nivel. Nivel 0 deja la imagen intacta, igual que la consulta
sintética, como control.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def _coeficientes_perspectiva(origen, destino):
    """Coeficientes de Image.transform(PERSPECTIVE) que llevan destino → origen."""
    filas = []
    for (x, y), (u, v) in zip(destino, origen):
        filas.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        filas.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = np.array(filas, dtype="float64")
    b = np.array(origen, dtype="float64").reshape(8)
    return np.linalg.solve(a, b).tolist()


def perspectiva(img: Image.Image, fuerza: float, rng: np.random.RandomState) -> Image.Image:
    ancho, alto = img.size
    d = 0.22 * fuerza
    esquinas = [(0, 0), (ancho, 0), (ancho, alto), (0, alto)]
    movidas = [
        (x + rng.uniform(-d, d) * ancho, y + rng.uniform(-d, d) * alto) for x, y in esquinas
    ]
    coef = _coeficientes_perspectiva(esquinas, movidas)
    return img.transform(img.size, Image.PERSPECTIVE, coef, Image.BICUBIC,
                         fillcolor=(190, 150, 120))


def movimiento(img: Image.Image, fuerza: float, rng: np.random.RandomState) -> Image.Image:
    """Promedio de copias desplazadas a lo largo de una dirección."""
    largo = max(1, int(round(2 + 18 * fuerza)))
    angulo = rng.uniform(0, np.pi)
    dx, dy = np.cos(angulo), np.sin(angulo)
    arr = np.asarray(img, dtype="float32")
    acumulado = np.zeros_like(arr)
    for k in range(largo):
        t = k - (largo - 1) / 2
        acumulado += np.roll(np.roll(arr, int(round(t * dy)), axis=0), int(round(t * dx)), axis=1)
    return Image.fromarray(np.clip(acumulado / largo, 0, 255).astype("uint8"))


def jpeg(img: Image.Image, fuerza: float, rng: np.random.RandomState) -> Image.Image:
    calidad = int(round(90 - 78 * fuerza))          # 90 → 12
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=max(5, calidad))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def baja_resolucion(img: Image.Image, fuerza: float, rng: np.random.RandomState) -> Image.Image:
    ancho, alto = img.size
    escala = max(0.12, 1.0 - 0.85 * fuerza)
    chica = img.resize((max(8, int(ancho * escala)), max(8, int(alto * escala))), Image.BILINEAR)
    return chica.resize((ancho, alto), Image.BILINEAR)


def dominante(img: Image.Image, fuerza: float, rng: np.random.RandomState) -> Image.Image:
    ganancias = 1.0 + rng.uniform(-0.35, 0.35, size=3) * fuerza
    arr = np.asarray(img, dtype="float32") * ganancias[None, None, :]
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


def ruido(img: Image.Image, fuerza: float, rng: np.random.RandomState) -> Image.Image:
    arr = np.asarray(img, dtype="float32")
    arr = arr + rng.normal(0, 28 * fuerza, size=arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


DISTORSIONES = [perspectiva, movimiento, jpeg, baja_resolucion, dominante, ruido]


def consulta_ood(img: Image.Image, nivel: float, rng: np.random.RandomState) -> Image.Image:
    """Tres distorsiones fuera de distribución, con intensidad proporcional al nivel."""
    salida = img.convert("RGB")
    if nivel <= 0:
        return salida
    elegidas = rng.choice(len(DISTORSIONES), size=3, replace=False)
    for i in sorted(elegidas):
        salida = DISTORSIONES[i](salida, nivel, rng)
    return salida
