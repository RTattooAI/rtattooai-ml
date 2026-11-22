"""Pruebas de la lógica del ensamble y del semáforo conformal (sin modelos ni GPU).

Ejecutar desde rtattooai-ml:
    python -m unittest serving.tests.test_ensamble
"""
from __future__ import annotations

import unittest

import numpy as np

from serving.ensamble import (Calibracion, calibrar, cuantil_conformal, decidir, indices_identidad,
                              maximo_por_identidad, puntajes_por_modelo)

DA, DB = 6, 4


def unitarios(rng, n, d):
    v = rng.normal(size=(n, d)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def concatenar(a, b):
    return np.concatenate([a, b], axis=1)


def cal_fija(tau, ma=0.0, sa=1.0, mb=0.0, sb=1.0):
    return Calibracion(ma, sa, mb, sb, tau=tau, alfa=0.01, n_desconocidos=100, n_identidades=3)


class PuntajesTests(unittest.TestCase):
    def test_maximo_por_identidad_con_vistas_desordenadas(self) -> None:
        codigos, ids = indices_identidad(["b", "a", "b", "c", "a"])
        S = np.array([[0.1, 0.9, 0.5, 0.3, 0.2]], dtype="float32")

        out = maximo_por_identidad(S, codigos, len(ids))

        self.assertEqual(ids, ["b", "a", "c"])
        np.testing.assert_allclose(out, [[0.5, 0.9, 0.3]])

    def test_cada_modelo_se_puntua_con_su_propio_bloque(self) -> None:
        rng = np.random.default_rng(0)
        ga, gb = unitarios(rng, 2, DA), unitarios(rng, 2, DB)
        G = concatenar(ga, gb)
        codigos, _ = indices_identidad(["x", "y"])

        sa, sb = puntajes_por_modelo(G[:1], G, codigos, 2, (DA, DB))

        self.assertAlmostEqual(float(sa[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(sb[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(sa[0, 1]), float(ga[0] @ ga[1]), places=5)
        self.assertAlmostEqual(float(sb[0, 1]), float(gb[0] @ gb[1]), places=5)

    def test_galeria_vacia_no_falla(self) -> None:
        out = maximo_por_identidad(np.zeros((3, 0), dtype="float32"), np.zeros(0, dtype=np.int64), 0)
        self.assertEqual(out.shape, (3, 0))


class CuantilTests(unittest.TestCase):
    def test_usa_el_orden_de_muestra_finita(self) -> None:
        x = np.arange(1, 100, dtype=float)          # n = 99
        self.assertEqual(cuantil_conformal(x, 0.95), 95.0)

    def test_con_pocos_datos_no_promete_lo_que_no_puede(self) -> None:
        # Con 50 desconocidos no se puede garantizar el 1 %: la línea queda en infinito
        # y nada se acepta, en vez de aceptar con una garantía falsa.
        self.assertEqual(cuantil_conformal(np.ones(50), 0.99), float("inf"))


class DecisionTests(unittest.TestCase):
    def test_sin_candidatos_es_rojo(self) -> None:
        d = decidir(np.array([0.1, 0.2]), np.array([0.1, 0.2]), cal_fija(tau=1.0))
        self.assertEqual(d.verdict, "no_match")

    def test_un_candidato_con_consenso_es_verde(self) -> None:
        d = decidir(np.array([0.9, 0.1]), np.array([0.8, 0.2]), cal_fija(tau=0.5))
        self.assertEqual((d.verdict, d.consenso, d.candidatos), ("match", True, 1))

    def test_sin_consenso_nunca_es_verde(self) -> None:
        # El generalista elige la identidad 0 con mucha seguridad y el especialista
        # la 1: aunque solo una pase la línea, no se afirma.
        d = decidir(np.array([2.0, 0.0]), np.array([0.4, 0.5]), cal_fija(tau=1.0))
        self.assertEqual(d.candidatos, 1)
        self.assertFalse(d.consenso)
        self.assertEqual(d.verdict, "review")

    def test_varios_candidatos_es_amarillo(self) -> None:
        d = decidir(np.array([0.9, 0.85]), np.array([0.9, 0.8]), cal_fija(tau=0.5))
        self.assertEqual((d.verdict, d.candidatos), ("review", 2))

    def test_la_normalizacion_iguala_las_escalas(self) -> None:
        # El especialista da cosenos mucho más altos a todo: sin normalizar dominaría.
        cal = cal_fija(tau=10, ma=0.2, sa=0.1, mb=0.8, sb=0.02)
        d = decidir(np.array([0.5, 0.2]), np.array([0.80, 0.82]), cal)
        # generalista: z = [3, 0]; especialista: z = [0, 1] → fusionado [1.5, 0.5]
        np.testing.assert_allclose(d.fusionado, [1.5, 0.5], rtol=1e-5)
        self.assertEqual(int(d.orden[0]), 0)


class EscalaMostradaTests(unittest.TestCase):
    def test_la_linea_conformal_se_muestra_como_70(self) -> None:
        from serving.ensamble import similitud_mostrada
        out = similitud_mostrada(np.array([1.5, 1.0, 0.0]), tau=1.5)
        self.assertAlmostEqual(float(out[0]), 0.70, places=5)
        self.assertAlmostEqual(float(out[1]), 0.55, places=5)
        self.assertLess(float(out[2]), 0.55)

    def test_un_verde_siempre_se_ve_como_alta(self) -> None:
        # La pantalla solo pinta verde con similitud >= 0.70: un veredicto «match»
        # tiene que mostrarse en esa franja o la pantalla contradice al modelo.
        d = decidir(np.array([0.9, 0.1]), np.array([0.8, 0.2]), cal_fija(tau=0.5))
        self.assertEqual(d.verdict, "match")
        self.assertGreaterEqual(float(d.similitud[d.orden[0]]), 0.70)


class CalibracionTests(unittest.TestCase):
    def test_la_tasa_de_desconocidos_aceptados_queda_bajo_alfa(self) -> None:
        rng = np.random.default_rng(1)
        n_id, vistas = 40, 3
        G = concatenar(unitarios(rng, n_id * vistas, DA), unitarios(rng, n_id * vistas, DB))
        catalogo = [f"tat_{i:03d}" for i in range(n_id) for _ in range(vistas)]
        fuera = []
        for _ in range(30):
            cal_d = concatenar(unitarios(rng, 400, DA), unitarios(rng, 400, DB))
            prueba = concatenar(unitarios(rng, 400, DA), unitarios(rng, 400, DB))
            cal = calibrar(cal_d, G, catalogo, (DA, DB), alfa=0.05)
            codigos, ids = indices_identidad(catalogo)
            sa, sb = puntajes_por_modelo(prueba, G, codigos, len(ids), (DA, DB))
            fuera.append(float((cal.fusionar(sa, sb).max(1) >= cal.tau).mean()))
        # Garantía en promedio: ≤ α (+ margen por azar de 30 repeticiones).
        self.assertLessEqual(float(np.mean(fuera)), 0.05 + 0.01)

    def test_la_linea_sube_cuando_crece_la_galeria(self) -> None:
        rng = np.random.default_rng(2)
        desconocidos = concatenar(unitarios(rng, 500, DA), unitarios(rng, 500, DB))
        G = concatenar(unitarios(rng, 400, DA), unitarios(rng, 400, DB))
        ids = [f"tat_{i:03d}" for i in range(400)]
        chica = calibrar(desconocidos, G[:20], ids[:20], (DA, DB), 0.01)
        grande = calibrar(desconocidos, G, ids, (DA, DB), 0.01)
        # Más identidades = más ocasiones de parecerse por azar: el máximo de un
        # desconocido sube, y la línea en coseno tiene que subir con él.
        self.assertGreater(grande.media_a, chica.media_a)

    def test_sin_galeria_no_hay_calibracion(self) -> None:
        rng = np.random.default_rng(3)
        self.assertIsNone(calibrar(concatenar(unitarios(rng, 10, DA), unitarios(rng, 10, DB)),
                                   np.zeros((0, DA + DB), dtype="float32"), [], (DA, DB), 0.01))


if __name__ == "__main__":
    unittest.main()
