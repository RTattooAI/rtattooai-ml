"""Pruebas de la galería persistente.

Ejecutar desde rtattooai-ml:
    python -m unittest serving.tests.test_gallery_store
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from serving.gallery_store import CatalogIdOcupado, GalleryStore

DIM = 8


def vistas(semilla: int, n: int = 3) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    v = rng.normal(size=(n, DIM)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class GalleryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.carpeta = tempfile.TemporaryDirectory()
        self.ruta = Path(self.carpeta.name) / "gallery.npz"
        self.galeria = GalleryStore(self.ruta, dim=DIM)

    def tearDown(self) -> None:
        self.carpeta.cleanup()

    def recargar(self) -> GalleryStore:
        otra = GalleryStore(self.ruta, dim=DIM)
        otra.load()
        return otra

    def test_usa_el_codigo_que_elige_quien_llama(self) -> None:
        cid = self.galeria.enroll_views(vistas(1), band="", catalog_id="tat_00040", image_sha256="abc")

        self.assertEqual(cid, "tat_00040")
        self.assertEqual(self.recargar().entries(),
                         [{"catalog_id": "tat_00040", "views": 3, "image_sha256": "abc"}])

    def test_rechaza_un_codigo_que_ya_tiene_vectores_sin_tocar_la_galeria(self) -> None:
        # Es el incidente del 11/09/2026: el mismo código acabó con dos tatuajes.
        self.galeria.enroll_views(vistas(1), band="", catalog_id="tat_00026")

        with self.assertRaises(CatalogIdOcupado):
            self.galeria.enroll_views(vistas(2), band="", catalog_id="tat_00026")

        self.assertEqual(self.recargar().entries()[0]["views"], 3)

    def test_sin_codigo_asigna_el_siguiente_libre(self) -> None:
        self.galeria.enroll_views(vistas(1), band="", catalog_id="tat_00007")

        self.assertEqual(self.galeria.enroll_views(vistas(2), band=""), "tat_00008")

    def test_quitar_borra_todas_las_vistas_y_deja_respaldo(self) -> None:
        self.galeria.enroll_views(vistas(1), band="", catalog_id="tat_00001")
        self.galeria.enroll_views(vistas(2), band="", catalog_id="tat_00002")

        self.assertEqual(self.galeria.remove("tat_00001"), 3)

        recargada = self.recargar()
        self.assertEqual([e["catalog_id"] for e in recargada.entries()], ["tat_00002"])
        self.assertEqual(recargada.embeddings.shape, (3, DIM))
        self.assertTrue(any("antes-quitar-tat_00001" in p.name for p in self.ruta.parent.iterdir()))

    def test_quitar_un_codigo_inexistente_no_escribe_nada(self) -> None:
        self.assertEqual(self.galeria.remove("tat_09999"), 0)
        self.assertFalse(self.ruta.exists())

    def test_detecta_un_vector_compartido_por_dos_codigos(self) -> None:
        comun = vistas(7, n=1)
        self.galeria.enroll_views(np.concatenate([comun, vistas(8, n=2)]), band="", catalog_id="tat_00026")
        self.galeria.enroll_views(np.concatenate([comun, vistas(9, n=2)]), band="", catalog_id="tat_00027")
        self.galeria.enroll_views(vistas(10), band="", catalog_id="tat_00028")

        self.assertEqual(self.galeria.shared_vectors(), [["tat_00026", "tat_00027"]])

    def test_una_galeria_anterior_sin_huellas_se_sigue_leyendo(self) -> None:
        np.savez(self.ruta, embeddings=vistas(1), catalog_ids=np.array(["tat_00001"] * 3),
                 bands=np.array([""] * 3), names=np.array(["tren"] * 3))

        galeria = self.recargar()

        self.assertEqual(galeria.entries(), [{"catalog_id": "tat_00001", "views": 3, "image_sha256": None}])
        galeria.enroll_views(vistas(2), band="", catalog_id="tat_00002", image_sha256="def")
        self.assertEqual(len(self.recargar().image_hashes), 6)


if __name__ == "__main__":
    unittest.main()
