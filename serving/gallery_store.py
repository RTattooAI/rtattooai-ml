"""Galería persistente (embeddings + catalog_id + band) con backend local o Azure Blob.

Esquema del npz (una fila por VISTA; varias vistas pueden compartir catalog_id):
  - embeddings  : (N, dim) float32, L2-normalizados
  - catalog_ids : (N,) str   -- "tat_00001", repetido por cada vista de la identidad
  - bands       : (N,) str   -- banda asociada (constante dentro de un catalog_id)
  - names       : (N,) str   -- nombre de diseño original (trazabilidad/debug)
  - image_sha256: (N,) str   -- huella de la imagen enrolada; vacía en las filas
                                anteriores a que existiera. Es lo que permite al
                                backend comprobar que un código de la galería
                                corresponde a la misma imagen que su ficha del catálogo.

Backend:
  - Si hay AZURE_STORAGE_CONNECTION_STRING: la galería vive en Blob; se descarga
    al arrancar y se vuelve a subir en cada `save()` (last-write-wins, aceptable
    para esta demo según el contrato). También se cachea una copia local.
  - Si no: el archivo local es la única fuente de verdad.

Concurrencia: un `threading.Lock` protege append+save (enrolamiento es de baja
frecuencia; basta para no corromper el estado en memoria entre requests).
"""
from __future__ import annotations

import io
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class CatalogIdOcupado(Exception):
    """El código pedido ya tiene vectores en la galería.

    Pasó el 11/09/2026: una reconstrucción de la galería dejó fuera un tatuaje que el
    catálogo seguía teniendo como `tat_00026`, la galería volvió a repartir ese código
    a otro tatuaje y el catálogo acabó atribuyéndole la banda del primero. Mezclar dos
    tatuajes bajo un mismo código es peor que rechazar el enrolamiento, así que se
    rechaza y quien llama elige otro código.
    """


class GalleryStore:
    def __init__(self, local_path: Path, dim: int,
                 azure_connection_string: str | None = None,
                 container: str = "models", blob_name: str = "gallery_dinov2.npz"):
        self.local_path = Path(local_path)
        self.dim = dim
        self._azure_conn_str = azure_connection_string
        self._container = container
        self._blob_name = blob_name
        self._lock = threading.Lock()

        self.embeddings = np.zeros((0, dim), dtype="float32")
        self.catalog_ids: list[str] = []
        self.bands: list[str] = []
        self.names: list[str] = []
        self.image_hashes: list[str] = []
        # Huella de la cabeza con la que se calcularon estos vectores. Sirve para
        # detectar que la galeria y el modelo dejaron de ser la misma pareja.
        self.head_fingerprint: str | None = None

    @property
    def _use_blob(self) -> bool:
        return bool(self._azure_conn_str)

    def _blob_client(self):
        from azure.storage.blob import BlobServiceClient
        service = BlobServiceClient.from_connection_string(self._azure_conn_str)
        container_client = service.get_container_client(self._container)
        if not container_client.exists():
            container_client.create_container()
        return container_client.get_blob_client(self._blob_name)

    def load(self) -> None:
        """Carga la galería: Blob (si está configurado) o archivo local."""
        raw_bytes: bytes | None = None
        if self._use_blob:
            client = self._blob_client()
            if client.exists():
                raw_bytes = client.download_blob().readall()
                self.local_path.parent.mkdir(parents=True, exist_ok=True)
                self.local_path.write_bytes(raw_bytes)
        if raw_bytes is None and self.local_path.exists():
            raw_bytes = self.local_path.read_bytes()

        if raw_bytes is None:
            return  # galería vacía: arranca desde cero

        data = np.load(io.BytesIO(raw_bytes), allow_pickle=True)
        self.embeddings = data["embeddings"].astype("float32")
        self.catalog_ids = [str(c) for c in data["catalog_ids"]]
        self.bands = [str(b) for b in data["bands"]]
        self.names = [str(n) for n in data["names"]] if "names" in data else list(self.catalog_ids)
        self.image_hashes = ([str(h) for h in data["image_sha256"]] if "image_sha256" in data
                             else [""] * len(self.catalog_ids))
        self.head_fingerprint = str(data["head_fingerprint"]) if "head_fingerprint" in data else None

    def save(self) -> None:
        """Persiste el estado actual: local siempre, y Blob si está configurado."""
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        extra = {}
        if self.head_fingerprint:
            extra["head_fingerprint"] = np.array(self.head_fingerprint)
        np.savez(buf, embeddings=self.embeddings.astype("float32"),
                 catalog_ids=np.array(self.catalog_ids, dtype=str), bands=np.array(self.bands, dtype=str),
                 names=np.array(self.names, dtype=str), image_sha256=np.array(self.image_hashes, dtype=str),
                 **extra)
        raw_bytes = buf.getvalue()
        self.local_path.write_bytes(raw_bytes)
        if self._use_blob:
            client = self._blob_client()
            client.upload_blob(raw_bytes, overwrite=True)

    def next_catalog_id(self) -> str:
        max_n = 0
        for cid in self.catalog_ids:
            if cid.startswith("tat_"):
                try:
                    max_n = max(max_n, int(cid[4:]))
                except ValueError:
                    continue
        return f"tat_{max_n + 1:05d}"

    def append(self, embeddings: np.ndarray, catalog_id: str, band: str,
               names: list[str] | None = None, image_sha256: str = "") -> None:
        """Agrega vistas (V, dim) con un catalog_id ya conocido y persiste. Thread-safe."""
        with self._lock:
            self._append_locked(embeddings, catalog_id, band, names, image_sha256)

    def enroll_views(self, embeddings: np.ndarray, band: str,
                     names: list[str] | None = None, catalog_id: str | None = None,
                     image_sha256: str = "") -> str:
        """Agrega las vistas atómicamente y devuelve el catalog_id.

        Si `catalog_id` llega, se usa ese y se lanza `CatalogIdOcupado` cuando ya tiene
        vectores. Sin él, la galería asigna el siguiente libre: es el comportamiento
        anterior, que no sabe nada de los códigos que existen en el catálogo del backend.
        """
        with self._lock:
            if catalog_id:
                if catalog_id in self.catalog_ids:
                    raise CatalogIdOcupado(catalog_id)
                elegido = catalog_id
            else:
                elegido = self.next_catalog_id()
            self._append_locked(embeddings, elegido, band, names, image_sha256)
            return elegido

    def _append_locked(self, embeddings: np.ndarray, catalog_id: str, band: str,
                       names: list[str] | None = None, image_sha256: str = "") -> None:
        n_views = embeddings.shape[0]
        names = names or [catalog_id] * n_views
        self.embeddings = np.concatenate([self.embeddings, embeddings.astype("float32")], axis=0)
        self.catalog_ids.extend([catalog_id] * n_views)
        self.bands.extend([band] * n_views)
        self.names.extend(names)
        self.image_hashes.extend([image_sha256] * n_views)
        self.save()

    def remove(self, catalog_id: str) -> int:
        """Quita todas las vistas de un catalog_id y persiste. Devuelve cuántas quitó.

        Antes de escribir deja una copia fechada del archivo: borrar vectores no tiene
        vuelta atrás sin la imagen de origen.
        """
        with self._lock:
            conservar = [i for i, cid in enumerate(self.catalog_ids) if cid != catalog_id]
            quitadas = len(self.catalog_ids) - len(conservar)
            if quitadas == 0:
                return 0

            if self.local_path.exists():
                marca = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                respaldo = self.local_path.with_name(
                    f"{self.local_path.stem}.antes-quitar-{catalog_id}-{marca}{self.local_path.suffix}")
                shutil.copy2(self.local_path, respaldo)

            self.embeddings = (self.embeddings[conservar] if conservar
                               else np.zeros((0, self.dim), dtype="float32"))
            self.catalog_ids = [self.catalog_ids[i] for i in conservar]
            self.bands = [self.bands[i] for i in conservar]
            self.names = [self.names[i] for i in conservar]
            self.image_hashes = [self.image_hashes[i] for i in conservar]
            self.save()
            return quitadas

    def entries(self) -> list[dict]:
        """Una entrada por catalog_id, en el orden en que aparecen: vistas y huella."""
        vistas: dict[str, int] = {}
        huellas: dict[str, str] = {}
        for cid, huella in zip(self.catalog_ids, self.image_hashes):
            vistas[cid] = vistas.get(cid, 0) + 1
            if huella and not huellas.get(cid):
                huellas[cid] = huella
        return [{"catalog_id": cid, "views": n, "image_sha256": huellas.get(cid) or None}
                for cid, n in vistas.items()]

    def shared_vectors(self) -> list[list[str]]:
        """Grupos de catalog_id distintos que contienen exactamente el mismo vector.

        Dos tatuajes distintos no producen el mismo vector: si ocurre, es que la misma
        imagen se enroló bajo dos códigos. Es la huella que dejó el incidente del
        11/09/2026, y la única forma de verlo en las filas anteriores a la huella de
        imagen.
        """
        if self.embeddings.shape[0] == 0:
            return []
        grupos: dict[bytes, set[str]] = {}
        for fila, cid in zip(np.round(self.embeddings, 5), self.catalog_ids):
            grupos.setdefault(fila.tobytes(), set()).add(cid)
        distintos = {tuple(sorted(ids)) for ids in grupos.values() if len(ids) > 1}
        return [list(ids) for ids in sorted(distintos)]

    def search(self, query: np.ndarray) -> list[tuple[str, str, float]]:
        """Coseno contra todas las vistas; máximo por catalog_id. Devuelve
        [(catalog_id, band, similarity)] ordenado desc. Vacío si la galería está vacía."""
        if self.embeddings.shape[0] == 0:
            return []
        sims = self.embeddings @ query
        best: dict[str, tuple[str, float]] = {}
        for cid, band, s in zip(self.catalog_ids, self.bands, sims):
            s = float(s)
            if cid not in best or s > best[cid][1]:
                best[cid] = (band, s)
        ranked = sorted(((cid, band, s) for cid, (band, s) in best.items()), key=lambda t: -t[2])
        return ranked
