"""Galería persistente (embeddings + catalog_id + band) con backend local o Azure Blob.

Esquema del npz (una fila por VISTA; varias vistas pueden compartir catalog_id):
  - embeddings : (N, dim) float32, L2-normalizados
  - catalog_ids: (N,) str   -- "tat_00001", repetido por cada vista de la identidad
  - bands      : (N,) str   -- banda asociada (constante dentro de un catalog_id)
  - names      : (N,) str   -- nombre de diseño original (trazabilidad/debug)

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
import threading
from pathlib import Path

import numpy as np


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
        self.head_fingerprint = str(data["head_fingerprint"]) if "head_fingerprint" in data else None

    def save(self) -> None:
        """Persiste el estado actual: local siempre, y Blob si está configurado."""
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        extra = {}
        if self.head_fingerprint:
            extra["head_fingerprint"] = np.array(self.head_fingerprint)
        np.savez(buf, embeddings=self.embeddings.astype("float32"),
                 catalog_ids=np.array(self.catalog_ids), bands=np.array(self.bands),
                 names=np.array(self.names), **extra)
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
               names: list[str] | None = None) -> None:
        """Agrega vistas (V, dim) con un catalog_id ya conocido y persiste. Thread-safe."""
        with self._lock:
            self._append_locked(embeddings, catalog_id, band, names)

    def enroll_views(self, embeddings: np.ndarray, band: str,
                      names: list[str] | None = None) -> str:
        """Asigna un catalog_id nuevo y agrega las vistas atómicamente (evita que
        dos enrolamientos concurrentes reciban el mismo id). Devuelve el catalog_id."""
        with self._lock:
            catalog_id = self.next_catalog_id()
            self._append_locked(embeddings, catalog_id, band, names)
            return catalog_id

    def _append_locked(self, embeddings: np.ndarray, catalog_id: str, band: str,
                        names: list[str] | None = None) -> None:
        n_views = embeddings.shape[0]
        names = names or [catalog_id] * n_views
        self.embeddings = np.concatenate([self.embeddings, embeddings.astype("float32")], axis=0)
        self.catalog_ids.extend([catalog_id] * n_views)
        self.bands.extend([band] * n_views)
        self.names.extend(names)
        self.save()

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
