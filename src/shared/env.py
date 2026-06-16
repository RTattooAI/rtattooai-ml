"""Carga de variables de entorno desde un archivo .env en la raíz del repo.

Permite poner secretos (p. ej. ROBOFLOW_API_KEY) en un archivo local
`.env` (gitignored) en lugar de exportarlos a mano en cada terminal.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Se cargan en orden; .env.local tiene prioridad sobre .env.
ENV_FILES = (ROOT / ".env", ROOT / ".env.local")


def _parse_into_environ(path: Path) -> None:
    import os
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ[k.strip()] = v.strip().strip('"').strip("'")


def load_env() -> bool:
    """Carga .env y/o .env.local de la raíz. True si se cargó alguno."""
    loaded = False
    for env_file in ENV_FILES:
        if not env_file.exists():
            continue
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file, override=True)
        except ImportError:
            _parse_into_environ(env_file)
        loaded = True
    return loaded
