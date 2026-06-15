"""Carga de variables de entorno desde un archivo .env en la raíz del repo.

Permite poner secretos (p. ej. ROBOFLOW_API_KEY) en un archivo local
`.env` (gitignored) en lugar de exportarlos a mano en cada terminal.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"


def load_env() -> bool:
    """Carga .env si existe. Devuelve True si se cargó un archivo."""
    if not ENV_FILE.exists():
        return False
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE, override=False)
        return True
    except ImportError:
        # Fallback mínimo sin python-dotenv: parseo KEY=VALUE línea a línea.
        import os
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        return True
