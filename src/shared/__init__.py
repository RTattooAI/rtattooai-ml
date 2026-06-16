"""Utilidades compartidas entre las etapas de detección y embedding."""
# La consola de Windows (cp1252) rompe al imprimir acentos/Unicode. Forzamos
# UTF-8 en stdout/stderr para que los scripts no crasheen al imprimir reportes.
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
