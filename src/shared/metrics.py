"""Helpers para construir tablas de métricas y reportes en markdown/csv."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

# Metas del proyecto (informe de tesis RTattooAI)
BASELINE_MAP50 = 0.79     # Pocevičė et al., 2024 — YOLOv5 sobre DeMSI
GOAL_MAP50 = 0.85         # meta de la tesis: mAP@0.5 > 0.85
GOAL_PRECISION = 0.80     # meta de la tesis: precisión >= 80%


def write_markdown_table(rows: Sequence[dict], headers: Sequence[str], path: Path) -> str:
    """Escribe `rows` como tabla markdown en `path` y devuelve el texto."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def write_csv(rows: Sequence[dict], headers: Sequence[str], path: Path) -> None:
    """Escribe `rows` como CSV en `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(headers), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def goal_verdict(map50: float, precision: float) -> str:
    """Devuelve un veredicto legible sobre el cumplimiento de la meta."""
    parts = []
    parts.append(
        f"mAP@0.5 = {map50:.3f}  "
        f"({'CUMPLE' if map50 > GOAL_MAP50 else 'NO cumple'} meta > {GOAL_MAP50}; "
        f"{'supera' if map50 > BASELINE_MAP50 else 'no supera'} baseline {BASELINE_MAP50})"
    )
    parts.append(
        f"Precisión = {precision:.3f}  "
        f"({'CUMPLE' if precision >= GOAL_PRECISION else 'NO cumple'} meta >= {GOAL_PRECISION})"
    )
    return "\n".join(parts)
