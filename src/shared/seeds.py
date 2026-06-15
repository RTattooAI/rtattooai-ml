"""Fijado de semillas para reproducibilidad.

Se usa la MISMA semilla (por defecto 42) en descarga, preparación del
dataset, splits y entrenamiento, para que los resultados sean reproducibles.
"""
from __future__ import annotations

import os
import random

SEED = 42


def set_seed(seed: int = SEED, deterministic: bool = True) -> int:
    """Fija la semilla global de Python, NumPy y PyTorch (CPU + CUDA).

    Devuelve la semilla usada para que el caller pueda registrarla.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            # cudnn determinista: reproducible a costa de algo de velocidad
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
