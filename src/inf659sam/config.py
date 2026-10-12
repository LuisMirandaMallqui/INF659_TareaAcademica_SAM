"""Rutas, versiones fijadas y ajuste del entorno de ejecucion.

Todas las rutas pesadas (datos, pesos, caches) se pueden mover con variables de
entorno para no guardarlas dentro de carpetas sincronizadas (OneDrive, Drive):

    INF659_DATA     carpeta de Sen1Floods11 descargado      (por defecto ./data)
    INF659_MODEL    carpeta con los pesos de SAM ViT-B       (por defecto ./models/sam-vit-base)
    INF659_CACHE    caches de embeddings y modelos ajustados (por defecto ./cache)
    INF659_THREADS  hilos de CPU para PyTorch                (por defecto 4)
    INF659_LOW_PRIORITY=1  baja la prioridad del proceso (util si la PC se usa para otra cosa)
"""
from __future__ import annotations

import os
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("INF659_DATA", REPO_ROOT / "data"))
MODEL_DIR = Path(os.environ.get("INF659_MODEL", REPO_ROOT / "models" / "sam-vit-base"))
CACHE_DIR = Path(os.environ.get("INF659_CACHE", REPO_ROOT / "cache"))
RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figuras"

# Modelo: pesos oficiales de Meta en Hugging Face, revision inmutable (commit) fijada.
MODEL_ID = "facebook/sam-vit-base"
MODEL_REVISION = "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f"
MODEL_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")

# Dataset: Sen1Floods11 v1.1 (Bonafilia et al., CVPRW 2020), bucket publico oficial.
S1F11_BASE = "https://storage.googleapis.com/sen1floods11/v1.1"

# Protocolo comun a todos los experimentos (fijado antes de ejecutar, ver docs/).
BAND_USED = "VH"
N_POS = 3
N_NEG = 2
NODATA_EXCLUSION_THRESHOLD = 0.95  # fraccion de pixeles sin etiqueta por encima de la cual se excluye un chip
SEEDS = (0, 1, 2)

N_THREADS = int(os.environ.get("INF659_THREADS", "4"))


def get_device() -> str:
    """'cuda' si hay GPU disponible (o lo que indique INF659_DEVICE), si no 'cpu'."""
    forced = os.environ.get("INF659_DEVICE")
    if forced:
        return forced
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def setup_runtime(seed: int = 0, threads: int | None = None, verbose: bool = True) -> dict:
    """Fija semillas, hilos de CPU y (opcional) baja la prioridad del proceso."""
    import numpy as np
    import torch

    threads = threads or N_THREADS
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    info = {"seed": seed, "torch_threads": torch.get_num_threads()}

    if os.environ.get("INF659_LOW_PRIORITY", "0") == "1":
        try:
            import psutil

            p = psutil.Process()
            if hasattr(psutil, "IDLE_PRIORITY_CLASS"):
                p.nice(psutil.IDLE_PRIORITY_CLASS)
            else:
                p.nice(19)
            info["priority"] = "baja"
        except Exception as exc:  # no es critico
            info["priority"] = f"sin cambio ({exc})"

    for d in (RESULTS_DIR, FIG_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if verbose:
        print("Entorno:", info)
    return info


def environment_report() -> dict:
    """Versiones exactas de las librerias usadas (para el README y los JSON)."""
    import platform
    import sys

    import numpy as np
    import scipy
    import skimage
    import torch
    import transformers

    cpu = platform.processor() or platform.machine()
    try:
        import psutil

        ram_gb = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        ram_gb = None
    dev = get_device()
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": cpu,
        "ram_gb": ram_gb,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "skimage": skimage.__version__,
        "device": dev,
        "gpu": gpu,
        "cuda": torch.version.cuda if dev == "cuda" else None,
        "torch_threads": torch.get_num_threads(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
