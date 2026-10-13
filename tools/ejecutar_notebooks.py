"""Ejecuta los notebooks en orden y guarda las salidas dentro de cada .ipynb.

Uso (desde la carpeta del repositorio):
    python tools/ejecutar_notebooks.py                 # todos
    python tools/ejecutar_notebooks.py 01 03           # solo los que empiezan con esos prefijos
    python tools/ejecutar_notebooks.py --baja-prioridad    # no compite con otras aplicaciones

Variables de entorno utiles: INF659_DATA, INF659_MODEL, INF659_CACHE, INF659_THREADS.
"""
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "notebooks"


def main(argv):
    low = "--baja-prioridad" in argv
    prefixes = [a for a in argv if not a.startswith("--")]
    if low:
        os.environ["INF659_LOW_PRIORITY"] = "1"
        try:
            import psutil

            psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if hasattr(psutil, "IDLE_PRIORITY_CLASS") else 19)
        except Exception as exc:
            print("No se pudo bajar la prioridad:", exc)
    nbs = sorted(NB_DIR.glob("[0-9][0-9]_*.ipynb"))
    if prefixes:
        nbs = [p for p in nbs if any(p.name.startswith(x) for x in prefixes)]
    for path in nbs:
        t0 = time.time()
        print(f"Ejecutando {path.name} ...", flush=True)
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(nb, timeout=None, kernel_name="python3",
                                resources={"metadata": {"path": str(NB_DIR)}})
        try:
            client.execute()
        finally:
            nbformat.write(nb, path)
        print(f"  listo en {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
