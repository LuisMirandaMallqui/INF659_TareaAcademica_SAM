"""Cola sin vigilancia del bloque de la seccion 2d de la bitacora. Corre cada paso cuando ya se puede.

    python tools/cola_exploracion.py

Orden: espera la descarga -> mascaras JRC -> espera R2 (libera la GPU) -> figuras finales ->
prueba de humo -> exploracion completa. Si un paso falla, se detiene y lo dice; se puede relanzar.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inf659sam import config  # noqa: E402

WEAK = Path(config.DATA_DIR) / "weak"
CINCO = config.RESULTS_DIR / "hacia_060" / "robustez" / "cinco_semillas.json"


def contar(sub):
    return len(list((WEAK / sub).glob("*.tif"))) if (WEAK / sub).exists() else 0


def esperar(msg, cond):
    print(f"[{time.strftime('%H:%M')}] Esperando: {msg}", flush=True)
    while not cond():
        time.sleep(30)
    print(f"[{time.strftime('%H:%M')}] Listo: {msg}", flush=True)


def correr(*args):
    print(f"\n[{time.strftime('%H:%M')}] >>> {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, str(Path(__file__).parent / args[0]), *args[1:]])
    if r.returncode != 0:
        raise SystemExit(f"Falló {args[0]} (código {r.returncode}). La cola se detiene; se puede relanzar.")


def main():
    esperar("descarga de etiquetas débiles y S1Weak", lambda: contar("S2IndexLabelWeak") >= 4384 and contar("S1Weak") >= 4384)
    correr("preparar_jrc_debiles.py")
    esperar("R2 (cinco semillas) para liberar la GPU", CINCO.exists)
    correr("figuras_finales_hacia060.py")
    correr("exploracion_unet.py", "--rapido")
    correr("exploracion_unet.py")
    print(f"\n[{time.strftime('%H:%M')}] Cola terminada.")


if __name__ == "__main__":
    main()
