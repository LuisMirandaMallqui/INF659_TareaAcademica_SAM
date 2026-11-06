"""E2b (bitacora, seccion 2d): mascara "ocurrencia de agua JRC >= 50 %" para cada chip debil.

    python tools/preparar_jrc_debiles.py

Baja los mosaicos publicos de JRC Global Surface Water (10 x 10 grados, 0.00025 grados por pixel, 2021) a
<INF659_DATA>/jrc_occurrence/ y guarda una mascara por chip en <INF659_DATA>/weak/JRCOcc50/<chip>.npy
(np.packbits). Vecino mas cercano. Se usa solo para corregir etiquetas debiles de entrenamiento.
"""
from __future__ import annotations

import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from tqdm import tqdm  # noqa: E402

from inf659sam import config  # noqa: E402

URL = "https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/"
RES, SIZE = 0.00025, 512
UMBRAL = 50


def nombre_mosaico(w: int, n: int) -> str:
    return f"occurrence_{abs(w)}{'E' if w >= 0 else 'W'}_{abs(n)}{'N' if n >= 0 else 'S'}v1_4_2021.tif"


def bajar(nombre: str, dest: Path) -> Path | None:
    f = dest / nombre
    if f.exists():
        return f
    try:
        urllib.request.urlretrieve(URL + nombre, f.with_suffix(".part"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:  # mosaico sin tierra
            return None
        raise
    f.with_suffix(".part").replace(f)
    return f


def main():
    data_dir = Path(config.DATA_DIR)
    labs = sorted((data_dir / "weak" / "S2IndexLabelWeak").glob("*.tif"))
    out_dir = data_dir / "weak" / "JRCOcc50"
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = data_dir / "jrc_occurrence"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    geo, por_mosaico = {}, {}
    for f in tqdm(labs, desc="georreferencia"):
        chip = f.name.replace("_S2IndexLabelWeak.tif", "")
        if (out_dir / f"{chip}.npy").exists():
            continue
        with tifffile.TiffFile(f) as t:
            g = t.pages[0].geotiff_tags
        s, lon0, lat0 = g["ModelPixelScale"][0], g["ModelTiepoint"][3], g["ModelTiepoint"][4]
        lons = lon0 + (np.arange(SIZE) + 0.5) * s
        lats = lat0 - (np.arange(SIZE) + 0.5) * s
        geo[chip] = (lons, lats)
        for w in {math.floor(lons[0] / 10) * 10, math.floor(lons[-1] / 10) * 10}:
            for n in {math.ceil(lats[0] / 10) * 10, math.ceil(lats[-1] / 10) * 10}:
                por_mosaico.setdefault((w, n), []).append(chip)
    print(f"{len(geo)} chips por procesar en {len(por_mosaico)} mosaicos")

    masks = {c: np.zeros((SIZE, SIZE), bool) for c in geo}
    for (w, n), chips in tqdm(sorted(por_mosaico.items()), desc="mosaicos"):
        f = bajar(nombre_mosaico(w, n), tiles_dir)
        if f is None:
            continue
        occ = tifffile.imread(f)
        for c in chips:
            lons, lats = geo[c]
            cols = np.floor((lons - w) / RES).astype(int)
            rows = np.floor((n - lats) / RES).astype(int)
            ok_c, ok_r = (cols >= 0) & (cols < occ.shape[1]), (rows >= 0) & (rows < occ.shape[0])
            if not ok_c.any() or not ok_r.any():
                continue
            ic, ir = np.nonzero(ok_c)[0], np.nonzero(ok_r)[0]
            win = occ[np.ix_(rows[ir], cols[ic])]
            masks[c][np.ix_(ir, ic)] |= (win >= UMBRAL) & (win != 255)
        del occ
    for c, m in masks.items():
        np.save(out_dir / f"{c}.npy", np.packbits(m))
    frac = np.mean([m.mean() for m in masks.values()]) if masks else float("nan")
    print(f"Listo: {len(masks)} máscaras, fracción media de agua JRC {frac:.4f}")


if __name__ == "__main__":
    main()
