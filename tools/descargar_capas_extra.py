"""Descarga las capas del bloque exploratorio (bitacora, seccion 2d) del bucket publico de Sen1Floods11.

    python tools/descargar_capas_extra.py jrc      # JRCWaterHand (446 archivos, < 1 MB en total)
    python tools/descargar_capas_extra.py debiles  # S2IndexLabelWeak (31 MB) y S1Weak (7 GB)

Reanudable: salta lo que ya existe. Queda en <INF659_DATA>/JRCWaterHand y <INF659_DATA>/weak/.
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm  # noqa: E402

from inf659sam import config  # noqa: E402

API = "https://storage.googleapis.com/storage/v1/b/sen1floods11/o"
BASE = "https://storage.googleapis.com/sen1floods11/"
PREFIX = {
    "jrc": [("v1.1/data/flood_events/HandLabeled/JRCWaterHand/", "JRCWaterHand")],
    "debiles": [("v1.1/data/flood_events/WeaklyLabeled/S2IndexLabelWeak/", "weak/S2IndexLabelWeak"),
                ("v1.1/data/flood_events/WeaklyLabeled/S1Weak/", "weak/S1Weak")],
}


def listar(prefix: str) -> list[tuple[str, int]]:
    out, token = [], None
    while True:
        q = {"prefix": prefix, "maxResults": 1000, "fields": "items(name,size),nextPageToken"}
        if token:
            q["pageToken"] = token
        with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(q)}", timeout=120) as r:
            d = json.load(r)
        out += [(i["name"], int(i["size"])) for i in d.get("items", []) if not i["name"].endswith("/")]
        token = d.get("nextPageToken")
        if not token:
            return out


def bajar(args) -> int:
    name, size, dest = args
    f = dest / Path(name).name
    if f.exists() and f.stat().st_size == size:
        return 0
    tmp = f.with_suffix(f.suffix + ".part")
    for intento in range(4):
        try:
            with urllib.request.urlopen(BASE + urllib.parse.quote(name), timeout=120) as r, open(tmp, "wb") as o:
                while True:
                    b = r.read(1 << 18)
                    if not b:
                        break
                    o.write(b)
            if tmp.stat().st_size != size:
                raise OSError("tamaño distinto")
            tmp.replace(f)
            return size
        except Exception:
            if intento == 3:
                raise
    return 0


def main():
    capas = sys.argv[1:] or ["jrc"]
    for capa in capas:
        for prefix, sub in PREFIX[capa]:
            dest = Path(config.DATA_DIR) / sub
            dest.mkdir(parents=True, exist_ok=True)
            items = listar(prefix)
            total = sum(s for _, s in items)
            print(f"{sub}: {len(items)} archivos, {total / 1e6:,.0f} MB")
            with ThreadPoolExecutor(8) as ex, tqdm(total=total, unit="B", unit_scale=True, desc=sub) as bar:
                for n in ex.map(bajar, [(a, s, dest) for a, s in items]):
                    bar.update(n)
    print("Listo.")


if __name__ == "__main__":
    main()
