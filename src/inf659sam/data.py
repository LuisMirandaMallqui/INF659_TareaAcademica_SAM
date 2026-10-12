"""Descarga y lectura de Sen1Floods11 (v1.1, chips etiquetados a mano) y de los pesos de SAM.

Las descargas son secuenciales y con limite de ancho de banda (por defecto 2.5 MB/s)
para no saturar la red de la maquina. Cada archivo se verifica con SHA-256 y queda
registrado en un manifiesto JSON.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import tifffile

from . import config

SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "valid": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _throttle(done: int, t0: float, rate_limit_mb_s: float | None) -> None:
    if rate_limit_mb_s:
        expected = done / (rate_limit_mb_s * 1024 * 1024)
        elapsed = time.perf_counter() - t0
        if expected > elapsed:
            time.sleep(expected - elapsed)


def download(url: str, dest: Path, rate_limit_mb_s: float | None = 2.5, retries: int = 4,
             range_chunk_mb: int | None = None) -> Path:
    """Descarga `url` en `dest` (si no existe) leyendo en bloques y limitando la velocidad.

    Con `range_chunk_mb` la descarga se hace por rangos HTTP (reanudable), util para
    archivos grandes servidos por CDN como los pesos de Hugging Face.
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": "inf659-sam/1.0"}
    chunk = 256 * 1024

    if range_chunk_mb:
        step = range_chunk_mb * 1024 * 1024
        start = tmp.stat().st_size if tmp.exists() else 0
        t0, done = time.perf_counter(), 0
        with open(tmp, "ab") as out:
            while True:
                for attempt in range(1, retries + 1):
                    try:
                        req = urllib.request.Request(url, headers={**headers, "Range": f"bytes={start}-{start + step - 1}"})
                        with urllib.request.urlopen(req, timeout=60) as resp:
                            got = 0
                            while True:
                                block = resp.read(chunk)
                                if not block:
                                    break
                                out.write(block)
                                got += len(block)
                                done += len(block)
                                _throttle(done, t0, rate_limit_mb_s)
                        break
                    except urllib.error.HTTPError as exc:
                        if exc.code == 416:  # rango fuera del archivo: ya esta completo
                            got = 0
                            break
                        if attempt == retries:
                            raise
                        time.sleep(2 * attempt)
                    except Exception:
                        if attempt == retries:
                            raise
                        time.sleep(2 * attempt)
                start += got
                if got < step:
                    break
        tmp.replace(dest)
        return dest

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
                t0, done = time.perf_counter(), 0
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    _throttle(done, t0, rate_limit_mb_s)
            tmp.replace(dest)
            return dest
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"No se pudo descargar {url}: {exc}") from exc
            time.sleep(2 * attempt)
    return dest


# ----------------------------------------------------------------------------- particiones

def split_csv_url(split: str) -> str:
    return f"{config.S1F11_BASE}/splits/flood_handlabeled/{SPLIT_FILES[split]}"


def read_split(split: str, data_dir: Path | None = None) -> list[str]:
    """Nombres de chip (p. ej. 'Ghana_5079') en el orden exacto del CSV oficial."""
    data_dir = Path(data_dir or config.DATA_DIR)
    csv_path = download(split_csv_url(split), data_dir / "splits" / SPLIT_FILES[split], rate_limit_mb_s=None)
    names = []
    for line in csv_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        s1 = line.split(",")[0].strip()
        names.append(s1.replace("_S1Hand.tif", ""))
    return names


def region_of(chip: str) -> str:
    return chip.rsplit("_", 1)[0]


def stratified_holdout(names: list[str], frac: float = 0.2, seed: int = 0) -> tuple[list[str], list[str]]:
    """Separa `frac` de cada region (train-dev) para el barrido de hiperparametros de H2."""
    rng = np.random.default_rng(seed)
    by_region: dict[str, list[str]] = {}
    for n in names:
        by_region.setdefault(region_of(n), []).append(n)
    core, dev = [], []
    for r in sorted(by_region):
        group = sorted(by_region[r])
        k = max(1, int(round(frac * len(group))))
        pick = set(rng.choice(len(group), size=k, replace=False).tolist())
        for i, n in enumerate(group):
            (dev if i in pick else core).append(n)
    return core, dev


def first_chip_per_region(names: list[str]) -> list[str]:
    """Regla de seleccion del E1: primera fila de cada region en el orden del CSV."""
    seen, out = set(), []
    for n in names:
        r = region_of(n)
        if r not in seen:
            seen.add(r)
            out.append(n)
    return out


# ----------------------------------------------------------------------------- chips

def chip_urls(chip: str) -> dict:
    base = f"{config.S1F11_BASE}/data/flood_events/HandLabeled"
    return {
        "s1": f"{base}/S1Hand/{chip}_S1Hand.tif",
        "label": f"{base}/LabelHand/{chip}_LabelHand.tif",
    }


def chip_paths(chip: str, data_dir: Path | None = None) -> dict:
    data_dir = Path(data_dir or config.DATA_DIR)
    return {
        "s1": data_dir / "S1Hand" / f"{chip}_S1Hand.tif",
        "label": data_dir / "LabelHand" / f"{chip}_LabelHand.tif",
    }


def ensure_chips(chips: list[str], data_dir: Path | None = None, rate_limit_mb_s: float | None = 2.5,
                 log_every: int = 25) -> dict:
    """Descarga (si faltan) los chips pedidos y devuelve {archivo: sha256}."""
    hashes = {}
    for i, chip in enumerate(chips, 1):
        urls, paths = chip_urls(chip), chip_paths(chip, data_dir)
        for key in ("s1", "label"):
            download(urls[key], paths[key], rate_limit_mb_s=rate_limit_mb_s)
            hashes[paths[key].name] = sha256(paths[key])
        if log_every and (i % log_every == 0 or i == len(chips)):
            print(f"  chips listos: {i}/{len(chips)}", flush=True)
    return hashes


def load_chip(chip: str, data_dir: Path | None = None) -> dict:
    """VV y VH en dB (float32, NaN = sin dato del sensor) y LabelHand (-1 sin dato, 0 tierra, 1 agua)."""
    paths = chip_paths(chip, data_dir)
    s1 = tifffile.imread(paths["s1"]).astype(np.float32)
    label = tifffile.imread(paths["label"]).astype(np.int16)
    if s1.ndim != 3 or s1.shape[0] != 2:
        raise ValueError(f"{chip}: forma inesperada de S1Hand {s1.shape}")
    return {"chip": chip, "vv": s1[0], "vh": s1[1], "label": label}


def write_manifest(path: Path, split_to_chips: dict, hashes: dict, extra: dict | None = None) -> None:
    manifest = {
        "dataset": "Sen1Floods11 v1.1, chips etiquetados a mano (HandLabeled)",
        "base_url": config.S1F11_BASE,
        "splits": {k: {"csv": split_csv_url(k.split("_")[0]), "n_chips": len(v), "chips": v}
                   for k, v in split_to_chips.items()},
        "sha256": dict(sorted(hashes.items())),
    }
    if extra:
        manifest.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")


# ----------------------------------------------------------------------------- pesos de SAM

def ensure_model(model_dir: Path | None = None, rate_limit_mb_s: float | None = 3.0) -> Path:
    """Descarga config, preprocesador y model.safetensors de la revision fijada (sin pickle)."""
    model_dir = Path(model_dir or config.MODEL_DIR)
    for fname in config.MODEL_FILES:
        url = f"https://huggingface.co/{config.MODEL_ID}/resolve/{config.MODEL_REVISION}/{fname}"
        big = fname.endswith(".safetensors")
        download(url, model_dir / fname, rate_limit_mb_s=rate_limit_mb_s, range_chunk_mb=8 if big else None)
    weights = model_dir / "model.safetensors"
    if sha256(weights) != MODEL_SAFETENSORS_SHA256:
        raise RuntimeError("El SHA-256 de model.safetensors no coincide con el publicado por Hugging Face")
    return model_dir


# SHA-256 publicado por Hugging Face (X-Linked-ETag) para model.safetensors en la revision fijada.
MODEL_SAFETENSORS_SHA256 = "892c410e496344e527255ccdcb2cb7244a609acb5389c7c4fdba1288f861c579"
