"""Cache de embeddings del codificador de imagen (y de tokens para el probing por capa).

Cada chip se codifica UNA vez por representacion de entrada. Se guarda:
  - el embedding del cuello (1, 256, 64, 64) en float32, que es lo que ve el decodificador;
  - una muestra estratificada de tokens (hasta 32 de agua y 32 de no agua) de los bloques
    0, 2, 5, 8, 11 y del cuello, para medir en que capa el agua es linealmente separable.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import config, data
from .faseA import encoder_states
from .sam_utils import SamWrapper

PROBE_LAYERS = (0, 2, 5, 8, 11)
TOKENS_PER_CLASS = 32


def cache_dir(rep: str) -> Path:
    d = config.CACHE_DIR / f"emb_{rep}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def token_labels(label: np.ndarray, grid: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Fraccion de agua y de pixeles etiquetados por token (cada token = 8x8 px del chip de 512)."""
    H, W = label.shape
    s = H // grid
    valid = (label != -1).reshape(grid, s, grid, s).mean(axis=(1, 3))
    water = (label == 1).reshape(grid, s, grid, s).sum(axis=(1, 3)) / np.maximum(
        (label != -1).reshape(grid, s, grid, s).sum(axis=(1, 3)), 1)
    return water, valid


def sample_tokens(label: np.ndarray, rng: np.random.Generator, k: int = TOKENS_PER_CLASS):
    water, valid = token_labels(label)
    ok = valid >= 0.5
    pos = np.argwhere(ok & (water >= 0.5))
    neg = np.argwhere(ok & (water < 0.5))
    sel_p = pos[rng.choice(len(pos), size=min(k, len(pos)), replace=False)] if len(pos) else np.zeros((0, 2), int)
    sel_n = neg[rng.choice(len(neg), size=min(k, len(neg)), replace=False)] if len(neg) else np.zeros((0, 2), int)
    idx = np.concatenate([sel_p, sel_n]).astype(np.int16)
    y = np.concatenate([np.ones(len(sel_p)), np.zeros(len(sel_n))]).astype(np.int8)
    return idx, y


def encode_and_cache(sam: SamWrapper, chips: list[str], rep: str = "vh3", probe: bool = True,
                     log_file: Path | None = None, seed: int = 0) -> dict:
    """Codifica los chips que falten en cache. Devuelve tiempos del codificador."""
    out_dir = cache_dir(rep)
    times = {}
    t_start = time.time()
    todo = [c for c in chips if not (out_dir / f"{c}.npz").exists()]
    for i, chip in enumerate(todo, 1):
        d = data.load_chip(chip)
        rgb = SamWrapper.sar_to_rgb(d["vv"], d["vh"], rep)
        pre = sam.preprocess(rgb)
        t0 = time.perf_counter()
        emb, states = encoder_states(sam.model, pre["pixel_values"], PROBE_LAYERS if probe else ())
        t_enc = time.perf_counter() - t0
        payload = {
            "emb": emb[0].numpy().astype(np.float32),
            "original_sizes": pre["original_sizes"].numpy(),
            "reshaped_input_sizes": pre["reshaped_input_sizes"].numpy(),
            "t_encoder_s": np.float32(t_enc),
        }
        if probe:
            # semilla derivada del nombre del chip: la muestra no depende del orden de proceso
            rng = np.random.default_rng(seed + int.from_bytes(chip.encode(), "little") % (2**32))
            idx, y = sample_tokens(d["label"], rng)
            payload["tok_idx"], payload["tok_y"] = idx, y
            for layer in PROBE_LAYERS:
                payload[f"tok_l{layer}"] = states[layer][idx[:, 0], idx[:, 1]].numpy().astype(np.float16)
            payload["tok_neck"] = emb[0][:, idx[:, 0], idx[:, 1]].T.numpy().astype(np.float16)
        np.savez(out_dir / f"{chip}.npz", **payload)
        times[chip] = t_enc
        if log_file is not None:
            el = time.time() - t_start
            eta = el / i * (len(todo) - i)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{rep}] {i}/{len(todo)} {chip} enc={t_enc:.1f}s  transcurrido={el/60:.1f}min  restante~{eta/60:.1f}min\n")
    return times


def load_embedding(chip: str, rep: str = "vh3") -> dict:
    z = np.load(cache_dir(rep) / f"{chip}.npz")
    return {
        "emb": torch.from_numpy(z["emb"])[None],
        "original_sizes": torch.from_numpy(z["original_sizes"]),
        "reshaped_input_sizes": torch.from_numpy(z["reshaped_input_sizes"]),
        "t_encoder_s": float(z["t_encoder_s"]),
    }


def load_probe_tokens(chips: list[str], rep: str = "vh3") -> dict:
    """Devuelve {capa: (X, y, chip_id)} con los tokens muestreados de cada chip."""
    out = {}
    keys = [f"tok_l{l}" for l in PROBE_LAYERS] + ["tok_neck"]
    for k in keys:
        Xs, ys, gs = [], [], []
        for ci, chip in enumerate(chips):
            z = np.load(cache_dir(rep) / f"{chip}.npz")
            if k not in z or len(z["tok_y"]) == 0:
                continue
            Xs.append(z[k].astype(np.float32))
            ys.append(z["tok_y"])
            gs.append(np.full(len(z["tok_y"]), ci))
        out[k] = (np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs))
    return out


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_np_default), encoding="utf-8")


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
