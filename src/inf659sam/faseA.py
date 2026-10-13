"""Fase A: diseccion instrumentada de SAM ViT-B.

- Parametros por bloque (12 bloques del ViT, cuello, codificador de prompts, decodificador).
- FLOPs MEDIDOS por bloque con torch.utils.flop_counter (contrastados con la formula analitica).
- Caracteristicas intermedias: PCA a 3 componentes de la salida de cada bloque.
- Campo receptivo efectivo (Luo et al., 2016): |d salida(token central) / d entrada| por capa.
- Perfilado de latencia y memoria del pipeline SAR completo, etapa por etapa.
"""
from __future__ import annotations

import re
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


# ----------------------------------------------------------------------------- parametros

def _n(module) -> int:
    return sum(p.numel() for p in module.parameters())


def param_table(model) -> list[dict]:
    enc, pe, dec = model.vision_encoder, model.prompt_encoder, model.mask_decoder
    gidx = set(model.config.vision_config.global_attn_indexes)
    rows = [{"bloque": "vision_encoder.patch_embed", "params": _n(enc.patch_embed)},
            {"bloque": "vision_encoder.pos_embed", "params": enc.pos_embed.numel()}]
    for i, layer in enumerate(enc.layers):
        tipo = "global" if i in gidx else "ventana 14x14"
        rows.append({"bloque": f"vision_encoder.layers.{i} ({tipo})", "params": _n(layer)})
    rows.append({"bloque": "vision_encoder.neck", "params": _n(enc.neck)})
    rows.append({"bloque": "prompt_encoder (total)", "params": _n(pe)})
    rows.append({"bloque": "mask_decoder.transformer (two-way, 2 capas)", "params": _n(dec.transformer)})
    up = _n(dec.upscale_conv1) + _n(dec.upscale_conv2) + _n(dec.upscale_layer_norm)
    rows.append({"bloque": "mask_decoder.upscaling (2 conv transpuestas)", "params": up})
    rows.append({"bloque": "mask_decoder.output_hypernetworks_mlps (4)", "params": _n(dec.output_hypernetworks_mlps)})
    rows.append({"bloque": "mask_decoder.iou_prediction_head", "params": _n(dec.iou_prediction_head)})
    rows.append({"bloque": "mask_decoder.tokens (iou + 4 mascara)", "params": _n(dec.iou_token) + _n(dec.mask_tokens)})
    total = _n(model)
    for r in rows:
        r["pct_total"] = 100 * r["params"] / total
    return rows


# ----------------------------------------------------------------------------- FLOPs

def measure_flops(model, pixel_values: torch.Tensor, decode_fn=None) -> dict:
    """FLOPs contados por PyTorch (2 FLOPs por multiplicacion-suma) por submodulo."""
    from torch.utils.flop_counter import FlopCounterMode

    enc = model.vision_encoder
    pixel_values = pixel_values.to(next(model.parameters()).device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        enc(pixel_values)
    counts = fc.get_flop_counts()
    per_layer, other = {}, {}
    for name, ops in counts.items():
        total = sum(ops.values())
        m = re.search(r"layers\.(\d+)$", name)
        if m:
            per_layer[int(m.group(1))] = total
        elif name.endswith("patch_embed") or name.endswith("neck"):
            other[name.split(".")[-1]] = total
    enc_total = sum(sum(v.values()) for k, v in counts.items() if k == "Global")
    out = {"encoder_total": int(enc_total), "per_layer": {k: int(v) for k, v in sorted(per_layer.items())},
           "patch_embed": int(other.get("patch_embed", 0)), "neck": int(other.get("neck", 0)),
           "raw_keys_sample": list(counts.keys())[:8]}
    if decode_fn is not None:
        with torch.no_grad(), FlopCounterMode(display=False) as fc2:
            decode_fn()
        out["prompt_mas_decoder_total"] = int(sum(sum(v.values()) for k, v in fc2.get_flop_counts().items()
                                                   if k == "Global"))
    return out


def analytical_flops(vc) -> dict:
    """Formula del equipo (fase_a_diseccion.py), recalculada desde la configuracion."""
    D, L, P, S, W = vc.hidden_size, vc.num_hidden_layers, vc.patch_size, vc.image_size, vc.window_size
    N = (S // P) ** 2
    grid = S // P
    pad = (W - grid % W) % W
    nwin = ((grid + pad) // W) ** 2
    tok = 24 * N * D * D
    glob = tok + 4 * N * N * D
    win = tok + 4 * nwin * (W * W) ** 2 * D
    patch = N * D * (P * P * 3) * 2
    ng = len(vc.global_attn_indexes)
    return {"global_block": glob, "window_block": win, "patch_embed": patch,
            "total": patch + ng * glob + (L - ng) * win}


# ----------------------------------------------------------------------------- caracteristicas

@torch.no_grad()
def encoder_states(model, pixel_values: torch.Tensor, layers=range(12)) -> tuple[torch.Tensor, dict]:
    """Forward manual del codificador (identico a SamVisionEncoder.forward) guardando estados (en CPU)."""
    enc = model.vision_encoder
    h = enc.patch_embed(pixel_values.to(next(model.parameters()).device))
    if enc.pos_embed is not None:
        h = h + enc.pos_embed
    states = {}
    layers = set(layers)
    for i, layer in enumerate(enc.layers):
        h = layer(h)
        if i in layers:
            states[i] = h[0].float().cpu()
    emb = enc.neck(h)
    return emb.float().cpu(), states


def pca_rgb(feat: torch.Tensor | np.ndarray, basis=None) -> tuple[np.ndarray, tuple]:
    """(H, W, C) -> (H, W, 3) en [0, 1] proyectando en las 3 primeras componentes principales."""
    f = torch.as_tensor(feat, dtype=torch.float32)
    H, W, C = f.shape
    X = f.reshape(-1, C)
    if basis is None:
        mu = X.mean(0, keepdim=True)
        _, _, V = torch.pca_lowrank(X - mu, q=3, center=False, niter=4)
        basis = (mu, V[:, :3])
    mu, V = basis
    Y = (X - mu) @ V
    lo = torch.quantile(Y, 0.01, dim=0)
    hi = torch.quantile(Y, 0.99, dim=0)
    Y = ((Y - lo) / (hi - lo + 1e-8)).clamp(0, 1)
    return Y.reshape(H, W, 3).numpy(), basis


# ----------------------------------------------------------------------------- campo receptivo efectivo

def effective_receptive_field(model, pixel_values: torch.Tensor, layer_idx: int,
                              token: tuple[int, int] = (32, 32)) -> np.ndarray:
    """|d sum_c h_l[token, c] / d x| sumado sobre RGB -> mapa 1024x1024 (Luo et al., NeurIPS 2016).

    Usa checkpointing por bloque para que la memoria pico sea la de un solo bloque.
    """
    enc = model.vision_encoder
    x = pixel_values.clone().to(next(model.parameters()).device).requires_grad_(True)
    with torch.enable_grad():
        h = enc.patch_embed(x)
        if enc.pos_embed is not None:
            h = h + enc.pos_embed
        for i in range(layer_idx + 1):
            h = checkpoint(enc.layers[i], h, use_reentrant=False)
        target = h[0, token[0], token[1], :].sum()
        target.backward()
    g = x.grad[0].abs().sum(0)
    return g.float().cpu().numpy()


def erf_extent(erf: np.ndarray, mass: float = 0.95) -> dict:
    """Area (en pixeles de entrada) que concentra `mass` de la magnitud del gradiente."""
    v = np.sort(erf.ravel())[::-1]
    c = np.cumsum(v) / v.sum()
    n = int(np.searchsorted(c, mass) + 1)
    ys, xs = np.nonzero(erf > 0)
    bbox = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())) if len(ys) else None
    return {"pixeles_95pct_masa": n, "lado_equivalente_px": float(np.sqrt(n)),
            "bbox_gradiente_no_nulo_yx": bbox}


# ----------------------------------------------------------------------------- perfilado

class StageTimer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        timer = self

        class _Ctx:
            def __enter__(self_):
                self_.t0 = time.perf_counter()

            def __exit__(self_, *exc):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # el trabajo en GPU es asincrono
                timer.t.setdefault(name, []).append((time.perf_counter() - self_.t0) * 1000)

        return _Ctx()

    def summary(self) -> dict:
        return {k: {"media_ms": float(np.mean(v)), "desv_ms": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                    "n": len(v)} for k, v in self.t.items()}


def rss_mib() -> dict:
    import psutil

    mi = psutil.Process().memory_info()
    out = {"rss_mib": mi.rss / 2**20}
    if hasattr(mi, "peak_wset"):
        out["pico_working_set_mib"] = mi.peak_wset / 2**20
    return out
