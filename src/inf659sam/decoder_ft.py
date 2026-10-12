"""H2: ajuste exclusivo del decodificador de mascaras con ambos codificadores congelados.

Protocolo fijado ANTES de ejecutar (docs/protocolo_preregistrado.md):
  - Entrenamiento en la particion train; barrido de tasa de aprendizaje en una subparticion
    interna (train-dev) separada de train; evaluacion en valid; test queda sin tocar.
  - Se congelan el codificador de imagen y el de prompts (incluida la codificacion posicional
    compartida). Solo se actualizan los 4 058 340 parametros del mask decoder.
  - Salida de una sola mascara (multimask_output=False), prompts automaticos de Otsu.
  - Perdida: BCE + Dice sobre logits re-escalados a 512x512, ignorando pixeles sin etiqueta.
  - AdamW, weight decay 0.01, lote 4, 12 epocas, sin early stopping (se usa la ultima epoca).
  - Semillas 0, 1, 2 (controlan orden de los datos y muestreo de prompts).

Control (H2-c): sonda lineal 1x1 sobre el embedding congelado (257 parametros), mismo
presupuesto, para saber cuanta informacion de agua ya esta en las caracteristicas.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import config, data
from .baseline import auto_prompts, otsu_masks, sample_random_points
from .cache import load_embedding

LOSS_SIZE = 512


def load_training_set(chips: list[str], rep: str = "vh3") -> list[dict]:
    """Embeddings + etiquetas + mascaras Otsu en memoria (aprox. 4.3 MB por chip)."""
    items = []
    for chip in chips:
        d = data.load_chip(chip)
        om = otsu_masks(d["vv"], d["vh"])
        c = load_embedding(chip, rep)
        items.append({
            "chip": chip,
            "emb": c["emb"][0],
            "gt": torch.from_numpy((d["label"] == 1).astype(np.float32)),
            "valid": torch.from_numpy((d["label"] != -1).astype(np.float32)),
            "opened": om["opened"],
            "neg_region": ~om["opened"] & om["sensor_valid"],
            "osz": c["original_sizes"][0],
            "rsz": c["reshaped_input_sizes"][0],
        })
    return items


def masked_bce_dice(logits: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    g = gt * valid
    inter = (p * g).sum(dim=(-2, -1))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(-2, -1)) + g.sum(dim=(-2, -1)) + 1)
    return bce + dice.mean()


def _prompt_batch(batch: list[dict], rng: np.random.Generator, n_pos: int, n_neg: int):
    pts, lbs = [], []
    for it in batch:
        pos = sample_random_points(it["opened"], n_pos, rng)
        neg = sample_random_points(it["neg_region"], n_neg, rng)
        p = pos + neg
        l = [1] * len(pos) + [0] * len(neg)
        sx = float(it["rsz"][1]) / float(it["osz"][1])
        sy = float(it["rsz"][0]) / float(it["osz"][0])
        p = [[x * sx, y * sy] for x, y in p]
        while len(p) < n_pos + n_neg:  # relleno ignorado por el codificador de prompts
            p.append([0.0, 0.0])
            l.append(-10)
        pts.append([p])
        lbs.append([l])
    return torch.tensor(pts, dtype=torch.float32), torch.tensor(lbs, dtype=torch.int64)


def train_decoder(sam, items: list[dict], seed: int, lr: float = 1e-4, epochs: int = 12, batch_size: int = 4,
                  weight_decay: float = 0.01, log_file: Path | None = None, tag: str = "",
                  callback=None) -> dict:
    """Ajusta SOLO sam.model.mask_decoder (in place). Devuelve historial de perdida."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = sam.model
    dev = sam.device
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.mask_decoder.parameters():
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.mask_decoder.parameters(), lr=lr, weight_decay=weight_decay)
    usable = [it for it in items if it["opened"].any()]
    hist = []
    t0 = time.time()
    model.mask_decoder.train()
    for ep in range(epochs):
        order = rng.permutation(len(usable))
        losses = []
        for s in range(0, len(order), batch_size):
            batch = [usable[i] for i in order[s:s + batch_size]]
            emb = torch.stack([it["emb"] for it in batch]).to(dev)
            pts, lbs = _prompt_batch(batch, rng, config.N_POS, config.N_NEG)
            out = model(image_embeddings=emb, input_points=pts.to(dev), input_labels=lbs.to(dev), multimask_output=False)
            logits = F.interpolate(out.pred_masks[:, 0], size=(LOSS_SIZE, LOSS_SIZE), mode="bilinear",
                                   align_corners=False)[:, 0]
            gt = torch.stack([it["gt"] for it in batch]).to(dev)
            valid = torch.stack([it["valid"] for it in batch]).to(dev)
            loss = masked_bce_dice(logits, gt, valid)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        hist.append({"epoca": ep + 1, "perdida_media": float(np.mean(losses)), "min": (time.time() - t0) / 60})
        if callback is not None:  # p. ej., evaluar en train-dev al final de cada epoca
            model.mask_decoder.eval()
            hist[-1].update(callback(ep + 1) or {})
            model.mask_decoder.train()
        if log_file is not None:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[H2{tag} seed={seed} lr={lr}] epoca {ep+1}/{epochs} perdida={np.mean(losses):.4f} "
                        f"t={(time.time()-t0)/60:.1f}min\n")
    model.mask_decoder.eval()
    for p in model.mask_decoder.parameters():
        p.requires_grad_(False)
    return {"historial": hist, "n_chips_entrenamiento": len(usable), "params_entrenables": n_train,
            "minutos": (time.time() - t0) / 60}


class LinearProbe(torch.nn.Module):
    """Sonda 1x1 sobre el embedding (256 -> 1) re-escalada a la resolucion del chip."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, emb):
        return F.interpolate(self.conv(emb), size=(LOSS_SIZE, LOSS_SIZE), mode="bilinear", align_corners=False)[:, 0]


def train_probe(items: list[dict], seed: int, lr: float = 1e-3, epochs: int = 12, batch_size: int = 4,
                log_file: Path | None = None, device: str | None = None) -> tuple[LinearProbe, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device or config.get_device()
    probe = LinearProbe().to(dev)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        order = rng.permutation(len(items))
        losses = []
        for s in range(0, len(order), batch_size):
            batch = [items[i] for i in order[s:s + batch_size]]
            emb = torch.stack([it["emb"] for it in batch]).to(dev)
            logits = probe(emb)
            loss = masked_bce_dice(logits, torch.stack([it["gt"] for it in batch]).to(dev),
                                   torch.stack([it["valid"] for it in batch]).to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        hist.append({"epoca": ep + 1, "perdida_media": float(np.mean(losses))})
        if log_file is not None:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[sonda seed={seed}] epoca {ep+1}/{epochs} perdida={np.mean(losses):.4f}\n")
    probe.eval()
    return probe, {"historial": hist, "minutos": (time.time() - t0) / 60}


@torch.no_grad()
def probe_predict(probe: LinearProbe, emb: torch.Tensor) -> np.ndarray:
    dev = next(probe.parameters()).device
    return (probe(emb.to(dev))[0] > 0).cpu().numpy()


def decoder_state(sam) -> dict:
    return copy.deepcopy(sam.model.mask_decoder.state_dict())


def load_decoder_state(sam, state: dict) -> None:
    sam.model.mask_decoder.load_state_dict(state)
    sam.model.mask_decoder.eval()


def save_history(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
