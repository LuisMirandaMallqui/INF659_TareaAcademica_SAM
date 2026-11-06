"""Hacia IoU 0.60, paso 3 (pre-registrado en docs/bitacora_hacia_060.md, seccion 2c, ANTES de ejecutar).

  3a  `unet_db3`              Linea base: U-Net desde cero con la misma entrada, chips y aumentos.
  3b  `lora_db3_otsu_tversky` El mejor del paso 2 con una sola diferencia: el termino Dice de la perdida
                              se reemplaza por Tversky (alfa 0.3 para FP, beta 0.7 para FN).

Igual que en el paso 2: hiperparametros fijados de antemano, sin seleccion, ultima epoca, umbral 0.5,
semillas 0, 1 y 2 en todo `train`, evaluacion en `valid` (individual, TTA D4, ensamble).
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from . import lora_sam as L
from .decoder_ft import masked_bce_dice
from .hacia060 import compare

OUT_DIR = config.RESULTS_DIR / "hacia_060" / "paso3"
MODEL_DIR = config.CACHE_DIR / "modelos" / "hacia_060_paso3"
BITACORA = config.REPO_ROOT / "docs" / "bitacora_hacia_060.md"
THRESHOLD = 0.5

TV_ALPHA, TV_BETA = 0.3, 0.7  # peso de FP y de FN en Tversky (Dice = 0.5 / 0.5)

UNET_BASE = 32
UNET_EPOCHS, UNET_WARMUP = 100, 2
UNET_LR, UNET_WD, UNET_BATCH = 1e-3, 1e-4, 8

# nombre -> definicion (orden de ejecucion)
EXPERIMENTS = {
    "unet_db3": {"tipo": "unet", "rep": "db3", "prompts": "sin", "paso": "3a",
                 "que": f"U-Net desde cero (base {UNET_BASE}), [VV, VH, VV−VH] con rango fijo, BCE + Dice"},
    "lora_db3_otsu_tversky": {"tipo": "lora", "rep": "db3", "prompts": "otsu", "paso": "3b",
                              "que": f"`lora_db3_otsu` con BCE + Tversky (α {TV_ALPHA}, β {TV_BETA}) en vez de BCE + Dice"},
}
PASO2_REF = ("db3", "otsu")  # referencia directa: lora_db3_otsu del paso 2


def seed_file(name, seed):
    return OUT_DIR / f"{name}_seed{seed}.json"


def config_file(name):
    return OUT_DIR / f"config_{name}.json"


def probs_file(name, seed, tta=False):
    return MODEL_DIR / f"{name}_seed{seed}_probs{'_tta' if tta else ''}_valid.npz"


# ----------------------------------------------------------------------------- perdida Tversky

def masked_bce_tversky(logits: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor,
                       alpha: float = TV_ALPHA, beta: float = TV_BETA) -> torch.Tensor:
    """Como `masked_bce_dice`, pero con Tversky en vez de Dice. Con alfa = beta = 0.5 es Dice."""
    bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    g = gt * valid
    tp = (p * g).sum(dim=(-2, -1))
    fp = (p * (1 - g)).sum(dim=(-2, -1))
    fn = ((1 - p) * g).sum(dim=(-2, -1))
    tv = 1 - (tp + 1) / (tp + alpha * fp + beta * fn + 1)
    return bce + tv.mean()


# ----------------------------------------------------------------------------- U-Net

class _Double(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """U-Net clasica (Ronneberger et al. 2015) con BatchNorm: 4 bajadas, base 32 canales (~7.8 M parametros)."""

    def __init__(self, cin: int = 3, base: int = UNET_BASE):
        super().__init__()
        ch = [base * 2 ** i for i in range(5)]
        self.enc = nn.ModuleList([_Double(cin, ch[0])] + [_Double(ch[i], ch[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([_Double(2 * ch[i], ch[i]) for i in reversed(range(4))])
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([skips.pop(), up(x)], 1))
        return self.out(x)[:, 0]


def unet_input(rgb_u8: torch.Tensor) -> torch.Tensor:
    return (rgb_u8.float() / 255.0 - 0.5) / 0.25


def train_unet(model, items, *, seed, epochs=UNET_EPOCHS, lr=UNET_LR, batch_size=UNET_BATCH,
               on_step=None, on_epoch=None) -> dict:
    rng = np.random.default_rng(seed)
    dev = next(model.parameters()).device  # los chips pueden estar en la CPU (conjuntos grandes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=UNET_WD)
    spe = math.ceil(len(items) / batch_size)
    total, warm = spe * epochs, spe * UNET_WARMUP
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    hist, t0 = [], time.time()
    model.train()
    for ep in range(epochs):
        t_ep = time.time()
        order = rng.permutation(len(items))
        losses, TP, FP, FN = [], 0, 0, 0
        for s in range(0, len(order), batch_size):
            batch = [items[i] for i in order[s:s + batch_size]]
            ks = rng.integers(0, 4, size=len(batch))
            fl = rng.integers(0, 2, size=len(batch))
            rgb = torch.stack([L.tf_img(it["rgb"], int(k), int(f)) for it, k, f in zip(batch, ks, fl)]).to(dev)
            gt = torch.stack([L.tf_img(it["gt"], int(k), int(f)) for it, k, f in zip(batch, ks, fl)]).to(dev)
            valid = torch.stack([L.tf_img(it["valid"], int(k), int(f)) for it, k, f in zip(batch, ks, fl)]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(unet_input(rgb)).float()
            loss = masked_bce_dice(logits, gt.float(), valid.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            with torch.no_grad():
                pred = (logits > 0) & valid
                TP += int((pred & gt).sum())
                FP += int((pred & ~gt).sum())
                FN += int((~pred & gt).sum())
            if on_step:
                on_step(len(batch), loss=losses[-1])
        hist.append({"epoca": ep + 1, "perdida_media": float(np.mean(losses)),
                     "iou_entrenamiento": TP / max(TP + FP + FN, 1), "lr_final": sched.get_last_lr()[0],
                     "seg_epoca": time.time() - t_ep})
        if on_epoch:
            on_epoch(ep + 1, hist[-1])
    model.eval()
    return {"historial": hist, "n_chips_entrenamiento": len(items), "minutos": (time.time() - t0) / 60,
            "params_entrenables": sum(p.numel() for p in model.parameters())}


@torch.no_grad()
def predict_unet(model, items, tta=False, batch_size=8, on_chip=None) -> list[np.ndarray]:
    model.eval()
    dev = next(model.parameters()).device
    tfs = L.D4 if tta else ((0, 0),)
    out = []
    for s in range(0, len(items), batch_size):
        b = items[s:s + batch_size]
        acc = None
        for k, f in tfs:
            rgb = torch.stack([L.tf_img(it["rgb"], k, f) for it in b]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(unet_input(rgb)).float()
            p = torch.sigmoid(L.inv_img(lg, k, f))
            acc = p if acc is None else acc + p
        acc /= len(tfs)
        out.extend(acc[j].half().cpu().numpy() for j in range(len(b)))
        if on_chip:
            on_chip(len(b))
    return out


# ----------------------------------------------------------------------------- referencias

def load_paso2_ref(valid: list[str]) -> dict | None:
    """Filas por chip de `lora_db3_otsu` (paso 2): por semilla y del ensamble."""
    rep, pr = PASO2_REF
    files = [L.seed_file(rep, pr, s) for s in config.SEEDS]
    if not all(f.exists() for f in files) or not L.config_file(rep, pr).exists():
        return None
    c = json.loads(L.config_file(rep, pr).read_text(encoding="utf-8"))
    if c["chips_valid"] != valid:
        return None
    return {"semillas": {s: json.loads(f.read_text(encoding="utf-8"))["valid"]["individual"]["por_chip"]
                         for s, f in zip(config.SEEDS, files)},
            "ensamble": c["ensamble"]["individual"]["por_chip"],
            "ensamble_tta": c["ensamble"]["tta"]["por_chip"]}


def comparisons(rows, seed, refs, paso2, regions, tta=False) -> dict:
    out = {"vs_otsu": compare(rows, refs["otsu"], regions)}
    if seed is not None:
        out["vs_h2"] = compare(rows, refs["h2"][seed], regions)
        if paso2:
            out["vs_paso2"] = compare(rows, paso2["semillas"][seed], regions)
    elif paso2:
        out["vs_paso2_ensamble"] = compare(rows, paso2["ensamble_tta" if tta else "ensamble"], regions)
    return out


# ----------------------------------------------------------------------------- bitacora

MARK_START = "<!-- RESULTADOS_PASO3:INICIO (se regenera solo; no editar a mano dentro) -->"
MARK_END = "<!-- RESULTADOS_PASO3:FIN -->"


def _partial(name) -> str:
    try:
        e = json.loads((OUT_DIR / "estado.json").read_text(encoding="utf-8"))
        if e.get("config_actual") == name and not e.get("terminado") and not e.get("interrumpido"):
            return f"⏳ En curso ({e.get('detalle', '')})"
    except Exception:
        pass
    return "⬜ Pendiente"


def _f(v):
    return "—" if v is None else f"{v:.3f}"


def results_markdown() -> str:
    lines = ["| Paso | Experimento | Qué es | Semillas | **IoU global valid** (individual) | Con TTA | Ensamble | "
             "Ensamble + TTA | IoU medio por chip | Precisión | Sensibilidad | Δ vs paso 2 `lora_db3_otsu` (misma semilla) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    n_done = 0
    for name, ex in EXPERIMENTS.items():
        seeds = [json.loads(seed_file(name, s).read_text(encoding="utf-8"))
                 for s in config.SEEDS if seed_file(name, s).exists()]
        if not seeds:
            lines.append(f"| {ex['paso']} | `{name}` | {ex['que']} | {_partial(name)} | | | | | | | | |")
            continue
        ind = [r["valid"]["individual"]["agregado"] for r in seeds]
        g = [a["global_iou"] for a in ind]
        gt = [r["valid"]["tta"]["agregado"]["global_iou"] for r in seeds]
        dp = [r["valid"]["individual"]["vs_paso2"]["bootstrap"]["global_iou"]["diff_b_menos_a"] for r in seeds
              if "vs_paso2" in r["valid"]["individual"]]
        ens = ens_t = "—"
        if config_file(name).exists():
            n_done += 1
            c = json.loads(config_file(name).read_text(encoding="utf-8"))
            ens = _f(c["ensamble"]["individual"]["agregado"]["global_iou"])
            ens_t = _f(c["ensamble"]["tta"]["agregado"]["global_iou"])
        estado = f"✅ {len(seeds)}/3" if len(seeds) == len(config.SEEDS) else f"⏳ {len(seeds)}/3"
        lines.append(f"| {ex['paso']} | `{name}` | {ex['que']} | {estado} | **{L._ms(g)}** | {L._ms(gt)} | {ens} | "
                     f"{ens_t} | {L._ms([a['mean_chip_iou'] for a in ind])} | {L._ms([a['precision'] for a in ind])} | "
                     f"{L._ms([a['recall'] for a in ind])} | {(f'{np.mean(dp):+.3f}' if dp else '—')} |")
    head = (f"**Avance del paso 3: {n_done}/{len(EXPERIMENTS)} experimentos terminados.** "
            + ("✅ Terminado." if n_done == len(EXPERIMENTS) else "⏳ En curso o pendiente.") + "\n\n")
    out = head + "\n".join(lines)
    out += ("\n\nReferencias en valid (86 chips): H2 **0.431 ± 0.038** · paso 2 `lora_db3_otsu` **0.601 ± 0.010** "
            "(TTA 0.611, ensamble 0.612, ensamble + TTA 0.615) · meta **0.60**. Métrica principal en negrita: media de "
            "3 semillas, modelo individual, sin TTA, umbral 0.5.")
    out += f"\n\n_Actualizado automáticamente: {time.strftime('%Y-%m-%d %H:%M')}_"
    return out


def update_bitacora() -> None:
    if not BITACORA.exists():
        return
    txt = BITACORA.read_text(encoding="utf-8")
    if MARK_START not in txt or MARK_END not in txt:
        return
    a = txt.index(MARK_START) + len(MARK_START)
    b = txt.index(MARK_END)
    try:
        block = results_markdown()
    except Exception as exc:
        block = f"(No se pudo regenerar la tabla: {exc!r}. Los JSON de results/hacia_060/paso3/ están completos.)"
    BITACORA.write_text(txt[:a] + "\n\n" + block + "\n\n" + txt[b:], encoding="utf-8")
