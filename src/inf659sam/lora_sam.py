"""Hacia IoU 0.60, paso 2 (H5): LoRA en el codificador de imagen de SAM + decodificador ajustado,
con aumentos D4 (rotaciones de 90 grados y espejado) al vuelo.

Protocolo pre-registrado en docs/bitacora_hacia_060.md (seccion 2b) ANTES de ejecutar. Resumen:
  - LoRA de rango 8 (alfa 16) en la proyeccion qkv de los 12 bloques del codificador, mas el
    `mask_decoder` completo. Codificador de prompts, cuello y resto del codificador congelados.
  - Hiperparametros fijados a priori, SIN seleccion: 15 epocas (1 de calentamiento + coseno),
    lr 1e-4, lote 4, AdamW wd 0.01, BCE + Dice, umbral 0.5, se usa la ultima epoca.
  - Entrenamiento en todo `train` (250 chips) con semillas 0, 1 y 2; evaluacion en `valid`.
  - Metrica principal: media de las 3 semillas, modelo individual, sin TTA. Secundarias, fijadas
    de antemano: con TTA D4 (8 transformaciones) y ensamble de las 3 semillas (con y sin TTA).

Toda la entrada se prepara en la GPU (uint8 -> 1024 bilineal -> normalizacion ImageNet), igual en
entrenamiento y en evaluacion. El post-procesamiento de la mascara es el de SamProcessor.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from . import config, data
from .baseline import auto_prompts, otsu_masks
from .decoder_ft import masked_bce_dice
from .hacia060 import _postprocess, _row, compare
from .sam_utils import SamWrapper

OUT_DIR = config.RESULTS_DIR / "hacia_060" / "paso2"
MODEL_DIR = config.CACHE_DIR / "modelos" / "hacia_060_paso2"
BITACORA = config.REPO_ROOT / "docs" / "bitacora_hacia_060.md"

LORA_R, LORA_ALPHA = 8, 16
EPOCHS, WARMUP_EPOCHS = 15, 1
LR, WEIGHT_DECAY, BATCH = 1e-4, 0.01, 4
THRESHOLD = 0.5
N_BOOT = 5000
SIZE, INPUT = 512, 1024
D4 = tuple((k, f) for f in (0, 1) for k in range(4))  # (rotaciones de 90 grados, espejado)

CONFIGS = (("vhdb3", "otsu"), ("db3", "otsu"), ("vhdb3", "sin"))
REP_INFO = {"vh3": "VH x3, percentiles por chip", "vhdb3": "VH x3, rango fijo en dB",
            "db3": "[VV, VH, VV-VH], rangos fijos en dB"}
PROMPT_INFO = {"otsu": "prompts Otsu 3+2", "sin": "sin prompts"}

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def config_name(rep: str, prompts: str) -> str:
    return f"lora_{rep}_{prompts}"


def seed_file(rep, prompts, seed) -> Path:
    return OUT_DIR / f"{config_name(rep, prompts)}_seed{seed}.json"


def config_file(rep, prompts) -> Path:
    return OUT_DIR / f"config_{config_name(rep, prompts)}.json"


def probs_file(rep, prompts, seed) -> Path:
    return MODEL_DIR / f"{config_name(rep, prompts)}_seed{seed}_probs_valid.npz"


# ----------------------------------------------------------------------------- LoRA

class LoRALinear(nn.Module):
    """y = W x + (alfa / r) * B A x, con W congelada y B inicializada en cero (arranca igual a SAM)."""

    def __init__(self, base: nn.Linear, r: int = LORA_R, alpha: int = LORA_ALPHA):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale


def build_model(device: str, lora: bool = True):
    """SAM con atencion SDPA, LoRA en qkv y checkpointing de gradiente (sin reentrada)."""
    from transformers import SamModel

    model = SamModel.from_pretrained(str(config.MODEL_DIR), attn_implementation="sdpa")
    for p in model.parameters():
        p.requires_grad_(False)
    if lora:
        for layer in model.vision_encoder.layers:
            layer.attn.qkv = LoRALinear(layer.attn.qkv)
    model.to(device)
    # use_reentrant=False es obligatorio: con reentrada los gradientes no llegan a LoRA
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def trainable_parameters(model) -> list:
    ps = [p for n, p in model.named_parameters() if "lora_" in n]
    ps += list(model.mask_decoder.parameters())
    return ps


def trained_state(model) -> dict:
    return {n: p.detach().cpu().clone().contiguous() for n, p in model.named_parameters()
            if "lora_" in n or n.startswith("mask_decoder.")}


# ----------------------------------------------------------------------------- datos y D4

def _candidates(mask: np.ndarray, min_dist: float = 2.0) -> np.ndarray:
    """(x, y) donde `sample_random_points` puede muestrear: interior a >= 2 px del borde."""
    if not mask.any():
        return np.zeros((0, 2), np.int16)
    dist = distance_transform_edt(mask)
    ys, xs = np.nonzero(dist >= min_dist)
    if len(ys) == 0:
        ys, xs = np.nonzero(mask)
    return np.stack([xs, ys], 1).astype(np.int16)


def prepare_chips(chips: list[str], rep: str, device: str, on_chip=None) -> list[dict]:
    """Imagen uint8 de 3 canales, etiqueta y prompts de cada chip, ya en la GPU."""
    items = []
    for chip in chips:
        d = data.load_chip(chip)
        rgb = SamWrapper.sar_to_rgb(d["vv"], d["vh"], rep)
        label = d["label"]
        lv = label != -1
        om = otsu_masks(d["vv"], d["vh"])
        opened, sv = om["opened"], om["sensor_valid"]
        pts, lbs, npos = auto_prompts(opened, sv, config.N_POS, config.N_NEG)
        items.append({
            "chip": chip,
            "rgb": torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).to(device),
            "gt": torch.from_numpy((label == 1) & lv).to(device),
            "valid": torch.from_numpy(lv).to(device),
            "pos_c": _candidates(opened), "neg_c": _candidates(~opened & sv),
            "eval_pts": [list(p) for p in pts], "eval_lbs": list(lbs), "npos": npos,
        })
        if on_chip:
            on_chip()
    return items


def tf_img(x: torch.Tensor, k: int, flip: int) -> torch.Tensor:
    if flip:
        x = torch.flip(x, dims=[-1])
    return torch.rot90(x, k, dims=[-2, -1]) if k else x


def inv_img(x: torch.Tensor, k: int, flip: int) -> torch.Tensor:
    if k:
        x = torch.rot90(x, -k, dims=[-2, -1])
    return torch.flip(x, dims=[-1]) if flip else x


def tf_pts(pts, k: int, flip: int, n: int = SIZE) -> list[list[float]]:
    """Mismo movimiento que `tf_img` para puntos (x = columna, y = fila)."""
    out = []
    for x, y in pts:
        if flip:
            x = n - 1 - x
        for _ in range(k):  # torch.rot90 con k=1: (fila, col) -> (n-1-col, fila)
            x, y = y, n - 1 - x
        out.append([x, y])
    return out


def pixel_values(rgb_u8: torch.Tensor) -> torch.Tensor:
    """(B, 3, 512, 512) uint8 -> (B, 3, 1024, 1024) normalizado como SamProcessor."""
    x = rgb_u8.float() / 255.0
    x = F.interpolate(x, (INPUT, INPUT), mode="bilinear", align_corners=False)
    return (x - _MEAN.to(x.device)) / _STD.to(x.device)


def _sample(cands: np.ndarray, n: int, rng) -> list[list[float]]:
    if len(cands) == 0 or n <= 0:
        return []
    idx = rng.choice(len(cands), size=min(n, len(cands)), replace=False)
    return [[float(cands[i, 0]), float(cands[i, 1])] for i in idx]


# ----------------------------------------------------------------------------- entrenamiento

def train(model, items: list[dict], *, seed: int, prompts: str, epochs: int = EPOCHS, lr: float = LR,
          batch_size: int = BATCH, on_step=None, on_epoch=None, loss_fn=None) -> dict:
    """`loss_fn(logits, gt, valid)`; por defecto BCE + Dice (paso 2). El paso 3 pasa BCE + Tversky."""
    loss_fn = loss_fn or masked_bce_dice
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    dev = next(model.parameters()).device
    params = trainable_parameters(model)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    usable = items if prompts == "sin" else [it for it in items if len(it["pos_c"])]
    spe = math.ceil(len(usable) / batch_size)
    total, warm = spe * epochs, spe * WARMUP_EPOCHS
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scale = INPUT / SIZE
    hist, t0 = [], time.time()
    model.train()
    for ep in range(epochs):
        t_ep = time.time()
        order = rng.permutation(len(usable))
        losses, TP, FP, FN = [], 0, 0, 0
        for s in range(0, len(order), batch_size):
            batch = [usable[i] for i in order[s:s + batch_size]]
            ks = rng.integers(0, 4, size=len(batch))
            fl = rng.integers(0, 2, size=len(batch))
            rgb = torch.stack([tf_img(it["rgb"], int(k), int(f)) for it, k, f in zip(batch, ks, fl)])
            gt = torch.stack([tf_img(it["gt"], int(k), int(f)) for it, k, f in zip(batch, ks, fl)])
            valid = torch.stack([tf_img(it["valid"], int(k), int(f)) for it, k, f in zip(batch, ks, fl)])
            kw = {}
            if prompts == "otsu":
                pts_b, lbs_b = [], []
                for it, k, f in zip(batch, ks, fl):
                    pos = _sample(it["pos_c"], config.N_POS, rng)
                    neg = _sample(it["neg_c"], config.N_NEG, rng)
                    p = [[x * scale, y * scale] for x, y in tf_pts(pos + neg, int(k), int(f))]
                    lab = [1] * len(pos) + [0] * len(neg)
                    while len(p) < config.N_POS + config.N_NEG:  # relleno: embedding cero
                        p.append([0.0, 0.0])
                        lab.append(-10)
                    pts_b.append([p])
                    lbs_b.append([lab])
                kw = {"input_points": torch.tensor(pts_b, dtype=torch.float32, device=dev),
                      "input_labels": torch.tensor(lbs_b, dtype=torch.int64, device=dev)}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(pixel_values=pixel_values(rgb), multimask_output=False, **kw)
            logits = F.interpolate(out.pred_masks[:, 0].float(), size=(SIZE, SIZE), mode="bilinear",
                                   align_corners=False)[:, 0]
            loss = loss_fn(logits, gt.float(), valid.float())
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
    for p in params:
        p.requires_grad_(False)
    return {"historial": hist, "n_chips_entrenamiento": len(usable), "minutos": (time.time() - t0) / 60,
            "params_entrenables": sum(p.numel() for p in params)}


# ----------------------------------------------------------------------------- evaluacion

@torch.no_grad()
def predict_probs(model, items: list[dict], prompts: str, tta: bool = False, batch_size: int = 4,
                  on_chip=None) -> list[np.ndarray | None]:
    """Probabilidad de agua por pixel (512x512, float16). None = sin prompt positivo (mascara vacia)."""
    model.eval()
    dev = next(model.parameters()).device
    tfs = D4 if tta else ((0, 0),)
    scale = INPUT / SIZE
    out_probs: list = [None] * len(items)
    groups: dict = {}
    for i, it in enumerate(items):
        key = "sin" if prompts == "sin" else (None if it["npos"] == 0 else len(it["eval_pts"]))
        groups.setdefault(key, []).append(i)
    for key, idxs in groups.items():
        for s in range(0, len(idxs), batch_size):
            b = idxs[s:s + batch_size]
            if key is None:
                if on_chip:
                    on_chip(len(b))
                continue
            acc = torch.zeros(len(b), SIZE, SIZE, device=dev)
            for k, f in tfs:
                rgb = torch.stack([tf_img(items[i]["rgb"], k, f) for i in b])
                kw = {}
                if key != "sin":
                    pts = [[[[x * scale, y * scale] for x, y in tf_pts(items[i]["eval_pts"], k, f)]] for i in b]
                    kw = {"input_points": torch.tensor(pts, dtype=torch.float32, device=dev),
                          "input_labels": torch.tensor([[items[i]["eval_lbs"]] for i in b], dtype=torch.int64,
                                                       device=dev)}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = model(pixel_values=pixel_values(rgb), multimask_output=False, **kw)
                logit = _postprocess(o.pred_masks[:, 0].float(), (INPUT, INPUT), (SIZE, SIZE))[:, 0]
                acc += torch.sigmoid(inv_img(logit, k, f))
            acc /= len(tfs)
            for j, i in enumerate(b):
                out_probs[i] = acc[j].half().cpu().numpy()
            if on_chip:
                on_chip(len(b))
    return out_probs


def rows_from_probs(probs: list, items: list[dict], thr: float = THRESHOLD) -> list[dict]:
    rows = []
    for p, it in zip(probs, items):
        gt, valid = it["gt"].cpu().numpy(), it["valid"].cpu().numpy()
        n_gt = int(gt.sum())
        if p is None:
            rows.append(_row(0, 0, n_gt))
            continue
        pred = (p.astype(np.float32) > thr) & valid
        tp = int((pred & gt).sum())
        rows.append(_row(tp, int((pred & ~gt).sum()), n_gt - tp))
    return rows


def ensemble(prob_lists: list[list]) -> list:
    out = []
    for ps in zip(*prob_lists):
        if all(p is None for p in ps):
            out.append(None)
        else:
            out.append(np.mean([np.zeros((SIZE, SIZE), np.float32) if p is None else p.astype(np.float32)
                                for p in ps], axis=0).astype(np.float16))
    return out


def save_probs(path: Path, probs: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"p{i}": (np.zeros(0, np.float16) if p is None else p) for i, p in enumerate(probs)})


def load_probs(path: Path, n: int) -> list:
    z = np.load(path)
    return [None if z[f"p{i}"].size == 0 else z[f"p{i}"] for i in range(n)]


# ----------------------------------------------------------------------------- referencias y bitacora

def load_step1_best(valid: list[str]) -> dict | None:
    f = config.RESULTS_DIR / "hacia_060" / "config_vhdb3_otsu.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text(encoding="utf-8"))
    if r["chips_valid"] != valid:
        return None
    return {int(k): v for k, v in r["final"]["por_chip"].items()}


def comparisons(rows, seed, refs, step1, regions) -> dict:
    """Modelo individual: contra la misma semilla de H2 y del paso 1. Ensamble (seed=None): contra
    cada una de las 3 semillas de cada referencia."""
    out = {"vs_otsu": compare(rows, refs["otsu"], regions)}
    if seed is not None:
        out["vs_h2"] = compare(rows, refs["h2"][seed], regions)
        if step1:
            out["vs_paso1"] = compare(rows, step1[seed], regions)
    else:
        out["vs_h2_por_semilla"] = {str(s): compare(rows, refs["h2"][s], regions) for s in config.SEEDS}
        if step1:
            out["vs_paso1_por_semilla"] = {str(s): compare(rows, step1[s], regions) for s in config.SEEDS}
    return out


MARK_START = "<!-- RESULTADOS_PASO2:INICIO (se regenera solo; no editar a mano dentro) -->"
MARK_END = "<!-- RESULTADOS_PASO2:FIN -->"


def _ms(vals):
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}"


def results_markdown() -> str:
    lines = ["| Configuración | Entrada | Prompts | Semillas listas | **IoU global valid** (modelo individual) | "
             "Con TTA D4 | Ensamble 3 semillas | Ensamble + TTA | IoU medio por chip | Δ vs H2 | Δ vs paso 1 (`vhdb3_otsu`) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    best = (None, -1)
    for rep, prompts in CONFIGS:
        name = config_name(rep, prompts)
        seeds = [json.loads(seed_file(rep, prompts, s).read_text(encoding="utf-8"))
                 for s in config.SEEDS if seed_file(rep, prompts, s).exists()]
        if not seeds:
            st = _partial(name)
            lines.append(f"| `{name}` | {REP_INFO[rep]} | {PROMPT_INFO[prompts]} | {st} | | | | | | | |")
            continue
        g = [r["valid"]["individual"]["agregado"]["global_iou"] for r in seeds]
        gt = [r["valid"]["tta"]["agregado"]["global_iou"] for r in seeds]
        m = [r["valid"]["individual"]["agregado"]["mean_chip_iou"] for r in seeds]
        dh = [r["valid"]["individual"]["vs_h2"]["bootstrap"]["global_iou"]["diff_b_menos_a"] for r in seeds
              if "vs_h2" in r["valid"]["individual"]]
        dp = [r["valid"]["individual"]["vs_paso1"]["bootstrap"]["global_iou"]["diff_b_menos_a"] for r in seeds
              if "vs_paso1" in r["valid"]["individual"]]
        ens = ens_t = "—"
        if config_file(rep, prompts).exists():
            c = json.loads(config_file(rep, prompts).read_text(encoding="utf-8"))
            ens = f"{c['ensamble']['individual']['agregado']['global_iou']:.3f}"
            ens_t = f"{c['ensamble']['tta']['agregado']['global_iou']:.3f}"
        estado = f"✅ {len(seeds)}/3" if len(seeds) == len(config.SEEDS) else f"⏳ {len(seeds)}/3"
        lines.append(f"| `{name}` | {REP_INFO[rep]} | {PROMPT_INFO[prompts]} | {estado} | **{_ms(g)}** | {_ms(gt)} | "
                     f"{ens} | {ens_t} | {_ms(m)} | {(f'{np.mean(dh):+.3f}' if dh else '—')} | "
                     f"{(f'{np.mean(dp):+.3f}' if dp else '—')} |")
        if len(seeds) == len(config.SEEDS) and np.mean(g) > best[1]:
            best = (name, float(np.mean(g)))
    n_done = sum(config_file(r, p).exists() for r, p in CONFIGS)
    head = (f"**Avance del paso 2: {n_done}/{len(CONFIGS)} configuraciones terminadas.** "
            + ("✅ Paso 2 terminado." if n_done == len(CONFIGS) else "⏳ En curso o pendiente.") + "\n\n")
    out = head + "\n".join(lines)
    out += ("\n\nReferencias en valid (86 chips): Otsu **0.256** · H2 **0.431 ± 0.038** · paso 1 `vhdb3_otsu` "
            "**0.472 ± 0.017** · meta **0.60**. La métrica principal es la columna en negrita (media de 3 semillas, "
            "modelo individual, sin TTA). Las demás columnas son secundarias, fijadas de antemano.")
    if best[0]:
        crosses = "✅ **supera 0.60**" if best[1] >= 0.60 else f"todavía a {0.60 - best[1]:.3f} de 0.60"
        out += f"\n\nMejor configuración completa (métrica principal): `{best[0]}` = {best[1]:.3f} → {crosses}."
    out += f"\n\n_Actualizado automáticamente: {time.strftime('%Y-%m-%d %H:%M')}_"
    return out


def _partial(name) -> str:
    try:
        e = json.loads((OUT_DIR / "estado.json").read_text(encoding="utf-8"))
        if e.get("config_actual") == name and not e.get("terminado") and not e.get("interrumpido"):
            return f"⏳ En curso ({e.get('detalle', '')})"
    except Exception:
        pass
    return "⬜ Pendiente"


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
        block = f"(No se pudo regenerar la tabla: {exc!r}. Los JSON de results/hacia_060/paso2/ están completos.)"
    BITACORA.write_text(txt[:a] + "\n\n" + block + "\n\n" + txt[b:], encoding="utf-8")
