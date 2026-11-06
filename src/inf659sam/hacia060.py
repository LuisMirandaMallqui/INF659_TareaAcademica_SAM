"""Hacia IoU 0.60, paso 1 (H4): entrada SAR con rangos fijos en dB, canales VV/VH, entrenamiento
del decodificador con programa de lr coseno, seleccion de epoca y umbral en train-dev, y una
variante sin prompts.

El protocolo se registro ANTES de ejecutar en docs/bitacora_hacia_060.md (seccion "Pre-registro H4").
Resumen:
  - Rangos fijos: percentiles 1 y 99 de los pixeles de `train` (VV, VH y VV-VH), un solo rango
    para todos los chips. Valid no interviene.
  - Configuraciones: representacion {db3, vhdb3, vh3} x prompts {otsu, sin}.
  - Por configuracion: barrido de lr {3e-5, 1e-4, 3e-4} con semilla 0 entrenando en train-core y
    midiendo en train-dev al final de cada epoca (20 epocas, 1 de calentamiento y coseno). Se fija
    la pareja (lr, epoca) con mejor IoU global en train-dev y luego el umbral de probabilidad, tambien
    en train-dev. Recien entonces se entrena con semillas 0, 1 y 2 en todo train (mismo programa,
    detenido en la epoca elegida) y se evalua en valid.
  - La evaluacion reproduce exactamente `experiments.evaluate_chip(..., "sam_single")`: mismos
    prompts deterministas v2.1, mismo post-procesamiento de SamProcessor y mismos pixeles.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import config, data, stats
from .baseline import auto_prompts, otsu_masks
from .cache import load_embedding
from .decoder_ft import LOSS_SIZE, _prompt_batch, masked_bce_dice

OUT_DIR = config.RESULTS_DIR / "hacia_060"
RANGES_FILE = OUT_DIR / "rangos_db.json"
MODEL_DIR = config.CACHE_DIR / "modelos" / "hacia_060"
BITACORA = config.REPO_ROOT / "docs" / "bitacora_hacia_060.md"

PCT = (1.0, 99.0)
LRS = (3e-5, 1e-4, 3e-4)
EPOCHS = 20
WARMUP_EPOCHS = 1
BATCH = 4
WEIGHT_DECAY = 0.01
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
N_BOOT = 5000

# orden de ejecucion: primero lo que mas esperamos que ayude, para tener resultados pronto
CONFIGS = (("db3", "otsu"), ("db3", "sin"), ("vhdb3", "otsu"), ("vhdb3", "sin"), ("vh3", "otsu"), ("vh3", "sin"))
REP_INFO = {
    "vh3": "VH x3, percentiles 2-98 por chip (como H2)",
    "vhdb3": "VH x3, rango fijo en dB",
    "db3": "[VV, VH, VV-VH], rangos fijos en dB",
}
PROMPT_INFO = {"otsu": "prompts Otsu 3+2 (como H2)", "sin": "sin prompts (embedding por defecto)"}


def config_name(rep: str, prompts: str) -> str:
    return f"{rep}_{prompts}"


def result_file(rep: str, prompts: str) -> Path:
    return OUT_DIR / f"config_{config_name(rep, prompts)}.json"


# ----------------------------------------------------------------------------- rangos fijos

def compute_db_ranges(chips: list[str], step: int = 4, on_chip=None) -> dict:
    """Percentiles 1-99 de VV, VH y VV-VH sobre los pixeles validos de `chips` (submuestreo fijo)."""
    vals = {"vv": [], "vh": [], "diff": []}
    for chip in chips:
        d = data.load_chip(chip)
        vv, vh = d["vv"][::step, ::step], d["vh"][::step, ::step]
        ok = np.isfinite(vv) & np.isfinite(vh)
        vals["vv"].append(vv[ok])
        vals["vh"].append(vh[ok])
        vals["diff"].append((vv - vh)[ok])
        if on_chip:
            on_chip()
    rangos = {}
    for k, v in vals.items():
        v = np.concatenate(v).astype(np.float64)
        lo, hi = np.percentile(v, PCT)
        rangos[k] = {"lo": float(lo), "hi": float(hi), "mediana": float(np.median(v))}
    return {"regla": f"percentiles {PCT[0]:g} y {PCT[1]:g} de los pixeles con dato del sensor en train, "
                     f"submuestreo 1 de cada {step} filas y columnas", "n_chips": len(chips), "rangos": rangos}


# ----------------------------------------------------------------------------- evaluacion rapida

def load_eval_set(chips: list[str], rep: str) -> list[dict]:
    """Todo lo necesario para evaluar en GPU por lotes, con los prompts deterministas v2.1."""
    items = []
    for chip in chips:
        d = data.load_chip(chip)
        label = d["label"]
        lv = label != -1
        gt = (label == 1) & lv
        om = otsu_masks(d["vv"], d["vh"])
        pts, lbs, npos = auto_prompts(om["opened"], om["sensor_valid"], config.N_POS, config.N_NEG)
        c = load_embedding(chip, rep)
        oh, ow = [int(v) for v in c["original_sizes"][0]]
        rh, rw = [int(v) for v in c["reshaped_input_sizes"][0]]
        items.append({
            "chip": chip, "emb": c["emb"][0],
            "gt": torch.from_numpy(gt), "valid": torch.from_numpy(lv),
            "pts": [[x * (rw / ow), y * (rh / oh)] for x, y in pts], "lbs": list(lbs), "npos": npos,
            "osz": (oh, ow), "rsz": (rh, rw),
        })
    return items


def _postprocess(low: torch.Tensor, rsz, osz) -> torch.Tensor:
    """Igual que SamProcessor.post_process_masks (sin binarizar): 256 -> 1024 -> recorte -> chip."""
    m = F.interpolate(low, (1024, 1024), mode="bilinear", align_corners=False)
    m = m[..., : rsz[0], : rsz[1]]
    return F.interpolate(m, osz, mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate(model, items: list[dict], prompts: str, thresholds=(0.5,), batch_size: int = 8) -> dict:
    """{umbral: [fila por chip]} con tp, fp, fn e iou, en el mismo orden que `items`.

    Con prompts Otsu se agrupan los chips con igual cantidad de puntos (relleno = 0) para que el
    resultado sea identico a decodificar chip por chip. Un chip sin puntos positivos recibe la
    mascara vacia, igual que en `evaluate_chip`.
    """
    dev = next(model.parameters()).device
    thr_logit = torch.tensor([math.log(t / (1 - t)) for t in thresholds], device=dev)
    rows = {t: [None] * len(items) for t in thresholds}
    groups: dict = {}
    for i, it in enumerate(items):
        key = "sin" if prompts == "sin" else (None if it["npos"] == 0 else len(it["pts"]))
        groups.setdefault(key, []).append(i)

    def put(i, logit):
        gt, valid = items[i]["gt"].to(dev), items[i]["valid"].to(dev)
        n_gt = int(gt.sum())
        if logit is None:
            for t in thresholds:
                rows[t][i] = _row(0, 0, n_gt)
            return
        pred = (logit[None] > thr_logit[:, None, None]) & valid[None]
        tp = (pred & gt[None]).sum(dim=(1, 2)).tolist()
        fp = (pred & ~gt[None]).sum(dim=(1, 2)).tolist()
        for k, t in enumerate(thresholds):
            rows[t][i] = _row(int(tp[k]), int(fp[k]), n_gt - int(tp[k]))

    for key, idxs in groups.items():
        for s in range(0, len(idxs), batch_size):
            b = idxs[s:s + batch_size]
            if key is None:
                for i in b:
                    put(i, None)
                continue
            emb = torch.stack([items[i]["emb"] for i in b]).to(dev)
            if key == "sin":
                out = model(image_embeddings=emb, multimask_output=False)
            else:
                pts = torch.tensor([[items[i]["pts"]] for i in b], dtype=torch.float32, device=dev)
                lbs = torch.tensor([[items[i]["lbs"]] for i in b], dtype=torch.int64, device=dev)
                out = model(image_embeddings=emb, input_points=pts, input_labels=lbs, multimask_output=False)
            low = out.pred_masks[:, 0].float()  # (B, 1, 256, 256)
            for j, i in enumerate(b):
                put(i, _postprocess(low[j:j + 1], items[i]["rsz"], items[i]["osz"])[0, 0])
    return rows


def _row(tp: int, fp: int, fn: int) -> dict:
    u = tp + fp + fn
    return {"iou": tp / u if u > 0 else None, "tp": tp, "fp": fp, "fn": fn, "union": u}


def global_iou(rows: list[dict]) -> float:
    return stats.aggregate(rows)["global_iou"]


# ----------------------------------------------------------------------------- entrenamiento

def train_decoder_v2(model, items: list[dict], *, seed: int, lr: float, prompts: str,
                     epochs_total: int = EPOCHS, stop_at: int | None = None, batch_size: int = BATCH,
                     weight_decay: float = WEIGHT_DECAY, warmup_epochs: int = WARMUP_EPOCHS,
                     on_epoch=None, on_step=None) -> dict:
    """Como `decoder_ft.train_decoder`, con calentamiento lineal + coseno y evaluacion por epoca.

    `stop_at` corta el entrenamiento en esa epoca sin cambiar el programa de lr (que sigue
    definido para `epochs_total`), asi la corrida final reproduce la trayectoria elegida.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    dev = next(model.parameters()).device
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.mask_decoder.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.mask_decoder.parameters(), lr=lr, weight_decay=weight_decay)
    usable = items if prompts == "sin" else [it for it in items if it["opened"].any()]
    spe = math.ceil(len(usable) / batch_size)
    total, warm = spe * epochs_total, spe * warmup_epochs

    def lr_factor(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    hist, t0 = [], time.time()
    for ep in range(stop_at or epochs_total):
        model.mask_decoder.train()
        order = rng.permutation(len(usable))
        losses = []
        for s in range(0, len(order), batch_size):
            batch = [usable[i] for i in order[s:s + batch_size]]
            emb = torch.stack([it["emb"] for it in batch]).to(dev)
            if prompts == "sin":
                out = model(image_embeddings=emb, multimask_output=False)
            else:
                pts, lbs = _prompt_batch(batch, rng, config.N_POS, config.N_NEG)
                out = model(image_embeddings=emb, input_points=pts.to(dev), input_labels=lbs.to(dev),
                            multimask_output=False)
            logits = F.interpolate(out.pred_masks[:, 0], size=(LOSS_SIZE, LOSS_SIZE), mode="bilinear",
                                   align_corners=False)[:, 0]
            gt = torch.stack([it["gt"] for it in batch]).to(dev)
            valid = torch.stack([it["valid"] for it in batch]).to(dev)
            loss = masked_bce_dice(logits, gt, valid)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            if on_step:
                on_step(len(batch))
        hist.append({"epoca": ep + 1, "perdida_media": float(np.mean(losses)), "lr_final": sched.get_last_lr()[0],
                     "min": (time.time() - t0) / 60})
        if on_epoch:
            model.mask_decoder.eval()
            hist[-1].update(on_epoch(ep + 1, hist[-1]) or {})
    model.mask_decoder.eval()
    for p in model.mask_decoder.parameters():
        p.requires_grad_(False)
    return {"historial": hist, "n_chips_entrenamiento": len(usable), "minutos": (time.time() - t0) / 60}


def decoder_state_cpu(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.mask_decoder.state_dict().items()}


# ----------------------------------------------------------------------------- referencias previas

def load_references(valid: list[str]) -> dict:
    """Filas por chip de Otsu y de H2 (3 semillas) en valid, desde results/H2_resultados.json."""
    h2 = json.loads((config.RESULTS_DIR / "H2_resultados.json").read_text(encoding="utf-8"))
    pc = h2["por_chip"]
    if pc["chips"] != valid:
        raise RuntimeError("El orden de los chips de valid no coincide con H2_resultados.json")
    return {"otsu": pc["otsu"], "h2": {int(k): v for k, v in pc["ft"].items()}}


def compare(rows: list[dict], ref_rows: list[dict], regions: list[str]) -> dict:
    return {
        "bootstrap": stats.paired_bootstrap(ref_rows, rows, n_boot=N_BOOT),
        "bootstrap_region": stats.paired_bootstrap(ref_rows, rows, groups=regions, n_boot=N_BOOT),
        "wilcoxon": stats.wilcoxon_paired([r["iou"] for r in ref_rows], [r["iou"] for r in rows]),
    }


# ----------------------------------------------------------------------------- bitacora

MARK_START = "<!-- RESULTADOS_PASO1:INICIO (se regenera solo; no editar a mano dentro) -->"
MARK_END = "<!-- RESULTADOS_PASO1:FIN -->"


def _fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def results_markdown() -> str:
    """Tabla de resultados del paso 1 a partir de los JSON de cada configuracion."""
    lines = ["| Configuración | Entrada | Prompts | Estado | lr | Época | Umbral | IoU train-dev | "
             "**IoU global valid** (3 semillas) | IoU medio por chip | Δ vs H2 (misma semilla) | Δ vs Otsu |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    best_dev, best_name = -1.0, None
    n_done = sum(result_file(r, p).exists() for r, p in CONFIGS)
    head = (f"**Avance del paso 1: {n_done}/{len(CONFIGS)} configuraciones terminadas.** "
            + ("✅ Paso 1 terminado." if n_done == len(CONFIGS) else "⏳ En curso o pendiente.") + "\n\n")
    for rep, prompts in CONFIGS:
        f = result_file(rep, prompts)
        name = config_name(rep, prompts)
        if not f.exists():
            st = _partial_status(rep, prompts)
            lines.append(f"| `{name}` | {REP_INFO[rep]} | {PROMPT_INFO[prompts]} | {st} | | | | | | | | |")
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        s = r["seleccion"]
        fin = r["final"]
        g, m = fin["iou_global"], fin["iou_medio"]

        def mean_diff(key):
            v = [fin[key][k]["bootstrap"]["global_iou"]["diff_b_menos_a"] for k in fin.get(key, {})]
            return f"{np.mean(v):+.3f}" if v else "—"

        lines.append(f"| `{name}` | {REP_INFO[rep]} | {PROMPT_INFO[prompts]} | ✅ Terminado | {s['lr']:g} | "
                     f"{s['epoca']} | {s['umbral']:g} | {s['iou_train_dev']:.3f} | "
                     f"**{g['media']:.3f} ± {g['desv']:.3f}** | {m['media']:.3f} ± {m['desv']:.3f} | "
                     f"{mean_diff('vs_h2')} | {mean_diff('vs_otsu')} |")
        if s["iou_train_dev"] > best_dev:
            best_dev, best_name = s["iou_train_dev"], name
    out = head + "\n".join(lines)
    out += ("\n\nReferencia (valid, 86 chips): Otsu con apertura **0.256**, H2 **0.431 ± 0.038**. "
            "Δ = diferencia media de IoU global entre semillas; los IC 95 % pareados están en "
            "`results/hacia_060/config_*.json`.")
    if best_name:
        out += (f"\n\n**Candidata del paso 1 según train-dev (regla pre-registrada): `{best_name}`** "
                f"(IoU train-dev {best_dev:.3f}). Se elige por train-dev, no por valid.")
    out += f"\n\n_Actualizado automáticamente: {time.strftime('%Y-%m-%d %H:%M')}_"
    return out


def _partial_status(rep, prompts) -> str:
    estado = OUT_DIR / "estado.json"
    try:
        e = json.loads(estado.read_text(encoding="utf-8"))
        if (e.get("config_actual") == config_name(rep, prompts) and not e.get("terminado")
                and not e.get("interrumpido")):
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
    except Exception as exc:  # la bitacora nunca debe tumbar un entrenamiento largo
        block = f"(No se pudo regenerar la tabla: {exc!r}. Los JSON de results/hacia_060/ están completos.)"
    BITACORA.write_text(txt[:a] + "\n\n" + block + "\n\n" + txt[b:], encoding="utf-8")
