"""Figuras cualitativas del camino hacia 0.60: imagen, predicciones de cada etapa y etiqueta.

    python tools/figuras_hacia060.py

No entrena nada: usa los pesos guardados de H2 y del paso 1 (decodificadores, sobre los embeddings
en cache) y las probabilidades del paso 2 ya calculadas en valid. Tarda menos de un minuto.

Genera en results/figuras/:
  * hacia060_cualitativo.png  VH | Otsu | H2 | paso 1 | paso 2 (2 configuraciones) | LabelHand
  * hacia060_errores.png      mapa de aciertos y errores (TP, FP, FN) de H2, paso 1 y paso 2
  * hacia060_iou_por_chip.png IoU de cada chip de valid: H2 contra paso 2

Todas las predicciones son de la semilla 0, fijada de antemano para no elegir la mejor semilla.
Los chips son los mismos de H2_cualitativo.png mas los dos que mas error concentran en el paso 2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from inf659sam import config, data  # noqa: E402
from inf659sam import hacia060 as H  # noqa: E402
from inf659sam import lora_sam as L  # noqa: E402
from inf659sam.baseline import normalize_band, otsu_masks  # noqa: E402

SEED = 0
CHIPS = ("Mekong_977338", "Paraguay_581976", "Paraguay_284928", "Somalia_12849", "Ghana_1033830", "Nigeria_984831")
P2 = (("db3", "otsu"), ("vhdb3", "sin"))
WATER = "#08306b"


@torch.no_grad()
def decoder_probs(model, items: list[dict]) -> list:
    """Probabilidad de agua por chip con prompts Otsu (como `H.evaluate`, chip por chip)."""
    dev = next(model.parameters()).device
    out = []
    for it in items:
        if it["npos"] == 0:
            out.append(None)
            continue
        pts = torch.tensor([[it["pts"]]], dtype=torch.float32, device=dev)
        lbs = torch.tensor([[it["lbs"]]], dtype=torch.int64, device=dev)
        r = model(image_embeddings=it["emb"][None].to(dev), input_points=pts, input_labels=lbs,
                  multimask_output=False)
        logit = H._postprocess(r.pred_masks[:, 0].float(), it["rsz"], it["osz"])[0, 0]
        out.append(torch.sigmoid(logit).cpu().numpy())
    return out


def counts(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    pred = pred & valid
    tp = int((pred & gt).sum())
    return H._row(tp, int((pred & ~gt).sum()), int(gt.sum()) - tp)


def title(name: str, row: dict) -> str:
    return f"{name}\nIoU {row['iou']:.3f}" if row["iou"] is not None else f"{name}\nIoU indefinido"


def main() -> None:
    from inf659sam.sam_utils import SamWrapper

    config.setup_runtime(SEED, verbose=False)
    fig_dir = config.FIG_DIR
    valid = json.loads(L.config_file(*P2[0]).read_text(encoding="utf-8"))["chips_valid"]
    idx = {c: valid.index(c) for c in CHIPS}

    sam = SamWrapper()
    model = sam.model
    # H2: decodificador ajustado sobre VH x3 con percentiles por chip
    model.mask_decoder.load_state_dict(load_file(str(config.CACHE_DIR / "modelos" / f"mask_decoder_ft_seed{SEED}.safetensors")))
    p_h2 = decoder_probs(model, H.load_eval_set(list(CHIPS), "vh3"))
    # paso 1: mejor en valid, vhdb3_otsu (umbral elegido 0.5)
    model.mask_decoder.load_state_dict(load_file(str(H.MODEL_DIR / f"vhdb3_otsu_seed{SEED}.safetensors")))
    p_p1 = decoder_probs(model, H.load_eval_set(list(CHIPS), "vhdb3"))
    # paso 2: probabilidades guardadas en valid
    p_p2 = {}
    for rep, pr in P2:
        allp = L.load_probs(L.probs_file(rep, pr, SEED), len(valid))
        p_p2[(rep, pr)] = [allp[idx[c]] for c in CHIPS]

    # control: el IoU recalculado debe coincidir con el guardado en los JSON por semilla
    for rep, pr in P2:
        saved = json.loads(L.seed_file(rep, pr, SEED).read_text(encoding="utf-8"))["valid"]["individual"]["por_chip"]
        for c, p in zip(CHIPS, p_p2[(rep, pr)]):
            d = data.load_chip(c)
            lab = d["label"]
            row = counts(np.zeros_like(lab, bool) if p is None else p.astype(np.float32) > 0.5, lab == 1, lab != -1)
            assert row["tp"] == saved[idx[c]]["tp"] and row["fp"] == saved[idx[c]]["fp"], (rep, pr, c)

    cols = ["VH", "Otsu", "H2\n(decodificador)", "Paso 1\n(rango fijo dB)",
            "Paso 2 LoRA\n[VV,VH,VV−VH] + Otsu", "Paso 2 LoRA\nVH sin prompts", "LabelHand"]
    fig1, ax1 = plt.subplots(len(CHIPS), len(cols), figsize=(2.3 * len(cols), 2.45 * len(CHIPS)))
    err_names = ["H2", "Paso 1", "Paso 2 LoRA [VV,VH,VV−VH] + Otsu"]
    fig2, ax2 = plt.subplots(len(CHIPS), 2 + len(err_names), figsize=(2.6 * (2 + len(err_names)), 2.6 * len(CHIPS)))
    # 0 sin etiqueta, 1 TN, 2 TP, 3 FP, 4 FN
    err_cmap = ListedColormap(["#bdbdbd", "#ffffff", WATER, "#e6550d", "#fdd835"])
    blues = ListedColormap(["#f7fbff", WATER]).with_extremes(bad="#bdbdbd")

    for r, c in enumerate(CHIPS):
        d = data.load_chip(c)
        lab = d["label"]
        gt, lv = lab == 1, lab != -1
        om = otsu_masks(d["vv"], d["vh"])
        preds = [("Otsu", om["opened"].astype(bool))]
        for name, p in (("H2", p_h2[r]), ("Paso 1", p_p1[r]),
                        ("P2a", p_p2[P2[0]][r]), ("P2b", p_p2[P2[1]][r])):
            preds.append((name, np.zeros_like(gt) if p is None else np.asarray(p, np.float32) > 0.5))
        vh = normalize_band(d["vh"])
        water_pct = 100 * gt.sum() / max(lv.sum(), 1)

        ax1[r, 0].imshow(vh, cmap="gray", interpolation="nearest")
        ax1[r, 0].set_title(f"VH {c}\nagua {water_pct:.1f} % del chip", fontsize=8)
        for j, (_, m) in enumerate(preds, start=1):
            ax1[r, j].imshow(m, cmap=blues, vmin=0, vmax=1, interpolation="nearest")
            ax1[r, j].set_title(title(cols[j], counts(m, gt, lv)), fontsize=8)
        ax1[r, -1].imshow(np.where(lv, gt.astype(float), np.nan), cmap=blues, vmin=0, vmax=1, interpolation="nearest")
        ax1[r, -1].set_title("LabelHand\n(gris = sin etiqueta, no cuenta)", fontsize=8)

        ax2[r, 0].imshow(vh, cmap="gray", interpolation="nearest")
        ax2[r, 0].set_title(f"VH {c}", fontsize=8)
        ax2[r, 1].imshow(np.where(lv, gt.astype(float), np.nan), cmap=blues, vmin=0, vmax=1, interpolation="nearest")
        ax2[r, 1].set_title(f"LabelHand (agua {water_pct:.1f} %)", fontsize=8)
        for j, (name, (_, m)) in enumerate(zip(err_names, [preds[1], preds[2], preds[3]]), start=2):
            code = np.where(~lv, 0, np.where(m & gt, 2, np.where(m & ~gt, 3, np.where(~m & gt, 4, 1))))
            row = counts(m, gt, lv)
            ax2[r, j].imshow(code, cmap=err_cmap, vmin=0, vmax=4, interpolation="nearest")
            ax2[r, j].set_title(f"{title(name, row)}\nFP {row['fp'] / 1000:.0f} mil · FN {row['fn'] / 1000:.0f} mil px",
                                fontsize=8)

    for a in list(ax1.flat) + list(ax2.flat):
        a.axis("off")
    fig1.suptitle(f"Valid, semilla {SEED}: cómo cambia la predicción en cada etapa (azul = agua)", fontsize=11)
    fig1.tight_layout(rect=(0, 0, 1, 0.98))
    fig1.savefig(fig_dir / "hacia060_cualitativo.png", dpi=130)
    fig2.legend(handles=[Patch(color=WATER, label="acierto: agua (TP)"),
                         Patch(facecolor="#e6550d", label="falso positivo: marca agua donde no hay (FP)"),
                         Patch(facecolor="#fdd835", label="agua no detectada (FN)"),
                         Patch(facecolor="#bdbdbd", label="sin etiqueta (no cuenta)")],
                loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig2.suptitle(f"Valid, semilla {SEED}: dónde acierta y dónde se equivoca cada etapa", fontsize=11)
    fig2.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig2.savefig(fig_dir / "hacia060_errores.png", dpi=130)

    # IoU por chip en los 86 chips: H2 contra paso 2
    h2_rows = H.load_references(valid)["h2"][SEED]
    p2_rows = json.loads(L.seed_file(*P2[0], SEED).read_text(encoding="utf-8"))["valid"]["individual"]["por_chip"]
    a = np.array([np.nan if x["iou"] is None else x["iou"] for x in h2_rows])
    b = np.array([np.nan if x["iou"] is None else x["iou"] for x in p2_rows])
    w = np.array([x["tp"] + x["fn"] for x in p2_rows], float)
    fig3, ax3 = plt.subplots(figsize=(6.4, 5.6))
    sc = ax3.scatter(a, b, s=12 + 140 * np.sqrt(w / w.max()), c=np.where(b >= a, WATER, "#e6550d"), alpha=0.7,
                     edgecolors="white", linewidths=0.5)
    ax3.plot([0, 1], [0, 1], color="#888", lw=1, ls="--")
    better, worse = int(np.nansum(b > a)), int(np.nansum(b < a))
    ax3.set_xlabel(f"IoU por chip, H2 (semilla {SEED})")
    ax3.set_ylabel(f"IoU por chip, paso 2 lora_db3_otsu (semilla {SEED})")
    ax3.set_xlim(-0.02, 1.02)
    ax3.set_ylim(-0.02, 1.02)
    ax3.set_title(f"Los 86 chips de valid: mejora en {better}, empeora en {worse}\n"
                  "(sobre la diagonal = mejor en el paso 2; tamaño = cantidad de agua real)", fontsize=10)
    for c in CHIPS:
        i = idx[c]
        if not np.isnan(a[i]) and not np.isnan(b[i]):
            ax3.annotate(c, (a[i], b[i]), fontsize=7, xytext=(4, -8), textcoords="offset points", color="#333")
    ax3.grid(alpha=0.25)
    fig3.tight_layout()
    fig3.savefig(fig_dir / "hacia060_iou_por_chip.png", dpi=130)
    print("Figuras guardadas en", fig_dir)


if __name__ == "__main__":
    main()
