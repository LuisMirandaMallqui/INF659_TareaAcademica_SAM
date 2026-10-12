"""Figuras en ingles para el resumen extendido y el poster de WAIMLAp (a partir de los JSON de resultados).

Uso: python tools/figuras_ingles.py   (despues de ejecutar los notebooks 03 y 04)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from inf659sam import config, data  # noqa: E402
from inf659sam.baseline import auto_prompts, normalize_band, otsu_masks  # noqa: E402

RES = REPO / "results"
OUT = RES / "figuras_en"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "savefig.dpi": 300, "font.family": "DejaVu Sans"})
PURPLE, GRAY, GOLD, GREEN, ORANGE = "#4B2A82", "#8a8f99", "#c9a227", "#2e8b57", "#d2691e"


def boot_ci(rows, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    tp = np.array([r["tp"] for r in rows], float)
    fp = np.array([r["fp"] for r in rows], float)
    fn = np.array([r["fn"] for r in rows], float)
    k = len(rows)
    vals = []
    for _ in range(n):
        i = rng.integers(0, k, k)
        vals.append(tp[i].sum() / (tp[i].sum() + fp[i].sum() + fn[i].sum()))
    g = tp.sum() / (tp.sum() + fp.sum() + fn.sum())
    return g, np.percentile(vals, [2.5, 97.5])


def main():
    per = json.loads((RES / "valid_resultados_por_chip.json").read_text(encoding="utf-8"))["vh3"]
    comp = json.loads((RES / "valid_comparaciones.json").read_text(encoding="utf-8"))
    h2 = json.loads((RES / "H2_resultados.json").read_text(encoding="utf-8"))
    pc = h2["por_chip"]

    # 1) global IoU with bootstrap CIs
    entries = [
        ("Otsu threshold + opening (baseline)", [r["otsu_opened"] for r in per], GRAY),
        ("SAM, automatic prompts (v2.1)", [r["sam_v2"] for r in per], PURPLE),
        ("SAM, single-mask output", [r["sam_single"] for r in per], PURPLE),
        ("Oracle prompts from labels (diagnostic)", [r["oracle_prompts"] for r in per], GOLD),
        ("Oracle best-of-3 mask (diagnostic)", [r["oracle_select"] for r in per], GOLD),
    ]
    seeds = sorted(pc["ft"].keys())
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    labels, ys = [], []
    for i, (name, rows, col) in enumerate(entries):
        g, (lo, hi) = boot_ci(rows)
        ax.barh(i, g, color=col, alpha=0.9)
        ax.errorbar(g, i, xerr=[[g - lo], [hi - g]], color="k", capsize=3, lw=1)
        ax.text(hi + 0.01, i, f"{g:.3f}", va="center", fontsize=8)
        labels.append(name)
    for j, (name, key, col) in enumerate((("Linear probe on frozen embedding (H2-c)", "sonda", GREEN),
                                          ("Decoder-only tuned SAM (H2)", "ft", ORANGE))):
        gs = []
        for s in seeds:
            rows = pc[key][s]
            gs.append(sum(r["tp"] for r in rows) / sum(r["tp"] + r["fp"] + r["fn"] for r in rows))
        g, (lo, hi) = boot_ci(pc[key][seeds[0]])
        i = len(entries) + j
        ax.barh(i, np.mean(gs), color=col, alpha=0.9)
        ax.errorbar(np.mean(gs), i, xerr=[[np.mean(gs) - lo], [hi - np.mean(gs)]], color="k", capsize=3, lw=1)
        ax.text(hi + 0.01, i, f"{np.mean(gs):.3f} ± {np.std(gs, ddof=1):.3f}", va="center", fontsize=8)
        labels.append(name)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"Global water IoU, {len(per)} validation chips (95% bootstrap CI)")
    ax.set_xlim(0, 0.72)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "fig_en_results.png")
    plt.close(fig)

    # 2) calibration + collapse
    sc, io = [], []
    for r in per:
        v = r["sam_v2"]
        if v.get("cand_scores"):
            for s, i_ in zip(v["cand_scores"], v["cand_ious"]):
                if i_ is not None:
                    sc.append(s)
                    io.append(i_)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    axes[0].scatter(sc, io, s=6, alpha=0.5, color=PURPLE)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.7)
    axes[0].set_xlabel("SAM predicted IoU")
    axes[0].set_ylabel("actual IoU vs labels")
    axes[0].set_title(f"IoU head is miscalibrated on SAR (ρ = {comp['D3_calibracion']['spearman_todas']:.2f})", fontsize=8.5)
    wf = [r["_meta"]["water_frac_gt"] for r in per]
    axes[1].scatter(wf, [r["sam_v2"]["area_frac"] for r in per], s=8, alpha=0.6, color=PURPLE, label="SAM (auto prompts)")
    ftrows = pc["ft"][seeds[0]]
    axes[1].scatter(wf, [r["area_frac"] for r in ftrows], s=8, alpha=0.6, color=ORANGE, label="decoder-tuned SAM")
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.7)
    axes[1].set_xlabel("true water fraction")
    axes[1].set_ylabel("predicted water fraction")
    axes[1].set_title("Over-segmentation (collapse to whole chip)", fontsize=8.5)
    axes[1].legend(fontsize=7, loc="center right")
    plt.tight_layout()
    plt.savefig(OUT / "fig_en_calibration.png")
    plt.close(fig)

    # 3) layer-wise linear probe
    pr = comp["P_sonda_por_capa"]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(range(len(pr) - 1), [r["AUC"] for r in pr[1:]], "o-", color=PURPLE, label="SAM tokens")
    ax.axhline(pr[0]["AUC"], color=GRAY, ls="--", label="VV/VH dB (2 features)")
    ax.set_xticks(range(len(pr) - 1))
    ax.set_xticklabels(["b0", "b2", "b5", "b8", "b11", "neck"])
    ax.set_ylabel("water vs land AUC")
    ax.set_title("Linear separability per block", fontsize=8.5)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "fig_en_probe.png")
    plt.close(fig)
    print("figuras listas en", OUT)


def qualitative(chips: list[str]):
    """Ejemplos cualitativos con el decodificador ajustado (semilla 0)."""
    import torch
    from safetensors.torch import load_file

    from inf659sam.cache import load_embedding
    from inf659sam.decoder_ft import decoder_state, load_decoder_state
    from inf659sam.sam_utils import SamWrapper

    config.setup_runtime(0, verbose=False)
    sam = SamWrapper()
    pre = decoder_state(sam)
    ft = load_file(str(config.CACHE_DIR / "modelos" / "mask_decoder_ft_seed0.safetensors"))
    fig, axes = plt.subplots(len(chips), 5, figsize=(7.2, 1.55 * len(chips)))
    for r, c in enumerate(chips):
        d = data.load_chip(c)
        lab = d["label"]
        om = otsu_masks(d["vv"], d["vh"])
        pts, lbs, k = auto_prompts(om["opened"], om["sensor_valid"])
        e = load_embedding(c)
        load_decoder_state(sam, pre)
        m0 = sam.decode(e["emb"], pts, lbs, e["original_sizes"], e["reshaped_input_sizes"])
        sam_mask = m0["masks"][int(np.argmax(m0["scores"]))]
        sam.model.mask_decoder.load_state_dict(ft)
        m1 = sam.decode(e["emb"], pts, lbs, e["original_sizes"], e["reshaped_input_sizes"], multimask=False)
        load_decoder_state(sam, pre)
        from inf659sam.baseline import iou_components
        panels = [("Sentinel-1 VH", normalize_band(d["vh"]), "gray", None),
                  ("Otsu", om["opened"], "Blues", om["opened"]),
                  ("SAM (auto prompts)", sam_mask, "Blues", sam_mask),
                  ("Decoder-tuned SAM", m1["masks"][0], "Blues", m1["masks"][0]),
                  ("Label", np.where(lab != -1, (lab == 1).astype(float), np.nan), "Blues", None)]
        for a, (t, im, cm, pred) in zip(axes[r], panels):
            a.imshow(im, cmap=cm, interpolation="nearest")
            if pred is not None:
                t = f"{t}\nIoU {iou_components(pred, lab == 1, lab != -1)['iou']:.2f}"
            if r == 0 or pred is not None:
                a.set_title(t, fontsize=7)
            a.set_xticks([])
            a.set_yticks([])
        axes[r, 0].set_ylabel(c.replace("_", " "), fontsize=7)
        if r == 0:
            axes[r, 0].scatter([p[0] for p in pts], [p[1] for p in pts], c=["lime" if l == 1 else "red" for l in lbs], s=6)
    plt.tight_layout()
    plt.savefig(OUT / "fig_en_qualitative.png")
    plt.close(fig)
    print("cualitativa lista")


if __name__ == "__main__":
    main()
    if len(sys.argv) > 1:
        qualitative(sys.argv[1:])
