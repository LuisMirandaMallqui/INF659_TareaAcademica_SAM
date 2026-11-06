"""R3 (bitacora, seccion 2d): figuras finales de test y Bolivia. No entrena nada.

    python tools/figuras_finales_hacia060.py             # todo (usa la GPU unos minutos)
    python tools/figuras_finales_hacia060.py --sin-gpu   # solo lo que sale de los JSON

Genera en results/figuras/:
  * hacia060_final_barras.png         IoU global y medio por chip de todos los metodos en test y Bolivia.
  * hacia060_iou_vs_agua.png          IoU de cada chip contra su % de agua: por que cae el IoU medio.
  * hacia060_final_cualitativo_<conjunto>.png  ejemplos (2 a favor de cada modelo y los 2 con mas agua).
Y en results/hacia_060/robustez/iou_por_cantidad_de_agua.json la tabla de la seccion 3i de la bitacora.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from inf659sam import config, data  # noqa: E402
from inf659sam import lora_sam as L  # noqa: E402
from inf659sam.baseline import normalize_band  # noqa: E402
from inf659sam.cache import save_json  # noqa: E402
from robustez_umbral import vh_filtrado  # noqa: E402

TEST_DIR = config.RESULTS_DIR / "hacia_060" / "test"
ROB = config.RESULTS_DIR / "hacia_060" / "robustez"
COL = {"Otsu": "#9e9e9e", "Umbral filtrado": "#c9a227", "SAM + LoRA": "#7b3294", "U-Net": "#08519c"}
SPLITS = {"test": "Test (regiones de entrenamiento)", "bolivia": "Bolivia (región no vista)"}
BINS = ((0, 1, "< 1 %"), (1, 10, "1–10 %"), (10, 30, "10–30 %"), (30, 101, "> 30 %"))
WATER = "#08306b"


def cargar():
    umbral = json.loads((ROB / "umbral_filtrado.json").read_text(encoding="utf-8"))["variantes"]["mediana_5x5"]
    out = {}
    for s in SPLITS:
        r = json.loads((TEST_DIR / f"resultados_{s}.json").read_text(encoding="utf-8"))
        out[s] = {"chips": r["chips"], "metodos": {
            "Otsu": r["otsu"], "Umbral filtrado": umbral["conjuntos"][s],
            "SAM + LoRA": r["modelos"]["sam"]["principal_ensamble_tta"],
            "U-Net": r["modelos"]["unet"]["principal_ensamble_tta"]}}
    return out, float(umbral["umbral_db"])


def figura_barras(d):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (s, titulo) in zip(axes, SPLITS.items()):
        names = list(d[s]["metodos"])
        x = np.arange(len(names))
        g = [d[s]["metodos"][k]["agregado"]["global_iou"] for k in names]
        m = [d[s]["metodos"][k]["agregado"]["mean_chip_iou"] for k in names]
        b1 = ax.bar(x - 0.2, g, 0.38, color=[COL[k] for k in names], label="IoU global")
        b2 = ax.bar(x + 0.2, m, 0.38, color=[COL[k] for k in names], alpha=0.45, label="IoU medio por chip")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{b.get_height():.3f}", ha="center",
                        fontsize=8)
        ax.axhline(0.60, ls="--", lw=1, color="#555")
        ax.text(len(names) - 0.5, 0.608, "meta 0.60", fontsize=8, ha="right", color="#555")
        ax.set_xticks(x, names, fontsize=9)
        ax.set_title(f"{titulo}, {len(d[s]['chips'])} chips", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("IoU (barra sólida: global; clara: medio por chip)")
    axes[0].set_ylim(0, 0.8)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "hacia060_final_barras.png", dpi=140)
    plt.close(fig)


def tabla_y_figura_agua(d):
    tabla = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, (s, titulo) in zip(axes, SPLITS.items()):
        tabla[s] = {}
        for k in ("SAM + LoRA", "U-Net"):
            filas = d[s]["metodos"][k]["por_chip"]
            agua = []
            for c in d[s]["chips"]:
                lab = data.load_chip(c)["label"]
                agua.append(100 * (lab == 1).sum() / max((lab != -1).sum(), 1))
            agua = np.array(agua)
            iou = np.array([np.nan if r["iou"] is None else r["iou"] for r in filas], float)
            tp = np.array([r["tp"] for r in filas], float)
            u = np.array([r["union"] for r in filas], float)
            ax.scatter(np.maximum(agua, 0.01), iou, s=22, alpha=0.65, color=COL[k], edgecolors="white", linewidths=0.4,
                       label=k)
            t = []
            for lo, hi, nom in BINS:
                m = (agua >= lo) & (agua < hi)
                t.append({"agua": nom, "chips": int(m.sum()), "chips_con_iou": int(np.isfinite(iou[m]).sum()),
                          "iou_medio": float(np.nanmean(iou[m])) if np.isfinite(iou[m]).any() else None,
                          "iou_global_del_grupo": float(tp[m].sum() / max(u[m].sum(), 1)),
                          "pct_de_los_pixeles_de_union": float(100 * u[m].sum() / max(u.sum(), 1))})
            tabla[s][k] = t
        ax.axvline(1, ls=":", color="#555", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("% de agua real en el chip (escala log)")
        ax.set_title(titulo, fontsize=10)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("IoU del chip (ensamble + TTA)")
    fig.suptitle("Los chips casi secos hunden el IoU medio por chip, pero casi no pesan en el IoU global", fontsize=11)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "hacia060_iou_vs_agua.png", dpi=140)
    plt.close(fig)
    save_json(tabla, ROB / "iou_por_cantidad_de_agua.json")
    return tabla


def elegir_chips(chips, sam, unet):
    """2 a favor de cada modelo y los 2 con mas agua (regla fija de la seccion 2d)."""
    d = np.array([(u["iou"] or 0) - (s["iou"] or 0) for s, u in zip(sam, unet)])
    agua = np.array([u["tp"] + u["fn"] for u in unet])
    orden = [int(i) for i in np.argsort(-d)[:2]] + [int(i) for i in np.argsort(d)[:2]]
    for i in np.argsort(-agua):
        if int(i) not in orden:
            orden.append(int(i))
        if len(orden) == 6:
            break
    return [chips[i] for i in orden]


def figuras_cualitativas(d, thr):
    import evaluar_test as E

    dev = config.get_device()
    for s in SPLITS:
        chips = elegir_chips(d[s]["chips"], d[s]["metodos"]["SAM + LoRA"]["por_chip"], d[s]["metodos"]["U-Net"]["por_chip"])
        vacios = set(E.no_sensor_chips(chips))
        chips = [c for c in chips if c not in vacios]
        items = L.prepare_chips(chips, "db3", dev)
        pred = {}
        for key, nom in (("sam", "SAM + LoRA"), ("unet", "U-Net")):
            ps = []
            for sd in config.SEEDS:
                torch.cuda.empty_cache()
                model = E.load_model(key, sd, dev)
                ps.append(E.predict(key, model, items, tta=True))
                del model
            pred[nom] = L.ensemble(ps)
        cols = ["VH", "Umbral filtrado", "SAM + LoRA", "U-Net", "LabelHand"]
        fig, ax = plt.subplots(len(chips), len(cols), figsize=(2.5 * len(cols), 2.6 * len(chips)), squeeze=False)
        blues = ListedColormap(["#f7fbff", WATER]).with_extremes(bad="#bdbdbd")
        idx = {c: i for i, c in enumerate(d[s]["chips"])}
        for r, c in enumerate(chips):
            x = data.load_chip(c)
            lab = x["label"]
            gt, lv = lab == 1, lab != -1
            vh, sv = vh_filtrado(x, True)
            ax[r, 0].imshow(normalize_band(x["vh"]), cmap="gray", interpolation="nearest")
            ax[r, 0].set_title(f"VH {c}\nagua {100 * gt.sum() / max(lv.sum(), 1):.1f} %", fontsize=8)
            masks = {"Umbral filtrado": (vh < thr) & sv, "SAM + LoRA": np.asarray(pred["SAM + LoRA"][r], np.float32) > 0.5,
                     "U-Net": np.asarray(pred["U-Net"][r], np.float32) > 0.5}
            for j, k in enumerate(("Umbral filtrado", "SAM + LoRA", "U-Net"), start=1):
                ax[r, j].imshow(masks[k], cmap=blues, vmin=0, vmax=1, interpolation="nearest")
                iou = d[s]["metodos"][k]["por_chip"][idx[c]]["iou"]
                ax[r, j].set_title(f"{k}\nIoU {iou:.3f}" if iou is not None else f"{k}\nIoU indefinido", fontsize=8)
            ax[r, 4].imshow(np.where(lv, gt.astype(float), np.nan), cmap=blues, vmin=0, vmax=1, interpolation="nearest")
            ax[r, 4].set_title("LabelHand\n(gris = sin etiqueta)", fontsize=8)
        for a in ax.flat:
            a.axis("off")
        fig.suptitle(f"{SPLITS[s]}: ejemplos (2 a favor de cada modelo y los 2 con más agua)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(config.FIG_DIR / f"hacia060_final_cualitativo_{s}.png", dpi=130)
        plt.close(fig)
        del items
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sin-gpu", action="store_true")
    args = ap.parse_args()
    d, thr = cargar()
    figura_barras(d)
    t = tabla_y_figura_agua(d)
    print(json.dumps(t, indent=1, ensure_ascii=False))
    if not args.sin_gpu:
        figuras_cualitativas(d, thr)
    print("Figuras en", config.FIG_DIR)


if __name__ == "__main__":
    main()
