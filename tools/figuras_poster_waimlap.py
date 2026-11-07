"""Figuras y cifras del poster final de WAIMLAp 2026. No entrena nada.

    python tools/figuras_poster_waimlap.py           # la primera vez usa la GPU ~2 min para las mascaras de ejemplo
    python tools/figuras_poster_waimlap.py --solo-cifras

Salidas en results/figuras_en/:
  * poster_cifras.json     cifras del poster y del abstract, leidas de los JSON de resultados
  * poster_t_*.png         miniaturas de la figura de metodo (se componen en el PowerPoint)
  * poster_r1.png ...      una figura por cada resultado
Las mascaras se guardan en INF659_CACHE/poster_waimlap/ y se comprueba que reproduzcan el IoU de cada chip guardado
en results/hacia_060/test/ y en results/valid_resultados_por_chip.json.
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
from matplotlib.patches import Rectangle  # noqa: E402

from inf659sam import config, stats  # noqa: E402

RES = config.RESULTS_DIR
HX = RES / "hacia_060"
OUT = RES / "figuras_en"
CACHE = config.CACHE_DIR / "poster_waimlap"
UMBRAL_DB = -23.3
N_BOOT = 5000  # el de hacia060.compare; reproduce los intervalos de la bitacora (secciones 3h y 3j)

INK, MUTED, GRID, CARD = "#28282D", "#5A5A64", "#E6E1F0", "#FAF9FC"
SAM, UNET, OTSU, UMBRAL = "#5B35A8", "#00918D", "#C3C3CC", "#8E8E9A"
ESCALERA = ["#BBA6E6", "#A388DB", "#8A6AD0", "#7350BF", "#5B35A8"]
ACIERTO, FALSA, OMISION, SECO, SIN_ETIQUETA = "#2563B8", "#F59E0B", "#E11D74", "#F1F1F5", "#BDBDC4"

CHIP_METODO = "Bolivia_129334"
CHIP_H1 = "India_1050276"
# Ejemplos tomados del conjunto que eligio la regla fija de la bitacora (seccion 2d): mayor diferencia entre modelos.
EJEMPLOS = {
    "Pakistan_694942": ("test", "A", "Test: SAM removes false alarms"),
    "Bolivia_129334": ("bolivia", "B", "Unseen region: SAM gives the cleanest map"),
    "India_747992": ("test", "C", "Failure: SAM misses most of the water"),
}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 20, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#B9B4C6",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 1.2, "savefig.dpi": 300, "figure.facecolor": CARD,
    "savefig.facecolor": CARD, "axes.facecolor": CARD, "pdf.fonttype": 42,
})


def j(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------- cifras

def ic_global(rows, n=N_BOOT, seed=0):
    """IC 95 % del IoU global remuestreando chips (mismo n y semilla que la bitacora)."""
    rng = np.random.default_rng(seed)
    tp, fp, fn = (np.array([r[k] for r in rows], float) for k in ("tp", "fp", "fn"))
    i = rng.integers(0, len(rows), (n, len(rows)))
    t = tp[i].sum(1)
    return [float(v) for v in np.percentile(t / (t + fp[i].sum(1) + fn[i].sum(1)), [2.5, 97.5])]


def resumen(a):
    return {k: a[k] for k in ("global_iou", "mean_chip_iou", "precision", "recall", "n_chips")}


def cifras() -> dict:
    umbral = j(HX / "robustez" / "umbral_filtrado.json")["variantes"]["mediana_5x5"]
    assert umbral["umbral_db"] == UMBRAL_DB
    c = {"umbral_db": umbral["umbral_db"]}
    for s in ("test", "bolivia"):
        r = j(HX / "test" / f"resultados_{s}.json")
        met = {"otsu": r["otsu"], "umbral": umbral["conjuntos"][s], "sam": r["modelos"]["sam"]["principal_ensamble_tta"],
               "unet": r["modelos"]["unet"]["principal_ensamble_tta"]}
        d = {"chips": r["chips"], "sin_radar": r["chips_sin_dato_radar"]}
        for k, v in met.items():
            d[k] = resumen(v["agregado"]) | {"ic95": ic_global(v["por_chip"])}
        for k in ("sam", "unet"):
            b = stats.paired_bootstrap(met["umbral"]["por_chip"], met[k]["por_chip"], n_boot=N_BOOT, seed=0)["global_iou"]
            d[f"{k}_vs_umbral"] = {"dif": b["diff_b_menos_a"], "ic95": b["ic95"]}
        u = r["unet_vs_sam"]
        d["unet_vs_sam"] = {"dif": u["bootstrap"]["global_iou"]["diff_b_menos_a"], "ic95": u["bootstrap"]["global_iou"]["ic95"],
                            "wilcoxon_p": u["wilcoxon"]["p_valor"], "unet_mejor": u["wilcoxon"]["b_mejor"],
                            "sam_mejor": u["wilcoxon"]["a_mejor"]}
        c[s] = d

    comp = j(RES / "valid_comparaciones.json")
    por_chip = j(RES / "valid_resultados_por_chip.json")["vh3"]
    h2 = j(RES / "H2_resultados.json")
    c["valid"] = {
        "otsu": comp["otsu_opened"]["agregado"]["global_iou"], "todo_agua": comp["all_water"]["agregado"]["global_iou"],
        "sam_preentrenado": comp["sam_v2"]["agregado"]["global_iou"],
        "sam_gana_a_otsu": comp["sam_v2"]["wilcoxon_vs_otsu"]["b_mejor"], "n": comp["sam_v2"]["agregado"]["n_chips"],
        "colapso_mas_90": sum(r["sam_v2"]["area_frac"] > 0.9 for r in por_chip),
        "spearman": comp["D3_calibracion"]["spearman_todas"],
        # D1 del abstract original: prompts desde las etiquetas con salida de una sola mascara.
        "oraculo_prompts": comp["oracle_prompts_single"]["agregado"]["global_iou"],
        "oraculo_mejor_mascara": comp["oracle_select"]["agregado"]["global_iou"],
    }
    # Cada paso se lee de su propio JSON (media y desviacion de las 3 semillas, modelo individual).
    p1 = next(v for v in j(HX / "config_vhdb3_otsu.json")["final"].values() if isinstance(v, dict) and "media" in v)
    fila_h2 = next(f for f in h2["tabla"] if f["método"].startswith("H2:"))
    h2_media, h2_desv = (float(x) for x in fila_h2["IoU global"].split(" ± "))
    p2 = j(HX / "paso2" / "config_lora_db3_otsu.json")
    p3 = j(HX / "paso3" / "config_unet_db3.json")
    cinco = j(HX / "robustez" / "cinco_semillas.json")
    c["valid"] |= {
        "h2": {"media": h2_media, "desv": h2_desv}, "fijo_vh": p1,
        "lora": p2["semillas_individual"], "lora_ens_tta": p2["ensamble"]["tta"]["agregado"]["global_iou"],
        "unet": p3["semillas_individual"], "unet_ens_tta": p3["ensamble"]["tta"]["agregado"]["global_iou"],
        "umbral": umbral["conjuntos"]["valid"]["agregado"]["global_iou"],
        "sam_5": cinco["sam"]["global"], "unet_5": cinco["unet"]["global"],
    }
    c["agua_test_sam"] = j(HX / "robustez" / "iou_por_cantidad_de_agua.json")["test"]["SAM + LoRA"]
    c["params"] = {"sam_entrenables": j(HX / "paso2" / "lora_db3_otsu_seed0.json")["entrenamiento"]["params_entrenables"],
                   "unet": j(HX / "paso3" / "unet_db3_seed0.json")["entrenamiento"]["params_entrenables"]}
    return c


# ------------------------------------------------------------------------------------------------- mascaras

def preparar_mascaras() -> dict:
    f = CACHE / "mascaras.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    import torch

    import evaluar_test as E
    from inf659sam import data
    from inf659sam import lora_sam as L
    from inf659sam.baseline import auto_prompts, otsu_masks
    from inf659sam.cache import load_embedding
    from inf659sam.sam_utils import SamWrapper
    from robustez_umbral import vh_filtrado

    dev = config.get_device()
    chips = list(dict.fromkeys([*EJEMPLOS, CHIP_METODO]))
    items = L.prepare_chips(chips, "db3", dev)
    pred = {}
    for key in ("sam", "unet"):
        ps = []
        for sd in config.SEEDS:
            torch.cuda.empty_cache()
            model = E.load_model(key, sd, dev)
            ps.append(E.predict(key, model, items, tta=True))
            del model
        pred[key] = L.ensemble(ps)
    out: dict = {}
    for r, c in enumerate(chips):
        x = data.load_chip(c)
        vh, sv = vh_filtrado(x, True)
        om = otsu_masks(x["vv"], x["vh"])
        out |= {f"{c}__rgb": items[r]["rgb"].cpu().numpy().transpose(1, 2, 0), f"{c}__label": x["label"],
                f"{c}__umbral": (vh < UMBRAL_DB) & sv, f"{c}__sam": np.asarray(pred["sam"][r], np.float32),
                f"{c}__unet": np.asarray(pred["unet"][r], np.float32), f"{c}__otsu": om["opened"],
                f"{c}__pts": np.array(items[r]["eval_pts"]), f"{c}__lbs": np.array(items[r]["eval_lbs"])}
    del items
    torch.cuda.empty_cache()

    config.setup_runtime(0, verbose=False)
    sam = SamWrapper()
    x = data.load_chip(CHIP_H1)
    om = otsu_masks(x["vv"], x["vh"])
    pts, lbs, _ = auto_prompts(om["opened"], om["sensor_valid"])
    e = load_embedding(CHIP_H1)
    m = sam.decode(e["emb"], pts, lbs, e["original_sizes"], e["reshaped_input_sizes"])
    out |= {f"{CHIP_H1}__vh": x["vh"], f"{CHIP_H1}__label": x["label"], f"{CHIP_H1}__otsu": om["opened"],
            f"{CHIP_H1}__sam": m["masks"][int(np.argmax(m["scores"]))], f"{CHIP_H1}__pts": np.array(pts),
            f"{CHIP_H1}__lbs": np.array(lbs)}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    return out


def iou(pred, lab) -> float | None:
    v = lab != -1
    p, g = pred.astype(bool) & v, (lab == 1) & v
    u = (p | g).sum()
    return float((p & g).sum() / u) if u else None


def verificar(m, c) -> dict:
    """Las mascaras regeneradas deben reproducir el IoU por chip guardado en la evaluacion final."""
    umbral = j(HX / "robustez" / "umbral_filtrado.json")["variantes"]["mediana_5x5"]["conjuntos"]
    tabla = {}
    for chip, (s, *_) in EJEMPLOS.items():
        r = j(HX / "test" / f"resultados_{s}.json")
        i = r["chips"].index(chip)
        ref = {"sam": r["modelos"]["sam"]["principal_ensamble_tta"]["por_chip"][i]["iou"],
               "unet": r["modelos"]["unet"]["principal_ensamble_tta"]["por_chip"][i]["iou"],
               "umbral": umbral[s]["por_chip"][i]["iou"]}
        lab = m[f"{chip}__label"]
        act = {"sam": iou(m[f"{chip}__sam"] > 0.5, lab), "unet": iou(m[f"{chip}__unet"] > 0.5, lab),
               "umbral": iou(m[f"{chip}__umbral"], lab)}
        for k in ref:
            if abs(ref[k] - act[k]) > 1e-3:
                raise SystemExit(f"{chip} {k}: guardado {ref[k]:.4f}, regenerado {act[k]:.4f}")
        tabla[chip] = {k: round(v, 4) for k, v in act.items()} | {
            "agua_pct": round(float(100 * (lab == 1).sum() / max((lab != -1).sum(), 1)), 1)}
    por_chip = j(RES / "valid_resultados_por_chip.json")["vh3"]
    fila = next(r for r in por_chip if r["_meta"].get("chip") == CHIP_H1)
    lab = m[f"{CHIP_H1}__label"]
    act = {"sam_v2": iou(m[f"{CHIP_H1}__sam"], lab), "otsu_opened": iou(m[f"{CHIP_H1}__otsu"], lab)}
    for k, v in act.items():
        if abs(fila[k]["iou"] - v) > 1e-3:
            raise SystemExit(f"{CHIP_H1} {k}: guardado {fila[k]['iou']:.4f}, regenerado {v:.4f}")
    tabla[CHIP_H1] = {k: round(v, 4) for k, v in act.items()} | {
        "area_sam_pct": round(float(100 * m[f"{CHIP_H1}__sam"].mean()), 1),
        "agua_pct": round(float(100 * (lab == 1).sum() / max((lab != -1).sum(), 1)), 1)}
    return tabla


# ------------------------------------------------------------------------------------------------- utilidades de imagen

def hexrgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.uint8)


def capa_errores(pred, lab):
    """Acierto, falsa alarma, omision y sin etiqueta. Lo que se considera agua real es acierto + omision."""
    pred = pred.astype(bool)
    v, g = lab != -1, lab == 1
    im = np.empty((*lab.shape, 3), np.uint8)
    im[...] = hexrgb(SECO)
    im[pred & g & v] = hexrgb(ACIERTO)
    im[pred & ~g & v] = hexrgb(FALSA)
    im[~pred & g & v] = hexrgb(OMISION)
    im[~v] = hexrgb(SIN_ETIQUETA)
    return im


def capa_etiqueta(lab):
    im = np.empty((*lab.shape, 3), np.uint8)
    im[...] = hexrgb(SECO)
    im[lab == 1] = hexrgb(ACIERTO)
    im[lab == -1] = hexrgb(SIN_ETIQUETA)
    return im


def gris(vh):
    """VH en escala de grises para mostrar; sin dato del radar en gris medio."""
    ok = np.isfinite(vh)
    lo, hi = np.percentile(vh[ok], [2, 98])
    g = np.clip((np.where(ok, vh, lo) - lo) / (hi - lo), 0, 1)
    im = (g * 255).astype(np.uint8)
    im[~ok] = 150
    return im


def leyenda_errores(fig, y, x=0.5, ha="center", con_etiqueta=True):
    items = [(ACIERTO, "Water detected"), (FALSA, "False alarm"), (OMISION, "Missed water")]
    if con_etiqueta:
        items.append((SIN_ETIQUETA, "Unlabelled"))
    h = [Rectangle((0, 0), 1, 1, color=c) for c, _ in items]
    fig.legend(h, [t for _, t in items], loc="center" if ha == "center" else "center left", bbox_to_anchor=(x, y),
               ncol=len(items), frameon=False, fontsize=20, handlelength=1.1, handleheight=1.1, columnspacing=1.8,
               handletextpad=0.5, borderaxespad=0)


def puntos(ax, pts, lbs, s=170):
    pos, neg = pts[lbs == 1], pts[lbs == 0]
    ax.scatter(pos[:, 0], pos[:, 1], s=s, c="#22C55E", marker="o", edgecolors="white", linewidths=2, zorder=5, clip_on=False)
    ax.scatter(neg[:, 0], neg[:, 1], s=s, c="#EF4444", marker="X", edgecolors="white", linewidths=2, zorder=5, clip_on=False)


def sin_ejes(ax, marco=None):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(marco or "#B9B4C6")
        sp.set_linewidth(3.5 if marco else 1)


# ------------------------------------------------------------------------------------------------- figuras

def miniaturas(m):
    """Miniaturas reales para la figura de metodo; el diagrama se arma con formas editables en el PowerPoint."""
    c = CHIP_METODO
    lab, rgb = m[f"{c}__label"], m[f"{c}__rgb"]

    def guarda(nombre, im=None, dibuja=None):
        fig = plt.figure(figsize=(2.4, 2.4))
        ax = fig.add_axes((0, 0, 1, 1))
        if im is not None:
            ax.imshow(im, interpolation="nearest")
        if dibuja:
            dibuja(ax)
        ax.axis("off")
        fig.savefig(OUT / f"poster_t_{nombre}.png", dpi=300)
        plt.close(fig)

    guarda("entrada", rgb)
    vh_rgb = np.repeat(rgb[..., 1:2], 3, axis=2)

    def otsu_y_puntos(ax):
        ax.imshow(np.ma.masked_where(~m[f"{c}__otsu"], m[f"{c}__otsu"]), cmap="winter", alpha=0.45, interpolation="nearest")
        puntos(ax, m[f"{c}__pts"], m[f"{c}__lbs"], s=120)
        ax.set_xlim(-14, 526)  # margen para que no se corten los puntos cercanos al borde
        ax.set_ylim(526, -14)

    guarda("prompts", vh_rgb, otsu_y_puntos)
    guarda("salida", capa_etiqueta(np.where(m[f"{c}__sam"] > 0.5, 1, 0)))
    guarda("etiqueta", capa_etiqueta(lab))


def fig_r1(m, c):
    k = CHIP_H1
    lab = m[f"{k}__label"]
    fig, ax = plt.subplots(1, 4, figsize=(14.7, 3.95))
    fig.subplots_adjust(left=0.012, right=0.996, top=0.875, bottom=0.115, wspace=0.05)
    ax[0].imshow(gris(m[f"{k}__vh"]), cmap="gray", interpolation="nearest")
    puntos(ax[0], m[f"{k}__pts"], m[f"{k}__lbs"])
    ax[1].imshow(capa_errores(m[f"{k}__otsu"], lab), interpolation="nearest")
    ax[2].imshow(capa_errores(m[f"{k}__sam"], lab), interpolation="nearest")
    ax[3].imshow(capa_etiqueta(lab), interpolation="nearest")
    t = ["Radar + automatic prompts", f"Otsu: IoU {iou(m[f'{k}__otsu'], lab):.2f}",
         f"Pretrained SAM: IoU {iou(m[f'{k}__sam'], lab):.2f}", "Hand label"]
    for a, s in zip(ax, t):
        a.set_title(s, fontsize=20, pad=7, fontweight="bold" if "SAM" in s else "normal")
        sin_ejes(a, SAM if "SAM" in s else None)
    leyenda_errores(fig, 0.05)
    fig.savefig(OUT / "poster_r1.png")
    plt.close(fig)


def fig_r2(c):
    v = c["valid"]
    filas = [("Pretrained SAM + automatic prompts", v["sam_preentrenado"], None),
             ("+ tuned mask decoder", v["h2"]["media"], v["h2"]["desv"]),
             ("+ fixed-range dB input", v["fijo_vh"]["media"], v["fijo_vh"]["desv"]),
             ("+ LoRA in encoder; VV, VH, VV−VH", v["lora"]["media"], v["lora"]["desv"]),
             ("+ 3-seed ensemble and TTA (final)", v["lora_ens_tta"], None)]
    fig, ax = plt.subplots(figsize=(14.7, 4.95))
    fig.subplots_adjust(left=0.365, right=0.985, top=0.80, bottom=0.135)
    y = np.arange(len(filas))[::-1]
    for yi, (nom, val, sd), col in zip(y, filas, ESCALERA):
        ax.barh(yi, val, height=0.56, color=col, zorder=2)
        if sd:
            ax.plot([val - sd, val + sd], [yi, yi], color=INK, lw=2.2, solid_capstyle="butt", zorder=4)
        txt = f"{val:.3f}" + (f" ± {sd:.3f}" if sd else "")
        ax.text(val + (sd or 0) + 0.008, yi, txt, va="center", ha="left", fontsize=20,
                fontweight="bold" if nom.endswith("(final)") else "normal", zorder=6,
                bbox=dict(facecolor=CARD, edgecolor="none", pad=1.5, alpha=0.9))
    ax.set_yticks(y, [f[0] for f in filas], fontsize=20, color=INK)
    ax.tick_params(axis="y", length=0)
    refs = [(v["otsu"], "Otsu", "center"), (v["umbral"], "Fixed\nthreshold", "center"), (v["unet_ens_tta"], "U-Net", "center")]
    for x, nombre, ha in refs:
        ax.axvline(x, color=MUTED, lw=1.6, ls=(0, (5, 4)), zorder=3)
        ax.text(x, len(filas) - 0.42, f"{nombre}\n{x:.3f}", ha=ha, va="bottom", fontsize=18, color=MUTED, linespacing=1.05)
    ax.set_xlim(0, 0.86)
    ax.set_ylim(-0.5, len(filas) - 0.5)
    ax.set_xticks(np.arange(0, 0.81, 0.2))
    ax.xaxis.grid(True, color=GRID, lw=1.1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Global water IoU, 86 validation chips", fontsize=20, labelpad=5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=18)
    fig.savefig(OUT / "poster_r2.png")
    plt.close(fig)


def fig_r3(c):
    metodos = [("otsu", "Otsu", OTSU, INK), ("umbral", "Fixed\nthreshold", UMBRAL, "white"), ("sam", "SAM +\nLoRA", SAM, "white"),
               ("unet", "U-Net", UNET, "white")]
    fig, axes = plt.subplots(1, 2, figsize=(14.7, 5.45), sharey=True)
    fig.subplots_adjust(left=0.067, right=0.985, top=0.905, bottom=0.165, wspace=0.05)
    for ax, (s, tit) in zip(axes, (("test", f"Test: {c['test']['sam']['n_chips']} chips, training regions"),
                                   ("bolivia", f"Bolivia: {c['bolivia']['sam']['n_chips']} chips, unseen region"))):
        for i, (k, nom, col, tinta) in enumerate(metodos):
            d = c[s][k]
            ax.bar(i, d["global_iou"], width=0.66, color=col, zorder=2)
            lo, hi = d["ic95"]
            ax.plot([i, i], [lo, hi], color=INK, lw=2.2, zorder=4)
            for q in (lo, hi):
                ax.plot([i - 0.07, i + 0.07], [q, q], color=INK, lw=2.2, zorder=4)
            ax.text(i, 0.035, f"{d['global_iou']:.3f}", ha="center", va="bottom", fontsize=23, fontweight="bold", color=tinta, zorder=5)
        ax.axhline(0.60, color=MUTED, lw=1.6, ls=(0, (5, 4)), zorder=3)
        ax.set_xticks(range(4), [m[1] for m in metodos], fontsize=20, color=INK)
        ax.set_title(tit, fontsize=21, fontweight="bold", pad=10)
        ax.yaxis.grid(True, color=GRID, lw=1.1, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=18)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.55, 3.55)
    axes[0].axhline(0.676, color=MUTED, lw=1.6, ls=(0, (1.5, 2.5)), zorder=3)
    for y, txt in ((0.607, "target: 0.60"), (0.683, "DeepLabv3+ [2]: 0.676")):
        axes[0].text(-0.5, y, txt, fontsize=18, color=MUTED, ha="left", va="bottom", zorder=6,
                     bbox=dict(facecolor=CARD, edgecolor="none", pad=0.8))
    axes[0].set_ylim(0, 0.84)
    axes[0].set_yticks(np.arange(0, 0.81, 0.2))
    axes[0].set_ylabel("Global water IoU", fontsize=20, labelpad=6)
    fig.savefig(OUT / "poster_r3.png")
    plt.close(fig)


def fig_r4(m):
    """Tres ejemplos en una tira a lo ancho; los colores muestran donde acierta y donde falla cada metodo."""
    ancho, alto, lado = 29.9, 3.82, 2.34
    fig = plt.figure(figsize=(ancho, alto))
    gap_p, gap_g = 0.05, 0.5
    grupo = 4 * lado + 3 * gap_p
    x0 = (ancho - (3 * grupo + 2 * gap_g)) / 2
    y_panel = 0.52
    for g, (chip, (_, letra, titulo)) in enumerate(EJEMPLOS.items()):
        x = x0 + g * (grupo + gap_g)
        lab, rgb = m[f"{chip}__label"], m[f"{chip}__rgb"]
        fig.text(x / ancho, (y_panel + lado + 0.62) / alto, f"{letra}. {titulo}", fontsize=22, fontweight="bold", ha="left", va="center")
        ps = [("Radar (VH)", rgb[..., 1], None), ("Threshold", m[f"{chip}__umbral"], None),
              ("SAM + LoRA", m[f"{chip}__sam"] > 0.5, SAM), ("U-Net", m[f"{chip}__unet"] > 0.5, None)]
        for i, (nom, im, marco) in enumerate(ps):
            ax = fig.add_axes(((x + i * (lado + gap_p)) / ancho, y_panel / alto, lado / ancho, lado / alto))
            if i == 0:
                ax.imshow(im, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
                ax.set_title(nom, fontsize=19, pad=5)
            else:
                ax.imshow(capa_errores(im, lab), interpolation="nearest")
                ax.set_title(f"{nom}: {iou(im, lab):.2f}", fontsize=19, pad=5, fontweight="bold" if marco else "normal")
            sin_ejes(ax, marco)
    leyenda_errores(fig, 0.23 / alto)
    fig.savefig(OUT / "poster_r4.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo-cifras", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    c = cifras()
    if a.solo_cifras:
        print(json.dumps({k: v for k, v in c.items() if k not in ("test", "bolivia")}, indent=1, ensure_ascii=False))
    else:
        m = preparar_mascaras()
        c["ejemplos"] = verificar(m, c)
        miniaturas(m)
        fig_r1(m, c)
        fig_r2(c)
        fig_r3(c)
        fig_r4(m)
    for s in ("test", "bolivia"):
        c[s].pop("chips", None)
    (OUT / "poster_cifras.json").write_text(json.dumps(c, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(c, indent=1, ensure_ascii=False)[:6000])
    print("Listo en", OUT)


if __name__ == "__main__":
    main()
