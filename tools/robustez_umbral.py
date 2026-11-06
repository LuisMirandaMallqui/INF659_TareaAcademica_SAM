"""R1 (bitacora, seccion 2d): linea base clasica fuerte. Umbral fijo sobre VH, con y sin filtro de mediana 5x5.

    python tools/robustez_umbral.py

El umbral se elige SOLO en train (barrido de -40 a +5 dB, paso 0.1, maximizando el IoU global).
Se evalua en valid, test y bolivia. Resultado: results/hacia_060/robustez/umbral_filtrado.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402

from inf659sam import config, data, stats  # noqa: E402
from inf659sam.cache import save_json  # noqa: E402
from inf659sam.hacia060 import _row, compare  # noqa: E402

OUT = config.RESULTS_DIR / "hacia_060" / "robustez"
GRID = np.round(np.arange(-40.0, 5.0001, 0.1), 1)
EDGES = np.concatenate([[-np.inf], GRID, [np.inf]])


def vh_filtrado(d, filtro: bool):
    vh = d["vh"].astype(np.float32)
    sv = np.isfinite(d["vv"]) & np.isfinite(vh)
    if not sv.any():
        return None, sv
    if filtro:
        vh = median_filter(np.where(sv, vh, np.median(vh[sv])), size=5)
    return vh, sv


def elegir_umbral(chips, filtro):
    """IoU global de train para cada umbral, con histogramas acumulados (agua si VH < umbral)."""
    h_agua, h_tierra = np.zeros(len(EDGES) - 1), np.zeros(len(EDGES) - 1)
    for c in chips:
        d = data.load_chip(c)
        lab = d["label"]
        vh, sv = vh_filtrado(d, filtro)
        if vh is None:
            continue
        m = sv & (lab != -1)
        h_agua += np.histogram(vh[m & (lab == 1)], EDGES)[0]
        h_tierra += np.histogram(vh[m & (lab == 0)], EDGES)[0]
    tp = np.cumsum(h_agua)[:-1]  # pixeles con VH < GRID[i]
    fp = np.cumsum(h_tierra)[:-1]
    fn = h_agua.sum() - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    i = int(np.argmax(iou))
    return float(GRID[i]), float(iou[i])


def evaluar(chips, filtro, thr):
    rows = []
    for c in chips:
        d = data.load_chip(c)
        lab = d["label"]
        gt, lv = lab == 1, lab != -1
        vh, sv = vh_filtrado(d, filtro)
        pred = np.zeros_like(gt) if vh is None else (vh < thr) & sv & lv
        tp = int((pred & gt).sum())
        rows.append(_row(tp, int((pred & ~gt).sum()), int(gt.sum()) - tp))
    return rows


def main():
    parts = json.loads((config.RESULTS_DIR / "particiones_usadas.json").read_text(encoding="utf-8"))
    sets = {"valid": parts["valid"]}
    modelos = {}
    for s in ("test", "bolivia"):
        r = json.loads((config.RESULTS_DIR / "hacia_060" / "test" / f"resultados_{s}.json").read_text(encoding="utf-8"))
        sets[s] = r["chips"]
        modelos[s] = {k: r["modelos"][k]["principal_ensamble_tta"]["por_chip"] for k in r["modelos"]}
    out = {"metodo": "umbral fijo en VH elegido en train", "variantes": {}}
    for nombre, filtro in (("sin_filtro", False), ("mediana_5x5", True)):
        thr, iou_tr = elegir_umbral(parts["train"], filtro)
        v = {"umbral_db": thr, "iou_global_train": iou_tr, "conjuntos": {}}
        print(f"{nombre}: umbral {thr} dB (IoU train {iou_tr:.4f})")
        for s, chips in sets.items():
            rows = evaluar(chips, filtro, thr)
            a = stats.aggregate(rows)
            v["conjuntos"][s] = {"agregado": a, "por_chip": rows}
            extra = ""
            if filtro and s in modelos:
                regs = [data.region_of(c) for c in chips]
                v["conjuntos"][s]["vs"] = {}
                for k, mr in modelos[s].items():
                    cmp_ = compare(mr, rows, regs)
                    v["conjuntos"][s]["vs"][k] = cmp_
                    b = cmp_["bootstrap"]["global_iou"]
                    extra += f" | {k}−umbral {b['diff_b_menos_a']:+.3f} [{b['ic95'][0]:+.3f},{b['ic95'][1]:+.3f}]"
            print(f"  {s}: IoU global {a['global_iou']:.4f} | medio {a['mean_chip_iou']:.4f} | "
                  f"P {a['precision']:.3f} R {a['recall']:.3f}{extra}")
        out["variantes"][nombre] = v
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(out, OUT / "umbral_filtrado.json")


if __name__ == "__main__":
    main()
