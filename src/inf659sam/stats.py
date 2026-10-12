"""Metricas agregadas e incertidumbre (bootstrap pareado por chip y por region, Wilcoxon)."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def aggregate(rows: list[dict]) -> dict:
    """rows: dicts con tp, fp, fn (y opcionalmente iou). Devuelve metricas globales y por chip."""
    tp = np.array([r["tp"] for r in rows], dtype=np.int64)
    fp = np.array([r["fp"] for r in rows], dtype=np.int64)
    fn = np.array([r["fn"] for r in rows], dtype=np.int64)
    union = tp + fp + fn
    per_chip = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    TP, FP, FN = int(tp.sum()), int(fp.sum()), int(fn.sum())
    return {
        "n_chips": len(rows),
        "global_iou": TP / (TP + FP + FN) if (TP + FP + FN) else None,
        "mean_chip_iou": float(np.nanmean(per_chip)) if np.isfinite(per_chip).any() else None,
        "median_chip_iou": float(np.nanmedian(per_chip)) if np.isfinite(per_chip).any() else None,
        "precision": TP / (TP + FP) if (TP + FP) else None,
        "recall": TP / (TP + FN) if (TP + FN) else None,
        "f1": 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else None,
        "iou_undefined": int((union == 0).sum()),
        "tp": TP, "fp": FP, "fn": FN,
    }


def _global_iou(tp, fp, fn, idx):
    t, f, n = tp[idx].sum(), fp[idx].sum(), fn[idx].sum()
    d = t + f + n
    return t / d if d else np.nan


def _mean_iou(tp, fp, fn, idx):
    u = tp[idx] + fp[idx] + fn[idx]
    v = np.where(u > 0, tp[idx] / np.maximum(u, 1), np.nan)
    return np.nanmean(v)


def paired_bootstrap(rows_a: list[dict], rows_b: list[dict], groups: list[str] | None = None,
                     n_boot: int = 10000, seed: int = 0) -> dict:
    """IC 95% de (B - A) para IoU global y IoU medio por chip, remuestreando chips (pareado).

    Si se pasan `groups` (region de cada chip), se remuestrean regiones completas
    (bootstrap por conglomerados), que es mas conservador cuando los chips de una
    misma region no son independientes.
    """
    rng = np.random.default_rng(seed)
    arr = {}
    for key, rows in (("a", rows_a), ("b", rows_b)):
        arr[key] = tuple(np.array([r[k] for r in rows], dtype=np.float64) for k in ("tp", "fp", "fn"))
    n = len(rows_a)
    assert n == len(rows_b), "los dos metodos deben evaluarse sobre los mismos chips"

    if groups is None:
        draws = [rng.integers(0, n, n) for _ in range(n_boot)]
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = {g: np.nonzero(groups == g)[0] for g in uniq}
        draws = []
        for _ in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            draws.append(np.concatenate([members[g] for g in pick]))

    out = {}
    for name, fn_ in (("global_iou", _global_iou), ("mean_chip_iou", _mean_iou)):
        point_a = fn_(*arr["a"], np.arange(n))
        point_b = fn_(*arr["b"], np.arange(n))
        diffs = np.array([fn_(*arr["b"], d) - fn_(*arr["a"], d) for d in draws])
        diffs = diffs[np.isfinite(diffs)]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        out[name] = {"a": float(point_a), "b": float(point_b), "diff_b_menos_a": float(point_b - point_a),
                     "ic95": [float(lo), float(hi)], "p_bootstrap_bilateral": float(min(1.0, p_two))}
    return out


def wilcoxon_paired(iou_a: list[float], iou_b: list[float]) -> dict:
    a = np.array([np.nan if v is None else v for v in iou_a], float)
    b = np.array([np.nan if v is None else v for v in iou_b], float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = b - a
    res = {"n": int(ok.sum()), "b_mejor": int((d > 0).sum()), "a_mejor": int((d < 0).sum()),
           "empates": int((d == 0).sum()), "mediana_diff": float(np.median(d)) if len(d) else None}
    if (d != 0).sum() >= 5:
        w = sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        res.update({"estadistico": float(w.statistic), "p_valor": float(w.pvalue)})
    return res


def mean_std(values: list[float]) -> dict:
    v = np.asarray(values, float)
    return {"media": float(v.mean()), "desv": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}
