"""Linea base Otsu y generacion automatica de prompts, identica a run_sam_pipeline_v2.py.

Separacion estricta (correccion v2):
  - sensor_valid: solo del radar (VV y VH finitos). Es lo unico que toca el umbral y los prompts.
  - label_valid:  solo de LabelHand (-1 = sin dato). Se usa exclusivamente para evaluar.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu
from skimage.measure import label as cc_label
from skimage.morphology import disk
from skimage.morphology import opening as binary_opening


def normalize_band(band: np.ndarray, lo_pct: float = 2, hi_pct: float = 98) -> np.ndarray:
    """Percentiles 2-98 -> uint8, NaN reemplazado por la mediana (igual que v2)."""
    band = np.where(np.isfinite(band), band, np.nanmedian(band))
    lo, hi = np.nanpercentile(band, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-6
    out = (band - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)


def otsu_masks(vv: np.ndarray, vh: np.ndarray) -> dict:
    """Otsu sobre VH (dB) con validez del sensor; mascara cruda y con apertura (disco r=1)."""
    sensor_valid = np.isfinite(vv) & np.isfinite(vh)
    thr = threshold_otsu(vh[sensor_valid])
    raw = (vh < thr) & sensor_valid
    opened = binary_opening(raw, disk(1)) & sensor_valid
    return {"threshold_db": float(thr), "raw": raw, "opened": opened, "sensor_valid": sensor_valid}


def sample_interior_points(mask: np.ndarray, n_points: int, stable: bool = True) -> list[tuple[int, int]]:
    """Puntos en el interior de los componentes mas grandes (maxima distancia al borde).

    Devuelve (x=columna, y=fila), la convencion de SAM. Es el muestreador de v2 con una
    sola diferencia (v2.1): `stable=True` usa un ordenamiento estable. La transformada de
    distancia tiene muchos empates y `np.argsort` por defecto los desempata segun la
    implementacion SIMD de la CPU (AVX2 frente a AVX-512), asi que v2 podia elegir puntos
    distintos en maquinas distintas. Con `stable=False` se reproduce v2 literal.
    """
    kind = "stable" if stable else None
    if mask.sum() == 0:
        return []
    comp = cc_label(mask)
    sizes = [(comp == i).sum() for i in range(1, comp.max() + 1)]
    order = np.argsort(sizes, kind=kind)[::-1] + 1
    points = []
    remaining = n_points
    for idx, comp_id in enumerate(order):
        if remaining <= 0:
            break
        comp_mask = comp == comp_id
        dist = distance_transform_edt(comp_mask)
        take = 1 if idx > 0 else min(remaining, max(1, n_points - (len(order) - 1)))
        take = min(take, remaining)
        flat_idx = np.argsort(dist.ravel(), kind=kind)[::-1]
        picked = 0
        seen = set()
        for fi in flat_idx:
            if dist.ravel()[fi] <= 0:
                break
            y, x = np.unravel_index(fi, dist.shape)
            if any(abs(y - py) < 15 and abs(x - px) < 15 for py, px in seen):
                continue
            points.append((int(x), int(y)))
            seen.add((y, x))
            picked += 1
            remaining -= 1
            if picked >= take or remaining <= 0:
                break
    return points[:n_points]


def sample_random_points(mask: np.ndarray, n_points: int, rng: np.random.Generator,
                         min_dist_px: float = 2.0) -> list[tuple[int, int]]:
    """Muestreo aleatorio de puntos interiores (para entrenamiento, con semilla)."""
    if mask.sum() == 0 or n_points <= 0:
        return []
    dist = distance_transform_edt(mask)
    ys, xs = np.nonzero(dist >= min_dist_px)
    if len(ys) == 0:
        ys, xs = np.nonzero(mask)
    idx = rng.choice(len(ys), size=min(n_points, len(ys)), replace=False)
    return [(int(xs[i]), int(ys[i])) for i in idx]


def auto_prompts(opened: np.ndarray, sensor_valid: np.ndarray, n_pos: int = 3, n_neg: int = 2,
                 stable: bool = True):
    """Prompts positivos desde la mascara Otsu y negativos desde su complemento (solo sensor)."""
    pos = sample_interior_points(opened, n_pos, stable=stable)
    neg = sample_interior_points(~opened & sensor_valid, n_neg, stable=stable)
    points = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    return points, labels, len(pos)


def verify_point_labels(points, labels_pt, mask) -> bool:
    """Cada punto cae en la clase asignada (x, y), mismo orden que recibe el procesador."""
    for (x, y), lab in zip(points, labels_pt):
        if bool(mask[y, x]) != bool(lab == 1):
            return False
    return True


def iou_components(pred, gt, valid) -> dict:
    pred = pred.astype(bool) & valid
    gt = gt.astype(bool) & valid
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    union = tp + fp + fn
    return {"iou": (tp / union) if union > 0 else None, "tp": tp, "fp": fp, "fn": fn, "union": union}
