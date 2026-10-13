"""Fases B y C: evaluacion de la linea base, diagnosticos y ablaciones sobre embeddings cacheados.

Todos los metodos se evaluan sobre los mismos chips y los mismos pixeles (LabelHand != -1),
de modo que las comparaciones son pareadas. Los metodos marcados como ORACULO usan la
etiqueta y existen solo como diagnostico (cotas superiores), nunca como metodo propuesto.
"""
from __future__ import annotations

import numpy as np

from . import config, data
from .baseline import auto_prompts, iou_components, otsu_masks, sample_interior_points, verify_point_labels
from .cache import load_embedding
from .sam_utils import SamWrapper, mask_to_prompt_logits

# descripcion corta de cada metodo, usada en tablas y figuras
METHOD_INFO = {
    "otsu_raw": "Otsu sobre VH, sin apertura",
    "otsu_opened": "Otsu sobre VH + apertura (linea base operativa)",
    "all_water": "Trivial: todo el soporte del sensor es agua",
    "sam_v2": "SAM, prompts Otsu 3+2, multimascara, mayor IoU predicho (pipeline v2)",
    "sam_single": "SAM, prompts Otsu 3+2, salida de una sola mascara",
    "sam_min_area": "SAM, multimascara, candidata de menor area",
    "sam_otsu_consistent": "SAM, multimascara, candidata mas parecida a Otsu (sin etiquetas)",
    "sam_maskprompt": "SAM, puntos 3+2 + prompt denso con la mascara Otsu, una mascara",
    "sam_p1_0": "SAM, 1 punto positivo, multimascara, mayor IoU predicho",
    "sam_p5_5": "SAM, 5+5 puntos, multimascara, mayor IoU predicho",
    "sam_p10_10": "SAM, 10+10 puntos, multimascara, mayor IoU predicho",
    "oracle_select": "ORACULO: de las 3 candidatas, la de mayor IoU real",
    "oracle_prompts": "ORACULO: prompts 3+2 tomados de LabelHand, multimascara, mayor IoU predicho",
    "oracle_prompts_single": "ORACULO: prompts 3+2 tomados de LabelHand, una mascara",
}


def _ev(mask, gt, label_valid, **extra) -> dict:
    r = iou_components(mask, gt, label_valid)
    r["area_frac"] = float(mask[label_valid].mean()) if label_valid.any() else 0.0
    r.update(extra)
    return r


def evaluate_chip(sam: SamWrapper, chip: str, rep: str = "vh3", methods: tuple | None = None,
                  decoder_model=None) -> dict:
    """Evalua todos los metodos pedidos en un chip. Devuelve {metodo: fila}."""
    d = data.load_chip(chip)
    label = d["label"]
    label_valid = label != -1
    gt = (label == 1) & label_valid
    om = otsu_masks(d["vv"], d["vh"])
    sv, opened = om["sensor_valid"], om["opened"]
    c = load_embedding(chip, rep)
    emb, osz, rsz = c["emb"], c["original_sizes"], c["reshaped_input_sizes"]
    methods = set(methods or METHOD_INFO.keys())
    out = {"_meta": {"chip": chip, "region": data.region_of(chip), "rep": rep,
                     "water_frac_gt": float(gt[label_valid].mean()) if label_valid.any() else 0.0,
                     "frac_nodata_label": float((~label_valid).mean()), "otsu_threshold_db": om["threshold_db"]}}

    if "otsu_raw" in methods:
        out["otsu_raw"] = _ev(om["raw"], gt, label_valid)
    if "otsu_opened" in methods:
        out["otsu_opened"] = _ev(opened, gt, label_valid)
    if "all_water" in methods:
        out["all_water"] = _ev(sv, gt, label_valid)

    empty = np.zeros_like(gt)
    pts, lbs, npos = auto_prompts(opened, sv, config.N_POS, config.N_NEG)
    out["_meta"]["n_pos"], out["_meta"]["n_neg"] = npos, len(pts) - npos
    out["_meta"]["prompts_ok"] = verify_point_labels(pts, lbs, opened) if pts else None

    def dec(p, l, multimask=True, input_masks=None):
        return sam.decode(emb, p, l, osz, rsz, multimask=multimask, input_masks=input_masks, model=decoder_model)

    def cand_ious(masks):
        return [iou_components(m, gt, label_valid)["iou"] for m in masks]

    need_mm = methods & {"sam_v2", "sam_min_area", "sam_otsu_consistent", "oracle_select"}
    if need_mm:
        if npos == 0:
            for m in need_mm:
                out[m] = _ev(empty, gt, label_valid, score=None, sin_prompts=True)
        else:
            r = dec(pts, lbs, multimask=True)
            masks, scores = r["masks"], r["scores"]
            ious = cand_ious(masks)
            areas = [float(m[label_valid].mean()) for m in masks]
            cand = {"cand_scores": scores.tolist(), "cand_ious": ious, "cand_areas": areas}
            if "sam_v2" in methods:
                b = int(np.argmax(scores))
                out["sam_v2"] = _ev(masks[b], gt, label_valid, score=float(scores[b]), elegida=b, **cand)
            if "sam_min_area" in methods:
                b = int(np.argmin([m.sum() for m in masks]))
                out["sam_min_area"] = _ev(masks[b], gt, label_valid, score=float(scores[b]), elegida=b)
            if "sam_otsu_consistent" in methods:
                cons = [iou_components(m, opened, sv)["iou"] or 0.0 for m in masks]
                b = int(np.argmax(cons))
                out["sam_otsu_consistent"] = _ev(masks[b], gt, label_valid, score=float(scores[b]), elegida=b)
            if "oracle_select" in methods:
                b = int(np.nanargmax([v if v is not None else -1 for v in ious]))
                out["oracle_select"] = _ev(masks[b], gt, label_valid, score=float(scores[b]), elegida=b)

    if "sam_single" in methods:
        if npos == 0:
            out["sam_single"] = _ev(empty, gt, label_valid, score=None, sin_prompts=True)
        else:
            r = dec(pts, lbs, multimask=False)
            out["sam_single"] = _ev(r["masks"][0], gt, label_valid, score=float(r["scores"][0]))

    if "sam_maskprompt" in methods:
        if npos == 0:
            out["sam_maskprompt"] = _ev(empty, gt, label_valid, score=None, sin_prompts=True)
        else:
            r = dec(pts, lbs, multimask=False, input_masks=mask_to_prompt_logits(opened))
            out["sam_maskprompt"] = _ev(r["masks"][0], gt, label_valid, score=float(r["scores"][0]))

    for name, (n_p, n_n) in {"sam_p1_0": (1, 0), "sam_p5_5": (5, 5), "sam_p10_10": (10, 10)}.items():
        if name not in methods:
            continue
        p, l, k = auto_prompts(opened, sv, n_p, n_n)
        if k == 0:
            out[name] = _ev(empty, gt, label_valid, score=None, sin_prompts=True)
        else:
            r = dec(p, l, multimask=True)
            b = int(np.argmax(r["scores"]))
            out[name] = _ev(r["masks"][b], gt, label_valid, score=float(r["scores"][b]), n_puntos=len(p))

    if methods & {"oracle_prompts", "oracle_prompts_single"}:
        p_pos = sample_interior_points(gt, config.N_POS)
        p_neg = sample_interior_points(~gt & label_valid & sv, config.N_NEG)
        p, l = p_pos + p_neg, [1] * len(p_pos) + [0] * len(p_neg)
        for name, mm in (("oracle_prompts", True), ("oracle_prompts_single", False)):
            if name not in methods:
                continue
            if not p_pos:
                out[name] = _ev(empty, gt, label_valid, score=None, sin_prompts=True)
                continue
            r = dec(p, l, multimask=mm)
            b = int(np.argmax(r["scores"]))
            out[name] = _ev(r["masks"][b], gt, label_valid, score=float(r["scores"][b]))
    return out


def evaluable_chips(chips: list[str]) -> tuple[list[str], list[dict]]:
    """Aplica el criterio de exclusion declarado (>95% de pixeles sin etiqueta)."""
    keep, log = [], []
    for chip in chips:
        lab = data.load_chip(chip)["label"]
        frac = float((lab == -1).mean())
        excl = frac > config.NODATA_EXCLUSION_THRESHOLD
        log.append({"chip": chip, "frac_nodata_label": frac, "excluido": excl})
        if not excl:
            keep.append(chip)
    return keep, log


def rows_for(results: list[dict], method: str) -> list[dict]:
    return [r[method] for r in results]
