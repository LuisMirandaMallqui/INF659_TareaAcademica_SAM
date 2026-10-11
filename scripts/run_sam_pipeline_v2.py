"""
Pipeline real v2: corrige la fuga senalada por la revision de ChatGPT sobre el v1.

Cambio central (punto 1 de la revision): v1 usaba `valid = label != -1` (validez de la
ETIQUETA) para restringir donde se muestreaban los prompts negativos. Eso significa que
la generacion de prompts, que se presenta como automatica y sin intervencion humana,
en realidad "sabia" en que pixeles el anotador humano habia dejado la mascara sin dato
(tipicamente por nubes en la imagen optica usada para etiquetar). No revela la clase
agua/no-agua, pero si filtra metadato derivado de la referencia hacia el paso de
inferencia, lo cual rompe la afirmacion de independencia del pipeline respecto a LabelHand.

v2 separa explicitamente dos variables que v1 confundia bajo un solo nombre:
  - sensor_valid: SOLO viene del propio radar (VV y VH finitos). Es lo unico que puede
    tocar la generacion de prompts y el umbral.
  - label_valid: SOLO viene de LabelHand (-1 = sin dato). Se usa EXCLUSIVAMENTE para
    evaluar (IoU), nunca para generar prompts.

Tambien se atienden, del mismo informe de revision:
  - punto 2: se reporta Otsu CON apertura (pipeline operativo) y SIN apertura (referencia).
  - punto 3: se verifica por codigo, no solo se afirma, que cada punto positivo cae en
    water_thr==True y cada punto negativo en water_thr==False (orden de coordenadas x,y).
  - punto 4: el chip sin prompts positivos ahora SIEMPRE tiene iou_sam definido (se compara
    una mascara vacia contra la verdad de terreno), nunca se excluye solo del lado SAM.
  - punto 5: se cuentan y reportan los casos de union=0 (IoU no definido), por metodo.
  - punto 6: se guardan TP/FP/FN, conteo de pixeles validos, prompts, scores de las 3
    mascaras candidatas de SAM, version de librerias y configuracion completa.
  - punto 7: el criterio de exclusion (nodata de ETIQUETA > 95%) se declara antes de
    aplicarlo y se registra para los 10 chips originales, no solo para el excluido.

No se toca el hallazgo central del v1 (SAM tiende a sobre-predecir agua); el objetivo de
v2 es confirmar si ese hallazgo se sostiene una vez cerrada la fuga, no asumirlo.
"""
import glob
import json
import os
import sys

import numpy as np
import scipy
import skimage
import tifffile
from skimage.filters import threshold_otsu
from skimage.morphology import opening as binary_opening, disk
from skimage.measure import label as cc_label
from scipy.ndimage import distance_transform_edt

import torch
import transformers
from transformers import SamModel, SamProcessor

CHIPS_DIR = "chips"
OUT_DIR = "results_v2"
os.makedirs(OUT_DIR, exist_ok=True)

N_POS = 3
N_NEG = 2
MODEL_ID = "facebook/sam-vit-base"
BAND_USED = "VH"
NODATA_EXCLUSION_THRESHOLD = 0.95  # declarado ANTES de aplicarlo (punto 7)


def normalize_band(band, lo_pct=2, hi_pct=98):
    band = np.where(np.isfinite(band), band, np.nanmedian(band))
    lo, hi = np.nanpercentile(band, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-6
    out = (band - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)


def sample_interior_points(mask, n_points):
    if mask.sum() == 0:
        return []
    comp = cc_label(mask)
    sizes = [(comp == i).sum() for i in range(1, comp.max() + 1)]
    order = np.argsort(sizes)[::-1] + 1
    points = []
    remaining = n_points
    for idx, comp_id in enumerate(order):
        if remaining <= 0:
            break
        comp_mask = comp == comp_id
        dist = distance_transform_edt(comp_mask)
        take = 1 if idx > 0 else min(remaining, max(1, n_points - (len(order) - 1)))
        take = min(take, remaining)
        flat_idx = np.argsort(dist.ravel())[::-1]
        picked = 0
        seen = set()
        for fi in flat_idx:
            if dist.ravel()[fi] <= 0:
                break
            y, x = np.unravel_index(fi, dist.shape)
            if any(abs(y - py) < 15 and abs(x - px) < 15 for py, px in seen):
                continue
            points.append((int(x), int(y)))  # (x=columna, y=fila), convencion de SAM
            seen.add((y, x))
            picked += 1
            remaining -= 1
            if picked >= take or remaining <= 0:
                break
    return points[:n_points]


def verify_point_labels(points, labels_pt, mask):
    """Punto 3 de la revision: confirmar por codigo que cada punto cae en la clase que
    se le asigno, usando el mismo orden de coordenadas (x,y) que recibe el procesador."""
    ok = True
    for (x, y), lab in zip(points, labels_pt):
        val = bool(mask[y, x])
        expected = bool(lab == 1)
        if val != expected:
            ok = False
    return ok


def iou_components(pred, gt, valid):
    pred = pred.astype(bool) & valid
    gt = gt.astype(bool) & valid
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    union = tp + fp + fn
    iou_val = (tp / union) if union > 0 else None
    return {"iou": iou_val, "tp": tp, "fp": fp, "fn": fn, "union": union}


def main():
    print("Entorno:", {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "skimage": skimage.__version__, "scipy": scipy.__version__, "numpy": np.__version__,
    })
    print(f"Cargando SAM oficial ({MODEL_ID}) via transformers...")
    model = SamModel.from_pretrained(MODEL_ID).to("cpu")
    model.eval()
    processor = SamProcessor.from_pretrained(MODEL_ID)
    print("Modelo cargado.\n")

    # --- punto 7: declarar y registrar el criterio de exclusion para TODOS los chips originales ---
    all_s1 = sorted(glob.glob(os.path.join(CHIPS_DIR, "*_S1Hand.tif")))
    exclusion_log = []
    for s1_path in all_s1:
        name = os.path.basename(s1_path).replace("_S1Hand.tif", "")
        lab_path = os.path.join(CHIPS_DIR, f"{name}_LabelHand.tif")
        label = tifffile.imread(lab_path)
        frac_nodata_label = float((label == -1).mean())
        excluded = frac_nodata_label > NODATA_EXCLUSION_THRESHOLD
        exclusion_log.append({"chip": name, "frac_nodata_label": frac_nodata_label, "excluded": excluded})
        print(f"  chequeo exclusion: {name:20s} nodata_etiqueta={frac_nodata_label*100:5.1f}%  excluido={excluded}")

    rows = []
    agg = {
        "sam": {"tp": 0, "fp": 0, "fn": 0, "undefined": 0},
        "thr_opened": {"tp": 0, "fp": 0, "fn": 0, "undefined": 0},
        "thr_raw": {"tp": 0, "fp": 0, "fn": 0, "undefined": 0},
    }
    per_chip = {"sam": [], "thr_opened": [], "thr_raw": []}
    coord_checks = []

    print()
    for entry in exclusion_log:
        name = entry["chip"]
        if entry["excluded"]:
            print(f"  [SKIP] {name}: {entry['frac_nodata_label']*100:.1f}% nodata de etiqueta (> {NODATA_EXCLUSION_THRESHOLD*100:.0f}%)")
            continue

        s1 = tifffile.imread(os.path.join(CHIPS_DIR, f"{name}_S1Hand.tif"))
        label = tifffile.imread(os.path.join(CHIPS_DIR, f"{name}_LabelHand.tif"))

        label_valid = label != -1          # SOLO para evaluacion
        vv, vh = s1[0], s1[1]
        sensor_valid = np.isfinite(vv) & np.isfinite(vh)   # SOLO para inferencia/prompts
        gt_water = (label == 1) & label_valid

        # --- baseline: Otsu real sobre VH, usando unicamente validez del sensor ---
        thr = threshold_otsu(vh[sensor_valid])
        water_thr_raw = (vh < thr) & sensor_valid
        water_thr_opened = binary_opening(water_thr_raw, disk(1)) & sensor_valid

        thr_raw_res = iou_components(water_thr_raw, gt_water, label_valid)
        thr_opened_res = iou_components(water_thr_opened, gt_water, label_valid)

        # --- prompts automaticos: SOLO desde el umbral + validez del SENSOR, nunca de la etiqueta ---
        pos_pts = sample_interior_points(water_thr_opened, N_POS)
        neg_pts = sample_interior_points(~water_thr_opened & sensor_valid, N_NEG)

        sam_mask = np.zeros_like(gt_water, dtype=bool)
        sam_scores = None
        coord_ok = None
        if pos_pts:
            points = pos_pts + neg_pts
            labels_pt = [1] * len(pos_pts) + [0] * len(neg_pts)
            coord_ok = verify_point_labels(points, labels_pt, water_thr_opened)
            coord_checks.append(coord_ok)

            vh_img = normalize_band(vh)
            rgb = np.stack([vh_img, vh_img, vh_img], axis=-1)
            inputs = processor(rgb, input_points=[[points]], input_labels=[[labels_pt]], return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu(),
            )[0]
            sam_scores = outputs.iou_scores[0, 0].detach().cpu().numpy().tolist()
            best = int(np.argmax(sam_scores))
            sam_mask = masks[0, best].numpy().astype(bool)

        # punto 4: SIEMPRE se evalua SAM, incluso con mascara vacia, para mantener pares completos
        sam_res = iou_components(sam_mask, gt_water, label_valid)

        water_frac_gt = float(gt_water[label_valid].mean()) if label_valid.sum() else 0.0
        water_frac_thr = float(water_thr_opened[label_valid].mean()) if label_valid.sum() else 0.0
        water_frac_sam = float(sam_mask[label_valid].mean()) if label_valid.sum() else 0.0

        row = {
            "chip": name,
            "otsu_threshold_dB": float(thr),
            "n_pos_prompts": len(pos_pts),
            "n_neg_prompts": len(neg_pts),
            "verificacion_coordenadas_ok": coord_ok,
            "water_frac_gt": water_frac_gt,
            "water_frac_threshold_opened": water_frac_thr,
            "water_frac_sam": water_frac_sam,
            "iou_threshold_raw": thr_raw_res,
            "iou_threshold_opened": thr_opened_res,
            "iou_sam": sam_res,
            "sam_mask_scores_candidatas": sam_scores,
        }
        rows.append(row)
        if name == "Somalia_12849":
            row["_vh"] = vh
            row["_water_thr_opened"] = water_thr_opened
            row["_sam_mask"] = sam_mask
            row["_gt_water"] = gt_water

        for key, res in (("thr_raw", thr_raw_res), ("thr_opened", thr_opened_res), ("sam", sam_res)):
            if res["iou"] is None:
                agg[key]["undefined"] += 1
            else:
                agg[key]["tp"] += res["tp"]; agg[key]["fp"] += res["fp"]; agg[key]["fn"] += res["fn"]
                per_chip[key].append(res["iou"])

        collapse = " [COLAPSO>=90%]" if water_frac_sam >= 0.9 else ""
        print(
            f"  {name:20s} agua_real={water_frac_gt*100:5.1f}%  agua_umbral={water_frac_thr*100:5.1f}%  "
            f"agua_SAM={water_frac_sam*100:5.1f}%  IoU_umbral_abierto={thr_opened_res['iou']}  "
            f"IoU_SAM={sam_res['iou']}  coords_ok={coord_ok}{collapse}"
        )

    def summarize(key):
        a = agg[key]
        union = a["tp"] + a["fp"] + a["fn"]
        global_iou = a["tp"] / union if union else None
        mean_iou = float(np.mean(per_chip[key])) if per_chip[key] else None
        return {"global_iou": global_iou, "mean_per_chip_iou": mean_iou, "casos_iou_no_definido": a["undefined"],
                "tp": a["tp"], "fp": a["fp"], "fn": a["fn"]}

    summary = {
        "modelo": MODEL_ID,
        "banda_usada": BAND_USED,
        "n_prompts": {"positivos": N_POS, "negativos": N_NEG},
        "criterio_exclusion": f"nodata_etiqueta > {NODATA_EXCLUSION_THRESHOLD}",
        "log_exclusion_10_chips_originales": exclusion_log,
        "n_chips_evaluados": len(rows),
        "verificacion_coordenadas_todas_ok": all(coord_checks) if coord_checks else None,
        "resultados": {
            "umbral_otsu_sin_apertura": summarize("thr_raw"),
            "umbral_otsu_con_apertura": summarize("thr_opened"),
            "sam_con_prompts_automaticos": summarize("sam"),
        },
        "entorno": {
            "torch": torch.__version__, "transformers": transformers.__version__,
            "skimage": skimage.__version__, "scipy": scipy.__version__, "numpy": np.__version__,
            "python": sys.version, "device": "cpu",
        },
        "per_chip": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
    }

    with open(os.path.join(OUT_DIR, "resultados_reales_v2.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== RESUMEN REAL v2 (fuga de etiqueta corregida) =====")
    for key, label_ in [("umbral_otsu_sin_apertura", "Otsu SIN apertura"),
                         ("umbral_otsu_con_apertura", "Otsu CON apertura (pipeline operativo)"),
                         ("sam_con_prompts_automaticos", "SAM con prompts automaticos")]:
        r = summary["resultados"][key]
        print(f"{label_:45s} IoU_global={r['global_iou']}  IoU_promedio_chip={r['mean_per_chip_iou']}  no_definidos={r['casos_iou_no_definido']}")
    print(f"\nVerificacion de coordenadas (todas correctas): {summary['verificacion_coordenadas_todas_ok']}")

    viz_row = next((r for r in rows if r["chip"] == "Somalia_12849"), None)
    if viz_row is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(normalize_band(viz_row["_vh"]), cmap="gray")
        axes[0].set_title(f"VH real ({viz_row['chip']})")
        axes[1].imshow(viz_row["_water_thr_opened"], cmap="Blues")
        axes[1].set_title(f"Umbral Otsu (abierto)\nIoU={viz_row['iou_threshold_opened']['iou']:.3f}")
        axes[2].imshow(viz_row["_sam_mask"], cmap="Blues")
        axes[2].set_title(f"SAM real v2\nIoU={viz_row['iou_sam']['iou']:.3f}")
        axes[3].imshow(viz_row["_gt_water"], cmap="Blues")
        axes[3].set_title("Verdad de terreno real")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        fig_path = os.path.join(OUT_DIR, "comparacion_real_v2.png")
        plt.savefig(fig_path, dpi=130)
        print(f"Figura guardada: {fig_path}")


if __name__ == "__main__":
    main()
