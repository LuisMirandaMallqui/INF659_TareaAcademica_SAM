"""R2 (bitacora, seccion 2d): semillas 3 y 4 de `lora_db3_otsu` y `unet_db3`, mismo protocolo exacto.

    python tools/hacia060_semillas_extra.py

Solo valid. Los pesos y probabilidades quedan junto a los de las semillas 0-2 (paso 2 y paso 3a), con
nombres de semilla nuevos (no se sobrescribe nada). Al final se calcula el resumen de 5 semillas en
results/hacia_060/robustez/cinco_semillas.json. Se puede cortar y relanzar: salta lo ya hecho.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hacia060_paso2 as T2  # noqa: E402  (activa tambien el "no suspender" de Windows)
import hacia060_paso3 as T3  # noqa: E402
from hacia060_paso1 import Avance  # noqa: E402
from inf659sam import config, data, stats  # noqa: E402
from inf659sam import lora_sam as L  # noqa: E402
from inf659sam import paso3 as P3  # noqa: E402
from inf659sam.cache import save_json  # noqa: E402
from inf659sam.hacia060 import compare  # noqa: E402

OUT = config.RESULTS_DIR / "hacia_060" / "robustez"
NEW_SEEDS = (3, 4)
ALL_SEEDS = (0, 1, 2, 3, 4)


def sam_seed_json(s):
    return L.seed_file("db3", "otsu", s) if s < 3 else OUT / f"lora_db3_otsu_seed{s}.json"


def unet_seed_json(s):
    return P3.seed_file("unet_db3", s) if s < 3 else OUT / f"unet_db3_seed{s}.json"


def resumen(va, valid):
    regions = [data.region_of(c) for c in valid]
    out = {"semillas": list(ALL_SEEDS), "chips_valid": valid}
    rows_by = {}
    for key, jf, pf in (("sam", sam_seed_json, lambda s, t: Path(str(L.probs_file("db3", "otsu", s)).replace(
                            "_probs_", "_probs_tta_")) if t else L.probs_file("db3", "otsu", s)),
                        ("unet", unet_seed_json, lambda s, t: P3.probs_file("unet_db3", s, t))):
        seeds = [json.loads(jf(s).read_text(encoding="utf-8")) for s in ALL_SEEDS]
        g = [r["valid"]["individual"]["agregado"]["global_iou"] for r in seeds]
        m = [r["valid"]["individual"]["agregado"]["mean_chip_iou"] for r in seeds]
        rows_by[key] = [r["valid"]["individual"]["por_chip"] for r in seeds]
        ens = {}
        for name, tta in (("ensamble", False), ("ensamble_tta", True)):
            probs = [L.load_probs(pf(s, tta), len(valid)) for s in ALL_SEEDS]
            rows = L.rows_from_probs(L.ensemble(probs), va, 0.5)
            ens[name] = stats.aggregate(rows)
            ens[name + "_por_chip"] = rows
        out[key] = {"global_por_semilla": g, "global": stats.mean_std(g), "medio_por_chip": stats.mean_std(m),
                    "global_3_semillas": stats.mean_std(g[:3]),
                    "ensamble_5": ens["ensamble"], "ensamble_5_tta": ens["ensamble_tta"]}
        out[key + "_ens_tta_por_chip"] = ens["ensamble_tta_por_chip"]
    pares = []
    for s in ALL_SEEDS:
        c = compare(rows_by["unet"][s], rows_by["sam"][s], regions)
        b = c["bootstrap"]["global_iou"]
        pares.append({"semilla": s, "diferencia": b["diff_b_menos_a"], "ic95": b["ic95"],
                      "excluye_cero": bool(b["ic95"][0] > 0 or b["ic95"][1] < 0),
                      "wilcoxon_p": c["wilcoxon"]["p_valor"]})
    out["unet_menos_sam_por_semilla"] = pares
    out["unet_menos_sam_ensamble_5_tta"] = compare(out["unet_ens_tta_por_chip"], out["sam_ens_tta_por_chip"], regions)
    return out


def main():
    config.setup_runtime(seed=0, verbose=False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = config.get_device()
    if dev != "cuda":
        raise SystemExit("Necesita la GPU (CUDA).")
    parts = json.loads((config.RESULTS_DIR / "particiones_usadas.json").read_text(encoding="utf-8"))
    P = {"train": parts["train"], "valid": parts["valid"], "epochs": L.EPOCHS, "epochs_lora": L.EPOCHS,
         "epochs_unet": P3.UNET_EPOCHS, "seeds": NEW_SEEDS}
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [("sam", s) for s in NEW_SEEDS if not sam_seed_json(s).exists()] + \
           [("unet", s) for s in NEW_SEEDS if not unet_seed_json(s).exists()]
    n = {"train": len(P["train"]), "valid": len(P["valid"])}
    units = (n["train"] + n["valid"]) * T2.W_PREP
    for k, _ in todo:
        units += (T2.W_BUILD + L.EPOCHS * n["train"] * T2.W_IMG + n["valid"] * (T2.W_EVAL + T2.W_TTA) + T2.W_STATS
                  if k == "sam" else T3.W_BUILD + P3.UNET_EPOCHS * n["train"] * T3.W_IMG["unet"] + T3.W_STATS)
    print(f"GPU: {torch.cuda.get_device_name(0)} | por correr: {todo}")
    av = Avance(units + 30, OUT)
    try:
        av.etapa("Preparando chips db3 (GPU)", n["train"] + n["valid"], "chip")
        tr = L.prepare_chips(P["train"], "db3", dev, on_chip=lambda: av.paso(T2.W_PREP))
        va = L.prepare_chips(P["valid"], "db3", dev, on_chip=lambda: av.paso(T2.W_PREP))
        for k, s in todo:
            av.info(f"\n=== R2 · {k} · semilla {s} ===")
            if k == "sam":
                res = T2.run_seed("db3", "otsu", s, tr, va, P, av, None, None, None, dev)
                save_json(res, sam_seed_json(s))
            else:
                res = T3.run_seed("unet_db3", s, tr, va, P, av, None, None, None, dev)
                save_json(res, unet_seed_json(s))
        av.etapa("Resumen de 5 semillas", 1, "paso")
        r = resumen(va, P["valid"])
        save_json(r, OUT / "cinco_semillas.json")
        for k in ("sam", "unet"):
            av.info(f"  [5 semillas] {k}: IoU global {r[k]['global']['media']:.4f} ± {r[k]['global']['desv']:.4f} "
                    f"| ensamble 5 + TTA {r[k]['ensamble_5_tta']['global_iou']:.4f}")
        exc = sum(p["excluye_cero"] for p in r["unet_menos_sam_por_semilla"])
        av.info(f"  U-Net − SAM: IC por semilla excluye el cero en {exc}/5")
        av.set(terminado=True, etapa="terminado")
    finally:
        av.cerrar()


if __name__ == "__main__":
    main()
