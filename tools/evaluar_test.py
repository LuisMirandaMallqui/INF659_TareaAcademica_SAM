"""Evaluacion final en `test` y `bolivia` (una sola vez, con los modelos ya fijados).

    python tools/evaluar_test.py                                  # prueba segura: corre todo sobre VALID
    python tools/evaluar_test.py --abrir-test                     # candidato SAM pre-registrado en test y bolivia
    python tools/evaluar_test.py --abrir-test --modelos sam,unet  # + la U-Net como linea base (si el equipo lo aprueba)

Mas facil: tools\\lanzar_evaluacion_test.bat con los mismos argumentos.

Que evalua (fijado en docs/bitacora_hacia_060.md, seccion 2c):
  * sam  -> `lora_db3_otsu` del paso 2, ensamble de las 3 semillas + TTA D4, umbral 0.5 (metrica principal).
            Tambien se reportan las 3 semillas por separado (media +- desviacion) como secundaria.
  * unet -> `unet_db3` del paso 3a, igual: ensamble de 3 semillas + TTA D4. Solo con --modelos ...,unet.
  * otsu -> Otsu con apertura, como referencia (siempre).
No entrena nada ni elige nada: carga los pesos guardados, predice y mide.

Protecciones:
  * Sin --abrir-test solo se usa `valid`, y se comprueba que reproduce los numeros conocidos
    (SAM 0.6145 y U-Net 0.6422 con ensamble + TTA, tolerancia 0.003). Sirve para verificar la tuberia
    sin mirar test.
  * Con --abrir-test, si ya hay resultados de un conjunto, el script se niega a repetirlo (se mira una
    sola vez). --rehacer lo permite, y queda anotado en el JSON.
  * Si faltan chips de test o bolivia, se descargan del bucket publico de Sen1Floods11 (~2.5 MB por chip,
    unos 260 MB en total) y se verifican con SHA-256.
  * Se aplica la misma exclusion que en train y valid: chips con mas del 95 % de pixeles sin etiqueta.
  * Un chip sin ningun pixel con dato del radar (en test: Paraguay_34417) se mantiene con la mascara
    vacia para todos los metodos (todo su agua cuenta como FN). Aparte se reporta la metrica sin esos chips.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if os.name == "nt":
    os.system("")
    try:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from hacia060_paso1 import Avance  # noqa: E402
from inf659sam import config, data, stats  # noqa: E402
from inf659sam import lora_sam as L  # noqa: E402
from inf659sam import paso3 as P3  # noqa: E402
from inf659sam.baseline import otsu_masks  # noqa: E402
from inf659sam.cache import save_json  # noqa: E402
from inf659sam.hacia060 import _row, compare  # noqa: E402

OUT_DIR = config.RESULTS_DIR / "hacia_060" / "test"
BITACORA = config.REPO_ROOT / "docs" / "bitacora_hacia_060.md"
MARK_START = "<!-- RESULTADOS_TEST:INICIO (se regenera solo; no editar a mano dentro) -->"
MARK_END = "<!-- RESULTADOS_TEST:FIN -->"

MODELOS = {
    "sam": {"nombre": "lora_db3_otsu (SAM ViT-B + LoRA, paso 2)", "tipo": "lora", "rep": "db3", "prompts": "otsu",
            "pesos": lambda s: L.MODEL_DIR / f"lora_db3_otsu_seed{s}.safetensors", "valid_ens_tta": 0.6145},
    "unet": {"nombre": "unet_db3 (U-Net desde cero, paso 3a)", "tipo": "unet", "rep": "db3", "prompts": "sin",
             "pesos": lambda s: P3.MODEL_DIR / f"unet_db3_seed{s}.safetensors", "valid_ens_tta": 0.6422},
}
TOL_VALID = 0.003
W_PREP, W_BUILD = 0.3, 5.0
W_PRED = {"lora": 0.08 + 0.6, "unet": 0.07}


def load_model(key, seed, dev):
    m = MODELOS[key]
    state = load_file(str(m["pesos"](seed)))
    if m["tipo"] == "unet":
        model = P3.UNet().to(dev)
        model.load_state_dict(state)
    else:
        model = L.build_model(dev)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(k.startswith("mask_decoder.") or "lora_" in k for k in missing):
            raise SystemExit(f"Los pesos de {m['pesos'](seed).name} no calzan con el modelo.")
    model.eval()
    return model


def predict(key, model, items, tta):
    if MODELOS[key]["tipo"] == "unet":
        return P3.predict_unet(model, items, tta=tta)
    return L.predict_probs(model, items, MODELOS[key]["prompts"], tta=tta)


def chips_of(split: str) -> tuple[list[str], list[dict]]:
    """Chips del CSV oficial, descargando los que falten, con la exclusion por falta de etiqueta."""
    names = data.read_split(split)
    missing = [c for c in names if not all(p.exists() for p in data.chip_paths(c).values())]
    if missing:
        print(f"  Descargando {len(missing)} chips de '{split}' (Sen1Floods11, bucket público)...")
        data.ensure_chips(missing)
    keep, log = [], []
    for c in names:
        lab = data.load_chip(c)["label"]
        frac = float((lab == -1).mean())
        exc = frac > config.NODATA_EXCLUSION_THRESHOLD
        log.append({"chip": c, "frac_nodata_label": frac, "excluido": exc})
        if not exc:
            keep.append(c)
    return keep, log


def otsu_rows(chips):
    rows = []
    for c in chips:
        d = data.load_chip(c)
        lab = d["label"]
        gt, lv = lab == 1, lab != -1
        try:
            pred = otsu_masks(d["vv"], d["vh"])["opened"] & lv
        except (ValueError, IndexError):  # chip sin dato del radar: mascara vacia
            pred = np.zeros_like(gt)
        tp = int((pred & gt).sum())
        rows.append(_row(tp, int((pred & ~gt).sum()), int(gt.sum()) - tp))
    return rows


def no_sensor_chips(chips) -> list[str]:
    """Chips sin ningun pixel con dato del radar: ningun metodo puede predecir ahi."""
    out = []
    for c in chips:
        d = data.load_chip(c)
        if not (np.isfinite(d["vv"]) & np.isfinite(d["vh"])).any():
            out.append(c)
    return out


def _full_rows(chips, ok, rows_ok):
    """Filas en el orden de `chips`; los chips sin dato del radar reciben la mascara vacia (todo FN)."""
    it = iter(rows_ok)
    out = []
    for c in chips:
        if c in ok:
            out.append(next(it))
        else:
            n_gt = int((data.load_chip(c)["label"] == 1).sum())
            out.append(_row(0, 0, n_gt))
    return out


def evaluate_set(split, chips, keys, dev, av) -> dict:
    regions = [data.region_of(c) for c in chips]
    empty = no_sensor_chips(chips)
    ok_list = [c for c in chips if c not in empty]
    ok = set(ok_list)
    if empty:
        av.info(f"  [{split}] sin ningún dato del radar (predicción vacía para todos los métodos): {', '.join(empty)}")
    av.etapa(f"{split}: preparando {len(ok_list)} chips", len(ok_list), "chip")
    items = L.prepare_chips(ok_list, "db3", dev, on_chip=lambda: av.paso(W_PREP))
    res = {"conjunto": split, "n_chips": len(chips), "chips": chips, "chips_sin_dato_radar": empty, "modelos": {}}
    ref_otsu = otsu_rows(chips)
    res["otsu"] = {"agregado": stats.aggregate(ref_otsu), "por_chip": ref_otsu}
    keep_idx = [i for i, c in enumerate(chips) if c in ok]
    ens_rows = {}
    for key in keys:
        m = MODELOS[key]
        per_seed, probs_tta, probs_ind = [], [], []
        for s in config.SEEDS:
            av.etapa(f"{split}: {key} semilla {s} (individual + TTA)", len(chips), "chip")
            torch.cuda.empty_cache()
            model = load_model(key, s, dev)
            av.paso(W_BUILD, n=0)
            p_ind = predict(key, model, items, tta=False)
            p_tta = predict(key, model, items, tta=True)
            av.paso(len(chips) * W_PRED[m["tipo"]], n=len(chips))
            del model
            probs_ind.append(p_ind)
            probs_tta.append(p_tta)
            rows = _full_rows(chips, ok, L.rows_from_probs(p_ind, items, L.THRESHOLD))
            per_seed.append({"semilla": s, "agregado": stats.aggregate(rows), "por_chip": rows})
        principal = _full_rows(chips, ok, L.rows_from_probs(L.ensemble(probs_tta), items, L.THRESHOLD))
        ens_ind = _full_rows(chips, ok, L.rows_from_probs(L.ensemble(probs_ind), items, L.THRESHOLD))
        ens_rows[key] = principal
        res["modelos"][key] = {
            "nombre": m["nombre"],
            "principal_ensamble_tta": {"agregado": stats.aggregate(principal), "por_chip": principal,
                                       "vs_otsu": compare(principal, ref_otsu, regions)},
            "principal_sin_chips_sin_radar": {"agregado": stats.aggregate([principal[i] for i in keep_idx])},
            "ensamble_sin_tta": {"agregado": stats.aggregate(ens_ind)},
            "semillas_individual": stats.mean_std([r["agregado"]["global_iou"] for r in per_seed]),
            "por_semilla": per_seed,
        }
        g = res["modelos"][key]["principal_ensamble_tta"]["agregado"]
        av.info(f"  [{split}] {key}: IoU global {g['global_iou']:.4f} (ensamble + TTA) | IoU medio {g['mean_chip_iou']:.4f}"
                f" | semillas {res['modelos'][key]['semillas_individual']['media']:.4f} ± "
                f"{res['modelos'][key]['semillas_individual']['desv']:.4f}")
    if "sam" in ens_rows and "unet" in ens_rows:
        res["unet_vs_sam"] = compare(ens_rows["unet"], ens_rows["sam"], regions)
    return res


def results_markdown() -> str:
    lines = ["| Conjunto | Chips | Otsu | **SAM `lora_db3_otsu`** (ensamble + TTA) | SAM, semillas por separado | "
             "U-Net `unet_db3` (ensamble + TTA) | U-Net − SAM (IC 95 % por chip) |", "|---|---|---|---|---|---|---|"]
    any_res = False
    for split in ("test", "bolivia"):
        f = OUT_DIR / f"resultados_{split}.json"
        if not f.exists():
            lines.append(f"| `{split}` | | ⬜ sin evaluar | | | | |")
            continue
        any_res = True
        r = json.loads(f.read_text(encoding="utf-8"))
        cell = {}
        for k in ("sam", "unet"):
            if k in r["modelos"]:
                a = r["modelos"][k]["principal_ensamble_tta"]["agregado"]
                cell[k] = f"{a['global_iou']:.3f} (medio por chip {a['mean_chip_iou']:.3f})"
            else:
                cell[k] = "—"
        sem = r["modelos"].get("sam", {}).get("semillas_individual")
        sem_txt = f"{sem['media']:.3f} ± {sem['desv']:.3f}" if sem else "—"
        d = "—"
        if "unet_vs_sam" in r:
            b = r["unet_vs_sam"]["bootstrap"]["global_iou"]
            d = f"{b['diff_b_menos_a']:+.3f} [{b['ic95'][0]:+.3f}, {b['ic95'][1]:+.3f}]"
        lines.append(f"| `{split}` | {r['n_chips']} | {r['otsu']['agregado']['global_iou']:.3f} | **{cell['sam']}** | "
                     f"{sem_txt} | "
                     f"{cell['unet']} | {d} |")
    head = ("**Evaluación final.** " + ("Resultados abajo; se miran una sola vez." if any_res else
                                       "⬜ Todavía no se abrió test.") + "\n\n")
    return (head + "\n".join(lines) + "\n\nMétrica principal: IoU global del candidato SAM pre-registrado "
            "(sección 2c). La U-Net es línea base: se compara, no se elige." +
            f"\n\n_Actualizado automáticamente: {time.strftime('%Y-%m-%d %H:%M')}_")


def update_bitacora():
    if not BITACORA.exists():
        return
    txt = BITACORA.read_text(encoding="utf-8")
    if MARK_START not in txt or MARK_END not in txt:
        return
    a, b = txt.index(MARK_START) + len(MARK_START), txt.index(MARK_END)
    try:
        block = results_markdown()
    except Exception as exc:
        block = f"(No se pudo regenerar la tabla: {exc!r}. Los JSON de results/hacia_060/test/ están completos.)"
    BITACORA.write_text(txt[:a] + "\n\n" + block + "\n\n" + txt[b:], encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--abrir-test", action="store_true", help="evaluar de verdad en test y bolivia")
    ap.add_argument("--modelos", default="sam", help="sam (pre-registrado) o sam,unet")
    ap.add_argument("--conjuntos", default="test,bolivia")
    ap.add_argument("--rehacer", action="store_true", help="repetir un conjunto ya evaluado (queda anotado)")
    args = ap.parse_args()
    keys = [k.strip() for k in args.modelos.split(",") if k.strip()]
    if not keys or any(k not in MODELOS for k in keys) or "sam" not in keys:
        raise SystemExit("--modelos debe incluir 'sam' (el candidato pre-registrado), y opcionalmente 'unet'.")

    config.setup_runtime(seed=0, verbose=False)
    dev = config.get_device()
    if dev != "cuda":
        raise SystemExit("La evaluación necesita la GPU (CUDA).")
    for k in keys:
        for s in config.SEEDS:
            if not MODELOS[k]["pesos"](s).exists():
                raise SystemExit(f"Faltan los pesos {MODELOS[k]['pesos'](s)}.")

    if not args.abrir_test:
        global OUT_DIR
        OUT_DIR = OUT_DIR / "prueba_en_valid"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        valid = json.loads((config.RESULTS_DIR / "particiones_usadas.json").read_text(encoding="utf-8"))["valid"]
        print("Modo prueba: se evalúa sobre VALID para verificar la tubería. Test no se toca.")
        av = Avance(len(valid) * (W_PREP + 3 * sum(W_PRED[MODELOS[k]["tipo"]] for k in keys)) + 6 * W_BUILD, OUT_DIR)
        try:
            r = evaluate_set("valid", valid, keys, dev, av)
        finally:
            av.cerrar()
        save_json(r, OUT_DIR / "resultados_valid.json")
        ok = True
        for k in keys:
            g = r["modelos"][k]["principal_ensamble_tta"]["agregado"]["global_iou"]
            exp = MODELOS[k]["valid_ens_tta"]
            good = abs(g - exp) <= TOL_VALID
            ok &= good
            print(f"  {k}: {g:.4f} en valid (esperado {exp:.4f}) -> {'OK' if good else 'NO COINCIDE'}")
        print("\nTubería verificada: lista para --abrir-test." if ok else "\nLa tubería NO reproduce valid: no abrir test.")
        sys.exit(0 if ok else 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.conjuntos.split(",") if s.strip()]
    for split in splits:
        if split not in ("test", "bolivia"):
            raise SystemExit("--conjuntos admite test y bolivia.")
        if (OUT_DIR / f"resultados_{split}.json").exists() and not args.rehacer:
            raise SystemExit(f"'{split}' ya fue evaluado (results/hacia_060/test/resultados_{split}.json). "
                             "Se mira una sola vez; usa --rehacer solo si hay una razón documentada.")
    av = Avance(1.0, OUT_DIR)
    try:
        for split in splits:
            av.etapa(f"{split}: lista de chips y descarga si falta", 1, "paso")
            chips, log = chips_of(split)
            av.ajustar_total(len(chips) * (W_PREP + 3 * sum(W_PRED[MODELOS[k]["tipo"]] for k in keys)) + 6 * W_BUILD)
            av.info(f"\n=== {split}: {len(chips)} chips evaluables ({len(log) - len(chips)} excluidos por falta de etiqueta) ===")
            r = evaluate_set(split, chips, keys, dev, av)
            r.update(exclusion=log, fecha=time.strftime("%Y-%m-%d %H:%M:%S"), rehecho=bool(args.rehacer),
                     modelos_pedidos=keys)
            save_json(r, OUT_DIR / f"resultados_{split}.json")
            update_bitacora()
        av.set(terminado=True, etapa="terminado")
    finally:
        av.cerrar()
        update_bitacora()
    print("\nListo. Resultados en", OUT_DIR, "| bitácora: sección 5 de", BITACORA.name)


if __name__ == "__main__":
    main()
