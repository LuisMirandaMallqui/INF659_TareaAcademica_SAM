"""Hacia IoU 0.60 - paso 3 (3a U-Net y 3b Tversky), con barra de avance visible.

    python tools/hacia060_paso3.py            # 2 experimentos x 3 semillas
    python tools/hacia060_paso3.py --rapido   # prueba de humo de pocos minutos (no toca resultados ni bitacora)

Mas facil: doble clic en tools/lanzar_hacia060_paso3.bat.

Mientras corre: la ventana muestra el % total y la epoca actual; results/hacia_060/paso3/estado.json y
ejecucion.log tienen lo mismo; la bitacora (seccion 3e) se actualiza sola al terminar cada semilla.
Se puede cortar con Ctrl+C: cada semilla terminada queda guardada y al relanzar se continua.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from pathlib import Path

if os.name == "nt":
    os.system("")
    try:  # mientras este proceso viva, Windows no entra en suspension por inactividad (se libera al salir)
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from hacia060_paso1 import Avance  # noqa: E402
from inf659sam import config, data, stats  # noqa: E402
from inf659sam import hacia060 as H1  # noqa: E402
from inf659sam import lora_sam as L  # noqa: E402
from inf659sam import paso3 as P3  # noqa: E402
from inf659sam.cache import save_json  # noqa: E402

# segundos aproximados por imagen en la RTX 3070 (medidos en la prueba de humo)
W_PREP, W_BUILD, W_STATS = 0.3, 5.0, 12.0
W_IMG = {"lora": 0.27, "unet": 0.022}
W_EVAL = {"lora": 0.08, "unet": 0.008}
W_TTA = {"lora": 0.6, "unet": 0.06}


def epochs_of(ex, P):
    return P["epochs_unet"] if ex["tipo"] == "unet" else P["epochs_lora"]


def plan_units(todo_seeds, todo_ens, n, P) -> float:
    u = n["train"] * W_PREP + n["valid"] * W_PREP
    for name, _ in todo_seeds:
        t = P3.EXPERIMENTS[name]["tipo"]
        u += W_BUILD + epochs_of(P3.EXPERIMENTS[name], P) * n["train"] * W_IMG[t]
        u += n["valid"] * (W_EVAL[t] + W_TTA[t]) + W_STATS
    return u + len(todo_ens) * W_STATS


def run_seed(name, seed, tr, va, P, av, refs, paso2, regions, dev) -> dict:
    ex = P3.EXPERIMENTS[name]
    t, prompts = ex["tipo"], ex["prompts"]
    ep_total = epochs_of(ex, P)
    usable = tr if (t == "unet" or prompts == "sin") else [it for it in tr if len(it["pos_c"])]
    av.ajustar_total((len(usable) - len(tr)) * ep_total * W_IMG[t])
    av.set(config_actual=name, detalle=f"semilla {seed}, construyendo el modelo")
    P3.update_bitacora()
    torch.cuda.empty_cache()
    if t == "unet":
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = P3.UNet().to(dev)
    else:
        model = L.build_model(dev)
    av.paso(W_BUILD, n=0)
    av.etapa(f"{name} · semilla {seed} · entrenando ({ep_total} épocas × {len(usable)} chips)",
             ep_total * len(usable), "img")
    ep_now = {"e": 1}

    def on_step(k, loss):
        av.paso(k * W_IMG[t], n=k, epoca=f"{ep_now['e']}/{ep_total}", perdida=f"{loss:.3f}")

    def on_epoch(ep, h):
        ep_now["e"] = ep + 1
        if t == "lora" or ep % 5 == 0 or ep == ep_total:
            av.info(f"  [{name} s{seed}] época {ep:3d}/{ep_total}  pérdida {h['perdida_media']:.4f}  "
                    f"IoU entrenamiento {h['iou_entrenamiento']:.4f}  lr {h['lr_final']:.2e}  ({h['seg_epoca']:.0f} s)")
        av.set(detalle=f"semilla {seed}, época {ep}/{ep_total}",
               ultimo={"config": name, "semilla": seed, "epoca": ep, "perdida": round(h["perdida_media"], 4),
                       "iou_entrenamiento": round(h["iou_entrenamiento"], 4)})

    if t == "unet":
        h = P3.train_unet(model, tr, seed=seed, epochs=ep_total, on_step=on_step, on_epoch=on_epoch)
        state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    else:
        h = L.train(model, tr, seed=seed, prompts=prompts, epochs=ep_total, on_step=on_step, on_epoch=on_epoch,
                    loss_fn=functools.partial(P3.masked_bce_tversky, alpha=P3.TV_ALPHA, beta=P3.TV_BETA))
        state = L.trained_state(model)
    P3.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_file(state, str(P3.MODEL_DIR / f"{name}_seed{seed}.safetensors"))

    av.etapa(f"{name} · semilla {seed} · evaluando valid (individual y TTA D4)", 2 * len(va), "chip",
             detalle=f"semilla {seed}, evaluando")
    if t == "unet":
        p_ind = P3.predict_unet(model, va, tta=False, on_chip=lambda k: av.paso(k * W_EVAL[t], n=k))
        p_tta = P3.predict_unet(model, va, tta=True, on_chip=lambda k: av.paso(k * W_TTA[t], n=k))
    else:
        p_ind = L.predict_probs(model, va, prompts, tta=False, on_chip=lambda k: av.paso(k * W_EVAL[t], n=k))
        p_tta = L.predict_probs(model, va, prompts, tta=True, on_chip=lambda k: av.paso(k * W_TTA[t], n=k))
    L.save_probs(P3.probs_file(name, seed), p_ind)
    L.save_probs(P3.probs_file(name, seed, tta=True), p_tta)
    del model
    torch.cuda.empty_cache()

    out = {"config": name, "definicion": ex, "semilla": seed, "entrenamiento": h, "valid": {}}
    for key, probs in (("individual", p_ind), ("tta", p_tta)):
        rows = L.rows_from_probs(probs, va, P3.THRESHOLD)
        res = {"agregado": stats.aggregate(rows), "por_chip": rows}
        if refs is not None:
            res.update(P3.comparisons(rows, seed, refs, paso2, regions))
        out["valid"][key] = res
    av.paso(W_STATS, n=0)
    a, tt = out["valid"]["individual"]["agregado"], out["valid"]["tta"]["agregado"]
    av.info(f"  [{name} s{seed}] IoU global VALID {a['global_iou']:.4f} (con TTA {tt['global_iou']:.4f}) | "
            f"IoU medio {a['mean_chip_iou']:.4f} | precisión {H1._fmt(a['precision'])} | "
            f"sensibilidad {H1._fmt(a['recall'])} | {h['minutos']:.1f} min de entrenamiento")
    av.set(ultimo={"config": name, "semilla": seed, "iou_global_valid": round(a["global_iou"], 4),
                   "iou_global_valid_tta": round(tt["global_iou"], 4)})
    return out


def run_ensemble(name, va, P, av, refs, paso2, regions) -> dict:
    n = len(va)
    out = {"config": name, "definicion": P3.EXPERIMENTS[name], "semillas": list(P["seeds"]),
           "protocolo": {"unet": {"base": P3.UNET_BASE, "epocas": P["epochs_unet"], "calentamiento": P3.UNET_WARMUP,
                                  "lr": P3.UNET_LR, "wd": P3.UNET_WD, "lote": P3.UNET_BATCH, "perdida": "BCE + Dice"},
                         "lora": {"r": L.LORA_R, "alpha": L.LORA_ALPHA, "epocas": P["epochs_lora"], "lr": L.LR,
                                  "lote": L.BATCH, "perdida": f"BCE + Tversky (a {P3.TV_ALPHA}, b {P3.TV_BETA})"},
                         "umbral": P3.THRESHOLD, "aumentos": "D4 al azar por muestra"},
           "ensamble": {}, "chips_valid": P["valid"]}
    for key, tta in (("individual", False), ("tta", True)):
        probs = [L.load_probs(P3.probs_file(name, s, tta), n) for s in P["seeds"]]
        rows = L.rows_from_probs(L.ensemble(probs), va, P3.THRESHOLD)
        res = {"agregado": stats.aggregate(rows), "por_chip": rows}
        if refs is not None:
            res.update(P3.comparisons(rows, None, refs, paso2, regions, tta=tta))
        out["ensamble"][key] = res
    seeds = [json.loads(P3.seed_file(name, s).read_text(encoding="utf-8")) for s in P["seeds"]]
    for key in ("individual", "tta"):
        out[f"semillas_{key}"] = stats.mean_std([r["valid"][key]["agregado"]["global_iou"] for r in seeds])
    av.paso(W_STATS, n=0)
    av.info(f"  [{name}] TERMINADO: IoU global valid {out['semillas_individual']['media']:.4f} ± "
            f"{out['semillas_individual']['desv']:.4f} (modelo individual) | con TTA "
            f"{out['semillas_tta']['media']:.4f} | ensamble {out['ensamble']['individual']['agregado']['global_iou']:.4f}"
            f" | ensamble + TTA {out['ensamble']['tta']['agregado']['global_iou']:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rapido", action="store_true", help="prueba de humo: pocos chips, 2 épocas, 1 semilla")
    args = ap.parse_args()

    config.setup_runtime(seed=0, verbose=False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dev = config.get_device()
    if dev != "cuda":
        raise SystemExit("El paso 3 necesita la GPU (CUDA).")
    parts = json.loads((config.RESULTS_DIR / "particiones_usadas.json").read_text(encoding="utf-8"))
    P = {"train": parts["train"], "valid": parts["valid"], "epochs_unet": P3.UNET_EPOCHS,
         "epochs_lora": L.EPOCHS, "seeds": tuple(config.SEEDS)}
    if args.rapido:
        import shutil

        P3.OUT_DIR = P3.OUT_DIR / "prueba_rapida"
        P3.MODEL_DIR = P3.OUT_DIR / "modelos"
        P3.BITACORA = P3.OUT_DIR / "no_existe.md"
        shutil.rmtree(P3.OUT_DIR, ignore_errors=True)
        P.update(train=P["train"][:16], valid=P["valid"][:6], epochs_unet=2, epochs_lora=1, seeds=(0,))
    P3.OUT_DIR.mkdir(parents=True, exist_ok=True)
    config.SEEDS = P["seeds"]

    todo_seeds = [(nm, s) for nm in P3.EXPERIMENTS for s in P["seeds"] if not P3.seed_file(nm, s).exists()]
    todo_ens = [nm for nm in P3.EXPERIMENTS if not P3.config_file(nm).exists()]
    n = {"train": len(P["train"]), "valid": len(P["valid"])}
    print(f"GPU: {torch.cuda.get_device_name(0)} | datos: {config.DATA_DIR}")
    print(f"Semillas por correr: {len(todo_seeds)} | ensambles por calcular: {len(todo_ens)}")
    av = Avance(plan_units(todo_seeds, todo_ens, n, P), P3.OUT_DIR)
    P3.update_bitacora()
    try:
        refs = H1.load_references(P["valid"]) if not args.rapido else None
        paso2 = P3.load_paso2_ref(P["valid"]) if not args.rapido else None
        regions = [data.region_of(c) for c in P["valid"]]
        prepared: dict = {}

        def items_for(rep):
            if rep not in prepared:
                av.etapa(f"Preparando chips con la entrada {rep} (en la GPU)", n["train"] + n["valid"], "chip",
                         detalle=f"preparando {rep}")
                prepared[rep] = (L.prepare_chips(P["train"], rep, dev, on_chip=lambda: av.paso(W_PREP)),
                                 L.prepare_chips(P["valid"], rep, dev, on_chip=lambda: av.paso(W_PREP)))
            return prepared[rep]

        for name, ex in P3.EXPERIMENTS.items():
            pending = [s for s in P["seeds"] if (name, s) in todo_seeds]
            if not pending and name not in todo_ens:
                continue
            av.info(f"\n=== Paso {ex['paso']} · {name}: {ex['que']} ===")
            tr, va = items_for(ex["rep"])
            for seed in pending:
                res = run_seed(name, seed, tr, va, P, av, refs, paso2, regions, dev)
                save_json(res, P3.seed_file(name, seed))
                P3.update_bitacora()
            res = run_ensemble(name, va, P, av, refs, paso2, regions)
            save_json(res, P3.config_file(name))
            av.estado["configs_terminadas"].append(name)
            av.set(config_actual=None, detalle="")
            P3.update_bitacora()
        av.set(terminado=True, etapa="terminado")
        av.bar.n = 100.0
        av.bar.refresh()
    except KeyboardInterrupt:
        av.set(interrumpido=True)
        av.info("\nInterrumpido. Cada semilla terminada quedó guardada; al relanzar se continúa.")
    finally:
        av.cerrar()
        P3.update_bitacora()
    print("\nListo. Resultados en", P3.OUT_DIR, "| bitácora:", P3.BITACORA)


if __name__ == "__main__":
    main()
