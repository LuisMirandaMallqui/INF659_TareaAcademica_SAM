"""Hacia IoU 0.60 - paso 2 (H5): LoRA en el codificador de SAM, con barra de avance visible.

    python tools/hacia060_paso2.py            # 3 configuraciones x 3 semillas (unas 3 h en la RTX 3070)
    python tools/hacia060_paso2.py --rapido   # prueba de humo de pocos minutos (no toca resultados ni bitacora)

Mas facil: doble clic en tools/lanzar_hacia060_paso2.bat.

Mientras corre: la ventana muestra el % total y la epoca actual (perdida e IoU de entrenamiento);
results/hacia_060/paso2/estado.json y ejecucion.log tienen lo mismo; la bitacora se actualiza sola
al terminar cada semilla. Se puede cortar con Ctrl+C: cada semilla terminada queda guardada.
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
    try:  # mientras este proceso viva, Windows no entra en suspension por inactividad (se libera al salir)
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from hacia060_paso1 import Avance  # noqa: E402
from inf659sam import config, data, stats  # noqa: E402
from inf659sam import hacia060 as H1  # noqa: E402
from inf659sam import lora_sam as L  # noqa: E402
from inf659sam.cache import save_json  # noqa: E402

# segundos aproximados en la RTX 3070, para que el % siga al tiempo real
W_PREP, W_IMG, W_EVAL, W_TTA, W_BUILD, W_STATS = 0.3, 0.27, 0.08, 0.6, 5.0, 12.0


def plan_units(todo_seeds, todo_ens, reps, n, epochs, verify) -> float:
    u = (n["valid"] * (W_PREP + W_EVAL) + W_BUILD) if verify else 0.0
    u += len(reps) * (n["train"] + n["valid"]) * W_PREP
    u += len(todo_seeds) * (W_BUILD + epochs * n["train"] * W_IMG + n["valid"] * (W_EVAL + W_TTA) + W_STATS)
    u += len(todo_ens) * W_STATS
    return u


def verificar(dev, valid_chips, av: Avance) -> dict:
    """Con LoRA en cero y el decodificador de H2, la evaluacion del paso 2 debe dar ~0.4684."""
    f = config.CACHE_DIR / "modelos" / "mask_decoder_ft_seed0.safetensors"
    if not f.exists():
        av.info("  (no están los pesos de H2: se omite la verificación)")
        return {"omitida": True}
    av.etapa("Verificando el preprocesamiento en GPU contra H2", len(valid_chips), "chip")
    items = L.prepare_chips(valid_chips, "vh3", dev, on_chip=lambda: av.paso(W_PREP))
    model = L.build_model(dev)
    model.mask_decoder.load_state_dict(load_file(str(f)))
    av.paso(W_BUILD, n=0)
    probs = L.predict_probs(model, items, "otsu", on_chip=lambda k: av.paso(k * W_EVAL, n=0))
    g = H1.global_iou(L.rows_from_probs(probs, items))
    del model, items
    torch.cuda.empty_cache()
    res = {"iou_global_nuevo": g, "iou_global_h2": 0.46844701756716195, "diferencia": g - 0.46844701756716195}
    if len(valid_chips) == 86:
        av.info(f"  Verificación: IoU global {g:.4f} con el preprocesamiento en GPU (H2 publicado: 0.4684, "
                f"diferencia {g - 0.4684:+.4f}; tolerancia ±0.01)")
        if abs(res["diferencia"]) > 0.01:
            raise SystemExit("El preprocesamiento en GPU no reproduce H2 dentro de ±0.01: se detiene.")
    return res


def run_seed(rep, prompts, seed, items_train, items_valid, P, av, refs, step1, regions, dev) -> dict:
    name = L.config_name(rep, prompts)
    usable = len(items_train) if prompts == "sin" else sum(bool(len(it["pos_c"])) for it in items_train)
    av.ajustar_total((usable - len(items_train)) * P["epochs"] * W_IMG)
    av.set(config_actual=name, detalle=f"semilla {seed}, construyendo el modelo")
    L.update_bitacora()
    torch.cuda.empty_cache()
    model = L.build_model(dev)
    av.paso(W_BUILD, n=0)
    av.etapa(f"{name} · semilla {seed} · entrenando ({P['epochs']} épocas × {usable} chips)",
             P["epochs"] * usable, "img")
    ep_now = {"e": 1}

    def on_step(k, loss):
        av.paso(k * W_IMG, n=k, epoca=f"{ep_now['e']}/{P['epochs']}", perdida=f"{loss:.3f}")

    def on_epoch(ep, h):
        ep_now["e"] = ep + 1
        av.info(f"  [{name} s{seed}] época {ep:2d}/{P['epochs']}  pérdida {h['perdida_media']:.4f}  "
                f"IoU entrenamiento {h['iou_entrenamiento']:.4f}  lr {h['lr_final']:.2e}  ({h['seg_epoca']:.0f} s)")
        av.set(detalle=f"semilla {seed}, época {ep}/{P['epochs']}",
               ultimo={"config": name, "semilla": seed, "epoca": ep, "perdida": round(h["perdida_media"], 4),
                       "iou_entrenamiento": round(h["iou_entrenamiento"], 4)})

    h = L.train(model, items_train, seed=seed, prompts=prompts, epochs=P["epochs"], on_step=on_step,
                on_epoch=on_epoch)
    L.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_file(L.trained_state(model), str(L.MODEL_DIR / f"{name}_seed{seed}.safetensors"))

    av.etapa(f"{name} · semilla {seed} · evaluando valid (individual y TTA D4)", 2 * len(items_valid), "chip",
             detalle=f"semilla {seed}, evaluando")
    p_ind = L.predict_probs(model, items_valid, prompts, tta=False, on_chip=lambda k: av.paso(k * W_EVAL, n=k))
    p_tta = L.predict_probs(model, items_valid, prompts, tta=True, on_chip=lambda k: av.paso(k * W_TTA, n=k))
    L.save_probs(L.probs_file(rep, prompts, seed), p_ind)
    L.save_probs(Path(str(L.probs_file(rep, prompts, seed)).replace("_probs_", "_probs_tta_")), p_tta)
    del model
    torch.cuda.empty_cache()

    out = {"config": name, "rep": rep, "prompts": prompts, "semilla": seed, "entrenamiento": h, "valid": {}}
    for key, probs in (("individual", p_ind), ("tta", p_tta)):
        rows = L.rows_from_probs(probs, items_valid)
        res = {"agregado": stats.aggregate(rows), "por_chip": rows}
        if refs is not None:
            res.update(L.comparisons(rows, seed, refs, step1, regions))
        out["valid"][key] = res
    av.paso(W_STATS, n=0)
    a, t = out["valid"]["individual"]["agregado"], out["valid"]["tta"]["agregado"]
    av.info(f"  [{name} s{seed}] IoU global VALID {a['global_iou']:.4f} (con TTA {t['global_iou']:.4f}) | "
            f"IoU medio {a['mean_chip_iou']:.4f} | precisión {H1._fmt(a['precision'])} | "
            f"sensibilidad {H1._fmt(a['recall'])} | {h['minutos']:.1f} min de entrenamiento")
    av.set(ultimo={"config": name, "semilla": seed, "iou_global_valid": round(a["global_iou"], 4),
                   "iou_global_valid_tta": round(t["global_iou"], 4)})
    return out


def run_ensemble(rep, prompts, items_valid, P, av, refs, step1, regions) -> dict:
    name = L.config_name(rep, prompts)
    n = len(items_valid)
    out = {"config": name, "rep": rep, "prompts": prompts, "semillas": list(P["seeds"]),
           "protocolo": {"lora_r": L.LORA_R, "lora_alpha": L.LORA_ALPHA, "epocas": P["epochs"],
                         "calentamiento": L.WARMUP_EPOCHS, "lr": L.LR, "wd": L.WEIGHT_DECAY, "lote": L.BATCH,
                         "umbral": L.THRESHOLD, "aumentos": "D4 al azar por muestra", "perdida": "BCE + Dice"},
           "ensamble": {}, "chips_valid": P["valid"]}
    for key, tag in (("individual", "_probs_"), ("tta", "_probs_tta_")):
        probs = [L.load_probs(Path(str(L.probs_file(rep, prompts, s)).replace("_probs_", tag)), n) for s in P["seeds"]]
        rows = L.rows_from_probs(L.ensemble(probs), items_valid)
        res = {"agregado": stats.aggregate(rows), "por_chip": rows}
        if refs is not None:
            res.update(L.comparisons(rows, None, refs, step1, regions))
        out["ensamble"][key] = res
    seeds = [json.loads(L.seed_file(rep, prompts, s).read_text(encoding="utf-8")) for s in P["seeds"]]
    for key in ("individual", "tta"):
        out[f"semillas_{key}"] = stats.mean_std([r["valid"][key]["agregado"]["global_iou"] for r in seeds])
    av.paso(W_STATS, n=0)
    av.info(f"  [{name}] TERMINADO: IoU global valid {out['semillas_individual']['media']:.4f} ± "
            f"{out['semillas_individual']['desv']:.4f} (modelo individual) | con TTA "
            f"{out['semillas_tta']['media']:.4f} | ensamble {out['ensamble']['individual']['agregado']['global_iou']:.4f}"
            f" | ensamble + TTA {out['ensamble']['tta']['agregado']['global_iou']:.4f}")
    return out


def figura_resumen():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    names, vals = [], {"individual": [], "tta": [], "ens": [], "ens_tta": []}
    for rep, prompts in L.CONFIGS:
        seeds = [json.loads(L.seed_file(rep, prompts, s).read_text(encoding="utf-8"))
                 for s in config.SEEDS if L.seed_file(rep, prompts, s).exists()]
        if not seeds:
            continue
        for r in seeds:
            hh = r["entrenamiento"]["historial"]
            axes[0].plot([e["epoca"] for e in hh], [e["perdida_media"] for e in hh], alpha=0.7,
                         label=f"{r['config']} s{r['semilla']}")
        names.append(L.config_name(rep, prompts))
        vals["individual"].append(np.mean([r["valid"]["individual"]["agregado"]["global_iou"] for r in seeds]))
        vals["tta"].append(np.mean([r["valid"]["tta"]["agregado"]["global_iou"] for r in seeds]))
        c = L.config_file(rep, prompts)
        if c.exists():
            cc = json.loads(c.read_text(encoding="utf-8"))
            vals["ens"].append(cc["ensamble"]["individual"]["agregado"]["global_iou"])
            vals["ens_tta"].append(cc["ensamble"]["tta"]["agregado"]["global_iou"])
        else:
            vals["ens"].append(np.nan)
            vals["ens_tta"].append(np.nan)
    if not names:
        return
    axes[0].set(xlabel="época", ylabel="pérdida BCE + Dice (entrenamiento)", title="Paso 2: entrenamiento con LoRA")
    axes[0].legend(fontsize=6)
    x = np.arange(len(names))
    for j, (k, lab) in enumerate((("individual", "individual"), ("tta", "+ TTA"), ("ens", "ensamble"),
                                  ("ens_tta", "ensamble + TTA"))):
        axes[1].bar(x + (j - 1.5) * 0.2, vals[k], width=0.2, label=lab)
    for y, lab in ((0.431, "H2"), (0.472, "paso 1"), (0.60, "meta 0.60")):
        axes[1].axhline(y, ls="--", lw=1, color="gray")
        axes[1].text(len(names) - 0.5, y + 0.005, lab, fontsize=7, ha="right")
    axes[1].set_xticks(x, names, rotation=15, fontsize=7)
    axes[1].set(ylabel="IoU global valid", ylim=(0, 0.8), title="Paso 2 hacia 0.60")
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.FIG_DIR / "hacia060_paso2.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rapido", action="store_true", help="prueba de humo: pocos chips, 1 época, 1 semilla")
    args = ap.parse_args()

    config.setup_runtime(seed=0, verbose=False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = config.get_device()
    if dev != "cuda":
        raise SystemExit("El paso 2 necesita la GPU (CUDA).")
    parts = json.loads((config.RESULTS_DIR / "particiones_usadas.json").read_text(encoding="utf-8"))
    P = {"train": parts["train"], "valid": parts["valid"], "epochs": L.EPOCHS, "seeds": tuple(config.SEEDS)}
    if args.rapido:
        import shutil

        L.OUT_DIR = L.OUT_DIR / "prueba_rapida"
        L.MODEL_DIR = L.OUT_DIR / "modelos"
        L.BITACORA = L.OUT_DIR / "no_existe.md"
        config.FIG_DIR = L.OUT_DIR
        shutil.rmtree(L.OUT_DIR, ignore_errors=True)
        P.update(train=P["train"][:8], valid=P["valid"][:6], epochs=1, seeds=(0,))
    L.OUT_DIR.mkdir(parents=True, exist_ok=True)
    config.SEEDS = P["seeds"]  # results_markdown y figura leen config.SEEDS

    todo_seeds = [(r, p, s) for r, p in L.CONFIGS for s in P["seeds"] if not L.seed_file(r, p, s).exists()]
    todo_ens = [(r, p) for r, p in L.CONFIGS if not L.config_file(r, p).exists()]
    reps = list(dict.fromkeys(r for r, p, _ in todo_seeds)) + [r for r, p in todo_ens if not any(
        r == rr for rr, _, _ in todo_seeds)]
    reps = list(dict.fromkeys(reps))
    n = {"train": len(P["train"]), "valid": len(P["valid"])}
    print(f"GPU: {torch.cuda.get_device_name(0)} | datos: {config.DATA_DIR}")
    print(f"Semillas por correr: {len(todo_seeds)} | ensambles por calcular: {len(todo_ens)}")
    av = Avance(plan_units(todo_seeds, todo_ens, reps, n, P["epochs"], verify=bool(todo_seeds)), L.OUT_DIR)
    L.update_bitacora()
    try:
        ver = verificar(dev, P["valid"], av) if todo_seeds else {"omitida": True}
        av.set(verificacion=ver)
        refs = H1.load_references(P["valid"]) if not args.rapido else None
        step1 = L.load_step1_best(P["valid"]) if not args.rapido else None
        regions = [data.region_of(c) for c in P["valid"]]
        prepared: dict = {}

        def items_for(rep):
            if rep not in prepared:
                av.etapa(f"Preparando chips con la entrada {rep} (en la GPU)", n["train"] + n["valid"], "chip",
                         detalle=f"preparando {rep}")
                prepared[rep] = (L.prepare_chips(P["train"], rep, dev, on_chip=lambda: av.paso(W_PREP)),
                                 L.prepare_chips(P["valid"], rep, dev, on_chip=lambda: av.paso(W_PREP)))
            return prepared[rep]

        for rep, prompts in L.CONFIGS:
            name = L.config_name(rep, prompts)
            pending = [s for s in P["seeds"] if (rep, prompts, s) in todo_seeds]
            if not pending and (rep, prompts) not in todo_ens:
                continue
            av.info(f"\n=== {name}: {L.REP_INFO[rep]} · {L.PROMPT_INFO[prompts]} · LoRA r={L.LORA_R} ===")
            tr, va = items_for(rep)
            for seed in pending:
                res = run_seed(rep, prompts, seed, tr, va, P, av, refs, step1, regions, dev)
                save_json(res, L.seed_file(rep, prompts, seed))
                L.update_bitacora()
                try:
                    figura_resumen()
                except Exception as exc:
                    av.info(f"  (no se pudo dibujar la figura: {exc!r})")
            res = run_ensemble(rep, prompts, va, P, av, refs, step1, regions)
            save_json(res, L.config_file(rep, prompts))
            av.estado["configs_terminadas"].append(name)
            av.set(config_actual=None, detalle="")
            L.update_bitacora()
            try:
                figura_resumen()
            except Exception as exc:
                av.info(f"  (no se pudo dibujar la figura: {exc!r})")
        av.set(terminado=True, etapa="terminado")
        av.bar.n = 100.0
        av.bar.refresh()
    except KeyboardInterrupt:
        av.set(interrumpido=True)
        av.info("\nInterrumpido. Cada semilla terminada quedó guardada; al relanzar se continúa.")
    finally:
        av.cerrar()
        L.update_bitacora()
    print("\nListo. Resultados en", L.OUT_DIR, "| bitácora:", L.BITACORA)


if __name__ == "__main__":
    main()
