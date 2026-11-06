"""Hacia IoU 0.60 - paso 1 (H4). Ejecuta todo con barra de avance visible.

    python tools/hacia060_paso1.py                 # las 6 configuraciones (unas 2 a 3 h en la RTX 3070)
    python tools/hacia060_paso1.py --configs db3_otsu,db3_sin
    python tools/hacia060_paso1.py --rapido        # prueba de humo de ~2 min (no toca resultados ni bitacora)

Mas facil: doble clic en tools/lanzar_hacia060.bat (abre una ventana con el avance).

Mientras corre:
  * La ventana muestra el % total, la etapa actual y el tiempo restante aproximado.
  * results/hacia_060/estado.json tiene lo mismo en texto (lo puede leer cualquiera).
  * results/hacia_060/ejecucion.log guarda una linea por epoca.
  * docs/bitacora_hacia_060.md se actualiza sola al terminar cada configuracion.
Se puede cortar con Ctrl+C: lo ya terminado queda guardado y al relanzar se salta.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if os.name == "nt":
    os.system("")  # activa los codigos ANSI de la consola de Windows (barras de tqdm en dos lineas)
    try:  # sin "edicion rapida": un clic en la ventana congelaria el entrenamiento hasta pulsar una tecla
        import ctypes

        _k32 = ctypes.windll.kernel32
        _h, _modo = _k32.GetStdHandle(-10), ctypes.c_uint32()
        if _k32.GetConsoleMode(_h, ctypes.byref(_modo)):
            _k32.SetConsoleMode(_h, (_modo.value & ~0x0040) | 0x0080)  # quita QUICK_EDIT, pone EXTENDED_FLAGS
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from tqdm import tqdm  # noqa: E402

from inf659sam import baseline, config, data, stats  # noqa: E402
from inf659sam import hacia060 as H  # noqa: E402
from inf659sam.cache import cache_dir, encode_and_cache, save_json  # noqa: E402
from inf659sam.decoder_ft import load_training_set  # noqa: E402

# pesos aproximados (segundos en la RTX 3070) para que el % refleje tiempo y no cantidad de pasos
W_RANGE, W_ENC, W_LOAD, W_EVAL, W_BOOT = 0.15, 0.8, 0.12, 0.03, 15.0
W_STEP = {"otsu": 0.05, "sin": 0.03}


class Avance:
    """Barra total (en %) + barra de la etapa actual + estado.json + log."""

    def __init__(self, total: float, out_dir: Path):
        self.total, self.done = max(total, 1e-9), 0.0
        self.t0 = time.time()
        self.out_dir = out_dir
        self.estado_path, self.log_path = out_dir / "estado.json", out_dir / "ejecucion.log"
        self.estado = {"inicio": time.strftime("%Y-%m-%d %H:%M:%S"), "terminado": False, "configs_terminadas": []}
        self.bar = tqdm(total=100.0, position=0, desc="AVANCE TOTAL", dynamic_ncols=True,
                        bar_format="{desc}: {percentage:5.1f}% |{bar}| {elapsed} transcurrido, ~{remaining} restante")
        self.sub = None
        self._last_write = 0.0

    def etapa(self, desc: str, total: int, unit: str = "paso", **estado):
        if self.sub is not None:
            self.sub.close()
        self.sub = tqdm(total=total, position=1, desc=f"  {desc}", unit=unit, leave=False, dynamic_ncols=True,
                        bar_format="{desc}: {n_fmt}/{total_fmt} {unit} |{bar}| {rate_fmt}{postfix}")
        self.estado.update(etapa=desc, **estado)
        self._write(force=True)

    def paso(self, units: float, n: int = 1, **postfix):
        self.done = min(self.done + units, self.total)
        self.bar.n = 100.0 * self.done / self.total
        self.bar.refresh()
        if self.sub is not None:
            self.sub.update(n)
            if postfix:
                self.sub.set_postfix(postfix, refresh=False)
        self._write()

    def ajustar_total(self, delta: float):
        self.total = max(self.total + delta, self.done + 1e-9)
        self.paso(0, n=0)

    def info(self, msg: str):
        tqdm.write(msg)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    def set(self, **kw):
        self.estado.update(kw)
        self._write(force=True)

    def _write(self, force: bool = False):
        now = time.time()
        if not force and now - self._last_write < 2:
            return
        self._last_write = now
        pct = 100.0 * self.done / self.total
        el = now - self.t0
        self.estado.update(actualizado=time.strftime("%Y-%m-%d %H:%M:%S"), avance_pct=round(pct, 1),
                           transcurrido_min=round(el / 60, 1),
                           restante_min_aprox=round(el / max(pct, 1e-6) * (100 - pct) / 60, 1) if pct > 0.5 else None)
        try:
            self.estado_path.write_text(json.dumps(self.estado, indent=1, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    def cerrar(self):
        if self.sub is not None:
            self.sub.close()
        self.bar.close()


def plan_units(todo, reps_to_encode, n, epochs, n_lr, n_seeds) -> float:
    u = n["train"] * W_RANGE if not H.RANGES_FILE.exists() else 0.0
    u += sum(reps_to_encode.values()) * W_ENC
    for rep, prompts in todo:
        sweep_ep = n["core"] * W_STEP[prompts] + n["dev"] * W_EVAL
        final_ep = n["train"] * W_STEP[prompts]
        u += (n["core"] + n["dev"]) * W_LOAD + n_lr * epochs * sweep_ep + n["dev"] * W_EVAL
        u += (n["train"] + n["valid"]) * W_LOAD + n_seeds * (epochs * final_ep + n["valid"] * W_EVAL) + W_BOOT
    return u


def load_items(chips, rep, av: Avance, fn):
    out = []
    for c in chips:
        out.extend(fn([c], rep))
        av.paso(W_LOAD)
    return out


def verificar(sam, valid, av: Avance) -> dict:
    """La evaluacion rapida debe reproducir H2 (semilla 0) chip por chip antes de confiar en ella."""
    f = config.CACHE_DIR / "modelos" / "mask_decoder_ft_seed0.safetensors"
    if not f.exists() or not (cache_dir("vh3") / f"{valid[0]}.npz").exists():
        av.info("  (sin pesos de H2 o sin cache vh3: se omite la verificacion)")
        return {"omitida": True}
    av.etapa("Verificando que la evaluación nueva reproduce H2", len(valid), "chip")
    pre = H.decoder_state_cpu(sam.model)
    ev = []
    for c in valid:
        ev.extend(H.load_eval_set([c], "vh3"))
        av.paso(0, n=1)
    sam.model.mask_decoder.load_state_dict(load_file(str(f)))
    rows = H.evaluate(sam.model, ev, "otsu", (0.5,))[0.5]
    sam.model.mask_decoder.load_state_dict(pre)
    ref = H.load_references(valid)["h2"][0] if len(valid) == 86 else None
    g = H.global_iou(rows)
    res = {"iou_global_nuevo": g}
    if ref is not None:
        g_ref = H.global_iou(ref)
        dmax = max(abs(a["tp"] - b["tp"]) + abs(a["fp"] - b["fp"]) for a, b in zip(rows, ref))
        res.update(iou_global_h2=g_ref, diferencia=g - g_ref, max_dif_pixeles_por_chip=int(dmax))
        av.info(f"  Verificación: IoU global {g:.4f} (H2 semilla 0 publicado: {g_ref:.4f}, "
                f"diferencia {g - g_ref:+.5f}; máx. {dmax} píxeles distintos en un chip)")
        if abs(g - g_ref) > 2e-3:
            raise SystemExit("La evaluación rápida NO reproduce H2: se detiene para no producir números incomparables.")
    return res


def run_config(sam, rep, prompts, P, av: Avance, refs, regions) -> dict:
    name = H.config_name(rep, prompts)
    av.set(config_actual=name, detalle="cargando datos")
    H.update_bitacora()
    av.info(f"\n=== Configuración {name}: {H.REP_INFO[rep]} · {H.PROMPT_INFO[prompts]} ===")
    model = sam.model
    PRE = H.decoder_state_cpu(model)
    t_cfg = time.time()

    av.etapa(f"{name} · cargando train-core y train-dev", len(P["core"]) + len(P["dev"]), "chip")
    core = load_items(P["core"], rep, av, load_training_set)
    dev = load_items(P["dev"], rep, av, H.load_eval_set)
    n_core_usable = len(core) if prompts == "sin" else sum(it["opened"].any() for it in core)

    # ---------------------------------------------------------------- barrido lr x epoca en train-dev
    sweep, best = [], {"iou": -1.0}
    for lr in P["lrs"]:
        model.mask_decoder.load_state_dict(PRE)
        av.etapa(f"{name} · barrido lr={lr:g} (train-core → train-dev)", P["epochs"] * n_core_usable, "chip",
                 detalle=f"barrido lr={lr:g}")
        curva = []

        def on_epoch(ep, h, lr=lr, curva=curva):
            rows = H.evaluate(model, dev, prompts, (0.5,))[0.5]
            iou = H.global_iou(rows)
            av.paso(len(dev) * W_EVAL, n=0, epoca=f"{ep}/{P['epochs']}", iou_dev=f"{iou:.3f}")
            curva.append({"epoca": ep, "iou_dev": iou, "perdida": h["perdida_media"]})
            av.info(f"  [{name}] lr={lr:g} época {ep:2d}/{P['epochs']}  pérdida {h['perdida_media']:.4f}  "
                    f"IoU train-dev {iou:.4f}")
            if iou > best["iou"]:
                best.update(iou=iou, lr=lr, epoca=ep, state=H.decoder_state_cpu(model))
            av.set(ultimo={"config": name, "lr": lr, "epoca": ep, "iou_train_dev": round(iou, 4),
                           "mejor_hasta_ahora": round(best["iou"], 4)})
            return {"iou_dev": iou}

        h = H.train_decoder_v2(model, core, seed=0, lr=lr, prompts=prompts, epochs_total=P["epochs"],
                               on_epoch=on_epoch, on_step=lambda k: av.paso(k * W_STEP[prompts], n=k))
        sweep.append({"lr": lr, "curva": curva, "minutos": h["minutos"],
                      "mejor_epoca": max(curva, key=lambda e: e["iou_dev"])["epoca"],
                      "mejor_iou_dev": max(e["iou_dev"] for e in curva)})

    # ---------------------------------------------------------------- umbral en train-dev
    model.mask_decoder.load_state_dict(best["state"])
    by_thr = H.evaluate(model, dev, prompts, H.THRESHOLDS)
    av.paso(len(dev) * W_EVAL, n=0)
    iou_thr = {t: H.global_iou(r) for t, r in by_thr.items()}
    thr = max(H.THRESHOLDS, key=lambda t: (round(iou_thr[t], 6), -abs(t - 0.5)))
    sel = {"lr": best["lr"], "epoca": best["epoca"], "umbral": thr, "iou_train_dev_umbral_0.5": best["iou"],
           "iou_train_dev": iou_thr[thr], "iou_train_dev_por_umbral": {str(t): v for t, v in iou_thr.items()}}
    av.info(f"  [{name}] ELEGIDO en train-dev: lr={best['lr']:g}, época {best['epoca']}, umbral {thr:g} "
            f"→ IoU train-dev {iou_thr[thr]:.4f}")
    # las epocas no entrenadas en la corrida final ya no se haran
    av.ajustar_total(-(P["epochs"] - best["epoca"]) * len(P["seeds"]) * len(P["train"]) * W_STEP[prompts])
    del core, dev

    # ---------------------------------------------------------------- final: 3 semillas en todo train
    av.etapa(f"{name} · cargando train completo y valid", len(P["train"]) + len(P["valid"]), "chip",
             detalle="final: cargando datos")
    train = load_items(P["train"], rep, av, load_training_set)
    valid = load_items(P["valid"], rep, av, H.load_eval_set)
    n_train_usable = len(train) if prompts == "sin" else sum(it["opened"].any() for it in train)
    final = {"por_semilla": {}, "vs_h2": {}, "vs_otsu": {}, "por_chip": {}}
    H.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for seed in P["seeds"]:
        model.mask_decoder.load_state_dict(PRE)
        av.etapa(f"{name} · final semilla {seed} (train completo, {best['epoca']} épocas)",
                 best["epoca"] * n_train_usable, "chip", detalle=f"final semilla {seed}")

        def on_epoch_f(ep, h, seed=seed):
            av.info(f"  [{name}] final semilla {seed} época {ep:2d}/{best['epoca']}  pérdida {h['perdida_media']:.4f}")

        h = H.train_decoder_v2(model, train, seed=seed, lr=best["lr"], prompts=prompts, epochs_total=P["epochs"],
                               stop_at=best["epoca"], on_epoch=on_epoch_f,
                               on_step=lambda k: av.paso(k * W_STEP[prompts], n=k))
        ev = H.evaluate(model, valid, prompts, tuple(sorted({thr, 0.5})))
        av.paso(len(valid) * W_EVAL, n=0)
        rows = ev[thr]
        a = stats.aggregate(rows)
        final["por_semilla"][str(seed)] = {"agregado": a, "iou_global_umbral_0.5": H.global_iou(ev[0.5]),
                                           "historial": h["historial"], "minutos": h["minutos"]}
        final["por_chip"][str(seed)] = rows
        save_file({k: v.contiguous() for k, v in H.decoder_state_cpu(model).items()},
                  str(H.MODEL_DIR / f"{name}_seed{seed}.safetensors"))
        if refs is not None:
            final["vs_h2"][str(seed)] = H.compare(rows, refs["h2"][seed], regions)
            final["vs_otsu"][str(seed)] = H.compare(rows, refs["otsu"], regions)
        av.info(f"  [{name}] semilla {seed}: IoU global VALID {H._fmt(a['global_iou'], 4)} "
                f"(IoU medio {H._fmt(a['mean_chip_iou'], 4)}, precisión {H._fmt(a['precision'])}, "
                f"sensibilidad {H._fmt(a['recall'])})")
        av.set(ultimo={"config": name, "semilla": seed, "iou_global_valid": round(a["global_iou"], 4)})
    av.paso(W_BOOT, n=0)
    model.mask_decoder.load_state_dict(PRE)

    g = [final["por_semilla"][str(s)]["agregado"]["global_iou"] for s in P["seeds"]]
    m = [final["por_semilla"][str(s)]["agregado"]["mean_chip_iou"] for s in P["seeds"]]
    final["iou_global"], final["iou_medio"] = stats.mean_std(g), stats.mean_std(m)
    out = {"config": name, "rep": rep, "prompts": prompts, "descripcion": f"{H.REP_INFO[rep]} · {H.PROMPT_INFO[prompts]}",
           "protocolo": {"lrs": list(P["lrs"]), "epocas_programa": P["epochs"], "calentamiento_epocas": H.WARMUP_EPOCHS,
                         "lote": H.BATCH, "weight_decay": H.WEIGHT_DECAY, "umbrales": list(H.THRESHOLDS),
                         "semillas": list(P["seeds"]), "perdida": "BCE + Dice", "n_boot": H.N_BOOT},
           "barrido": sweep, "seleccion": sel, "final": final, "chips_valid": P["valid"],
           "minutos_total": (time.time() - t_cfg) / 60}
    av.info(f"  [{name}] TERMINADO: IoU global valid {final['iou_global']['media']:.4f} ± "
            f"{final['iou_global']['desv']:.4f} ({out['minutos_total']:.1f} min)")
    return out


def figura_resumen():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    done = [(r, p) for r, p in H.CONFIGS if H.result_file(r, p).exists()]
    if not done:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    names, means, sds = [], [], []
    for rep, prompts in done:
        r = json.loads(H.result_file(rep, prompts).read_text(encoding="utf-8"))
        s = r["seleccion"]
        curva = next(x["curva"] for x in r["barrido"] if x["lr"] == s["lr"])
        axes[0].plot([e["epoca"] for e in curva], [e["iou_dev"] for e in curva], label=f"{r['config']} (lr={s['lr']:g})")
        names.append(r["config"])
        means.append(r["final"]["iou_global"]["media"])
        sds.append(r["final"]["iou_global"]["desv"])
    axes[0].set(xlabel="época", ylabel="IoU global train-dev (umbral 0.5)", title="Curvas en train-dev (lr elegida)")
    axes[0].legend(fontsize=7)
    axes[1].bar(names, means, yerr=sds, capsize=3, color="#4C78A8")
    for y, lab in ((0.256, "Otsu 0.256"), (0.431, "H2 0.431"), (0.60, "meta 0.60")):
        axes[1].axhline(y, ls="--", lw=1, color="gray")
        axes[1].text(len(names) - 0.5, y + 0.005, lab, fontsize=7, ha="right")
    axes[1].set(ylabel="IoU global valid (3 semillas)", title="Paso 1 hacia 0.60", ylim=(0, 0.8))
    axes[1].tick_params(axis="x", rotation=30, labelsize=7)
    plt.tight_layout()
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.FIG_DIR / "hacia060_paso1.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default=",".join(H.config_name(r, p) for r, p in H.CONFIGS))
    ap.add_argument("--rapido", action="store_true", help="prueba de humo con pocos chips y 2 épocas")
    ap.add_argument("--rehacer", action="store_true", help="vuelve a correr configuraciones ya terminadas")
    args = ap.parse_args()

    config.setup_runtime(seed=0, verbose=False)
    parts = json.loads((config.RESULTS_DIR / "particiones_usadas.json").read_text(encoding="utf-8"))
    P = {"train": parts["train"], "core": parts["train_core"], "dev": parts["train_dev"], "valid": parts["valid"],
         "lrs": H.LRS, "epochs": H.EPOCHS, "seeds": config.SEEDS}
    wanted = [c for c in H.CONFIGS if H.config_name(*c) in args.configs.split(",")]

    if args.rapido:  # todo va a carpetas aparte: no toca rangos, caches, resultados ni bitacora reales
        tmp_res = H.OUT_DIR / "prueba_rapida"
        H.OUT_DIR = tmp_res
        H.RANGES_FILE = baseline.DB_RANGES_FILE = tmp_res / "rangos_db.json"
        H.MODEL_DIR = tmp_res / "modelos"
        H.BITACORA = tmp_res / "no_existe.md"
        config.FIG_DIR = tmp_res
        P.update(core=P["core"][:6], dev=P["dev"][:4], valid=P["valid"][:6], lrs=(1e-4,), epochs=2, seeds=(0,))
        P["train"] = P["core"] + P["dev"]
        wanted = [("db3", "otsu"), ("db3", "sin")]
        args.rehacer = True
    H.OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [c for c in wanted if args.rehacer or not H.result_file(*c).exists()]

    from inf659sam.sam_utils import SamWrapper

    print(f"Dispositivo: {config.get_device()} | datos: {config.DATA_DIR} | caché: {config.CACHE_DIR}")
    print(f"Configuraciones a correr: {', '.join(H.config_name(*c) for c in todo) or 'ninguna (todo terminado)'}")
    sam = SamWrapper()

    real_cache = config.CACHE_DIR
    if args.rapido:
        import shutil

        config.CACHE_DIR = real_cache / "prueba_rapida"
        for d in (config.CACHE_DIR, H.OUT_DIR):  # la prueba siempre empieza de cero
            shutil.rmtree(d, ignore_errors=True)
        H.OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chips = P["train"] + P["valid"]
    reps_to_encode = {}
    for rep in dict.fromkeys(r for r, _ in todo):
        if rep != "vh3":
            reps_to_encode[rep] = sum(not (cache_dir(rep) / f"{c}.npz").exists() for c in all_chips)
    n = {k: len(P[k]) for k in ("train", "core", "dev", "valid")}
    av = Avance(plan_units(todo, reps_to_encode, n, P["epochs"], len(P["lrs"]), len(P["seeds"])), H.OUT_DIR)
    H.update_bitacora()
    try:
        # 0. verificacion de la evaluacion rapida contra H2 (sobre la cache real vh3)
        config.CACHE_DIR = real_cache
        ver = verificar(sam, P["valid"], av)
        if args.rapido:
            config.CACHE_DIR = real_cache / "prueba_rapida"
        av.set(verificacion=ver)

        # 1. rangos fijos en dB, solo con train
        if not H.RANGES_FILE.exists():
            av.etapa("Calculando rangos fijos en dB (solo train)", len(P["train"]), "chip")
            rng = H.compute_db_ranges(P["train"], on_chip=lambda: av.paso(W_RANGE))
            save_json(rng, H.RANGES_FILE)
        rng = json.loads(H.RANGES_FILE.read_text(encoding="utf-8"))
        av.info("Rangos fijos en dB (train): " + ", ".join(
            f"{k} [{v['lo']:.1f}, {v['hi']:.1f}]" for k, v in rng["rangos"].items()))

        refs = H.load_references(P["valid"]) if not args.rapido else None
        regions = [data.region_of(c) for c in P["valid"]]
        for rep, prompts in todo:
            # 2. embeddings de la representacion (una sola vez por representacion)
            if reps_to_encode.get(rep):
                stamp = cache_dir(rep) / "rangos_db.json"
                if stamp.exists() and json.loads(stamp.read_text(encoding="utf-8")) != rng["rangos"]:
                    raise SystemExit(f"La caché {cache_dir(rep)} se hizo con otros rangos en dB: bórrala y relanza.")
                save_json(rng["rangos"], stamp)
                missing = [c for c in all_chips if not (cache_dir(rep) / f"{c}.npz").exists()]
                av.etapa(f"Codificando {len(missing)} chips con la entrada {rep} (codificador de SAM)", len(missing),
                         "chip", config_actual=f"embeddings {rep}", detalle="codificador")
                for c in missing:
                    encode_and_cache(sam, [c], rep=rep, probe=True)
                    av.paso(W_ENC)
                reps_to_encode[rep] = 0
            res = run_config(sam, rep, prompts, P, av, refs, regions)
            save_json(res, H.result_file(rep, prompts))
            av.estado["configs_terminadas"].append(res["config"])
            av.set(config_actual=None, detalle="")
            H.update_bitacora()
            try:
                figura_resumen()
            except Exception as exc:
                av.info(f"  (no se pudo dibujar la figura resumen: {exc!r})")
        av.set(terminado=True, etapa="terminado")
        av.bar.n = 100.0
        av.bar.refresh()
    except KeyboardInterrupt:
        av.set(interrumpido=True)
        av.info("\nInterrumpido por el usuario. Lo terminado quedó guardado; al relanzar se continúa.")
    finally:
        av.cerrar()
        H.update_bitacora()
    print("\nListo. Resultados en", H.OUT_DIR, "| bitácora:", H.BITACORA)


if __name__ == "__main__":
    main()
