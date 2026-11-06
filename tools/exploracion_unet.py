"""E1 y E2 (bitacora, seccion 2d). Exploratorio: test y Bolivia ya se usaron, nada de esto cambia el candidato.

    python tools/exploracion_unet.py                 # las 3 variantes x 3 semillas
    python tools/exploracion_unet.py --rapido        # prueba de humo (no toca resultados)
    python tools/exploracion_unet.py --esperar       # espera a que terminen R2 y la descarga, y luego corre

Variantes:
  unet_db3_jrc        4.o canal con la capa JRCWaterHand; receta de unet_db3.
  unet_db3_debil      preentrena 10 epocas con S1Weak + S2IndexLabelWeak (sin Bolivia), luego receta de unet_db3.
  unet_db3_debil_jrc  igual, con las etiquetas debiles corregidas con ocurrencia JRC >= 50 %.

Resultados: results/hacia_060/exploracion/. Tabla automatica en la bitacora (seccion 3j).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import evaluar_test as E  # noqa: E402  (activa tambien el "no suspender" de Windows)
from hacia060_paso1 import Avance  # noqa: E402
from inf659sam import config, data, stats  # noqa: E402
from inf659sam import lora_sam as L  # noqa: E402
from inf659sam import paso3 as P3  # noqa: E402
from inf659sam.cache import save_json  # noqa: E402
from inf659sam.hacia060 import compare  # noqa: E402
from inf659sam.sam_utils import SamWrapper  # noqa: E402

OUT = config.RESULTS_DIR / "hacia_060" / "exploracion"
MODELS = config.CACHE_DIR / "modelos" / "hacia_060_exploracion"
BITACORA = config.REPO_ROOT / "docs" / "bitacora_hacia_060.md"
MARK_START = "<!-- RESULTADOS_EXPLORACION:INICIO (se regenera solo; no editar a mano dentro) -->"
MARK_END = "<!-- RESULTADOS_EXPLORACION:FIN -->"
VARIANTES = {
    "unet_db3_jrc": {"canal_jrc": True, "debil": None, "que": "E1: U-Net con 4.º canal de agua permanente JRC"},
    "unet_db3_debil": {"canal_jrc": False, "debil": "cruda", "que": "E2a: preentrenada con etiquetas débiles crudas"},
    "unet_db3_debil_jrc": {"canal_jrc": False, "debil": "jrc", "que": "E2b: preentrenada con débiles corregidas con JRC"},
}
EPOCAS_DEBIL = 10
W_IMG, W_PRED, W_WEAK = 0.022, 0.07, 0.0225


def con_jrc(items):
    for it in items:
        j = tifffile.imread(Path(config.DATA_DIR) / "JRCWaterHand" / f"{it['chip']}_JRCWaterHand.tif")
        canal = torch.from_numpy((j == 1).astype(np.uint8) * 255)[None].to(it["rgb"].device)
        it["rgb"] = torch.cat([it["rgb"], canal])
    return items


def chips_debiles(corregir: bool, limite=None) -> list[dict]:
    """En la CPU (RAM): imagen db3, etiqueta debil y validez. Sin Bolivia."""
    d = Path(config.DATA_DIR) / "weak"
    items, omitidos = [], 0
    labs = sorted((d / "S2IndexLabelWeak").glob("*.tif"))
    for f in labs:
        chip = f.name.replace("_S2IndexLabelWeak.tif", "")
        s1f = d / "S1Weak" / f"{chip}_S1Weak.tif"
        if data.region_of(chip) == "Bolivia" or not s1f.exists():
            omitidos += 1
            continue
        s1 = tifffile.imread(s1f).astype(np.float32)
        lab = tifffile.imread(f).astype(np.int16)
        sv = np.isfinite(s1[0]) & np.isfinite(s1[1])
        if not sv.any():
            omitidos += 1
            continue
        gt, valid = lab == 1, lab != -1
        if corregir:
            occ = np.unpackbits(np.load(d / "JRCOcc50" / f"{chip}.npy")).reshape(512, 512).astype(bool) & sv
            gt = gt | occ
            valid = valid | occ
        if not valid.any():
            omitidos += 1
            continue
        if not items:  # control: S1Weak debe venir en dB, igual que S1Hand
            med = float(np.median(s1[1][sv]))
            print(f"  Control de escala: mediana de VH en {chip} = {med:.1f} dB")
            if not -45 < med < 0:
                raise SystemExit("S1Weak no parece estar en dB: se detiene para no entrenar con otra escala.")
        rgb = SamWrapper.sar_to_rgb(s1[0], s1[1], "db3")
        items.append({"chip": chip, "rgb": torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))),
                      "gt": torch.from_numpy(gt & valid), "valid": torch.from_numpy(valid)})
        if limite and len(items) >= limite:
            break
    print(f"  Chips débiles: {len(items)} usados, {omitidos} omitidos (Bolivia, sin radar o sin etiqueta)")
    return items


def conjuntos(parts, rapido):
    sets = {"valid": parts["valid"]}
    for s in ("test", "bolivia"):
        r = json.loads((config.RESULTS_DIR / "hacia_060" / "test" / f"resultados_{s}.json").read_text(encoding="utf-8"))
        sets[s] = r["chips"]
    if rapido:
        sets = {k: v[:6] for k, v in sets.items()}
    return sets


def correr_semilla(nombre, v, seed, tr, debil, eval_items, P, av, dev):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = P3.UNet(cin=4 if v["canal_jrc"] else 3).to(dev)
    ep = {"n": 0, "tot": 0}

    def on_step(k, loss):
        av.paso(k * (W_WEAK if ep["debil"] else W_IMG), n=k, epoca=f"{ep['n'] + 1}/{ep['tot']}", perdida=f"{loss:.3f}")

    def on_epoch(e, h):
        ep["n"] = e
        if ep["debil"] or e % 10 == 0 or e == ep["tot"]:
            av.info(f"  [{nombre} s{seed}{' débil' if ep['debil'] else ''}] época {e}/{ep['tot']}  pérdida "
                    f"{h['perdida_media']:.4f}  IoU entrenamiento {h['iou_entrenamiento']:.4f}  ({h['seg_epoca']:.0f} s)")
        av.set(detalle=f"{nombre}, semilla {seed}, época {e}/{ep['tot']}")

    h_debil = None
    if debil is not None:
        # el preentrenamiento se guarda: si la corrida se corta en el ajuste, no hay que repetirlo
        MODELS.mkdir(parents=True, exist_ok=True)
        ck, ck_h = MODELS / f"{nombre}_seed{seed}_preentreno.safetensors", MODELS / f"{nombre}_seed{seed}_preentreno.json"
        if ck.exists() and ck_h.exists():
            model.load_state_dict(load_file(str(ck)))
            h_debil = json.loads(ck_h.read_text(encoding="utf-8"))
            av.paso(P["ep_debil"] * len(debil) * W_WEAK, n=0)
            av.info(f"  [{nombre} s{seed}] preentrenamiento ya guardado: se reutiliza")
        else:
            ep.update(n=0, tot=P["ep_debil"], debil=True)
            av.etapa(f"{nombre} s{seed} · preentrenando con débiles", P["ep_debil"] * len(debil), "img")
            h_debil = P3.train_unet(model, debil, seed=seed, epochs=P["ep_debil"], on_step=on_step, on_epoch=on_epoch)
            save_file({k: t.detach().cpu().contiguous() for k, t in model.state_dict().items()}, str(ck))
            ck_h.write_text(json.dumps(h_debil), encoding="utf-8")
        model.train()
    ep.update(n=0, tot=P["ep"], debil=False)
    av.etapa(f"{nombre} s{seed} · ajustando con manuales", P["ep"] * len(tr), "img")
    h = P3.train_unet(model, tr, seed=seed, epochs=P["ep"], on_step=on_step, on_epoch=on_epoch)
    MODELS.mkdir(parents=True, exist_ok=True)
    save_file({k: t.detach().cpu().contiguous() for k, t in model.state_dict().items()},
              str(MODELS / f"{nombre}_seed{seed}.safetensors"))
    out = {"config": nombre, "semilla": seed, "preentrenamiento": h_debil, "entrenamiento": h, "conjuntos": {}}
    for s, (items, chips, ok) in eval_items.items():
        av.etapa(f"{nombre} s{seed} · evaluando {s}", 2 * len(items), "chip")
        p_ind = P3.predict_unet(model, items, tta=False)
        p_tta = P3.predict_unet(model, items, tta=True)
        av.paso(len(items) * W_PRED, n=2 * len(items))
        L.save_probs(MODELS / f"{nombre}_seed{seed}_{s}_tta.npz", p_tta)
        rows = E._full_rows(chips, ok, L.rows_from_probs(p_ind, items, 0.5))
        out["conjuntos"][s] = {"agregado": stats.aggregate(rows), "por_chip": rows}
    del model
    a = out["conjuntos"]["valid"]["agregado"]
    av.info(f"  [{nombre} s{seed}] IoU global VALID {a['global_iou']:.4f} | IoU medio {a['mean_chip_iou']:.4f} | "
            f"P {a['precision']:.3f} R {a['recall']:.3f}")
    return out


def ensamble(nombre, seeds, eval_items, ref):
    out = {"config": nombre, "semillas": list(seeds), "que": VARIANTES[nombre]["que"], "conjuntos": {}}
    for s, (items, chips, ok) in eval_items.items():
        probs = [L.load_probs(MODELS / f"{nombre}_seed{sd}_{s}_tta.npz", len(items)) for sd in seeds]
        rows = E._full_rows(chips, ok, L.rows_from_probs(L.ensemble(probs), items, 0.5))
        indiv = [json.loads((OUT / f"{nombre}_seed{sd}.json").read_text(encoding="utf-8"))["conjuntos"][s]["agregado"]
                 for sd in seeds]
        res = {"ensamble_tta": {"agregado": stats.aggregate(rows), "por_chip": rows},
               "semillas": stats.mean_std([a["global_iou"] for a in indiv]),
               "medio_por_chip": stats.mean_std([a["mean_chip_iou"] for a in indiv])}
        if ref and s in ref:
            res["vs_unet_db3"] = compare(rows, ref[s], [data.region_of(c) for c in chips])
        out["conjuntos"][s] = res
    return out


def referencia_unet(sets):
    """Filas del ensamble + TTA de unet_db3 (la linea base sin extras) en cada conjunto."""
    ref = {}
    c = json.loads(P3.config_file("unet_db3").read_text(encoding="utf-8"))
    if c["chips_valid"] == sets["valid"]:
        ref["valid"] = c["ensamble"]["tta"]["por_chip"]
    for s in ("test", "bolivia"):
        r = json.loads((config.RESULTS_DIR / "hacia_060" / "test" / f"resultados_{s}.json").read_text(encoding="utf-8"))
        if "unet" in r["modelos"] and r["chips"] == sets[s]:
            ref[s] = r["modelos"]["unet"]["principal_ensamble_tta"]["por_chip"]
    return ref


def results_markdown() -> str:
    filas = ["| Variante | Qué es | IoU global valid (3 semillas) | Ensamble + TTA valid | Δ vs `unet_db3` valid "
             "(IC 95 %) | Test, ensamble + TTA | Bolivia, ensamble + TTA | IoU medio valid |",
             "|---|---|---|---|---|---|---|---|"]
    for nombre, v in VARIANTES.items():
        f = OUT / f"config_{nombre}.json"
        if not f.exists():
            hechas = sum((OUT / f"{nombre}_seed{s}.json").exists() for s in config.SEEDS)
            filas.append(f"| `{nombre}` | {v['que']} | ⏳ {hechas}/3 semillas | | | | | |")
            continue
        r = json.loads(f.read_text(encoding="utf-8"))["conjuntos"]
        va = r["valid"]
        d = "—"
        if "vs_unet_db3" in va:
            b = va["vs_unet_db3"]["bootstrap"]["global_iou"]
            d = f"{b['diff_b_menos_a']:+.3f} [{b['ic95'][0]:+.3f}, {b['ic95'][1]:+.3f}]"
        filas.append(f"| `{nombre}` | {v['que']} | **{va['semillas']['media']:.3f} ± {va['semillas']['desv']:.3f}** | "
                     f"{va['ensamble_tta']['agregado']['global_iou']:.3f} | {d} | "
                     f"{r['test']['ensamble_tta']['agregado']['global_iou']:.3f} | "
                     f"{r['bolivia']['ensamble_tta']['agregado']['global_iou']:.3f} | "
                     f"{va['medio_por_chip']['media']:.3f} |")
    return ("**Exploratorio: test y Bolivia ya se habían usado.** Referencia `unet_db3`: valid 0.637 ± 0.003 (ensamble + TTA "
            "0.642), test 0.666, Bolivia 0.628.\n\n" + "\n".join(filas) +
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
        block = f"(No se pudo regenerar la tabla: {exc!r}. Los JSON de results/hacia_060/exploracion/ están completos.)"
    BITACORA.write_text(txt[:a] + "\n\n" + block + "\n\n" + txt[b:], encoding="utf-8")


def main():
    global OUT, MODELS, BITACORA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--esperar", action="store_true", help="espera a R2, a la descarga y a las máscaras JRC")
    args = ap.parse_args()
    if args.esperar:
        cinco = config.RESULTS_DIR / "hacia_060" / "robustez" / "cinco_semillas.json"
        d = Path(config.DATA_DIR) / "weak"
        print("Esperando a R2 (5 semillas), a S1Weak y a las máscaras JRC...", flush=True)
        while not (cinco.exists() and len(list((d / "S1Weak").glob("*.tif"))) >= 4384
                   and len(list((d / "JRCOcc50").glob("*.npy"))) >= 4384):
            time.sleep(60)
    config.setup_runtime(seed=0, verbose=False)
    torch.backends.cudnn.benchmark = True
    dev = config.get_device()
    if dev != "cuda":
        raise SystemExit("Necesita la GPU (CUDA).")
    parts = json.loads((config.RESULTS_DIR / "particiones_usadas.json").read_text(encoding="utf-8"))
    P = {"train": parts["train"], "ep": P3.UNET_EPOCHS, "ep_debil": EPOCAS_DEBIL, "seeds": tuple(config.SEEDS)}
    limite = None
    if args.rapido:
        import shutil

        OUT = OUT / "prueba_rapida"
        MODELS = OUT / "modelos"
        BITACORA = OUT / "no_existe.md"
        shutil.rmtree(OUT, ignore_errors=True)
        P.update(train=P["train"][:16], ep=1, ep_debil=1, seeds=(0,))
        limite = 16
    OUT.mkdir(parents=True, exist_ok=True)
    config.SEEDS = P["seeds"]
    sets = conjuntos(parts, args.rapido)
    n_tr = len(P["train"])
    todo = [(k, s) for k in VARIANTES for s in P["seeds"] if not (OUT / f"{k}_seed{s}.json").exists()]
    units = 200.0
    for k, _ in todo:
        units += P["ep"] * n_tr * W_IMG + sum(len(v) for v in sets.values()) * W_PRED + 5
        if VARIANTES[k]["debil"]:
            units += P["ep_debil"] * (limite or 4100) * W_WEAK
    print(f"GPU: {torch.cuda.get_device_name(0)} | por correr: {len(todo)} semillas")
    av = Avance(units, OUT)
    update_bitacora()
    try:
        ref = referencia_unet(sets) if not args.rapido else None
        for nombre, v in VARIANTES.items():
            pendientes = [s for s in P["seeds"] if (nombre, s) in todo]
            if not pendientes and (OUT / f"config_{nombre}.json").exists():
                continue
            av.info(f"\n=== {nombre}: {v['que']} ===")
            av.etapa("Preparando chips", n_tr + sum(len(x) for x in sets.values()), "chip", config_actual=nombre)
            tr = L.prepare_chips(P["train"], "db3", dev, on_chip=lambda: av.paso(0.05))
            ev = {}
            for s, chips in sets.items():
                vacios = set(E.no_sensor_chips(chips))
                ok = [c for c in chips if c not in vacios]
                ev[s] = (L.prepare_chips(ok, "db3", "cpu", on_chip=lambda: av.paso(0.05)), chips, set(ok))
            if v["canal_jrc"]:
                con_jrc(tr)
                for s in ev:
                    con_jrc(ev[s][0])
            debil = None
            if v["debil"] and pendientes:
                av.etapa("Cargando chips débiles en la RAM", 1, "paso")
                debil = chips_debiles(v["debil"] == "jrc", limite)
            for seed in pendientes:
                res = correr_semilla(nombre, v, seed, tr, debil, ev, P, av, dev)
                save_json(res, OUT / f"{nombre}_seed{seed}.json")
                update_bitacora()
            del debil
            r = ensamble(nombre, P["seeds"], ev, ref)
            save_json(r, OUT / f"config_{nombre}.json")
            g = r["conjuntos"]
            av.info(f"  [{nombre}] TERMINADO: valid {g['valid']['semillas']['media']:.4f} ± {g['valid']['semillas']['desv']:.4f}"
                    f" | ensamble + TTA: valid {g['valid']['ensamble_tta']['agregado']['global_iou']:.4f}, "
                    f"test {g['test']['ensamble_tta']['agregado']['global_iou']:.4f}, "
                    f"bolivia {g['bolivia']['ensamble_tta']['agregado']['global_iou']:.4f}")
            del tr, ev
            torch.cuda.empty_cache()
            update_bitacora()
        av.set(terminado=True, etapa="terminado")
    finally:
        av.cerrar()
        update_bitacora()


if __name__ == "__main__":
    main()
