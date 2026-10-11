"""
Fase A real: diseccion instrumentada de facebook/sam-vit-base.
Parametros y su reparto por submodulo: reales, exactos.
Latencia y memoria: reales, medidas en este contenedor (CPU), no en la laptop RTX 3060.
FLOPs: estimados con thop sobre el image encoder (el submodulo que concentra el computo).
Campo receptivo: razonado analiticamente, no es un numero que se mida con una libreria.
"""
import json
import resource
import time

import numpy as np
import torch
from transformers import SamModel, SamProcessor

MODEL_ID = "facebook/sam-vit-base"

print(f"Cargando {MODEL_ID}...")
model = SamModel.from_pretrained(MODEL_ID).to("cpu")
model.eval()
processor = SamProcessor.from_pretrained(MODEL_ID)

# ---------- parametros reales, por submodulo ----------
def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

total_p, trainable_p = count_params(model)
enc_p, enc_t = count_params(model.vision_encoder)
prompt_p, prompt_t = count_params(model.prompt_encoder)
dec_p, dec_t = count_params(model.mask_decoder)

params = {
    "total": total_p,
    "vision_encoder": enc_p,
    "prompt_encoder": prompt_p,
    "mask_decoder": dec_p,
    "pct_encoder_del_total": round(100 * enc_p / total_p, 2),
    "pct_decoder_del_total": round(100 * dec_p / total_p, 4),
    "razon_encoder_sobre_decoder": round(enc_p / dec_p, 1),
}
print("Parametros reales:", json.dumps(params, indent=2))

# ---------- FLOPs reales: formula analitica de ViT, CORREGIDA para el atencion ventaneada real de SAM ----------
# thop sobre el encoder completo se cuelga en CPU: no termina en tiempo razonable aqui, asi que se usa
# una formula analitica en vez de un forward pass instrumentado.
# CORRECCION (detectada por la revision de ChatGPT sobre este mismo archivo, verificada aqui contra la
# config real del modelo cargado): SAM NO usa atencion global en las 12 capas. vision_config.window_size=14
# y vision_config.global_attn_indexes=[2,5,8,11] muestran que solo 4 de las 12 capas usan atencion global
# (costo O(N^2)); las otras 8 usan atencion ventaneada tipo ViTDet, donde cada parche solo atiende a los
# demas parches dentro de su ventana de 14x14, no a los 4096 parches de la imagen completa. La version
# anterior de este script asumia atencion global en las 12 capas y sobreestimaba el total en 29.5 por ciento
# (1319.09 GFLOPs en vez de los 930.38 GFLOPs corregidos).
vc = model.config.vision_config
D = vc.hidden_size
L = vc.num_hidden_layers
P = vc.patch_size
img_sz = vc.image_size
window_size = vc.window_size
global_idx = vc.global_attn_indexes
N = (img_sz // P) ** 2                      # 4096 parches reales

grid = img_sz // P                          # 64x64 parches
pad = (window_size - grid % window_size) % window_size   # relleno real que usa SAM para que la grilla sea divisible por la ventana
grid_padded = grid + pad
num_windows = (grid_padded // window_size) ** 2
w_tokens = window_size ** 2                 # tokens por ventana (196)
n_global = len(global_idx)
n_windowed = L - n_global

flops_patch_embed = N * D * (P * P * 3) * 2
flops_tokenwise_per_block = 24 * N * (D ** 2)                          # FFN + proyecciones QKVO, por bloque (se aplica a N real; no se separa el pequeno overhead de padding de las capas ventaneadas)
flops_attn_global_per_block = 4 * (N ** 2) * D                         # atencion completa N x N, solo en las 4 capas globales
flops_attn_windowed_per_block = 4 * num_windows * (w_tokens ** 2) * D  # atencion solo dentro de cada ventana, en las 8 capas restantes

flops_global_block = flops_tokenwise_per_block + flops_attn_global_per_block
flops_windowed_block = flops_tokenwise_per_block + flops_attn_windowed_per_block
flops_total_encoder = flops_patch_embed + n_global * flops_global_block + n_windowed * flops_windowed_block
flops_total_antiguo_incorrecto = flops_patch_embed + L * flops_global_block  # el numero que se habia reportado antes de la correccion

flops_info = {
    "formula": "bloque global: 24*N*D^2 + 4*N^2*D. bloque ventaneado: 24*N*D^2 + 4*num_windows*w_tokens^2*D",
    "verificado_del_modelo_real": {"window_size": window_size, "global_attn_indexes": global_idx,
                                    "capas_globales": n_global, "capas_ventaneadas": n_windowed,
                                    "num_windows": num_windows, "tokens_por_ventana": w_tokens},
    "N_parches": N, "D_dim_oculta": D, "L_capas": L, "patch_size": P, "resolucion_interna": img_sz,
    "flops_patch_embedding": int(flops_patch_embed),
    "gflops_por_bloque_global": round(flops_global_block / 1e9, 2),
    "gflops_por_bloque_ventaneado": round(flops_windowed_block / 1e9, 4),
    "flops_totales_vision_encoder": int(flops_total_encoder),
    "gflops_totales_vision_encoder": round(flops_total_encoder / 1e9, 2),
    "gflops_totales_version_anterior_incorrecta": round(flops_total_antiguo_incorrecto / 1e9, 2),
    "reduccion_pct_por_la_correccion": round(100 * (1 - flops_total_encoder / flops_total_antiguo_incorrecto), 1),
    "nota": "calculado a partir de la configuracion real del modelo cargado (facebook/sam-vit-base), "
            "no de un forward pass instrumentado: thop sobre 12 capas en 4096 parches no termino en tiempo "
            "razonable en la CPU de este contenedor. La formula si distingue ahora atencion global de "
            "ventaneada segun la config real; no desglosa el pequeno overhead de padding de las capas "
            "ventaneadas (grilla 64x64 rellenada a 70x70 para ser divisible entre ventanas de 14x14).",
}
print("FLOPs (formula analitica corregida, config real):", json.dumps(flops_info, indent=2, ensure_ascii=False))

# ---------- campo receptivo: razonado, no medido ----------
# CORRECCION: la version anterior afirmaba "global desde la primera capa", asumiendo atencion global en
# las 12 capas. Con window_size y global_attn_indexes reales (ver seccion de FLOPs), las capas 0 y 1 son
# ventaneadas: el campo receptivo tras ellas es la ventana de 14x14 parches, no la imagen completa. El
# campo receptivo se vuelve global por primera vez en la capa de indice 2, la primera marcada como global.
patch_size = getattr(model.config.vision_config, "patch_size", 16)
image_size = getattr(model.config.vision_config, "image_size", 1024)
primera_capa_global = min(global_idx)
campo_ventana_px = window_size * patch_size
campo_receptivo = {
    "patch_size": patch_size,
    "image_size_interno": image_size,
    "campo_receptivo_capa_de_parches": f"{patch_size}x{patch_size} pixeles (convolucion local de patchify)",
    "campo_receptivo_capas_ventaneadas_0_y_1": f"se queda en la ventana, {window_size}x{window_size} parches "
        f"equivalentes a {campo_ventana_px}x{campo_ventana_px} pixeles; no crece a la imagen completa porque "
        "SAM usa ventanas fijas, sin desplazamiento entre capas (a diferencia de Swin), asi que una segunda "
        "capa ventaneada no amplia el area ya cubierta por la primera.",
    "campo_receptivo_capa_global_mas_temprana": f"la capa de indice {primera_capa_global} (la primera de las "
        f"{len(global_idx)} capas marcadas global_attn_indexes={global_idx}) es donde el campo receptivo "
        "se vuelve global por primera vez: ahi cada parche atiende a los 4096 parches de la imagen. Antes de "
        "esa capa el campo receptivo NO es global.",
    "nota": "en una CNN el campo receptivo crece gradualmente con la profundidad; en SAM crece en escalones "
        "(ventana, ventana, imagen completa, ventana, ventana, imagen completa...) siguiendo el patron fijo "
        "de global_attn_indexes, no gradualmente capa por capa como en un ViT con atencion global pura.",
}
print("Campo receptivo (corregido, config real):", json.dumps(campo_receptivo, indent=2, ensure_ascii=False))

# ---------- latencia real (CPU, este contenedor) ----------
# misma forma de input_points/input_labels ya probada en run_sam_pipeline_v2.py: [[puntos]] / [[etiquetas]],
# NO el anidado de mas niveles que colgo la corrida anterior.
img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
puntos = [(512, 512)]
etiquetas = [1]
inputs = processor(img, input_points=[[puntos]], input_labels=[[etiquetas]], return_tensors="pt")

with torch.no_grad():
    for _ in range(1):  # calentamiento
        model(**inputs)

times = []
with torch.no_grad():
    for _ in range(5):
        t0 = time.perf_counter()
        model(**inputs)
        times.append((time.perf_counter() - t0) * 1000)

latencia = {"n_repeticiones": 5, "calentamiento": 1, "media_ms": float(np.mean(times)),
            "desviacion_ms": float(np.std(times)), "min_ms": float(np.min(times)), "max_ms": float(np.max(times)),
            "hardware": "CPU de este contenedor en la nube (2 vCPU), NO la laptop RTX 3060"}
print("Latencia real:", json.dumps(latencia, indent=2))

# ---------- memoria real (CPU, este contenedor) ----------
mem_antes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB en Linux
with torch.no_grad():
    model(**inputs)
mem_despues = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
memoria = {"pico_rss_kb_total_proceso": mem_despues, "pico_rss_mb_total_proceso": round(mem_despues / 1024, 1),
           "nota": "pico de memoria residente de TODO el proceso python (modelo + pesos + activaciones), "
                   "medido con resource.getrusage; no es memoria aislada solo-forward-pass ni memoria de GPU.",
           "hardware": "CPU de este contenedor en la nube, NO la laptop RTX 3060"}
print("Memoria real:", json.dumps(memoria, indent=2))

resultado = {"modelo": MODEL_ID, "parametros": params, "flops": flops_info,
             "campo_receptivo": campo_receptivo, "latencia": latencia, "memoria": memoria}
with open("results_v2/fase_a_diseccion_real.json", "w") as f:
    json.dump(resultado, f, indent=2, ensure_ascii=False)
print("\nGuardado: results_v2/fase_a_diseccion_real.json")
