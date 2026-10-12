# Estado consolidado del trabajo (actualizado al 12 de octubre de 2026)

SAM (ViT-B) aplicado a la segmentación de agua en Sentinel-1 SAR, con el benchmark Sen1Floods11 y proyección al Niño Costero de 2017 en Piura. Equipo: John Manuel Arzapalo Arana, Luis Alberto Miranda Mallqui y Juan Diego Zavala Viscardo. Curso INF659, Técnicas Avanzadas de Data Mining y Sistemas Inteligentes, PUCP 2026-2.

Este documento reemplaza al estado del 11 de octubre, que ahora está en `docs/historial/`.

## Las tres fases

| Fase | Estado | Evidencia (notebook) |
|---|---|---|
| A. Disección instrumentada | **Completa** | 01: parámetros por bloque (93.7 M; decodificador 4.06 M), **FLOPs medidos** (972 GFLOPs en el codificador frente a 930 de la fórmula; la diferencia es el relleno de las ventanas), PCA de características intermedias, **campo receptivo efectivo** (gradiente nulo fuera de la ventana antes del bloque 2) y latencia/memoria del pipeline SAR (4.4 s/chip en CPU y 0.60 s/chip en una RTX 3070; 2.5 GiB de RAM) |
| B. Intervención en el núcleo | **Completa (preliminar para E1)** | 03: ablaciones A1 a A4. 04: **H2**, ajuste exclusivo del decodificador con 3 semillas y control H2-c |
| C. Estrés y honestidad | **Completa en `valid`** | 03: oráculos D1 y D2 y calibración D3. 04: análisis de fallas. 05 y 06: análisis exploratorios de la sonda lineal y la curva de validación |

## Hipótesis

| Hipótesis | Resultado (86 chips de `valid`) |
|---|---|
| H1: SAM con prompts automáticos mejora a Otsu | **Rechazada.** IoU 0.115 frente a 0.256 (Δ −0.141, IC 95 % [−0.181, −0.103]); igual a "todo es agua" |
| H2: ajustar solo el decodificador mejora a SAM preentrenado | **Confirmada.** IoU **0.431 ± 0.038** (3 semillas), también por encima de Otsu en las 3 semillas |
| H2-c: una sonda lineal 1×1 rinde cerca de H2 | **No se cumplió** (0.073). Según el análisis exploratorio, por calibración del umbral y subajuste |
| H3: SAM como filtro de regiones candidatas | Pendiente (E3) |

## Hallazgos que conviene recordar

* El fallo de SAM **no viene de los prompts**: con prompts perfectos tampoco supera a Otsu. Viene de la cabeza de IoU, descalibrada en SAR (ρ = −0.05), que elige la máscara más grande en 80 de 86 chips.
* La información de agua **sí está** en el codificador (AUC 0.934 en bloques intermedios), aunque aporta poco sobre la retrodispersión cruda (0.906).
* El muestreador de prompts de v2 dependía de la CPU por empates en `np.argsort`. Se corrigió con un ordenamiento estable (**v2.1**).
* El entrenamiento de H2 generaliza hasta la época 12, muestra sobreajuste tardío, es inestable y no es determinista en GPU.

## Próximos pasos (E3, 16 de noviembre)

1. Evaluar en `test` y `bolivia`, hasta ahora reservadas.
2. Agregar una línea base U-Net entrenada con los mismos 250 chips.
3. Estabilizar H2 con decaimiento de la tasa de aprendizaje o EMA, selección de la época en `train-dev` y modo determinista, todo pre-registrado.
4. Adaptar parte del codificador con LoRA o adaptadores, usando VV y VH.
5. Evaluar Piura 2017 con las escenas de Shen et al. (2022) y MFF-Net (sección 3.6).
6. Completar las validaciones personales del cuaderno de decisiones y firmar el protocolo pre-registrado.
