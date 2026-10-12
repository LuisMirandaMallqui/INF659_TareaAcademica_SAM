# Cuaderno de decisiones (design log) y declaración de uso de IA

Proyecto: SAM sobre Sentinel-1 para segmentación de agua (Sen1Floods11) y proyección a Piura 2017. INF659, 2026-2.

La política del curso pide registrar cada decisión de arquitectura y de diseño con su justificación, e indicar dónde se usó asistencia de IA. **La columna "Validación del estudiante" la completa cada integrante, a mano y con sus palabras.** Una fila sin validación no acredita comprensión. La defensa oral individual es la salvaguarda.

Fuentes: los archivos del equipo (`estado_consolidado_del_trabajo.md`, `resumen_decisiones_equipo.md`, `e1_documento_integrado.md`) y la sesión de Claude Code del 12-oct-2026 que produjo esta carpeta.

## 1. Decisiones de diseño

| # | Decisión | Alternativas consideradas | Justificación | Quién decidió / IA usada | Validación del estudiante (nombre + qué verificó) |
|---|---|---|---|---|---|
| D01 | Modelo core: SAM, `facebook/sam-vit-base` | SAM ViT-L/H, SAM 2, DINOv2 | Banco de temas "El costo del prompt en SAM". ViT-B (93.7 M parámetros) cabe en CPU y en GPU de laptop; ViT-H (636 M) no permite iterar ablaciones con el cómputo disponible | Equipo | |
| D02 | Dataset de reto: Sen1Floods11 v1.1 (chips etiquetados a mano); Piura 2017 como aplicación futura | SA-1B, datasets médicos, MMFlood | Dominio SAR, fuera del preentrenamiento de SAM en imágenes RGB naturales (SA-1B): es exactamente lo que pide la Fase C. Tiene etiquetas manuales y particiones oficiales | Equipo | |
| D03 | Banda VH para el umbral y la entrada de SAM | VV, combinación | El agua lisa produce retrodispersión especular baja, más contrastada en VH en muchos escenarios. VV quedó como ablación | Equipo | |
| D04 | Línea base Otsu + apertura (disco r=1) y prompts derivados de esa máscara (3 positivos, 2 negativos) | Prompts manuales, cuadrículas de puntos | La pregunta es "SAM sin prompts humanos". Otsu es la referencia clásica de mapeo de agua en SAR | Equipo | |
| D05 | Corrección v2: la validez para prompts se toma **solo del sensor** y la etiqueta se usa solo para evaluar | v1 (usaba `label != -1`) | Fuga de metadato de la referencia hacia la inferencia, detectada en la revisión metodológica | Equipo + ChatGPT (revisión) + Claude (código v2) | |
| D06 | FLOPs contados con atención por ventanas en 8 capas y global en las capas 2, 5, 8 y 11 | Asumir atención global en las 12 | Verificado contra `window_size=14` y `global_attn_indexes` de la configuración real. El cálculo anterior sobreestimaba un 29.5 % | Claude (sesión previa), revisado por ChatGPT | |
| D07 | Fijar la revisión de los pesos (commit `70c1a07f…`) y usar `safetensors` | Descargar siempre "main"; `.bin` | Reproducibilidad inmutable y sin deserialización con pickle. El SHA-256 se verifica contra el publicado por Hugging Face | Claude Code (12-oct) | |
| D08 | **Cachear los embeddings del codificador** y ejecutar las ablaciones y H2 solo sobre el decodificador | Ejecutar el modelo completo en cada variante | El codificador concentra el 95.7 % de los parámetros y casi todo el cómputo. El decodificador cuesta unos 85 ms por llamada en CPU frente a unos 7 s del codificador. Así H2 se vuelve viable en CPU | Claude Code (12-oct) | |
| D09 | Ampliar H1 de 9 chips a **toda la partición valid**; no descargar `test` ni `bolivia` | Evaluar en test ahora | 9 chips con un chip por región no permiten inferencia. Test queda reservado para la evaluación final (E3) | Claude Code, siguiendo el protocolo del equipo (E1, sección 5) | |
| D10 | Diagnósticos **oráculo** (prompts tomados de la etiqueta; mejor candidata según la etiqueta) | Solo ablaciones sin etiqueta | Separan la causa del fallo: prompts, regla de selección o representación. Se reportan como cotas y nunca como método | Claude Code (12-oct) | |
| D11 | H2 con salida de una sola máscara (`multimask_output=False`) | Salida triple + cabeza de IoU | Con varios puntos el prompt no es ambiguo; es la recomendación de los autores de SAM y la práctica de MedSAM. Evita depender de la cabeza de IoU, que en SAR está descalibrada (D3) | Claude Code (propuesta, pre-registrada) | |
| D12 | Pérdida BCE + Dice ignorando píxeles sin etiqueta; AdamW, wd 0.01, lote 4, 12 épocas, sin *early stopping*; barrido de lr en `train-dev` | Focal; elegir la época en valid | BCE + Dice es estándar en el ajuste de SAM (MedSAM) y robusto al desbalance agua/tierra. Elegir hiperparámetros en `train-dev` evita usar valid para decidir | Claude Code (propuesta, pre-registrada) | |
| D13 | Control H2-c: sonda lineal 1×1 sobre el embedding congelado | Sin control | Distingue entre "el decodificador aprende algo" y "la información ya está en las características" | Claude Code (propuesta, pre-registrada) | |
| D14 | Sonda lineal **por capa** (bloques 0, 2, 5, 8, 11 y cuello) con el control VV/VH en dB | Solo visualización PCA | Responde cuantitativamente en qué capa el agua se vuelve linealmente separable y si SAM aporta más que la retrodispersión cruda | Claude Code (12-oct) | |
| D15 | Embeddings calculados en CPU con prioridad baja (la máquina estaba en uso); ajuste del decodificador, evaluaciones y perfilado final en GPU (RTX 3070) cuando la máquina quedó libre | Todo en GPU desde el inicio | El codificador es determinista: calcularlo en CPU o en GPU solo cambia redondeos (diferencias de IoU < 1e-3, verificadas). El entrenamiento en GPU es unas 6 veces más rápido. La latencia se reporta para ambos dispositivos | Claude Code (12-oct), con autorización del usuario para usar la GPU | |

## 2. Registro de uso de asistentes de IA

| Tarea | Herramienta | Qué se le pidió exactamente | Qué devolvió | Qué validó o entendió el estudiante antes de usarlo | Integrante responsable |
|---|---|---|---|---|---|
| Pipeline preliminar v1/v2 | Claude | Ejecutar Otsu, prompts, SAM y evaluación con datos reales | Scripts, JSON, figura; corrección v2 | | |
| Revisión metodológica | ChatGPT | Revisar código, resultados e informe | Fuga de etiqueta en negativos; atención por ventanas; observaciones de métricas | | |
| Disección de SAM (versión inicial) | Claude | Parámetros, FLOPs, latencia y memoria | Script y JSON; corrección de atención global/ventanas | | |
| Referencias y particiones | Claude | Verificar citas y conteos de CSV | Citas verificadas; conteos 252/89/90/15 | | |
| Redacción E1 y E2 (borradores) | ChatGPT | Integrar E1, preparar E2 | Documentos integrados | | |
| **Resolución E1 final, E2 y repositorio (esta carpeta)** | **Claude Code (Opus 5.5), 12-oct-2026** | "Resolver la tarea académica con lo que ya hizo el equipo, instalar lo necesario y dejar los documentos en una carpeta" | Notebooks 00 a 05 ejecutados; Fase A completada (FLOPs medidos, PCA, campo receptivo efectivo, perfilado en CPU y GPU); H1 en 86 chips; diagnósticos y ablaciones; H2 con 3 semillas; análisis exploratorio de la sonda (marcado como no pre-registrado); protocolo pre-registrado; E1 final; resumen extendido y póster de WAIMLAp; guía de defensa | | |
| Curva de validación de H2 | Claude Code, 12-oct-2026 | "La curva de pérdida me preocupa: ¿qué hizo realmente el modelo?"; agregar una curva de validación por época | Notebook 06: IoU por época en train-dev (24 épocas, 3 semillas). Sin sobreajuste sistemático, pero con entrenamiento inestable | | |
| Verificación de citas nuevas | Claude Code, 12-oct-2026 | Confirmar la fecha de MFF-Net y su caso Piura; buscar la fuente de datos de Piura | MFF-Net: Remote Sens. 18(1):123, publicado el 29-12-2025; la sección 3.6 usa Piura 2017 (20 imágenes VV/VH); su ref. [52] es Shen, Zhang y Wu (2022), ISPRS Annals | | |
| Requisitos de E2 | Claude Code, 12-oct-2026 | Revisar waimlap.org | E2 exige resumen extendido (2 págs., inglés, plantilla oficial) y póster A0 en inglés, en un ZIP, antes del 16-oct, mediante formulario | | |
