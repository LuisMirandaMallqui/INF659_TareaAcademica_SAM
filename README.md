# SAM aplicado a mapeo de inundaciones con Sentinel-1 SAR

Tarea Academica Final, curso INF659 (Deep Learning, PUCP). Pipeline real, de principio a fin: umbral de Otsu sobre la banda VH de Sentinel-1, generacion automatica de prompts de punto, inferencia con los pesos oficiales de SAM (`facebook/sam-vit-base`), e IoU real contra la verdad de terreno de Sen1Floods11, con proyeccion al Niño Costero de 2017 en Piura.

Punto de partida para seguir el trabajo desde una sesion cloud de Claude Code (con GPU) y aprovechar el credito de computo antes de que venza.

## Que hay aqui

```
scripts/
  run_sam_pipeline_v2.py    pipeline completo: Otsu -> prompts automaticos -> SAM -> IoU real
  run_sam_pipeline_v2.ipynb mismo pipeline como notebook (celdas = funciones del .py, sin reescribir ninguna linea)
  fase_a_diseccion.py       diseccion instrumentada de SAM: parametros, FLOPs, campo receptivo, latencia, memoria
  fase_a_diseccion.ipynb    misma diseccion como notebook, mismo orden de celdas que el .py
docs/
  Estado_consolidado_del_trabajo.md       tabla resumen de todo el avance (entregables, fases, hipotesis, proximos pasos)
  E1_Documento_integrado.md               documento integrado de la entrega E1 (las 8 secciones + anexos)
  E2_Texto_inscripcion_WAIMLAP.md         texto para la inscripcion del poster WAIMLAP (entrega E2)
  Resumen_decisiones_equipo.md            resumen corto de decisiones y avances para el equipo
  Verificacion_citas_Parte2.md            verificacion contra la fuente real de las 5 citas de la Parte 2 (estado del arte)
  E1_Borrador_ChatGPT.md                  borrador academico original de E1, las ocho partes
  Revision_del_material_Claude.md         revision critica que encontro las dos fallas reales corregidas abajo
  Correccion_v2_fuga_de_etiqueta.md       correccion 1: fuga de etiqueta en los prompts negativos (pipeline)
  Fase_A_diseccion_real.md                correccion 2: atencion ventaneada vs global mal asumida (diseccion del modelo)
  MANIFIESTO_reproducibilidad.md          URLs exactas + sha256 de cada archivo de datos usado
results/
  resultados_reales_v2.json      resultados completos por chip (TP/FP/FN, IoU, scores de SAM)
  fase_a_diseccion_real.json     parametros/FLOPs/latencia/memoria reales del modelo
  comparacion_real_v2.png        figura: VH real, Otsu abierto, SAM real, verdad de terreno
data/
  chips_sen1floods11_reales.zip  los 20 archivos reales (10 chips, S1Hand+LabelHand) ya usados en E1
```

## Como correr lo que ya existe

```bash
pip install -r requirements.txt
cd scripts
unzip ../data/chips_sen1floods11_reales.zip -d .   # crea scripts/chips/
python run_sam_pipeline_v2.py
python fase_a_diseccion.py
```

El requisito de "notebooks reproducibles" del curso se cubre con `run_sam_pipeline_v2.ipynb` y `fase_a_diseccion.ipynb`: cada funcion del `.py` correspondiente es su propia celda, sin reescribir ni reindentar ninguna linea, y la ultima celda llama `main()` (o, en el caso de `fase_a_diseccion`, que no tiene `main()`, las celdas siguen el mismo orden de arriba a abajo que el script). Ejecutar todas las celdas en orden en Jupyter es equivalente a correr el `.py` desde la terminal.

## Hallazgo principal (ya reportado en E1)

SAM con prompts automaticos pierde contra el umbral de Otsu en 8 de 9 chips reales (IoU global 0.0897 vs 0.1875), y colapsa a marcar entre 97% y 99.7% de la imagen como agua en 6 de 9 chips: un caso de "mask collapse" fuera de dominio, no una falla de implementacion (las coordenadas de los prompts se verificaron por codigo). El detalle completo, incluida la correccion por fuga de etiqueta, esta en `docs/Correccion_v2_fuga_de_etiqueta.md`.

## Que falta y como seguirlo en una sesion cloud con GPU

Todo esto esta pendiente porque corre lento o no corre en CPU (el forward pass de SAM aqui tarda ~12 segundos; con GPU deberia ser del orden de 100-200ms, como referencia el equipo ya tenia 135.5ms en una RTX 3060).

1. **H2 (Fase B, intervencion de nucleo)**: afinar solo el decodificador de mascaras (4.06M parametros, 22x mas chico que el codificador segun `results/fase_a_diseccion_real.json`) dejando el codificador de imagen congelado, sobre la particion de entrenamiento de Sen1Floods11. El bucket publico es el mismo que ya se usa en `MANIFIESTO_reproducibilidad.md` (`https://storage.googleapis.com/sen1floods11/v1.1/...`); el CSV de entrenamiento deberia estar en `v1.1/splits/flood_handlabeled/flood_train_data.csv` siguiendo el mismo patron que el de validacion ya confirmado, pero **verificar el nombre exacto listando el bucket antes de asumirlo** (`gsutil ls gs://sen1floods11/v1.1/splits/flood_handlabeled/` o el equivalente HTTP), no se confirmo en esta sesion.
2. **Ampliar de 9 a los 89 chips de validacion completos**: mismo patron de URL, iterar sobre las 89 filas del CSV ya referenciado en el manifiesto en vez de solo la primera fila por region.
3. **Evaluacion real en Piura**: requiere conseguir una escena Sentinel-1 real de la zona y fecha del Niño Costero 2017 con alguna mascara de referencia (o generar una por fotointerpretacion), eso todavia no esta resuelto.

Las 5 citas marcadas `[VERIFICAR]` en `docs/E1_Borrador_ChatGPT.md` (Parte 2) no son parte de este trabajo de computo; se verifican con busqueda bibliografica normal.

## Integraciones necesarias

- GPU: para que H2 y la ampliacion a 89 chips corran en minutos y no en horas. Confirmar en la UI de la sesion cloud que hardware estara disponible; no se asume de antemano.
- Red saliente al bucket publico de Google Cloud Storage (`storage.googleapis.com`) para bajar mas chips: no requiere autenticacion, es un bucket publico.
- Hugging Face (`huggingface.co`) para bajar `facebook/sam-vit-base`: tampoco requiere autenticacion ni token.
- No se necesita ninguna cuenta ni API key para nada de lo anterior.
