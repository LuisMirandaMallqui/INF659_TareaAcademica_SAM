# Protocolo pre-registrado de los experimentos nuevos (E1, versión final)

Redactado el **12 de octubre de 2026 a la 01:20 (hora de Lima)**, **antes** de ejecutar los experimentos que describe. Ningún número de esta página proviene de una corrida nueva. Los únicos resultados previos que se conocían al escribirlo eran los de la corrida v2 del equipo: 9 chips, SAM IoU global 0.0897 contra 0.1875 de Otsu con apertura.

> **Declaración de IA.** Claude (Claude Code, modelo Opus 5.5) redactó este protocolo a partir de las hipótesis que el equipo ya había registrado (H1, H2, H3 y el Anexo A del E1). Las expectativas razonadas de cada experimento son propuesta del asistente. **El equipo debe leerlas, corregirlas si no las comparte y firmar la tabla del final** antes de presentarlas como propias. La política del curso evalúa el modelo mental del equipo, no el del asistente.

## 1. Datos y particiones

| Partición | Uso | Chips |
|---|---|---|
| `train` (CSV oficial v1.1) | Ajuste del decodificador (H2) y de la sonda (H2-c) | 252 en el CSV |
| `train-dev` (20 % de `train`, estratificado por región, semilla 0) | Solo barrido de tasa de aprendizaje de H2 | ≈ 50 |
| `valid` (CSV oficial v1.1) | Todas las comparaciones del E1 | 89 en el CSV |
| `test` y `bolivia` | **No se descargan ni se miran.** Reservadas para el E3 | 90 + 15 |

Criterio de exclusión, el mismo de v2 y declarado antes: se excluye un chip si más del 95 % de sus píxeles no tiene etiqueta (LabelHand = −1). La métrica principal es el **IoU global de agua**, que suma TP, FP y FN de todos los chips antes de dividir. Se reportan además el IoU medio por chip, la precisión y la sensibilidad. Las comparaciones son **pareadas**: mismos chips, mismos píxeles y mismos prompts.

**Incertidumbre.** Se calcula bootstrap pareado por chip (10 000 remuestreos) y bootstrap por conglomerados de región, más conservador, para la diferencia de IoU global y de IoU medio. Se aplica además la prueba de Wilcoxon de rangos con signo sobre el IoU por chip. Con alrededor de diez regiones, el bootstrap por región es exploratorio.

## 2. Experimentos, hipótesis y expectativa previa

| Id | Experimento | Hipótesis (dirección) | Qué esperamos y por qué |
|---|---|---|---|
| R0 | Reproducir v2 en los 9 chips originales, en otra máquina (Windows, CPU) | Los números coinciden | El pipeline es determinista y las versiones de las librerías son las mismas. Esperamos IoU idénticos hasta el cuarto decimal. |
| H1-ext | H1 sobre **todo valid** (≈ 89 chips) | SAM v2 > Otsu con apertura | Esperamos que **H1 siga sin apoyo**. El colapso a máscaras de casi toda la imagen no parece depender del chip elegido. Diez veces más chips deberían estrechar el IC, no invertir el signo. |
| D1 | **Oráculo de prompts**: puntos tomados de LabelHand | Si SAM también colapsa con prompts perfectos, el fallo no está en Otsu | Esperamos que mejore algo, pero **siga muy lejos de un buen IoU**. El colapso ocurre con prompts ubicados correctamente (la verificación de coordenadas de v2 dio todo correcto). Eso apunta a la representación SAR, no al umbral. |
| D2 | **Oráculo de selección**: la mejor de las 3 candidatas según la etiqueta | Cota superior de cualquier regla de selección | Esperamos que suba claramente sobre v2: la candidata elegida por puntaje predicho suele ser la más grande. Si aun así queda bajo Otsu, ninguna regla de selección basta. |
| D3 | **Calibración** del IoU predicho frente al IoU real | Correlación positiva | Esperamos una **calibración rota**: puntajes de 0.92 a 0.99 con IoU reales menores a 0.2, como en v2, y correlación de Spearman cercana a cero. |
| A1 | Cantidad de prompts: 1+0, 3+2, 5+5 y 10+10 | Más puntos mejoran la sensibilidad hasta un límite (Anexo A) | Esperamos poco efecto, o ninguno, sobre el colapso. Más puntos negativos podrían recortar algo la máscara, pero los puntos heredan los errores de Otsu. |
| A2 | Regla de selección: salida única (`multimask_output=False`), menor área y consistencia con Otsu | Una regla sin etiquetas reduce la sobresegmentación (Anexo A) | **Salida única > multimáscara con puntaje máximo.** Los autores de SAM recomiendan salida única cuando hay varios puntos, porque el prompt deja de ser ambiguo. Menor área y consistencia con Otsu deberían reducir los FP. |
| A3 | Prompt denso: máscara Otsu como `input_masks` más los puntos | Restringe la máscara a la región Otsu | Esperamos un resultado cercano a Otsu, levemente por encima o por debajo. El decodificador copiaría el prompt denso sin agregar información. |
| A4 | Entrada [VV, VH, VV−VH] en lugar de VH×3 | Más información polarimétrica mejora el IoU (Anexo A) | Esperamos un efecto pequeño. El codificador congelado no "sabe" que los canales son polarizaciones; cambiar la entrada mueve la textura, no el significado. |
| P | Sonda lineal **por capa** (tokens de los bloques 0, 2, 5, 8, 11 y del cuello) | El agua es linealmente separable en algún bloque | Esperamos separabilidad **alta**, con AUC > 0.85: el agua en SAR es oscura y homogénea, fácil incluso para Otsu. Esperamos el máximo en bloques intermedios y no necesariamente en el cuello. |
| H2 | Ajustar **solo el decodificador** (4 058 340 parámetros) con prompts automáticos | IoU(SAM decodificador ajustado) > IoU(SAM preentrenado, salida única) | Esperamos que **H2 se cumpla con margen amplio** y que supere a Otsu. Si P confirma que el embedding ya contiene la señal de agua, el fallo de SAM está en cómo el decodificador la usa, y eso se puede reentrenar con 252 chips. |
| H2-c | Sonda lineal 1×1 (257 parámetros) sobre el embedding congelado, sin prompts | Control de H2 | Esperamos que quede **cerca de H2**, quizá apenas debajo. Si la sonda iguala a H2, la contribución del decodificador (y de los prompts) es pequeña. Eso sería un hallazgo importante para el artículo. |

## 3. Protocolo de H2 y H2-c, fijado antes de entrenar

* **Congelado**: el codificador de imagen y el codificador de prompts, incluida la codificación posicional compartida. **Se entrena**: solo `mask_decoder`.
* **Prompts de entrenamiento**: 3 positivos y 2 negativos muestreados al azar (con semilla) en el interior de la máscara Otsu con apertura y de su complemento dentro del soporte del sensor. Nunca se toman de la etiqueta. **Prompts de evaluación**: los deterministas de v2, idénticos a los de la línea base.
* **Pérdida**: BCE + Dice sobre logits reescalados a 512×512, ignorando píxeles sin etiqueta.
* **Optimización**: AdamW con decaimiento de pesos 0.01, lote 4 y **12 épocas**, sin *early stopping*: se evalúa la última época.
* **Barrido de tasa de aprendizaje**: {3e-5, 1e-4, 3e-4} con semilla 0, entrenando en `train` sin `train-dev` y midiendo en `train-dev`. La mejor tasa se fija y recién entonces se entrena con **semillas 0, 1 y 2** sobre todo `train`. Valid no se usa para elegir nada.
* **Sonda H2-c**: misma pérdida, épocas y semillas; tasa de aprendizaje 1e-3, fijada a priori por ser un modelo lineal.
* **Reporte**: media ± desviación estándar entre semillas. Se agregan el bootstrap pareado contra SAM preentrenado de salida única, contra SAM v2 y contra Otsu, el análisis por región y los casos de falla.

## 3 bis. Enmienda del 12-oct-2026, 01:55, antes de ejecutar H1-ext, D1–D3, A1–A4, P, H2 y H2-c

La reproducción R0 (notebook 02) mostró que el muestreador de prompts de v2 **no es determinista entre plataformas**. `np.argsort` sin orden estable desempata los valores iguales de la transformada de distancia según la implementación SIMD de la CPU, y en esta máquina (AVX2) cambió los puntos en 3 de los 9 chips del E1 frente a la corrida original. Para todos los experimentos nuevos se usa **v2.1**: el mismo muestreador con `kind="stable"`, que desempata por índice de píxel. La semántica no cambia (puntos de máxima distancia al borde) y el resultado deja de depender del hardware. R0 conserva el comportamiento literal de v2 para documentar la discrepancia. Esta enmienda se registra antes de ver cualquier resultado de los experimentos que afecta.

## 3 ter. Nota posterior (12-oct-2026, 03:00): análisis exploratorio NO pre-registrado

Después de ver que H2-c rindió muy por debajo de lo esperado, se agregó el notebook `05_exploratorio_sonda.ipynb` para diagnosticarlo: AUC por píxel, umbral elegido en train y sonda con más presupuesto. **Es exploratorio y no forma parte de este protocolo.** Sus conclusiones se reportan como hipótesis para el E3, no como evidencia confirmatoria. Además, la versión final de H2 y de las evaluaciones se ejecutó en GPU en vez de CPU (cambio de hardware sin cambio de protocolo). Las diferencias de IoU entre CPU y GPU fueron menores a 1e-3.

**Segundo análisis exploratorio (12-oct-2026, ~03:30).** A pedido del equipo se agregó `06_exploratorio_curva_validacion.ipynb`: entrena con el protocolo de H2 hasta 24 épocas y mide el IoU por época en `train-dev`, sin usar `valid`. Es exploratorio y no modifica el resultado pre-registrado de H2.

**Desviación declarada.** El protocolo indicaba 10 000 remuestreos bootstrap; los notebooks 03 y 04 usaron **5 000** para acortar la ejecución. Con 5 000 remuestreos, el error Monte Carlo en los percentiles 2.5 y 97.5 es del orden de milésimas de IoU, despreciable frente al ancho de los intervalos reportados. Ninguna conclusión depende de esa diferencia.

## 4. Qué cuenta como resultado negativo

Si H2 no supera a SAM preentrenado, o lo supera sin alcanzar a Otsu, **se reporta tal cual**. Lo mismo vale si la sonda H2-c iguala o supera al decodificador ajustado. Ningún resultado de este protocolo se omite del informe.

## 5. Firmas del equipo (completar a mano)

| Integrante | Leí y acepto o corrijo las expectativas (indicar cuáles cambió) | Fecha |
|---|---|---|
| [Nombre 1] | | |
| [Nombre 2] | | |
| [Nombre 3] | | |
