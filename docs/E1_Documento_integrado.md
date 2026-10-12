# Evaluación de SAM con prompts automáticos para segmentación de agua en Sentinel 1

Entrega E1 del curso INF659 Técnicas Avanzadas de Data Mining y Sistemas Inteligentes

Maestría PUCP. Fecha de entrega: 12 de octubre de 2026.

Integrantes: [Integrante 1], [Integrante 2], [Integrante 3].

El estudio evalúa SAM con prompts derivados de Otsu sobre Sen1Floods11 y plantea una evaluación posterior en Piura. La intervención preliminar corregida no mejora el baseline en la muestra exploratoria. Se presentan propuesta, disección, metodología, resultados, límites y cuaderno de decisiones; el ajuste del decodificador continúa pendiente.

# 1 Resumen e introducción

## Resumen

Este proyecto estudia la adaptación de Segment Anything Model (SAM), desarrollado por Meta, al mapeo de agua en imágenes de radar Sentinel-1, con proyección hacia la evaluación de inundaciones del Niño Costero de 2017 en Piura, Perú. El problema central es reducir la intervención humana requerida para definir los puntos que orientan la segmentación. Se propone generar prompts positivos y negativos mediante umbralización de Otsu sobre la retrodispersión SAR y evaluar si SAM mejora la máscara inicial. Sen1Floods11 constituye el benchmark de referencia, mientras que Piura corresponde a una evaluación local prevista, todavía no ejecutada. El estudio comprende la disección instrumentada del modelo, intervenciones controladas y evaluación fuera del dominio de preentrenamiento. Se contrastarán tres hipótesis: mejora respecto al umbral, beneficio del ajuste exclusivo del decodificador y utilidad de SAM para seleccionar regiones candidatas. La ejecución preliminar v2, corregida tras revisión para eliminar la dependencia de los prompts negativos respecto a las etiquetas, evalúa nueve chips y no respalda la mejora propuesta por H1. La contribución esperada es establecer condiciones de utilidad y falla, con métricas transparentes, para una aplicación relevante al contexto peruano.

## Introducción y motivación

El mapeo de inundaciones requiere procedimientos que permitan delimitar áreas afectadas de manera reproducible. En este proyecto, las imágenes SAR de Sentinel-1 constituyen la fuente de observación y la segmentación de agua constituye la tarea operacional inicial. Esta formulación exige distinguir entre detectar agua en una adquisición y identificar agua atribuible a una inundación: sin una referencia temporal o una delimitación previa del agua permanente, ambas categorías pueden confundirse. El trabajo comienza con el benchmark y reserva la evaluación de inundación del evento local para una etapa con datos y criterios explícitos.

El Niño Costero de 2017 en Piura ofrece una motivación territorial concreta. Sin embargo, los resultados disponibles corresponden a chips de Sen1Floods11 de otras regiones y no demuestran desempeño en Perú. La evaluación local requerirá comprobar disponibilidad, licencia, fechas de adquisición, máscaras de referencia y compatibilidad de preprocesamiento. La pertinencia del caso peruano reside en estudiar la transferencia del procedimiento a ese contexto, sin asumir que la transferencia será exitosa.

SAM permite orientar la segmentación mediante indicaciones espaciales, como puntos positivos y negativos. [Segment Anything](https://arxiv.org/abs/2304.02643). La definición manual de estas indicaciones introduce una tarea adicional por imagen. El proyecto propone automatizarla mediante una máscara de umbral que suministre candidatos de agua y de no agua. Así se podrá evaluar si la segmentación condicionada aporta información útil respecto al procedimiento clásico que genera sus propios prompts. La automatización constituye una hipótesis operacional; su eficacia debe medirse y no deducirse de la capacidad general del modelo.

# 2 Estado del arte

SAM introduce un paradigma de segmentación condicionada por prompts y organiza el procesamiento en un codificador de imagen, un codificador de prompts y un decodificador de máscaras. Esta estructura permite separar la representación visual de las indicaciones espaciales y de la producción de máscaras. Para el presente proyecto, esta separación facilita comparar el modelo preentrenado con una intervención restringida al decodificador. Su aplicación a SAR no garantiza que las características aprendidas permitan discriminar agua de otras superficies. [Segment Anything](https://arxiv.org/abs/2304.02643).

Sen1Floods11 proporciona una referencia georreferenciada para desarrollar y evaluar algoritmos de inundación con Sentinel-1. Su documentación especifica imágenes SAR con bandas VV y VH expresadas en dB y chips de 512 por 512 píxeles. Esta estructura permite contrastar procedimientos sobre observaciones y etiquetas comunes. El benchmark y el estudio local de Piura deben mantenerse como evaluaciones diferenciadas, porque compartir sensor no equivale a compartir distribución geográfica ni condiciones de adquisición. [Repositorio oficial de Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11).

TSM-SAM constituye un antecedente directo de automatización de la segmentación de inundaciones mediante umbralización y SAM. El capítulo del manual de Naciones Unidas y FAO describe un marco que también incorpora Isolation Forest. Por ello, el pipeline recibido debe presentarse como una implementación inspirada en ese patrón y no como una reproducción integral del método. La originalidad propuesta reside en el análisis experimental, las intervenciones y la transferencia local, no en haber inventado la conversión de umbrales en prompts. [An Automated Flood Detection Algorithm for Post-Disaster Assessment](https://fao-eostat.github.io/UN-Handbook/dis_china_floods.html).

MFF-Net es un trabajo relacionado sobre detección de inundaciones mediante SAR que evalúa el evento de 2017 en la cuenca del río Piura. Su arquitectura utiliza fusión de información de frecuencias e incertidumbre y constituye una referencia temática distinta de SAM. Sus resultados publicados no deben incorporarse como si fueran una comparación controlada con este experimento: se requeriría igualar datos, particiones, preprocesamiento y métricas. La referencia bibliográfica corresponde al volumen 18, número 1, artículo 123; la fecha exacta de publicación debe verificarse antes de normalizar la bibliografía. [MFF-Net](https://www.mdpi.com/2072-4292/18/1/123).

Como referencias adicionales, Segment anything, from space? permite contextualizar la evaluación de SAM en observación terrestre, sin asumir equivalencia entre imágenes remotas ópticas y SAR. [Ren et al., WACV 2024](https://arxiv.org/abs/2304.13000). SAM-Adapter: Adapting Segment Anything in Underperformed Scenes aporta un antecedente sobre adaptación del modelo en tareas donde su uso directo resulta insuficiente; no constituye por sí mismo una validación de ajustar exclusivamente el decodificador en SAR. [Chen et al., ICCVW 2023](https://openaccess.thecvf.com/content/ICCV2023W/VCL/papers/Chen_SAM-Adapter_Adapting_Segment_Anything_in_Underperformed_Scenes_ICCVW_2023_paper.pdf). CAT-SAM: Conditional Tuning for Few-Shot Adaptation of Segment Anything Model ofrece otra referencia para discutir adaptación con pocas muestras y distinguirla de la intervención específica propuesta aquí. [Xiao et al., ECCV 2024 (oral)](https://arxiv.org/abs/2402.03631).

Para ampliar la discusión de SAR, On Flood Detection Using Dual-Polarimetric SAR Observation orienta el análisis de la elección entre VV, VH y su combinación. [Kim, Lee y Park, Remote Sensing 17(11):1931](https://www.mdpi.com/2072-4292/17/11/1931). Unsupervised Rural Flood Mapping from Bi-Temporal Sentinel-1 Images Using an Improved Wavelet-Fusion Flood-Change Index (IWFCI) and an Uncertainty-Sensitive Markov Random Field (USMRF) Model distingue segmentación en una adquisición de detección temporal de inundación. [Mohsenifar, Mohammadzadeh y Jamali, Remote Sensing 17(6):1024](https://www.mdpi.com/2072-4292/17/6/1024). Las cinco referencias se verificaron contra la fuente original, no solo contra el título: título, autores y venue coinciden con lo citado y con el tema para el que se proponen (detalle en Verificacion_citas_Parte2.md). SAM-Adapter tiene además un preprint en arXiv (2304.09148) con un título más largo sobre el mismo trabajo; se cita la versión de workshop por ser la publicada.

# 3 Metodología y disección

## Datos, preprocesamiento y prompts

Se utilizarán las particiones oficiales de Sen1Floods11, con entrenamiento, selección de configuración y evaluación final separados. Se verificó contra los CSV reales del bucket público (v1.1/splits/flood_handlabeled/) que la distribución es exactamente 446 chips etiquetados manualmente: 252 de entrenamiento, 89 de validación, 90 de prueba y 15 de Bolivia; los cuatro conteos coinciden fila por fila con los archivos flood_train_data.csv, flood_valid_data.csv, flood_test_data.csv y flood_bolivia_data.csv. El registro preliminar contiene nueve chips evaluables y reporta la exclusión de Ghana_5079. La selección inicial es exploratoria y no representa una evaluación completa de la validación.

En cada chip se calculará Otsu sobre los valores finitos de la banda seleccionada, sin utilizar LabelHand para determinar el umbral. La primera configuración utiliza VH y clasifica como candidato de agua el píxel con retrodispersión inferior al umbral. El script aplica después una apertura morfológica con disco de radio un píxel. Por tanto, el baseline recibido es Otsu con apertura y debe nombrarse así. La ejecución v2 incluye además Otsu sin apertura para separar el efecto del procesamiento morfológico.

Los puntos positivos se seleccionan en el interior de componentes conectados de la máscara candidata mediante distancia al borde. Los negativos se toman de su complemento dentro del soporte válido del sensor. El pipeline v2, corregido tras revisión, configura tres positivos y dos negativos y define sensor_valid exclusivamente mediante valores finitos en VV y VH. LabelHand se reserva para evaluar y para aplicar el criterio explícito de exclusión por más de 95% de píxeles sin etiqueta. Se verificó en los nueve registros la correspondencia de las coordenadas con la máscara candidata. Esta comprobación no constituye una auditoría completa de las transformaciones internas del procesador. Para nuevas escenas también deberá comprobarse si existen valores finitos de nodata en los metadatos del radar.

La configuración inicial normaliza VH mediante percentiles 2 y 98, sustituye valores no finitos por la mediana y replica la banda en tres canales. Esta transformación crea una representación de entrada compatible con el procesamiento de imágenes del modelo, pero no añade información espectral. Los puntos se proporcionan con sus etiquetas positivas y negativas al procesador de SAM. El script selecciona la máscara con mayor puntuación de calidad predicha por el modelo; esa puntuación no equivale al IoU medido contra LabelHand. Se documentarán la transformación de coordenadas, el retorno a resolución original y el tratamiento de casos sin puntos positivos.

## Fases e hipótesis

La disección de Fase A reporta 93 735 728 parámetros totales: 89 670 912 en el codificador de imagen, 6 476 en el codificador de prompts y 4 058 340 en el decodificador de máscaras. La suma de estos tres submódulos difiere del total en 10 parámetros, cuya ubicación deberá documentarse. La razón entre codificador de imagen y decodificador es aproximadamente 22.1. Esto respalda una reducción de parámetros entrenables al ajustar únicamente el decodificador, pero no prueba todavía un ahorro de tiempo ni una mejora de exactitud.

El perfilado actualizado de Fase A usa una imagen RGB sintética aleatoria de 1024 por 1024 píxeles y un prompt positivo en CPU. Sobre cinco repeticiones tras un calentamiento, el forward completo tiene una latencia media de 12 424.1151 ms, desviación de 438.9304 ms, mínimo de 11 824.1785 ms y máximo de 13 101.0051 ms. El procesamiento previo de la entrada queda fuera del intervalo. El pico de memoria residente de todo el proceso es 2 500.9 MiB, aproximadamente 2.442 GiB. Se trata de mediciones reales de cómputo con entrada sintética, no del pipeline sobre SAR ni de una GPU. El costo con chips reales sigue como [latencia_pipeline_SAR_ms = pendiente] y [memoria_pipeline_SAR_GiB = pendiente].

La estimación analítica actualizada distingue cuatro capas globales, de índices 2, 5, 8 y 11, y ocho capas con ventanas de 14 por 14 parches. Reporta aproximadamente 109.52 GFLOPs por bloque global, 60.9324 por bloque con ventanas y 930.38 para la proyección de parches y los doce bloques. La convención cuenta dos operaciones por multiplicación y suma. La fórmula no es un perfilado automático y omite operaciones adicionales y el decodificador. Aunque el término de atención por ventanas incorpora 25 ventanas de 196 tokens, se debe confirmar cómo la implementación ejecuta las proyecciones de los tokens de relleno; la cifra se presenta como aproximación parcial, no como conteo exhaustivo de SAM. El tamaño relativo de los módulos en parámetros tampoco determina por sí solo su costo computacional.

La proyección de parches tiene campo receptivo local de 16 por 16 píxeles. Las primeras dos capas operan con ventanas de 14 por 14 parches, equivalentes a 224 por 224 píxeles. La primera capa global es la tercera, contando desde uno. Una vez que un token incorpora información global, las capas posteriores por ventanas no eliminan esa dependencia: debe distinguirse alcance directo de atención de campo receptivo teórico acumulado. La descripción no constituye una medición del campo receptivo efectivo. [Configuración oficial de SAM ViT-B](https://huggingface.co/facebook/sam-vit-base/raw/main/config.json).

H1 plantea que SAM preentrenado con prompts automáticos mejora el IoU respecto a la máscara Otsu con apertura que origina esos mismos prompts. La ejecución v2, corregida tras revisión, compara los mismos nueve chips sobre idéntico soporte de evaluación. El JSON reporta IoU global de 0.1875326163 para el baseline y 0.0897472400 para SAM, y promedios por chip de 0.1923513645 y 0.0819400793, respectivamente. Otsu sin apertura obtiene 0.1592855732 global y 0.1625590483 promedio por chip. Los agregados se comprobaron a partir de TP, FP y FN del JSON y los promedios individuales. Las cifras proceden de la ejecución de Claude, no de una nueva inferencia de ChatGPT. SAM pierde frente a Otsu con apertura en ocho de nueve chips, por lo que H1 no recibe apoyo en esta configuración y muestra exploratoria.

H2 plantea que entrenar exclusivamente el decodificador, manteniendo congelados los codificadores de imagen y de prompts, mejora el desempeño respecto a SAM preentrenado con la misma estrategia de prompts. El entrenamiento utilizará solamente la partición de entrenamiento y una función de pérdida definida previamente, con píxeles sin etiqueta excluidos. Los hiperparámetros se seleccionarán en validación y la prueba permanecerá reservada. Se registrarán semillas, presupuesto, parámetros entrenables y costo observado; el menor número de parámetros actualizados no demuestra por sí mismo una menor latencia de inferencia. El resultado se mantiene como [IoU_decoder_ajustado = pendiente].

H3 plantea que SAM puede contribuir a seleccionar regiones candidatas aun cuando no mejore directamente la máscara completa. Se comparará un conjunto de regiones derivado del baseline con otro filtrado mediante una regla de puntuación de SAM fijada en validación. Se medirá la reducción de falsos positivos junto con la conservación de agua verdadera, para evitar que descartar casi todas las regiones parezca una mejora. Esta intervención permanece pendiente y no tiene resultados medidos.

## Plan de evaluación

El IoU de agua se calculará como TP dividido entre TP más FP más FN sobre píxeles de evaluación válidos. El IoU global acumula esos conteos en todos los chips antes de efectuar la división. El promedio por chip calcula primero cada IoU y después su media, asignando el mismo peso a cada chip evaluable. El primero concede mayor influencia a las escenas con mayor unión; el segundo muestra el desempeño de la unidad espacial elegida. Se reportarán conjuntamente, acompañados por resultados individuales y fracción de agua predicha, sin escoger después de observar los resultados la métrica más favorable.

Se añadirá precisión y sensibilidad de agua, diferencia pareada de IoU y comparación con predicciones triviales de toda agua y de ninguna agua. Si la unión es cero, el IoU se registrará como no definido y se informará el número de casos, con la misma política para todos los métodos. Los casos sin prompts positivos deberán recibir una salida definida y seguir participando en las métricas cuando corresponda. Los conteos TP, FP y FN se conservarán por chip para reconstruir los agregados.

Para estimar incertidumbre se considerará bootstrap pareado por evento, remuestreando eventos completos y conservando dentro de cada réplica ambos métodos. Antes se verificará qué chips pertenecen al mismo evento mediante los metadatos. Remuestrear píxeles individuales produciría unidades de análisis incompatibles con su dependencia espacial. Con solo un chip por región y nueve chips, el bootstrap no recupera variabilidad interna no observada y sus intervalos serían exploratorios. Se priorizará ampliar la cobertura y reportar diferencias por evento antes de sostener inferencias generales.

# 4 Resultados preliminares y análisis de fallas

## Comparación agregada

La ejecución v2 utiliza nueve chips evaluables de los diez seleccionados mediante la primera fila de cada región en el manifiesto de validación. Ghana_5079 fue excluido por 100% de etiqueta sin dato. Los resultados corresponden a SAM oficial preentrenado en CPU, con VH replicada en tres canales y prompts automáticos. La evaluación excluye LabelHand igual a menos uno. La corrección separa validez del sensor y de etiquetas antes de generar prompts.

| Método | IoU global | IoU promedio por chip |
|---|---:|---:|
| Otsu sin apertura | 0.159286 | 0.162559 |
| Otsu con apertura | 0.187533 | 0.192351 |
| SAM con prompts automáticos v2 | 0.089747 | 0.081940 |

## Resultados por chip

| Chip | IoU Otsu con apertura | IoU SAM v2 | Agua predicha por SAM sobre píxeles etiquetados válidos |
|---|---:|---:|---:|
| India_1050276 | 0.4483 | 0.1885 | 97.56% |
| Mekong_1149855 | 0.4963 | 0.1296 | 44.16% |
| Nigeria_31096 | 0.0103 | 0.0035 | 97.49% |
| Pakistan_43105 | 0.0006 | 0.0004 | 21.18% |
| Paraguay_305760 | 0.3724 | 0.1670 | 99.70% |
| Somalia_12849 | 0.1770 | 0.1981 | 96.78% |
| Spain_1199913 | 0.1527 | 0.0358 | 97.34% |
| Sri-Lanka_85652 | 0.0176 | 0.0101 | 97.65% |
| USA_826217 | 0.0561 | 0.0046 | 73.56% |

SAM obtiene menor IoU que Otsu con apertura en ocho de nueve chips. En seis predice agua en más de 96% del soporte etiquetado válido. Somalia es el único caso con mayor IoU de SAM frente al baseline, pese a la sobresegmentación extensa. Esta observación muestra por qué un puntaje aislado no basta para valorar la utilidad de una máscara. La muestra no respalda H1 bajo esta configuración, pero tampoco identifica si la causa es la representación SAR, los prompts, la selección de máscaras u otro componente. H2 y H3 no se ejecutaron.

![Comparación del chip Somalia con la versión corregida](comparacion_real_v2.png)

Figura 1. Banda VH, baseline Otsu con apertura, predicción SAM v2 y referencia LabelHand del chip Somalia_12849. Los IoU corresponden a píxeles de etiqueta válidos; la figura no distingue visualmente todos los píxeles ignorados, por lo que debe interpretarse junto con esa política de evaluación.

# 5 Métricas y amenazas a la validez

## Revisión del plan de métricas

La distinción entre IoU global e IoU promedio por chip es adecuada porque responde a preguntas diferentes. El JSON v2 conserva TP, FP y FN por chip y método, lo que permite comprobar tanto los IoU globales como los promedios. Otsu sin apertura alcanza 0.1593 global y 0.1626 promedio, frente a 0.1875 y 0.1924 con apertura. La mejora del IoU coexiste con una reducción de TP de 120 191 a 113 915 y de FP de 589 699 a 442 577: la apertura reduce falsas detecciones pero también elimina agua correctamente detectada. Este intercambio justifica acompañar IoU con precisión y sensibilidad. No prueba que la apertura mejore universalmente todas las escenas. La v2 evalúa también una máscara SAM vacía cuando no hay prompts positivos y reporta cero casos de unión vacía en los tres métodos de esta corrida.

El bootstrap por evento es conceptualmente razonable frente al remuestreo por píxel, pero exige identificar los eventos y reconocer que nueve chips ofrecen cobertura limitada. Un chip por región no proporciona una estimación representativa del desempeño regional. Con pocos conglomerados, los intervalos deben interpretarse con cautela y no convertir la exploración en evidencia confirmatoria. El protocolo debe fijar de antemano la métrica principal, la política de unión vacía y los criterios de exclusión; ampliar la muestra es más informativo que generar intervalos precisos en apariencia sobre la selección inicial.

## Límites y amenazas a la validez

La evidencia disponible está limitada por la muestra exploratoria, el desbalance agua/no agua y la ausencia de evaluación en Piura. La entrada VH replicada, la umbralización, la ubicación de prompts y la selección de máscaras son explicaciones posibles de la sobresegmentación, pero los resultados actuales no identifican cuál causa el fallo. La dependencia de la validez de LabelHand al generar prompts negativos fue corregida en v2, y la reevaluación no alteró el patrón principal. Claude retiró la afirmación de ausencia de toda falla de implementación. Se mantiene pendiente una auditoría completa de normalización, transformaciones y posibles valores finitos de nodata. La disección actualizada distingue atención global y por ventanas; sus FLOPs son una aproximación parcial y el perfilado sintético debe mantenerse separado de la ejecución sobre datos SAR.

En seis chips, la fracción predicha de agua por SAM supera 0.96 según el JSON, lo que respalda describir sobresegmentación extensa. La figura de Somalia muestra que su mayor IoU respecto al baseline coexiste con una máscara casi completa; no permite atribuir causalidad ni negar toda coincidencia espacial correcta. La Fase C exige reportar estos casos y distinguir puntaje de utilidad. Además, segmentar agua no equivale a identificar exclusivamente inundación. La generalización al evento peruano, la eficacia del ajuste del decodificador y la superioridad frente a métodos especializados permanecen abiertas. La trazabilidad de datos, configuraciones, exclusiones y asistencia de IA será indispensable para que cada integrante pueda defender las decisiones.

# 6 Cronograma

Las fechas siguientes proceden del contexto del curso facilitado por el equipo y deberán contrastarse con el aula virtual.

| Fecha de 2026 | Entregable | Trabajo pendiente |
|---|---|---|
| 12 de octubre | E1: propuesta, disección, intervención preliminar y cuaderno | Integrar este borrador, revisar referencias, integrar la disección actualizada, completar responsables y validar las referencias. |
| 16 de octubre | E2: inscripción del póster WAIMLAP | Condensar problema, método y evidencia preliminar, confirmando requisitos y fecha de inscripción. |
| 16 de noviembre | E3: entrega completa | Corregir el protocolo, ejecutar ablaciones, evaluar en prueba y documentar la evaluación local si existen datos adecuados. |
| 18 al 20 de noviembre | Defensa oral individual | Preparar explicaciones personales de hipótesis, métricas, código, resultados negativos y uso de IA. |
| 30 de noviembre | Artículo final IEEE/LNCS, 4 a 8 páginas | Consolidar resultados, límites, referencias, reproducibilidad y formato editorial requerido. |

Entre E1 y E2 se priorizará auditar el pipeline y delimitar el alcance del póster. Posteriormente se completará la disección y se ejecutará H2 bajo un protocolo congelado, seguido de H3 y las ablaciones seleccionadas. Antes de E3 se reservará tiempo para revisar errores, comprobar las particiones y consolidar los archivos de reproducción. La evaluación de Piura quedará condicionada a disponer de referencia independiente y condiciones de uso verificadas; si no se concreta, se declarará como trabajo futuro.

# 7 Reproducibilidad y archivos de respaldo

El repositorio del proyecto es [INF659 Tarea Académica SAM](https://github.com/LuisMirandaMallqui/INF659_TareaAcademica_SAM), público. El manifiesto documenta URLs, SHA-256 y la regla de selección de chips; el ZIP contiene los veinte GeoTIFF de las diez parejas imagen y etiqueta. La integridad del ZIP se comprobó contra el manifiesto. Los resultados de segmentación se conservan en resultados_reales_v2.json y la disección en fase_a_diseccion_real.json. Los scripts correspondientes son run_sam_pipeline_v2.py y fase_a_diseccion.py.

El entorno del pipeline reporta Python 3.13.16, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-image 0.26.0, SciPy 1.18.1 y NumPy 2.5.3. Se utiliza facebook/sam-vit-base. Deberá fijarse una revisión inmutable de pesos y registrar todas las dependencias, incluida tifffile. La carpeta chips debe prepararse a partir del ZIP antes de ejecutar el pipeline; las salidas se escriben en results_v2. La disección emplea resource de Linux, lo cual requiere adaptación en Windows.

El muestreo de puntos es determinístico, pero eso no vuelve innecesaria toda semilla: la disección crea una imagen aleatoria y H2 requerirá controlar inicialización, orden de entrenamiento y otras fuentes de aleatoriedad. No consta una semilla fijada para la imagen sintética del script recibido. El requisito de notebooks reproducibles comunicado por el equipo debe atenderse junto con README, entorno y semillas: el equipo debe decidir si conserva los .py o los convierte a .ipynb antes de la entrega final, eso no se resolvió aquí. El contenido remoto del repositorio sí se verificó contra la API de GitHub: los 15 archivos (README, requirements, scripts, docs, resultados y el ZIP de chips) están efectivamente en la rama main.

# 8 Cuaderno de decisiones y declaración de IA

Las filas de ejemplo registran actividades de IA conocidas. La validación del estudiante permanece pendiente hasta que el integrante responsable la realice; no se atribuye comprensión que todavía no se ha acreditado.

| Tarea | Herramienta de IA usada | Qué se le pidió exactamente | Qué devolvió | Qué validó o entendió el estudiante antes de usarlo | Integrante responsable |
|---|---|---|---|---|---|
| Redacción del borrador E1 | ChatGPT | Revisar el material de Claude y ejecutar las ocho partes delegadas de redacción, métricas y ablaciones. | Este borrador con referencias candidatas y separación de evidencia y propuestas. | Pendiente: revisar cada párrafo, fuente y correspondencia con el protocolo. | [Nombre pendiente] |
| Revisión del experimento recibido | ChatGPT | Revisar script, JSON, informe y figura suministrados. | Identificación del uso de validez de LabelHand en prompts negativos y límites del diagnóstico. | Pendiente: explicar la dependencia original y revisar su corrección v2 con soporte del sensor. | [Nombre pendiente] |
| Pipeline preliminar real | Claude | Ejecutar Otsu, prompts automáticos, SAM y evaluación contra LabelHand en chips reales. | Script v2, corrección, JSON con nueve chips y figura v2 del caso Somalia. | Pendiente: reproducir, verificar selección de datos y auditar coordenadas, máscaras y métricas. | [Nombre pendiente] |
| [Tarea] | [Herramienta] | [Solicitud exacta] | [Respuesta y archivo] | [Validación personal y evidencia] | [Nombre] |
| [Tarea] | [Herramienta] | [Solicitud exacta] | [Respuesta y archivo] | [Validación personal y evidencia] | [Nombre] |
| [Tarea] | [Herramienta] | [Solicitud exacta] | [Respuesta y archivo] | [Validación personal y evidencia] | [Nombre] |
| [Tarea] | [Herramienta] | [Solicitud exacta] | [Respuesta y archivo] | [Validación personal y evidencia] | [Nombre] |
| [Tarea] | [Herramienta] | [Solicitud exacta] | [Respuesta y archivo] | [Validación personal y evidencia] | [Nombre] |
| [Tarea] | [Herramienta] | [Solicitud exacta] | [Respuesta y archivo] | [Validación personal y evidencia] | [Nombre] |
| [Tarea] | [Herramienta] | [Solicitud exacta] | [Respuesta y archivo] | [Validación personal y evidencia] | [Nombre] |

# Anexo A Intervenciones adicionales previstas

Ideas a evaluar, no resultados.

Umbral global frente a local. Hipótesis previa: el umbral local reduce falsos positivos en escenas heterogéneas respecto a Otsu global, manteniendo constantes banda, prompts y modelo.
Muestreo por componentes frente a cobertura espacial. Hipótesis previa: distribuir el mismo presupuesto de puntos en distintas zonas mejora el IoU frente a concentrarlo en componentes dominantes.
Cantidad de prompts. Hipótesis previa: aumentar el presupuesto de puntos mejora la sensibilidad hasta un límite, a partir del cual no compensa los errores de la máscara inicial.
Polarización de entrada. Hipótesis previa: sustituir VH por VV cambia el desempeño de SAM bajo el mismo procedimiento; se fijará por separado si también cambia la banda que genera prompts para evitar confundir dos efectos.
Composición de bandas. Hipótesis previa: una composición fija de VV, VH y su diferencia en dB mejora el IoU frente a VH replicada, manteniendo los prompts originales y normalizaciones documentadas.
Apertura morfológica. Control ya ejecutado en v2; para una nueva evaluación se deberá registrar previamente la hipótesis: la apertura reduce falsas detecciones pequeñas, pero puede eliminar agua estrecha; se compararán IoU y sensibilidad con y sin apertura bajo el mismo generador de puntos.
Selector de máscaras. Hipótesis previa: una regla basada en consistencia con prompts reduce la sobresegmentación frente al máximo de calidad predicha; se fijará la regla sin consultar etiquetas de evaluación.
Cuantización del modelo. Hipótesis previa: una configuración compatible de menor precisión reduce memoria y tiempo con pérdida de IoU dentro de una tolerancia predeclarada [tolerancia_IoU = pendiente], conservando entradas y prompts. Esta intervención requiere comprobar soporte y medirla en el mismo hardware.

Los cambios en prompts y preprocesamiento son controles útiles, pero no constituyen automáticamente una intervención en el núcleo de SAM. H2 y una intervención compatible de cuantización atenderán ese requisito; deberá confirmarse con la rúbrica original. Un modelo no promptable entrenado en las mismas particiones puede añadirse como comparación externa, separada de las ablaciones internas.

# Anexo B Manifiesto de datos


Dataset: Sen1Floods11 v1.1, split de validacion oficial.
CSV de split (89 filas reales): https://storage.googleapis.com/sen1floods11/v1.1/splits/flood_handlabeled/flood_valid_data.csv

Regla de seleccion: primera fila de cada region, en el orden en que aparecen las regiones en el CSV original, recorriendo Ghana, India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri-Lanka, USA. No hubo seleccion manual de chips favorables.

Chips descargados (10 de 89; 9 evaluables, Ghana_5079 excluido por 100% nodata de etiqueta):

| Chip | URL imagen S1Hand | URL mascara LabelHand |
|---|---|---|
| Ghana_5079 | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Ghana_5079_S1Hand.tif | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Ghana_5079_LabelHand.tif |
| India_1050276 | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/India_1050276_S1Hand.tif | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/India_1050276_LabelHand.tif |
| Mekong_1149855 | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Mekong_1149855_S1Hand.tif | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Mekong_1149855_LabelHand.tif |
| Nigeria_31096 | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Nigeria_31096_S1Hand.tif | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Nigeria_31096_LabelHand.tif |
| Pakistan_43105 | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Pakistan_43105_S1Hand.tif | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Pakistan_43105_LabelHand.tif |
| Paraguay_305760 | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Paraguay_305760_S1Hand.tif | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Paraguay_305760_LabelHand.tif |
| Somalia_12849 | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Somalia_12849_S1Hand.tif | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Somalia_12849_LabelHand.tif |
| Spain_1199913 | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Spain_1199913_S1Hand.tif | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Spain_1199913_LabelHand.tif |
| Sri-Lanka_85652 | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Sri-Lanka_85652_S1Hand.tif | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Sri-Lanka_85652_LabelHand.tif |
| USA_826217 | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/USA_826217_S1Hand.tif | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/USA_826217_LabelHand.tif |

Pesos del modelo: facebook/sam-vit-base, https://huggingface.co/facebook/sam-vit-base (pesos oficiales de Meta, servidos via transformers de Hugging Face).

sha256 de los 20 archivos de chips descargados (para verificar integridad bit a bit):
```
042647898ddeb5be7fd46494c5bc413e22413f21465b90d0d13f57d16f23ffa0  chips/Ghana_5079_LabelHand.tif
5a598a0050e522fc8faf3ffbe8ce4edc3ad642d16c18ad1faa54574adf76f3c7  chips/Ghana_5079_S1Hand.tif
b0f21694aa0a1e5a48ccc4f66e12f5a8b3f16e606164e6cafd789b6aed121ba9  chips/India_1050276_LabelHand.tif
14a6d3889427e4b3e4a2b13fbc664cbc9bc84c33a3f2a63553b67cc0743bb73f  chips/India_1050276_S1Hand.tif
e90aaedbb51c617c6250c06cbeff13a14c992c46e5df1fe5b8988a5b239308b6  chips/Mekong_1149855_LabelHand.tif
e53925ce5265c89099b9d3aa1915661173ffd34375c751e2e24c1b1a4bd07587  chips/Mekong_1149855_S1Hand.tif
c8da655114d488f6097de39492da9e27f065feef58764c70ef5d7b8683ae8542  chips/Nigeria_31096_LabelHand.tif
928f806af7f3713ffce13c152f09713ab25198b646902ea0fe4e780a8ab70350  chips/Nigeria_31096_S1Hand.tif
60fe4edb4e6ce61fad7cd3e8810e0f030712fccf2053e7996c786941ffcb0130  chips/Pakistan_43105_LabelHand.tif
1594a4ab57da8f3ef16816661bdbba94669236364605465832047a8560c2252c  chips/Pakistan_43105_S1Hand.tif
fcb5f3b21ae74bd8275226d843d6254a982448b7669f43dbff6fea44647db24a  chips/Paraguay_305760_LabelHand.tif
ff60d990884d9ac12ab66c6604a4a9b33d7829e9a9a1d332142cdd048ab042f8  chips/Paraguay_305760_S1Hand.tif
9b7cb1c8a378c193b1be9405c6ea200c38c3fc80300d190184c0226383c0afcb  chips/Somalia_12849_LabelHand.tif
da7ae40306d1615d3bd11627cfdf7086b6f3a7335c709a69a146e725e25484ee  chips/Somalia_12849_S1Hand.tif
27fedadf6819805196eb1061a21e11ccb7ba26648a2424917b9de3174d45f2e0  chips/Spain_1199913_LabelHand.tif
343e99bb775af47d2fb690d04c1c71c56c751f179c13aa56c84992fd4e6b740f  chips/Spain_1199913_S1Hand.tif
101a849cbddaa60858d1ef42bab6c88d762c5b4aad31ddb0a4b8f0d9bfedb842  chips/Sri-Lanka_85652_LabelHand.tif
401dedb1de802b0793ada0e2f7288b2bdf2ccd1abff647be9be0b96948229210  chips/Sri-Lanka_85652_S1Hand.tif
f2a6c8a54d0fe0e6f36ee7f8930f11c8c1190f11873e97275d33e3d89a81ba6d  chips/USA_826217_LabelHand.tif
7d39bd4c52057b00a368342756883e32ef288fcc533fbe1775a9357a9df3b205  chips/USA_826217_S1Hand.tif
```
