# Segmentación de agua en Sentinel 1 mediante SAM y prompts automáticos

Entrega E1 del curso INF659 Técnicas Avanzadas de Data Mining y Sistemas Inteligentes

Maestría PUCP. Fecha de entrega: 12 de octubre de 2026.

Integrantes: [Integrante 1], [Integrante 2], [Integrante 3].

# 1 Resumen e introducción

## Resumen

Este trabajo evalúa Segment Anything Model (SAM) para segmentar agua en imágenes de radar de apertura sintética (SAR) de Sentinel-1 mediante prompts automáticos. El objetivo es reducir la intervención manual y determinar si el modelo mejora una máscara inicial obtenida con Otsu. Se utiliza Sen1Floods11 como conjunto de referencia y se plantea una evaluación posterior del Niño Costero de 2017 en Piura, Perú. La metodología comprende la disección del modelo, una comparación preliminar con umbralización y el diseño de intervenciones sobre el decodificador. En nueve chips evaluables, SAM alcanzó un IoU global de 0.0897, frente a 0.1875 de Otsu con apertura y 0.1593 sin apertura. SAM obtuvo menor IoU que el método con apertura en ocho chips y presentó sobresegmentación extensa en seis. Este patrón persistió tras corregir una dependencia de las etiquetas en la generación de prompts negativos. Los resultados no respaldan la mejora esperada bajo la configuración evaluada. La contribución inicial consiste en documentar esta limitación y establecer una comparación reproducible. El ajuste del decodificador, la ampliación de la muestra y la evaluación en Piura permanecen pendientes.

## Introducción y motivación

El mapeo de inundaciones requiere delimitar áreas afectadas mediante procedimientos reproducibles. En este estudio, la segmentación de agua en imágenes SAR constituye el primer paso. Detectar agua en una adquisición, sin embargo, no permite atribuirla por sí sola a una inundación: para distinguirla del agua permanente se necesita una referencia temporal o información adicional. Por ello, la evaluación inicial se realiza sobre un conjunto de referencia y la aplicación al evento de Piura se plantea como una etapa posterior.

El Niño Costero de 2017 en Piura da al proyecto una motivación local. Su evaluación permitirá estudiar si el método puede transferirse al contexto peruano, siempre que se disponga de imágenes, etiquetas independientes y condiciones de uso adecuadas. Los resultados actuales proceden de otras regiones y no permiten anticipar su desempeño en Piura.

SAM segmenta imágenes a partir de indicaciones espaciales, denominadas prompts, que pueden incluir puntos positivos y negativos. Su selección manual añade trabajo por imagen. Proponemos obtenerlos de una máscara de Otsu y comprobar si SAM mejora esa misma máscara. La pregunta central es, por tanto, si la representación aprendida por el modelo aporta información útil cuando la entrada es SAR y los prompts también pueden contener errores. [Kirillov et al., Segment Anything](https://arxiv.org/abs/2304.02643).

# 2 Estado del arte

SAM combina un codificador de imagen, un codificador de prompts y un decodificador de máscaras. Esta organización permite orientar la segmentación mediante indicaciones espaciales y estudiar ajustes restringidos a determinados módulos. Su capacidad general no garantiza, sin embargo, que distinga agua en imágenes SAR. Esta transferencia debe evaluarse con datos del dominio de interés. [Kirillov et al., Segment Anything](https://arxiv.org/abs/2304.02643).

Sen1Floods11 ofrece datos georreferenciados para evaluar algoritmos de inundación con Sentinel-1. Las imágenes contienen bandas VV y VH en dB, organizadas en chips de 512 por 512 píxeles. El conjunto permite comparar métodos sobre observaciones y etiquetas comunes. La evaluación en Piura debe analizarse por separado, pues utilizar el mismo sensor no asegura condiciones geográficas o de adquisición equivalentes. [Bonafilia et al., Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11).

TSM-SAM es un antecedente directo de la combinación de umbralización y SAM para detectar inundaciones. El capítulo del manual de Naciones Unidas y FAO también incorpora Isolation Forest. Nuestro procedimiento toma ese patrón como referencia, sin reproducir todo el marco. La contribución propuesta se centra en las comparaciones controladas, el análisis de fallas y la futura evaluación local. [An Automated Flood Detection Algorithm for Post-Disaster Assessment](https://fao-eostat.github.io/UN-Handbook/dis_china_floods.html).

MFF-Net aborda la detección de inundaciones con SAR y evalúa el evento de 2017 en la cuenca del río Piura mediante una arquitectura de fusión de frecuencias e incertidumbre. Es una referencia pertinente para el caso de estudio, aunque sus resultados no constituyen una comparación directa con los nuestros. Para contrastarlos sería necesario igualar datos, particiones, preprocesamiento y métricas. [MFF-Net, Remote Sensing, volumen 18, artículo 123](https://www.mdpi.com/2072-4292/18/1/123).

Otros trabajos ayudan a situar la adaptación de SAM. Segment anything, from space? estudia su aplicación a observación terrestre; SAM-Adapter propone adaptarlo a escenarios donde el uso directo resulta insuficiente; y CAT-SAM examina la adaptación con pocas muestras. Estos antecedentes motivan estudiar intervenciones específicas, pero no demuestran que ajustar solo el decodificador mejore la segmentación SAR. [Ren et al., WACV 2024](https://arxiv.org/abs/2304.13000), [Chen et al., ICCVW 2023](https://openaccess.thecvf.com/content/ICCV2023W/VCL/papers/Chen_SAM-Adapter_Adapting_Segment_Anything_in_Underperformed_Scenes_ICCVW_2023_paper.pdf), [Xiao et al., ECCV 2024](https://arxiv.org/abs/2402.03631).

La literatura específica de SAR aporta dos criterios adicionales. El estudio de observaciones de doble polarización permite considerar VV, VH y sus combinaciones. Los métodos bitemporales, por su parte, ayudan a distinguir presencia de agua y cambio asociado a inundación. Ambos enfoques orientan las comparaciones futuras del proyecto. [Kim, Lee y Park, Remote Sensing 17, artículo 1931](https://www.mdpi.com/2072-4292/17/11/1931), [Mohsenifar, Mohammadzadeh y Jamali, Remote Sensing 17, artículo 1024](https://www.mdpi.com/2072-4292/17/6/1024).

# 3 Metodología y disección

## Datos y procedimiento

Sen1Floods11 contiene 446 chips etiquetados manualmente: 252 de entrenamiento, 89 de validación, 90 de prueba y 15 de Bolivia. Estos conteos fueron contrastados con los cuatro CSV de la versión 1.1. La exploración inicial seleccionó la primera fila de cada región en la validación. De diez chips, nueve resultaron evaluables; Ghana_5079 se excluyó por carecer completamente de etiqueta válida. Esta selección permite una primera inspección de fallas, pero no representa toda la partición de validación.

El procedimiento utiliza la banda VH. Otsu se calcula sobre píxeles con valores finitos en VV y VH, y los valores inferiores al umbral se clasifican como candidatos de agua. Se aplica una apertura morfológica con disco de radio un píxel. Se conserva también la máscara sin apertura para separar el efecto de este procesamiento.

A partir de la máscara con apertura se seleccionan hasta tres puntos positivos y dos negativos. Los positivos se ubican en el interior de componentes conectados mediante distancia al borde; los negativos, en el complemento de la máscara dentro del soporte válido del radar. La versión corregida, v2, obtiene ese soporte exclusivamente del sensor. LabelHand interviene en la evaluación y en el criterio de exclusión por más de 95% de píxeles sin etiqueta, pero no en la generación de prompts. Los nueve registros confirman la correspondencia entre las coordenadas seleccionadas y su clase en la máscara inicial.

Para entrar a SAM, VH se normaliza con los percentiles 2 y 98, se sustituyen los valores no finitos por la mediana y se replica la banda en tres canales. Esta representación no añade información espectral. SAM recibe los puntos y sus etiquetas, y se conserva la máscara con mayor puntuación de calidad predicha. Esa puntuación interna se distingue del IoU calculado posteriormente contra LabelHand. Si no hay puntos positivos, el procedimiento evalúa una máscara vacía y mantiene el chip en la comparación.

## Fase A Disección del modelo

El modelo facebook/sam-vit-base tiene 93 735 728 parámetros. El codificador de imagen concentra 89 670 912; el de prompts, 6 476; y el decodificador de máscaras, 4 058 340. La suma de los tres submódulos coincide exactamente con el total, sin parámetros fuera de ese desglose. La razón entre codificador de imagen y decodificador es aproximadamente 22.1. Ajustar solo el decodificador reduce los parámetros actualizados, aunque el ahorro de tiempo y la mejora de desempeño aún requieren medición.

La estimación analítica distingue cuatro capas de atención global, con índices 2, 5, 8 y 11, y ocho capas con ventanas de 14 por 14 parches. Reporta aproximadamente 109.52 GFLOPs por bloque global, 60.9324 por bloque con ventanas y 930.38 para la proyección de parches y los doce bloques. Se cuentan dos operaciones por multiplicación y suma. Es una aproximación parcial: no incluye el decodificador ni todas las operaciones adicionales, y requiere comprobar el tratamiento de las proyecciones de tokens de relleno.

La proyección inicial tiene un campo receptivo local de 16 por 16 píxeles. Las dos primeras capas atienden dentro de ventanas de 14 por 14 parches, equivalentes a 224 por 224 píxeles. La tercera capa introduce atención global. A partir de ella, las capas por ventanas conservan la información global incorporada: el alcance directo de la atención y el campo receptivo acumulado describen propiedades diferentes. Este análisis es teórico y no mide el campo receptivo efectivo. [Configuración oficial de SAM ViT-B](https://huggingface.co/facebook/sam-vit-base/raw/main/config.json).

El perfilado en CPU utiliza una imagen RGB sintética de 1024 por 1024 píxeles y un prompt positivo. Tras un calentamiento, cinco repeticiones del forward arrojan una latencia media de 12 424.1 ms y desviación de 438.9 ms. El pico de memoria residente de todo el proceso es 2 500.9 MiB. Estas mediciones excluyen el preprocesamiento y no corresponden al pipeline SAR ni a una GPU. Su costo permanece como [latencia_pipeline_SAR_ms = pendiente] y [memoria_pipeline_SAR_GiB = pendiente].

## Fase B Hipótesis e intervenciones

H1 plantea que SAM preentrenado con prompts automáticos mejora el IoU de la máscara Otsu con apertura que origina esos prompts. La comparación utiliza los mismos chips y píxeles de evaluación. Los resultados preliminares se presentan en la sección 4.

H2 plantea que ajustar exclusivamente el decodificador, con ambos codificadores congelados, mejora el desempeño frente a SAM preentrenado bajo la misma estrategia de prompts. El entrenamiento se realizará con la partición de entrenamiento; la selección de hiperparámetros, con validación; y la evaluación final, con prueba reservada. Se fijarán función de pérdida, semillas y presupuesto antes de ejecutar. El resultado continúa como [IoU_decoder_ajustado = pendiente].

H3 plantea que SAM puede ayudar a filtrar regiones candidatas aunque no mejore la máscara completa. Se compararán las regiones del método de referencia con las seleccionadas mediante una regla de puntuación fijada en validación. La reducción de falsos positivos se evaluará junto con la conservación de agua verdadera. Esta intervención aún no se ha ejecutado.

## Evaluación

El IoU de agua se calcula como TP dividido entre TP más FP más FN, excluyendo píxeles de LabelHand sin dato. El IoU global suma los conteos antes de dividir; el promedio por chip asigna igual peso a cada chip con IoU definido. Ambos se reportan porque describen aspectos distintos del desempeño. Se añadirán precisión, sensibilidad, diferencia pareada de IoU y referencias triviales de toda agua y ninguna agua.

Los casos con unión cero se registran como IoU no definido. El JSON v2 conserva TP, FP y FN y reporta cero casos indefinidos en esta corrida. Para estudiar incertidumbre se prevé bootstrap pareado por evento, tras verificar la agrupación mediante metadatos. Con nueve chips y un chip por región, cualquier intervalo sería exploratorio: remuestrear no sustituye la variabilidad que la selección inicial no observa.

# 4 Resultados preliminares y análisis de fallas

## Comparación agregada

La ejecución v2 utiliza SAM preentrenado en CPU, VH replicada en tres canales y prompts automáticos. La corrección de la dependencia de etiquetas no modificó el patrón principal del experimento. Los agregados se comprobaron a partir de los conteos por chip del JSON.

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

SAM obtiene menor IoU que Otsu con apertura en ocho de nueve chips. En seis predice agua en más de 96% del soporte etiquetado válido. Somalia es el único caso con mayor IoU de SAM, pese a la sobresegmentación extensa. Por tanto, H1 no recibe apoyo en esta muestra y configuración. Los resultados no permiten identificar qué componente causa el fallo ni anticipar el efecto de H2 o H3.

![Comparación del chip Somalia con la versión corregida](comparacion_real_v2.png)

Figura 1. Banda VH, Otsu con apertura, SAM v2 y referencia LabelHand de Somalia_12849. Los IoU excluyen los píxeles sin etiqueta, aunque la figura no los distingue visualmente del resto.

# 5 Discusión y amenazas a la validez

La apertura eleva el IoU global de Otsu de 0.1593 a 0.1875. Reduce los falsos positivos de 589 699 a 442 577, pero también los verdaderos positivos de 120 191 a 113 915. Este intercambio exige considerar precisión y sensibilidad; el aumento de IoU no demuestra una mejora universal en todas las escenas.

La principal limitación es la muestra exploratoria. Un chip por región no representa el desempeño regional, y el desbalance entre agua y no agua condiciona la interpretación de los agregados. Además, la comparación en validación orienta decisiones posteriores, por lo que la prueba debe permanecer reservada para la evaluación final.

La representación de VH en tres canales, los errores del umbral, la ubicación de prompts y la selección de máscaras son posibles explicaciones de la sobresegmentación. La corrección v2 resuelve la dependencia identificada de las etiquetas, pero no sustituye una auditoría de normalización, transformaciones y valores de nodata del sensor. Las ablaciones permitirán separar algunos de estos efectos. Los FLOPs analíticos y el perfilado sintético también deben interpretarse dentro de su alcance.

Finalmente, segmentar agua no equivale a identificar exclusivamente inundación. La evaluación en Piura requerirá referencia independiente y criterios para distinguir agua permanente. La contribución actual es documentar un resultado negativo reproducible y formular intervenciones comprobables, sin atribuir al método capacidades aún no demostradas.

# 6 Cronograma

| Fecha de 2026 | Entregable | Trabajo pendiente |
|---|---|---|
| 12 de octubre | E1 Propuesta y avance experimental | Completar integrantes y validaciones del cuaderno; revisar el formato y la presentación de la reproducibilidad. |
| 16 de octubre | E2 Inscripción del póster WAIMLAP | Completar autores y afiliaciones, confirmar requisitos y efectuar la inscripción. |
| 16 de noviembre | E3 Entrega completa | Ejecutar H2 y H3, ampliar la evaluación y estudiar Piura si se dispone de datos adecuados. |
| 18 al 20 de noviembre | Defensa oral individual | Preparar la explicación personal del diseño, código, resultados y uso de IA. |
| 30 de noviembre | Artículo IEEE o LNCS de 4 a 8 páginas | Consolidar resultados, discusión, referencias y formato editorial. |

Después de E1 se priorizará H2 con un protocolo fijado antes del entrenamiento. La ampliación de la muestra precederá a conclusiones generales. La evaluación en Piura quedará condicionada a contar con datos y etiquetas adecuados; si no se concreta, se declarará como trabajo futuro.

# 7 Reproducibilidad

El proyecto dispone del repositorio público [INF659 Tarea Académica SAM](https://github.com/LuisMirandaMallqui/INF659_TareaAcademica_SAM). Según la verificación registrada por el equipo, la rama main contiene scripts, documentos, resultados, README, requisitos y el ZIP de datos. El manifiesto conserva URLs, hashes SHA-256 y la regla de selección. Los veinte GeoTIFF del ZIP coinciden con sus hashes. La segmentación y la disección se registran en resultados_reales_v2.json y fase_a_diseccion_real.json, respectivamente.

El entorno reporta Python 3.13.16, PyTorch 2.14.1+cpu, transformers 5.19.0, scikit-image 0.26.0, SciPy 1.18.1 y NumPy 2.5.3. Los scripts utilizan facebook/sam-vit-base. Para ejecutar el pipeline se prepara la carpeta chips desde el ZIP; las salidas se escriben en results_v2. La disección emplea resource de Linux y requiere adaptación para Windows. Deben conservarse todas las dependencias y una revisión inmutable de los pesos.

El muestreo de prompts es determinístico. La imagen sintética del perfilado y el futuro entrenamiento, en cambio, requieren semillas y control de aleatoriedad. El requisito de notebooks reproducibles comunicado por el equipo debe atenderse junto con README y entorno documentado. La conversión de los scripts a notebooks permanece pendiente de decisión y ejecución.

# 8 Cuaderno de decisiones y declaración de IA

La asistencia de IA se utilizó en ejecución experimental, revisión de código y redacción. Cada integrante deberá registrar su responsabilidad y la validación personal realizada antes de incorporar los resultados. Los campos pendientes no acreditan todavía esa revisión.

| Tarea | Herramienta de IA usada | Qué se le pidió exactamente | Qué devolvió | Qué validó o entendió el estudiante antes de usarlo | Integrante responsable |
|---|---|---|---|---|---|
| Pipeline preliminar | Claude | Ejecutar Otsu, prompts, SAM y evaluación con datos reales. | Script, JSON y figura; corrección v2. | [Validación personal pendiente] | [Nombre pendiente] |
| Revisión metodológica | ChatGPT | Revisar código, resultados e informe. | Detección de dependencia de etiquetas y observaciones sobre métricas y perfilado. | [Validación personal pendiente] | [Nombre pendiente] |
| Disección de SAM | Claude | Obtener parámetros, estimar FLOPs y medir tiempo y memoria. | Script y registros; corrección de atención global y por ventanas tras revisión. | [Validación personal pendiente] | [Nombre pendiente] |
| Referencias y particiones | Claude | Verificar citas y conteos de los CSV. | Referencias corregidas y confirmación de las particiones, según el registro recibido. | [Validación personal pendiente] | [Nombre pendiente] |
| Redacción y síntesis | ChatGPT | Integrar E1, preparar E2 y revisar claridad académica. | Documentos integrados y resumen de decisiones. | [Validación personal pendiente] | [Nombre pendiente] |
| [Tarea] | [Herramienta] | [Solicitud] | [Respuesta] | [Validación personal] | [Nombre] |
| [Tarea] | [Herramienta] | [Solicitud] | [Respuesta] | [Validación personal] | [Nombre] |
| [Tarea] | [Herramienta] | [Solicitud] | [Respuesta] | [Validación personal] | [Nombre] |
| [Tarea] | [Herramienta] | [Solicitud] | [Respuesta] | [Validación personal] | [Nombre] |

# Anexo A Intervenciones adicionales

Ideas a evaluar, no resultados.

Umbralización local. Hipótesis: reduce falsos positivos frente a Otsu global en escenas heterogéneas, con banda, prompts y modelo constantes.

Distribución de prompts. Hipótesis: una mayor cobertura espacial mejora el IoU frente al muestreo concentrado en componentes dominantes, con igual número de puntos.

Cantidad de prompts. Hipótesis: aumentar el presupuesto mejora la sensibilidad hasta un límite, sin compensar necesariamente los errores del umbral.

Polarización. Se compararán VV y VH con una dirección de mejora fijada antes de ejecutar. La banda que genera prompts y la que entra al modelo se modificarán por separado para evitar confundir efectos.

Composición de bandas. Hipótesis: una combinación fija de VV, VH y su diferencia en dB mejora el IoU frente a VH replicada, con los mismos prompts.

Selección de máscaras. Hipótesis: una regla de consistencia con prompts reduce sobresegmentación frente al máximo de calidad predicha, sin consultar etiquetas de evaluación.

Cuantización. Hipótesis: una configuración compatible reduce memoria y tiempo dentro de una pérdida tolerada de IoU, fijada como [tolerancia_IoU = pendiente].

La apertura ya se comparó en v2. Una repetición deberá registrar previamente su hipótesis. Los controles de entrada complementan, pero no sustituyen, la intervención en el núcleo prevista mediante H2.


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
