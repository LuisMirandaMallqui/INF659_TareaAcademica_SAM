Estado consolidado del trabajo: INF659 Tarea Academica Final

SAM aplicado a mapeo de inundaciones con Sentinel-1 SAR, benchmark Sen1Floods11, con proyeccion al Niño Costero de 2017 en Piura. Ultima actualizacion: domingo 11 de octubre de 2026, tarde.

Entregables

| Entregable | Fecha | Puntos | Estado | Que falta |
|---|---|---|---|---|
| E1: propuesta, diseccion, intervencion preliminar, cuaderno de decisiones | lunes 12 de octubre | 4 | Material completo, falta ensamblar | Juntar el borrador de ChatGPT, la diseccion real de Fase A, la intervencion preliminar real v2 y el manifiesto en un solo documento; revisar las cinco citas marcadas VERIFICAR; completar el cuaderno de decisiones con los nombres reales de cada integrante |
| E2: poster e inscripcion WAIMLAP | 16 de octubre | 2 | Borrador corto listo | Inscripcion formal, confirmar requisitos del formulario |
| E3: entrega completa | 16 de noviembre | 2 | No iniciado | Ejecutar H2, fine-tuning del decoder; evaluacion en Piura real; ampliar a mas de los 89 chips de validacion; ablaciones adicionales de la Parte 6 del borrador |
| Defensa oral individual | 18 al 20 de noviembre | 6 | No iniciado | Cada integrante debe poder explicar personalmente el pipeline, los resultados y la correccion v2 |
| Articulo final IEEE o LNCS, 4 a 8 paginas | 30 de noviembre | 6 | No iniciado | Depende de E3 |

Las tres fases

| Fase | Que pide | Estado | Evidencia |
|---|---|---|---|
| Fase A, diseccion instrumentada | Parametros, FLOPs, campo receptivo, latencia, memoria | Completa, real, corregida | 93.7 millones de parametros totales, codificador de imagen 95.66 por ciento del total, decodificador 4.33 por ciento, razon 22 a 1; aproximadamente 930 GFLOPs por pasada, corregido tras verificar que SAM usa atencion ventaneada en 8 de 12 capas y global solo en las capas 2, 5, 8 y 11; campo receptivo local de 16 por 16 en la capa de parches, limitado a la ventana de 224 por 224 pixeles en las capas ventaneadas, y global por primera vez en la capa 2; latencia real 12.4 segundos y memoria real 2.5 gigabytes, ambas en CPU de este contenedor, no en la RTX 3060 |
| Fase B, intervencion con hipotesis previa | Control de entrada ya hecho; intervencion de nucleo, H2, pendiente | Parcial | Umbral de Otsu automatico y generacion automatica de prompts, reales, corregidos en v2 para no depender de la etiqueta; el ajuste del decoder en si, H2, todavia no se corrio |
| Fase C, estres y honestidad | Evaluacion fuera de dominio y diagnostico honesto de fallas | En curso | SAM pierde contra el umbral en ocho de nueve chips reales y colapsa a entre 97 y 99.7 por ciento de la imagen marcada como agua en seis de nueve; falta la evaluacion real sobre la escena de Piura |

Las tres hipotesis

| Hipotesis | Estado | Evidencia real |
|---|---|---|
| H1, SAM con prompts automaticos mejora el umbral | Evidencia real en contra, confirmada tras la correccion v2 | IoU global de SAM 0.0897 frente a IoU global del umbral 0.1875, sobre nueve chips reales de Sen1Floods11 |
| H2, afinar solo el decoder mejora el resultado | No probado | El reparto de parametros de Fase A, decoder veintidos veces mas chico que el codificador, respalda que es una intervencion barata de intentar; el entrenamiento en si no se ha corrido |
| H3, SAM sirve para seleccionar candidatos | No probado | Sin evidencia todavia |

Quien hizo que hasta ahora, resumen de alto nivel

| Quien | Que hizo |
|---|---|
| Claude, esta sesion | Pipeline real de principio a fin sobre datos reales de Sen1Floods11: umbral de Otsu real, prompts automaticos reales, SAM oficial real, IoU real; correccion v2 tras la revision recibida; diseccion real de Fase A; manifiesto de reproducibilidad con URLs y sha256 de los 20 archivos usados |
| ChatGPT | Redaccion academica de las ocho partes del borrador de E1, ya actualizadas con la correccion v2; revision critica del pipeline y de la diseccion de Fase A de Claude, encontro dos fallas reales: la fuga de etiqueta en los prompts negativos, corregida en v2, y el supuesto de atencion global en las doce capas del codificador, corregido junto con el campo receptivo tras verificar contra la configuracion real de SAM |
| El equipo | Propuesta original de usar SAM sobre Sentinel-1 para Piura; decision de direccion del proyecto; ensamblar el documento final; completar el cuaderno de decisiones con nombres y validaciones reales; la defensa oral |

Archivos que existen ahora mismo

De Claude, generados hoy: Intervencion_preliminar_real.md y su correccion Correccion_v2_fuga_de_etiqueta.md, los scripts run_sam_pipeline.py y run_sam_pipeline_v2.py, los resultados resultados_reales.json y resultados_reales_v2.json, las figuras comparacion_real.png y comparacion_real_v2.png, el manifiesto MANIFIESTO_reproducibilidad.md, los veinte archivos reales de chips comprimidos en chips_sen1floods11_reales.zip, la diseccion Fase_A_diseccion_real.md con su json, y este mismo documento.

De ChatGPT, en la maquina del usuario: E1_Borrador_ChatGPT.md con las ocho partes, y Revision_del_material_Claude.md con la revision que motivo la correccion v2.

Proximos pasos, en orden de prioridad

| Prioridad | Accion | Responsable |
|---|---|---|
| 1 | Ensamblar el documento final de E1 con todo lo anterior | El equipo, hoy |
| 2 | Verificar las cinco citas candidatas marcadas VERIFICAR en la Parte 2 del borrador | Claude puede hacerlo ahora con busqueda real si se lo piden |
| 3 | Completar el cuaderno de decisiones, Parte 5, con nombres reales de cada integrante | El equipo, no se puede delegar |
| 4 | Entregar E1 | El equipo, antes del lunes 12 de octubre |
| 5 | Evaluar H2, afinar el decoder, usando la particion de entrenamiento | Pendiente para E3, buen candidato para usar computo en la nube con GPU si esta disponible |
| 6 | Ampliar la evaluacion mas alla de nueve chips, y evaluar en la escena real de Piura si se resuelve el acceso y la licencia de los datos | Pendiente para E3 |
