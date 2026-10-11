Correccion v2 sobre la intervencion preliminar real: fuga de etiqueta en los prompts negativos

Este documento responde a la revision que ChatGPT hizo del material entregado antes (Intervencion_preliminar_real.md, resultados_reales.json, run_sam_pipeline.py) y dejó registrada en Revision_del_material_Claude.md. El punto central de esa revision era correcto y se corrigió. Los demás puntos se atendieron en la medida de lo posible antes de la entrega de E1.

Que estaba mal

El script original definia valid como label distinto de menos uno, es decir, la validez de la etiqueta de verdad de terreno, y usaba esa misma variable para dos cosas distintas: evaluar el resultado contra la verdad de terreno, lo cual es correcto, y restringir donde se muestreaban los puntos negativos automaticos para SAM, lo cual no es correcto. Esto no revela directamente si un pixel es agua o no agua, pero si permite que la generacion de prompts, presentada como automatica y sin intervencion humana, sepa en que zonas el anotador dejo la mascara sin dato, normalmente por nubes en la imagen optica usada para etiquetar. Eso compromete la afirmacion de que el pipeline es independiente de la etiqueta antes de la evaluacion.

Que se corrigio

El script ahora separa explicitamente dos variables. sensor_valid proviene unicamente del propio radar, de que las bandas VV y VH tengan un valor finito en ese pixel, y es lo unico que puede influir en el umbral de Otsu y en la generacion de prompts. label_valid proviene unicamente de LabelHand y se usa exclusivamente para evaluar, nunca para generar prompts. El script corregido se llama run_sam_pipeline_v2.py y quedo junto a este documento.

Resultado tras la correccion

Se volvio a correr el pipeline completo sobre los mismos nueve chips reales. Los resultados no cambiaron de forma relevante.

IoU global con umbral de Otsu con apertura: 0.1875, identico al valor anterior porque el umbral no dependia de la etiqueta y no se modifico.

IoU global de SAM con prompts automaticos: 0.0897, frente a 0.0896 antes de la correccion.

IoU promedio por chip de SAM: 0.0819, frente a 0.0820 antes de la correccion.

La diferencia esta en la tercera o cuarta cifra decimal. El hallazgo central se mantiene: SAM pierde contra el umbral en ocho de los nueve chips, y en seis de los nueve predice entre 97 y 99.7 por ciento de la imagen como agua. La correccion era necesaria por rigor metodologico, pero no cambia la conclusion reportada antes.

Otros puntos de la revision, atendidos en esta version

Se reporta ahora tambien el umbral de Otsu sin apertura morfologica, no solo con apertura. Su IoU global es 0.1593 y su IoU promedio por chip es 0.1626, ambos por debajo de la version con apertura, lo que indica que la limpieza morfologica si aporta sobre el umbral crudo.

El script verifica por codigo, no solo por inspeccion, que cada punto positivo cae dentro de la mascara de agua del umbral y cada punto negativo fuera de ella, respetando el orden de coordenadas x, y que espera el procesador de SAM. La verificacion dio correcta en los nueve chips y en todos los puntos.

El chip sin prompts positivos ya no queda fuera del agregado de SAM mientras sigue dentro del agregado del umbral: ahora siempre se evalua una mascara, vacia si no hubo prompts, contra la verdad de terreno, de modo que ambos metodos se comparan sobre el mismo conjunto de chips. En la muestra actual ningun chip llego a tener cero prompts positivos, asi que este caso no se activo, pero el script ya no tiene esa asimetria.

Se cuentan los casos en que el IoU queda indefinido por union igual a cero. En esta corrida fueron cero casos para los tres metodos.

El criterio de exclusion de chips, nodata de etiqueta superior a 95 por ciento, se declara ahora antes de aplicarse y se registra para los diez chips originales, no solo para el excluido. Ningun chip ademas de Ghana_5079 se acerca a ese umbral: el siguiente mas alto es Somalia con 27 por ciento de nodata de etiqueta.

El resultado JSON ahora guarda verdaderos positivos, falsos positivos y falsos negativos por chip y por metodo, no solo el IoU final, ademas de las tres puntuaciones de calidad que SAM predice internamente para sus mascaras candidatas, la configuracion exacta usada y las versiones de las librerias.

Se adjunta un manifiesto con la URL exacta de descarga de cada uno de los veinte archivos usados y su hash sha256, y los propios archivos de chips comprimidos, para que la corrida pueda reproducirse de principio a fin y no solo auditarse por lectura de codigo.

Lo que sigue pendiente

La revision señalaba, con razon, que afirmar que no hay ninguna falla de implementacion es una conclusion mas fuerte de lo que una revision estatica o incluso esta correccion pueden sostener. Se retira esa afirmacion. Lo que si puede sostenerse ahora es mas acotado: las coordenadas de los prompts se verificaron por codigo y son correctas, la fuga de etiqueta identificada se cerro y no cambio el resultado, y el pipeline corre de principio a fin sobre datos y pesos reales. No se audito si una normalizacion distinta a la de percentiles 2 y 98, u otra banda, cambiarian el comportamiento. Esas preguntas quedan abiertas para la Fase B.

La revision tambien señalaba que ajustar el umbral y los prompts son controles de entrada, utiles como intervencion preliminar, pero que la Fase B pide ademas una intervencion en el nucleo del modelo. Eso sigue sin hacerse: H2, afinar solo el decoder de SAM dejando el codificador de imagen congelado, continua pendiente y es, junto con la evaluacion en Piura, el trabajo real que falta despues de E1.
