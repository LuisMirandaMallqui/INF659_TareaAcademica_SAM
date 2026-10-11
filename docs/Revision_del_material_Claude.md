# Revisión actualizada del material de Claude

Se revisaron los 12 archivos vigentes en la carpeta indicada el 11 de octubre de 2026: documentos, scripts, JSON, figura v2 y ZIP. No se ejecutó nuevamente SAM ni se entrenó el modelo. Esta revisión sustituye la evaluación de v1.

## Corrección v2 y métricas

El código separa sensor_valid, obtenido de VV y VH finitos, de label_valid, obtenido de LabelHand. Los prompts ya no dependen de la validez de las etiquetas. Estas se utilizan para evaluar y para el criterio declarado de exclusión por más de 95% sin etiqueta. SAM se evalúa incluso cuando no tiene prompts positivos y produce una máscara vacía. Se reportan cero casos de unión vacía.

| Método | IoU global | IoU promedio por chip |
|---|---:|---:|
| Otsu sin apertura | 0.1592855732390801 | 0.16255904831766677 |
| Otsu con apertura | 0.1875326163364014 | 0.19235136452321197 |
| SAM v2 | 0.08974724001404935 | 0.08194007931763209 |

Se comprobaron los agregados y promedios con los registros por chip. SAM pierde frente a Otsu con apertura en ocho de nueve chips; en seis supera 96% de agua predicha sobre píxeles válidos de etiqueta. La figura v2 de Somalia concuerda con el JSON. No hay evaluación en Piura ni evidencia causal sobre el origen de la sobresegmentación.

La apertura reduce falsos positivos de 589 699 a 442 577, pero también verdaderos positivos de 120 191 a 113 915. Su mejora de IoU en esta muestra debe interpretarse junto con precisión y sensibilidad.

## Fase A: alcance y error pendiente

El JSON reporta 93 735 728 parámetros totales y 4 058 340 en el decoder. Los tres submódulos listados suman diez parámetros menos que el total: su ubicación requiere documentarse. La razón encoder/decoder de aproximadamente 22.1 respalda menos parámetros entrenables, no demuestra menor tiempo ni eficacia de H2.

La latencia de 11 944.4182 ms procede de una imagen RGB aleatoria sintética de 1024 por 1024 con un prompt positivo, cinco repeticiones y un calentamiento. Es una medición real de cómputo sintético en CPU; no es latencia SAR real. El intervalo incluye solo el forward, sin preparación de entradas. El pico RSS de todo el proceso es aproximadamente 2 532 MiB, no memoria exclusiva del modelo ni GPU. El script usa resource de Linux y necesita adaptación en Windows.

La estimación de 1 319.09 GFLOPs supone atención global en doce capas, lo cual no coincide con la configuración oficial: global_attn_indexes = [2, 5, 8, 11] y window_size = 14. La primera capa global es la tercera, contando desde uno; las primeras capas atienden por ventanas. Se retiró esa cifra como costo validado del borrador. Debe corregirse la aproximación distinguiendo bloques globales y por ventanas, padding y operaciones incluidas. No se rehízo la disección ni se inventó una cifra sustituta. [Configuración oficial de SAM ViT-B](https://huggingface.co/facebook/sam-vit-base/raw/main/config.json).

## Reproducibilidad y documentación

Se inspeccionó el ZIP y se comprobaron los SHA-256 de sus veinte GeoTIFF contra el manifiesto: todos coinciden. Esto verifica integridad respecto al manifiesto recibido, no una descarga independiente desde el origen. Hay URLs y regla de selección. El JSON no guarda coordenadas exactas de prompts ni máscaras producidas, pese al comentario del script que indica guardar prompts. Tampoco se fija una revisión inmutable de pesos ni se serializa toda la configuración de normalización y apertura.

Estado_consolidado_del_trabajo.md aún enumera archivos v1 eliminados y presenta Fase A como completa con la aproximación errónea de FLOPs y campo receptivo. El borrador actualizado evita trasladar esas afirmaciones. No existe entre los archivos revisados un registro que cierre las cinco citas candidatas; se mantienen [VERIFICAR], como también indica la tabla de estado.

## Redacción actualizada

E1_Borrador_ChatGPT.md conserva las ocho partes e incorpora v2, Otsu sin apertura, el alcance correcto del perfilado y los pendientes analíticos de Fase A. Se actualizaron resumen, metodología, cronograma, cuaderno, límites y póster. Los nombres y validaciones personales siguen pendientes. H2, H3 y Piura no tienen resultados. La propuesta de créditos y GitHub no forma parte de esta tarea de redacción.
