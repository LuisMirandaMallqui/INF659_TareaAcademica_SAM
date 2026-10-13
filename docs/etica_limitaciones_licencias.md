# Limitaciones, sesgos, ética y licencias

Sección obligatoria según el punto 10 del enunciado. Se incorpora al E1 y al artículo final.

## Licencias de modelos y datos

| Recurso | Licencia o términos | Cómo se respeta |
|---|---|---|
| SAM, `facebook/sam-vit-base` (Meta AI), revisión `70c1a07f…` | Apache-2.0, declarada en la ficha del modelo en Hugging Face | Se cita a Kirillov et al. (2023) y se usan los pesos sin redistribuirlos: los notebooks los descargan con hash verificado |
| Sen1Floods11 v1.1 (Cloud to Street, Bonafilia et al., 2020) | El repositorio público **no declara una licencia** (consultado el 12-10-2026: el README no la menciona y no hay archivo LICENSE). Los datos se publican en un bucket abierto con fines de investigación | Se cita el artículo. Los notebooks descargan los chips de la fuente oficial y verifican sus SHA-256; los 341 chips usados no se suben al repositorio. El único archivo de datos versionado es `data/chips_sen1floods11_reales.zip` (los 20 archivos del E1). **Pendiente del equipo:** confirmar el permiso con los autores (support@cloudtostreet.info) o retirarlo antes de la entrega final |
| Imágenes Sentinel-1 (Copernicus, ESA) | Acceso libre y abierto según la política de datos de Copernicus, con atribución | Se atribuye en figuras y referencias: "contiene datos Copernicus Sentinel modificados" |
| Plantillas de WAIMLAp 2026 | Proporcionadas por la organización para las contribuciones | Se conservan el encabezado y el pie, como pide la plantilla |

## Limitaciones del estudio

1. **Particiones y tamaño de muestra.** Las conclusiones del E1 se apoyan en la partición `valid` (86 chips evaluables, 10 regiones). Se usa para comparar y no para elegir hiperparámetros, que se eligen en `train-dev`; aun así, el número final debe confirmarse en `test` (90 chips) y en `bolivia` (15 chips, región nunca vista), que siguen sin abrir.
2. **Dependencia entre chips.** Los chips de una misma región comparten evento, fecha y paisaje. Por eso se reporta también el bootstrap por región, más conservador. Con 10 regiones, ese intervalo es exploratorio.
3. **Un solo modelo y una sola resolución.** Solo se evaluó SAM ViT-B a 1024 px, con un chip de 512 reescalado al doble. ViT-L/H, SAM 2 y otras resoluciones quedan fuera.
4. **Representación de entrada.** Las bandas SAR se convierten a 3 canales de 8 bits por percentiles por chip. Esa normalización depende de la escena y descarta la escala absoluta en dB.
5. **Medición de latencia.** La latencia se midió en una sola máquina de escritorio: CPU Ryzen 7 5700X con 8 hilos y GPU RTX 3070. No representa un despliegue en producción.
6. **Agua no es inundación.** El modelo segmenta agua en una sola fecha. Para hablar de inundación hace falta separar el agua permanente con una imagen previa o un mapa de agua permanente. Esto es crítico para Piura.
7. **Piura no se evaluó.** No se afirma ningún desempeño en Perú. Shen, Zhang y Wu (2022) y MFF-Net (sección 3.6) usan escenas Sentinel-1 de la cuenca del río Piura en 2017; son la ruta más directa para conseguir datos de evaluación local.

## Sesgos y representatividad

* Sen1Floods11 cubre **11 eventos en 11 países** y ninguno está en Sudamérica andina o costera. Las diferencias de paisaje (desierto costero, ríos estacionales y zonas urbanas de Piura) pueden cambiar mucho la distribución de retrodispersión.
* Las etiquetas manuales se hicieron con imágenes ópticas (Sentinel-2) y tienen huecos por nubes (−1). Las zonas más nubosas, típicas de un evento de lluvia intensa, quedan subrepresentadas en la evaluación.
* El desbalance agua/tierra es fuerte: hay chips con menos de 1 % de agua. El IoU global pesa más los chips con mucha agua y el IoU medio pesa igual cada chip; por eso se reportan ambos.

## Consideraciones éticas

* **Riesgo de uso en emergencias.** Un falso negativo (agua no detectada) puede dejar sin atención a población afectada, y un falso positivo puede desviar recursos. Este trabajo no es un sistema operativo. Cualquier uso real exige validación local, supervisión humana y comunicación explícita de la incertidumbre.
* **Privacidad.** Las imágenes SAR de 10 m no identifican personas y el dataset no contiene datos personales. El análisis de impacto sobre población (por ejemplo, cruzar con datos censales de Piura) sí requeriría cuidados de privacidad y agregación.
* **Huella de cómputo.** El diseño elegido (codificador congelado y embeddings cacheados) hace que cada experimento cueste minutos de CPU y no horas de GPU.
* **Uso de asistentes de IA.** Está declarado en `docs/cuaderno_de_decisiones.md`. La ejecución se delegó; la comprensión y la validación corresponden a cada integrante.
