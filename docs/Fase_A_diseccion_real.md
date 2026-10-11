Fase A real: diseccion instrumentada de facebook/sam-vit-base

Esto cierra el ultimo placeholder pendiente de Fase A en el borrador de ChatGPT. Los parametros, el reparto por submodulo y el campo receptivo son valores exactos o derivados directamente de la configuracion real del modelo cargado. Los FLOPs se calculan con una formula analitica sobre esa misma configuracion real, no con un perfilador automatico: intentar contar FLOPs con thop sobre el encoder completo no termino en tiempo razonable en la CPU de este contenedor. La latencia y la memoria si son mediciones reales, pero tomadas en la CPU de este contenedor en la nube, no en la laptop RTX 3060 del compañero, y con entrada sintetica (ruido aleatorio, no un chip real), solo para medir tiempo y memoria, no precision.

Correccion sobre la primera version de este documento: la revision de ChatGPT detecto que los FLOPs se habian calculado asumiendo atencion global en las doce capas del codificador. Eso es incorrecto para SAM. Se verifico contra la configuracion real del modelo cargado y se confirmo: SAM usa atencion ventaneada tipo ViTDet en ocho de las doce capas, con ventanas de 14 por 14 parches, y solo en las capas de indice 2, 5, 8 y 11 usa atencion global sobre los 4096 parches completos. Esto tambien obligaba a corregir la afirmacion sobre el campo receptivo, que decia "global desde la primera capa" y tampoco es correcto bajo este diseño. Ambas correcciones ya estan aplicadas abajo y en fase_a_diseccion.py.

Parametros reales

El modelo tiene 93735728 parametros en total. El codificador de imagen concentra 89670912, el 95.66 por ciento. El decodificador de mascaras tiene 4058340, el 4.33 por ciento. El codificador de prompts es casi insignificante, 6476 parametros. La razon entre el codificador y el decodificador es 22.1 a 1: el decodificador es veintidos veces mas pequeño. Este numero es evidencia directa a favor de H2: afinar solo el decodificador, dejando el codificador de imagen congelado, es una intervencion barata en terminos de parametros entrenables, no solo una suposicion razonable.

FLOPs estimados del codificador de imagen

Con la configuracion real cargada, 4096 parches de 16 por 16 pixeles sobre una imagen interna de 1024 por 1024 y dimension oculta 768, el codificador de imagen requiere aproximadamente 930 GFLOPs por pasada hacia adelante. Esta cifra ya distingue las dos arquitecturas de atencion reales de SAM: cuatro capas, en los indices 2, 5, 8 y 11, usan atencion global sobre los 4096 parches completos, a 109.5 GFLOPs cada una; las otras ocho capas usan atencion ventaneada, donde cada parche solo atiende a los demas parches dentro de su ventana de 14 por 14, a 60.9 GFLOPs cada una. La version anterior de este calculo asumio atencion global en las doce capas y llegaba a 1319 GFLOPs, un 29.5 por ciento mas alto que la cifra corregida: ese excedente es exactamente el costo de tratar las ocho capas ventaneadas como si fueran globales, una diferencia de 17.5 veces en el termino de atencion de cada una de esas capas. Esta cifra tampoco incluye el decodificador de mascaras, cuyo costo es marginal frente al del codificador dado el reparto de parametros ya mostrado, ni desglosa el pequeño relleno que SAM añade para que la grilla de 64 por 64 parches sea divisible entre ventanas de 14 por 14.

Campo receptivo

La capa de parches tiene un campo receptivo local de 16 por 16 pixeles, equivalente a una convolucion local. Las dos primeras capas de atencion, indices 0 y 1, son ventaneadas: el campo receptivo se queda en la ventana de 14 por 14 parches, equivalente a 224 por 224 pixeles, y no crece mas alla de eso porque SAM usa ventanas fijas sin desplazamiento entre capas, a diferencia de Swin Transformer. El campo receptivo se vuelve global por primera vez en la capa de indice 2, la primera de las cuatro capas marcadas para atencion global: ahi, y solo ahi en adelante cada vez que se repite una capa global, cada parche atiende a los 4096 parches de la imagen completa. El crecimiento del campo receptivo en SAM no es gradual capa por capa como en una red convolucional ni global desde el inicio como en un Vision Transformer con atencion completa en todas las capas: crece en escalones, ventana, ventana, imagen completa, siguiendo el patron fijo de capas globales.

Latencia real

Sobre cinco repeticiones reales, con un calentamiento previo, la latencia promedio de una pasada hacia adelante completa es 12424 milisegundos, con una desviacion de 439 milisegundos, un minimo de 11824 y un maximo de 13101. Esto se midio en la CPU de dos vCPU de este contenedor en la nube, sin GPU, con entrada sintetica (ruido aleatorio de 1024 por 1024 pixeles, no un chip real). Como referencia de contexto, el perfilado sintetico que ya tenia el equipo sobre la laptop RTX 3060, con SAM ViT-B en FP16, reportaba 135.5 milisegundos por imagen: la diferencia de casi cien veces es consistente con pasar de una GPU en FP16 a una CPU en FP32, y no implica ninguna contradiccion entre ambas mediciones.

Memoria real

El pico de memoria residente de todo el proceso de Python, modelo, pesos y activaciones incluidos, fue de 2501 megabytes, medido con resource.getrusage sobre la misma CPU de este contenedor.

Archivo con todos los valores: fase_a_diseccion_real.json. Script que los genero: fase_a_diseccion.py.
