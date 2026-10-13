# INF659 · SAM sobre Sentinel-1: segmentación de agua con prompts automáticos

Proyecto de la tarea académica final de INF659 (Técnicas Avanzadas de Data Mining y Sistemas Inteligentes, Maestría PUCP, 2026-2), a cargo de John Manuel Arzapalo Arana, Luis Alberto Miranda Mallqui y Juan Diego Zavala Viscardo, en el tema del banco **"El costo del prompt en SAM"**: caracterizar dónde y por qué falla el decodificador de máscaras sin prompts humanos y proponer una mitigación.

* **Modelo core:** Segment Anything (SAM) ViT-B, `facebook/sam-vit-base`, revisión fijada `70c1a07f894ebb5b307fd9eaaee97b9dfc16068f`.
* **Dataset de reto:** Sen1Floods11 v1.1, chips Sentinel-1 etiquetados a mano (VV y VH en dB, 512×512). Es un dominio SAR fuera del preentrenamiento de SAM.
* **Pregunta de investigación:** ¿puede SAM preentrenado mejorar una máscara de agua obtenida por umbral cuando sus prompts se generan automáticamente a partir de esa misma máscara, y qué componente (prompts, regla de selección, decodificador o representación del codificador) explica su desempeño fuera de dominio?

## Estructura

```
notebooks/
  00_datos_y_reproducibilidad.ipynb        descarga con SHA-256, particiones, exclusión, train-dev
  01_faseA_diseccion.ipynb                 parámetros y FLOPs medidos por bloque, PCA, campo receptivo efectivo, latencia/memoria
  02_H1_reproduccion_v2.ipynb              reproducción independiente del pipeline v2 (9 chips del E1)
  03_validacion_completa_diagnosticos.ipynb H1 en toda la validación, oráculos, calibración, ablaciones A1-A4, sonda por capa
  04_H2_ajuste_decodificador.ipynb         H2: ajuste exclusivo del decodificador (3 semillas) y control con sonda lineal
  05_exploratorio_sonda.ipynb              EXPLORATORIO (no pre-registrado): por qué falló la sonda lineal H2-c
  06_exploratorio_curva_validacion.ipynb   EXPLORATORIO (no pre-registrado): IoU por época en train-dev (¿generaliza o sobreajusta?)
src/inf659sam/                             código común (datos, línea base, SAM, Fase A, experimentos, estadística)
results/                                   JSON de resultados, figuras (ES y EN) y referencia del equipo (v2)
docs/                                      estado del trabajo, protocolo pre-registrado, cuaderno de decisiones, ética y licencias
docs/historial/                            documentos de etapas anteriores (borradores E1/E2, correcciones v2), ya reemplazados
tools/ejecutar_notebooks.py                ejecuta los notebooks en orden y guarda las salidas
tools/figuras_ingles.py                    figuras en inglés para el póster y el resumen de WAIMLAp
scripts/                                   pipeline v2 y disección originales (referencia; los reemplazan src/ y notebooks/)
data/chips_sen1floods11_reales.zip         los 20 archivos del E1 (el resto de los datos se descarga en el notebook 00)
```

**Estado actual y próximos pasos:** `docs/estado_consolidado_del_trabajo.md`.

## Cómo reproducir

```bash
python -m venv .venv
.venv\Scripts\activate            # en Linux/Mac: source .venv/bin/activate
pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python tools/ejecutar_notebooks.py               # los 6 notebooks en orden (unas 2 h en CPU; ~30 min con GPU)
python tools/figuras_ingles.py Mekong_977338 India_1050276   # figuras en inglés para el E2
```

* Los datos (unos 600 MB), los pesos (375 MB) y las cachés (unos 1.7 GB) se guardan por defecto en `data/`, `models/` y `cache/`, que están ignorados por git. Para moverlos fuera de una carpeta sincronizada (OneDrive, Drive) se usan las variables `INF659_DATA`, `INF659_MODEL` e `INF659_CACHE`.
* `INF659_THREADS` fija los hilos de CPU (4 por defecto). Con `python tools/ejecutar_notebooks.py --baja-prioridad` el proceso no compite con otras aplicaciones.
* Las semillas están fijadas (0, 1, 2) y las prompts de evaluación son deterministas. Se verificó que los IoU del pipeline v2 se reproducen en Windows con diferencias menores a 1e-4 frente a la corrida original del equipo en Linux.
* **No se descargan ni se usan** las particiones `test` y `bolivia`: quedan reservadas para la evaluación final (E3).

## Entorno con el que se generaron los resultados

Windows 11, Python 3.14.7, PyTorch 2.14.1 (CUDA 13.0), transformers 5.19.0, NumPy 2.5.3, SciPy 1.18.1, scikit-image 0.26.0 y scikit-learn 1.9.1. Hardware: AMD Ryzen 7 5700X, 32 GB de RAM y NVIDIA GeForce RTX 3070 de 8 GB. Los embeddings se calcularon en CPU; el ajuste del decodificador, las evaluaciones y el perfilado final se hicieron en GPU. El código detecta la GPU automáticamente; con `INF659_DEVICE=cpu` se fuerza la CPU. Para GPU: `pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu130`.

## Licencias y citas

Pesos de SAM bajo Apache-2.0 (Meta AI). Sen1Floods11 no declara licencia en su repositorio, así que los notebooks descargan los chips desde el bucket oficial y no los suben a git. La única excepción es `data/chips_sen1floods11_reales.zip` (20 archivos del E1), pendiente de confirmar con los autores. Ver `docs/etica_limitaciones_licencias.md`.

## Uso de asistentes de IA

Declarado en `docs/cuaderno_de_decisiones.md`, como exige la política del curso. El protocolo de los experimentos nuevos se registró **antes** de ejecutarlos (`docs/protocolo_preregistrado.md`).
