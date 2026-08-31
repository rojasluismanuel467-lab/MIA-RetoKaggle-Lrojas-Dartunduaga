# Guía del proyecto — Reto Kaggle CNN

Este archivo resume las reglas de trabajo y el contexto que deben conservarse al
editar este repositorio.

## Contexto de la competencia

- Competencia: [AAIV - 2026-II: Taller CNN - MIAA-MCD](https://www.kaggle.com/competitions/aaiv-2026-ii-taller-cnn-miaa-mcd).
- Objetivo: clasificar el estado del conductor como `awake` o `drowsy` y
  localizarlo mediante una bounding box.
- Métricas: accuracy para clasificación y coeficiente Dice para la bounding box.
- Datos publicados por Kaggle: `train.csv`, `test.csv`,
  `sample_submission.csv` e `images.zip` (aproximadamente 103,62 MB y 529
  archivos). Para descargarlos hay que iniciar sesión, unirse a la competencia
  y aceptar sus reglas.
- Formato obligatorio de submission:
  `filename,class,xmin,ymin,xmax,ymax`.
- El trabajo debe ser de exactamente dos integrantes, el notebook debe ser
  autocontenido y las output cells deben quedar visibles al entregarlo.
- Se debe evitar la fuga de datos, especialmente usar información del conjunto
  de prueba para entrenar o seleccionar hiperparámetros.

## Estructura de datos

La ejecución local espera:

```text
data/
├── train.csv
├── test.csv
├── sample_submission.csv
└── images/
    └── *.jpg
```

`data/`, `checkpoints/`, `.venv/` y `references/` no se versionan. No guardar
tokens de Kaggle en el repositorio.

## Configuración de ejecución

La configuración se lee desde `.env`; usar `.env.example` como plantilla.

```text
RUN_HEAVY_EXPERIMENTS=false
RUN_OPTIMIZATION=false
DEVICE=auto
REUSE_CHECKPOINTS=true
```

Valores válidos para `DEVICE`: `auto`, `cuda`, `mps` y `cpu`. `auto` prioriza
CUDA, después MPS en Apple Silicon y finalmente CPU. En Google Colab se debe
seleccionar una GPU NVIDIA para obtener CUDA.

`REUSE_CHECKPOINTS=true` hace que un experimento con el mismo nombre cargue su
archivo de `checkpoints/` en vez de volver a entrenarse. Para reentrenarlo,
cambiar temporalmente a `false` o usar un nombre nuevo.

MLflow queda habilitado con `MLFLOW_ENABLED=true`. El tracking store local está
en `mlflow.db` (SQLite), ignorado por Git. Cada entrenamiento nuevo registra parámetros,
métricas finales, métricas por época y el checkpoint como artefacto. Abrir la
interfaz con `mlflow ui --backend-store-uri sqlite:///./mlflow.db --port 5000`.

## Qué significa cada bandera

- `RUN_HEAVY_EXPERIMENTS`: activa los barridos exigidos por la rúbrica:
  optimizadores y learning rates, pesos de la loss de bbox, dropout, niveles de
  augmentation y tres backbones preentrenados. Cada variante es un entrenamiento
  independiente para poder comparar métricas.
- `RUN_OPTIMIZATION`: activa técnicas adicionales y costosas: grid search,
  Optuna, ensembles, pseudo-labeling, snapshots, pipeline two-stage
  bbox→crop→classify y matriz de robustez.

Para verificar que el notebook funciona, ambas banderas deben estar en `false`.
Para reproducir la investigación completa, se activan ambas y se requiere más
tiempo, memoria, datos y pesos preentrenados.

## Flujo del modelo

1. Leer CSV e imágenes.
2. Dividir `train` en entrenamiento y validación de forma estratificada.
3. Entrenar una CNN multitarea con cabeza de clasificación y cabeza de bbox.
4. Medir accuracy y Dice.
5. Comparar CNN propia, ResNet18, MobileNetV3-small y EfficientNet-B0.
6. Aplicar TTA horizontal y ensembles.
7. Opcionalmente recortar el conductor y clasificarlo con un segundo modelo.
8. Escribir submissions en `outputs/` y checkpoints en `checkpoints/`.

## Archivos importantes

- `reto_cnn.ipynb`: notebook autocontenido y entregable principal.
- `reto_cnn.py`: fuente Jupytext del notebook; mantenerlo sincronizado con el
  `.ipynb` cuando se edite el código fuente.
- `src/drowsy_cnn/`: implementación modular reutilizable.
- `tools/setup_data.sh`: descarga y descompresión del dataset.
- `tools/example_usage.py`: ejemplo mínimo de entrenamiento.
- `tools/two_stage.py`: pipeline de dos etapas; requiere checkpoints de stage 1.
- `run.sh`: ejecución end-to-end del notebook.
- `docs/pipeline_design.md`: diseño, justificaciones y referencias técnicas.
- `docs/QUICKSTART_GPU.md`: instalación y ejecución con GPU.
- `docs/Rúbrica de evaluación taller CNN.xlsx`: rúbrica suministrada por el curso.

## Reglas de desarrollo

- Ejecutar `pytest tests/ -v` después de modificar el paquete.
- Mantener las columnas y nombres de clase exactos; no cambiar el orden
  `awake=0`, `drowsy=1` sin actualizar todo el pipeline.
- No eliminar ni versionar datos, tokens, entornos virtuales o checkpoints pesados.
- Los checkpoints nuevos contienen pesos, configuración, métricas e historial;
  los cargadores deben continuar aceptando checkpoints históricos que solo
  contengan un `state_dict`.
- No usar el conjunto de prueba para entrenar. El pseudo-labeling es una técnica
  opcional y debe reportarse explícitamente como tal.
- Antes de entregar, ejecutar el notebook completo en el hardware elegido y
  conservar sus output cells.
- Los mensajes de commit no deben incluir nombres de asistentes ni líneas
  `Co-Authored-By`.
