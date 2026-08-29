# Pipeline de alto nivel — Taller CNN drowsiness

Diseño del proceso end-to-end. Cada decisión lleva su justificación (rúbrica + libros + papers).

---

## Panorama del problema

- **Task:** multitask (clasificación binaria awake/drowsy + regresión bbox del conductor).
- **Datos:** 420 train / 106 test, todas 1920×1080 RGB, 1 objeto por imagen, conductor casi siempre en zona central-inferior (cx std=8.7%, cy std=5.5%). Test **interleaved temporalmente** con train (mismo video GOPR0492).
- **Métricas:** Accuracy (cls), Dice = 2·IoU/(1+IoU) (bbox). Dice sobre bbox es **monótona en IoU** — se entrena con losses de la familia IoU/SmoothL1 y se reporta Dice (Medium – IoU/mAP/Dice).
- **Restricciones:** CPU-only, PyTorch, `albumentations` obligatorio, ≥2 backbones preentrenados distintos al del código base.

---

## Fases del pipeline

```
                          ┌────────────────────────────┐
                          │  CONFIG (dict global)      │
                          │  seeds, paths, hyperparams │
                          └─────────────┬──────────────┘
                                        │
   ┌─────────────┐   ┌──────────────┐   ▼   ┌──────────────┐   ┌─────────────┐
   │ 1. EDA      │──▶│ 2. Split     │──────▶│ 3. Data      │──▶│ 4. Metrics  │
   │ imágenes    │   │ estratificado│       │ pipeline     │   │ & losses    │
   │ + bbox stats│   │ + group ck   │       │ (Dataset +   │   │ (Dice, IoU, │
   └─────────────┘   └──────────────┘       │ albumentations)│  │ SmoothL1)   │
                                            └───────┬──────┘   └──────┬──────┘
                                                    │                 │
                                        ┌───────────▼─────────────────▼──────┐
                                        │ 5. train_one_config(cfg) → results │
                                        │    (bucle reutilizable)            │
                                        └───────────────┬────────────────────┘
                                                        │
                    ┌───────────────┬────────────────┬──┴────────────┬────────────────┐
                    ▼               ▼                ▼               ▼                ▼
              ┌──────────┐   ┌──────────┐    ┌────────────┐   ┌──────────┐    ┌────────────┐
              │ 6. PUNTO 1│  │ 7. PUNTOS │    │ 8. PUNTO 4 │   │ 9. PUNTO 5│  │10. PUNTO 6 │
              │ CNN prop. │  │ 2 y 3     │    │ Aug ablation│  │ Transfer  │  │ Best model │
              │ from scr. │  │ ≥3 hyps   │    │ (con/sin)   │  │ ≥2 backb. │  │ + análisis │
              └───────────┘  └──────────┘    └────────────┘   └──────────┘    └─────┬──────┘
                                                                                     │
                                                                          ┌──────────▼──────────┐
                                                                          │ 11. Inference + TTA │
                                                                          │     + submission.csv│
                                                                          └─────────────────────┘
```

---

## Detalle por fase

### 1. EDA (ya hecho)

**Fundamento:** Chollet §"EDA sobre imágenes" cap 8; hallazgos concretos guardados en memoria [[project-kaggle-cnn]].

**Entrega:** notebook con distribución de clases (240/180 = 57/43%), distribución de bbox (center/size), muestras visuales con bbox dibujada. Nota crítica sobre **interleave temporal** entre train y test.

### 2. Split train/val (anti-fuga) — estrategia **NESTED / two-stage**

**Fundamento:**
- Rúbrica sanciona -0.5 pt por fuga.
- Con dataset chico, un holdout único es vulnerable a sampling bias (±5-8% variance): [PMC guide on CV for medical imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC10388213/), [Raschka arxiv:1811.12808](https://arxiv.org/pdf/1811.12808).
- k=5 es el sweet spot (variance vs cost).

**Decisión:** two-stage por presupuesto CPU.
- **Etapa exploratoria (~15 experimentos):** single stratified split 80/20 (336/84) fijo. Rápido, comparable entre configs por usar el mismo split.
- **Etapa de selección final (top-3 candidatos):** `StratifiedKFold` k=5 sobre esos 3, promediando métricas para robustez.
- **Sanity check adicional:** group-shuffle por rangos temporales (10 bins de frame_idx) sobre el ganador — se reporta como control en la sustentación, no como criterio de selección.

**Semilla fija (42):** todos los splits usan `random_state=42` para reproducibilidad y comparabilidad de experimentos.

### 3. Data pipeline

**Fundamento:**
- Géron cap 14 [§Classification+Localization](references/handson-ml3/14_deep_computer_vision_with_cnns.ipynb) — patrón de salida `(logits, bbox)` compartiendo backbone.
- Docs oficiales de [albumentations bbox augmentations](https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/).
- Chollet cap 12 — normaliza coords bbox a 0-1 dividiendo por width/height.

**Decisiones:**
- **Homogenización de I/O:** todas las imgs abren con PIL/cv2 en RGB (evitar BGR de OpenCV), dtype `float32`. Ninguna imagen tiene forma distinta → no hay que resamplear tipos.
- **Encoding de labels:** `awake=0, drowsy=1` (LabelEncoder con orden alfabético fijado).
- **Bbox encoding — normalizado 0-1 en formato `(cx, cy, w, h)`** (convención YOLO/DETR).
  - Durante training: `(cx, cy, w, h)` en [0, 1] — target compatible con sigmoid en la head, scale-invariance para SmoothL1.
  - En el CSV final: convertir a `(xmin, ymin, xmax, ymax)` en pixels absolutos (formato exigido por Kaggle), `clip(0, W/H)`, redondear a int.
  - Fuentes: [Ultralytics YOLO norm issue](https://github.com/ultralytics/ultralytics/issues/2005), [HuggingFace DETR bbox format](https://github.com/huggingface/transformers/issues/32835).
- **Resize:** 224×224 (compatible con backbones ImageNet, distorsión aceptable). Albumentations reescala bboxes automáticamente.
- **Normalización de pixels:**
  - Con backbones **preentrenados** → stats de ImageNet `[0.485,0.456,0.406] / [0.229,0.224,0.225]` para preservar warm-start (Chollet cap 8, Géron cap 14).
  - Con **CNN desde cero** → stats calculadas sobre train `[0.475,0.459,0.456] / [0.250,0.235,0.216]` (fit-on-train, ya calculadas en `data/norm_stats.npz`).
- **Dataset class:** `DrowsyDataset(df, image_dir, transform)` — retorna `(img_tensor, label_int, bbox_cxcywh_norm_tensor)`.

### 4. Métricas & losses

**Fundamento:**
- Rúbrica exige Dice para bbox, Accuracy para cls.
- Chollet cap 12 (`intersection_over_union` líneas 520-540) — implementación base de IoU.
- LearnOpenCV / arxiv (IoU loss functions family) — SmoothL1 vs GIoU vs CIoU.

**Decisiones:**
- **`accuracy(y_true, y_pred)`**: from `sklearn.metrics.accuracy_score`.
- **`iou(box_a, box_b)` batched**: implementación PyTorch propia (5-10 líneas) siguiendo Chollet cap 12 L520-540.
- **`dice(box_a, box_b) = 2·IoU/(1+IoU)`**: derivada trivial de IoU.
- **Loss bbox — combo estilo DETR:** `L_bbox = 1.0·SmoothL1(cxcywh_norm) + 2.0·GIoU_loss(xyxy)`. Rationale: SmoothL1 estable a nivel de coordenadas + GIoU mejora convergencia geométrica ([DETR paper](https://arxiv.org/abs/2005.12872), [torchvision.ops.generalized_box_iou_loss](https://pytorch.org/vision/main/generated/torchvision.ops.generalized_box_iou_loss.html)).
- **Loss combinada total:** `L = 1·CE + λ·L_bbox` con `λ ∈ {1, 5, 10}` como hiperparámetro barrido (parte del experimento H2 del punto 7).
- **Class imbalance:** `nn.CrossEntropyLoss(weight=torch.tensor([0.75, 1.0]))` calculado como `n_total / (2·n_clase)` (Raschka cap 6).
- **Early stopping** integrado en el bucle: `patience=12`, `monitor=val_total_loss`, `min_delta=1e-3`, `max_epochs=60`. Fuente: [Keras EarlyStopping docs](https://keras.io/api/callbacks/early_stopping/), [MLM Early Stopping guide](https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/).

### 5. `train_one_config(cfg) → results` — bucle reutilizable

**Fundamento:** directriz 4 (pipeline mantenible).

**Firma:**
```python
def train_one_config(cfg: dict) -> dict:
    """
    cfg = {
        'name': str,               # etiqueta única del experimento
        'backbone': str,           # 'custom' | 'resnet18' | 'mobilenet_v3_small' | ...
        'input_size': int,         # 224, 320, ...
        'batch_size': int, 'epochs': int, 'lr': float,
        'optimizer': str,          # 'adam' | 'adamw' | 'sgd'
        'scheduler': str,          # None | 'cosine' | 'step'
        'bbox_loss': str,          # 'smoothl1' | 'giou' | 'ciou'
        'loss_weights': tuple,     # (w_cls, w_bbox)
        'augment': str,            # 'none' | 'basic' | 'strong'
        'class_weights': bool,
        'seed': 42,
    }
    → devuelve dict con history (train/val loss, acc, dice por epoch),
      best metrics, path del checkpoint, tiempo total.
    """
```

**Entrega:** función encapsulada + un `EXPERIMENTS_LOG` (lista de dicts / DataFrame global) que guarda cada corrida para comparación transversal. Todos los experimentos usan **la misma semilla y el mismo split** para comparabilidad.

### 6. **[PUNTO 1 rúbrica]** CNN custom desde cero (1 pt)

**Fundamento:** Raschka cap 14 part 2 (`ch14_part2.py:289-306`) — 4 bloques `Conv3x3(pad=1) → BN → ReLU → MaxPool → Dropout(0.5)`, canales 32→64→128→256, luego `GlobalAvgPool → heads`.

**Justificaciones defendibles en sustentación:**
- Kernel 3×3 con padding=1: preserva resolución (VGG-style, Simonyan & Zisserman 2014).
- BatchNorm después de conv: Ioffe & Szegedy 2015 (estabiliza gradiente, permite LR mayor).
- Dropout p=0.5: Srivastava 2014, apropiado con dataset pequeño.
- **GlobalAveragePooling** en vez de Flatten: reduce parámetros ~10×, previene overfit (Chollet cap 9).
- He/Kaiming init: Géron cap 11 (ReLU requiere init preservando varianza).

**Entrega:** clase `CustomCNN(nn.Module)` + entrenamiento con `train_one_config` + curvas train/val + reporte de métricas base.

### 7. **[PUNTOS 2 y 3 rúbrica]** Cabezas cls + bbox con ≥3 hiperparámetros comparados (2 pt)

**Fundamento:** rúbrica exige comparar variantes; directriz 2.

**Diseño experimental:** una **tabla comparativa** con al menos 3 barridos, ejecutados sobre el CustomCNN del punto anterior:

| Exp | Hiperparámetro variado | Valores probados |
|---|---|---|
| H1 | Optimizer + LR | Adam 1e-3 / AdamW 1e-3 / SGD 1e-2+momentum |
| H2 | `loss_weights` (w_cls, w_bbox) | (1,1) / (1,2) / (1,5) |
| H3 | `bbox_loss` | SmoothL1 / GIoU / CIoU |
| H4 (bonus) | Dropout rate | 0.3 / 0.5 / 0.7 |

**Entrega:** DataFrame de resultados + heatmap/barplot comparativo + una frase por hallazgo. Elegir la **mejor combinación** para servir de baseline al punto 5.

### 8. **[PUNTO 4 rúbrica]** Data augmentation con albumentations (0.5 pt)

**Fundamento:**
- Rúbrica exige `albumentations` explícitamente + ≥2 técnicas + análisis de impacto.
- Docs oficiales de [albumentations.BboxParams](https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/).
- Best practices online (agente): Flip horizontal, brightness/contrast, ShiftScaleRotate ±10°, blur son seguras. **Evitar** VerticalFlip y rotaciones grandes (irrealistas en cabina).

**Diseño experimental:** ablation study con 3 variantes:

| Aug | Composición |
|---|---|
| `none` | Solo resize + normalize |
| `basic` | + HorizontalFlip + RandomBrightnessContrast |
| `strong` | + ShiftScaleRotate(±10°) + HueSaturation + GaussNoise + MotionBlur |

Correr las 3 con la mejor config del punto 7. Reportar Acc y Dice en train/val, y sobretodo la **diferencia**: el efecto del augmentation debe verse en la brecha train↔val (menor overfit).

### 9. **[PUNTO 5 rúbrica]** Transfer learning con ≥2 backbones preentrenados (1 pt)

**Fundamento:**
- Rúbrica: 2 backbones distintos al del código base.
- Con CPU + 336 samples, **3 backbones ligeros son más informativos que 2 profundos** ([Lightweight benchmark arxiv:2505.03303](https://arxiv.org/html/2505.03303v1)).
- Estrategia freeze/unfreeze canónica: [TF Transfer Learning tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning), Chollet cap 8 §"Fine-tuning".

**Diseño experimental — 3 backbones:**

| Backbone | Params | Justificación |
|---|---|---|
| **`resnet18`** (torchvision, ImageNet) | 11M | Baseline sólido, arquitectura canónica (He et al. 2015) |
| **`mobilenet_v3_small`** (torchvision) | 2.5M | CPU-friendly, mejor en objetos pequeños (Howard et al. 2019) |
| **`efficientnet_b0`** (timm) | 5.3M | Sweet-spot accuracy/params (Tan & Le 2019) |

Cada uno con: backbone → GAP → head_cls (2) + head_bbox (4).

**Estrategia freeze/unfreeze en 2 fases** (Chollet cap 8, TF tutorial):
- **Fase A — feature extraction:** backbone congelado (`requires_grad=False`), solo se entrenan las heads. 10 epochs, LR = 1e-3.
- **Fase B — fine-tuning parcial:** descongelar únicamente el **último stage** del backbone. LR diferencial: `LR_backbone = LR_head / 10` (backbone 1e-4, head 1e-3). Otros 15-20 epochs con early stopping.

Nunca fine-tune completo desde el inicio (destruye el prior de ImageNet con dataset chico).

### 10. **[PUNTO 6 rúbrica]** Elección del mejor modelo + interpretabilidad (0.5 pt)

**Fundamento:** rúbrica exige análisis comparativo con métricas + gráficos + visualizaciones + discusión crítica. Añadimos **interpretabilidad de CNN** porque es material fuerte para la sustentación (78% de la nota) — permite defender "el modelo mira el rostro/postura del conductor, no artefactos del fondo".

**Entrega:**

**A. Comparación cuantitativa**
- Tabla consolidada de TODOS los experimentos (custom, hyperparam sweeps, augment ablation, transfer learning) con: nombre, backbone, epochs efectivos (post-early-stopping), best val_acc, best val_dice, tiempo, tamaño modelo (MB).
- Gráfico comparativo: bar (métrica por experimento) + scatter Acc vs Dice.
- Los **top-3 finalistas** entran a validación cruzada k=5 para desempate robusto.

**B. Visualización de predicciones del ganador**
- 8-12 imgs val: imagen + bbox_true (verde) + bbox_pred (rojo) + labels.
- Casos peores (menor Dice) y mejores para discutir dónde falla.

**C. Interpretabilidad del backbone ganador** (4 técnicas)
1. **Grad-CAM sobre la última conv** → dónde "mira" el modelo para decidir cls. Fuente: [Chollet ch10 3ª ed](references/deep-learning-with-python-notebooks/chapter10_interpreting-what-convnets-learn.ipynb) L619-785 (versión PyTorch en L655).
2. **Feature maps intermedios (post-ReLU)** de la primera conv sobre 4-6 imgs representativas — verifica que la red aprende bordes/texturas de rostro y no ruido de fondo. Fuente: mismo notebook L75-239 (patrón `activation_model`).
3. **Confusion matrix + galería de errores** (FN drowsy y FP awake) — con dataset chico cada error cuenta. Fuente: [Géron cap 3](references/handson-ml3/03_classification.ipynb) L562-758.
4. **t-SNE 2D de embeddings del backbone** (penúltima capa) coloreado por clase — revela si las clases son separables antes de la head; también expone outliers. Fuente: [Géron cap 8](references/handson-ml3/08_dimensionality_reduction.ipynb) L1929-2015 (plantilla MNIST directamente reutilizable con `sklearn.manifold.TSNE`).

**D. Discusión crítica**
- ¿Por qué gana este backbone? ¿trade-offs (accuracy vs latencia vs tamaño)?
- ¿Fallos comunes? ¿Qué clase es más difícil (drowsy suele ser minoritaria)?
- ¿Grad-CAM revela sesgos (mira siempre la misma zona, o realmente al conductor)?
- ¿Los embeddings t-SNE muestran clases separables?
- ¿Qué mejoraría con más datos / GPU / más tiempo?

### 11. Inferencia + submission

**Fundamento:** formato requerido por Kaggle.

**Decisiones:**
- Cargar checkpoint ganador.
- **TTA horizontal flip**: promediar logits, promediar bboxes reflejados. Fuente: Kaggle winners patterns (agente online).
- Des-normalizar bbox y `clip` a [0, 1920]/[0, 1080], redondear a int.
- Guardar `submission.csv` con columnas exactas `filename,class,xmin,ymin,xmax,ymax`.

---

## Arquitectura de código (cómo se organiza en el notebook)

```
reto_cnn.ipynb
├── §1  Setup + seed + config global (dict)
├── §2  EDA (imports, plots, hallazgos)
├── §3  Split (helper split_data(), imprime distribución)
├── §4  Dataset + transforms (clase DrowsyDataset + fn make_transforms(aug))
├── §5  Metrics + losses (dice(), iou(), CombinedLoss)
├── §6  Modelo custom (clase CustomCNN)
├── §7  Modelos con backbones (fn build_pretrained(name))
├── §8  train_one_config(cfg) — CORE
├── §9  Experiments log helper (append_result(), summarize(), plot_all())
│
├── §10 [PUNTO 1] Corrida baseline con CustomCNN
├── §11 [PUNTOS 2+3] Barridos H1..H4 sobre CustomCNN
├── §12 [PUNTO 4] Ablation de augmentation
├── §13 [PUNTO 5] Transfer learning con 3 backbones
├── §14 [PUNTO 6] Análisis consolidado + mejor modelo elegido
│
└── §15 Inference con TTA + submission.csv
```

**Ventaja de esta arquitectura:** cambiar un hiperparámetro es solo cambiar el `cfg` dict, y llamar `train_one_config(cfg)`. Todos los experimentos son comparables porque comparten pipeline, seed, split y métrica.

---

## Tabla resumen de decisiones (con fuente)

| Decisión | Elección | Fuente |
|---|---|---|
| Framework | PyTorch 2.13 CPU | Instalado, canónico en Raschka |
| Preprocesamiento coord bbox | Normalizar a 0-1 | Chollet cap 12 |
| Resize | 224×224 cuadrado | Backbones ImageNet estándar |
| Normalización pixels (pretrained) | ImageNet stats | Chollet cap 8, Géron cap 14 |
| Normalización pixels (from scratch) | Stats de train | Fit-on-train, sklearn convention |
| Split principal | Random estratificado 80/20 | Matches test público interleaved |
| Split sanity | Group-shuffle 10 bins | Generalización honesta |
| Backbone from scratch | 4×[Conv3-BN-ReLU-Pool-Dropout] 32→256 + GAP | Raschka cap 14, Chollet cap 9 |
| Loss cls | CrossEntropy + class_weights | Raschka cap 6 |
| Loss bbox | SmoothL1 base, GIoU variante | Géron cap 12, torchvision.ops |
| Loss combinada | `1·CE + 2·SmoothL1` | Géron cap 14 (0.8/0.2 análogo) |
| Metric | Dice = 2·IoU/(1+IoU) | Definición, monotónica en IoU |
| Backbones preentrenados | ResNet18, MobileNetV3-small, EfficientNet-B0 | Cross: agente online + He/Howard/Tan |
| Aug seguras (con bbox) | HFlip, RandomBrightnessContrast, ShiftScaleRotate±10° | Albumentations docs |
| Optimizer default | AdamW + Cosine schedule | Kaggle winners pattern |
| TTA final | Horizontal flip | Kaggle winners pattern |
| Early stopping | patience=12, monitor=val_total_loss, max_epochs=60 | Keras/MLM guides + adaptación small-data |
| Freeze strategy | Fase A feature-extract (10ep) + Fase B unfreeze último stage (LR diferencial) | Chollet cap 8, TF tutorial |
| Regularización combo | Aug fuerte + wd=5e-4 + dropout=0.3 en head + cosine schedule | Perez&Wang 2017, evidencia small-data |
| Interpretabilidad (punto 6) | Grad-CAM + feature maps + confusion matrix + t-SNE embeddings | Chollet cap 10, Géron cap 3 y 8 |
