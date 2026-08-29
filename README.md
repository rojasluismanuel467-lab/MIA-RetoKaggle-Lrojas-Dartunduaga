# MIA-RetoKaggle-Lrojas-Dartunduaga

Taller CNN — **AAIV 2026-II · MIAA-MCD** (Icesi).
Competencia Kaggle: [aaiv-2026-ii-taller-cnn-miaa-mcd](https://www.kaggle.com/competitions/aaiv-2026-ii-taller-cnn-miaa-mcd).

**Objetivo:** clasificar el estado del conductor (`awake` / `drowsy`) y predecir la bounding box que lo localiza.
**Métricas:** Accuracy (clasificación) · Dice coefficient (bbox regression).
**Fechas:** entrega notebook **17-sep-2026 · 23:59** · sustentación **21–26 sep 2026**.
**Integrantes (2):** Luis Manuel Rojas Correa · Dartunduaga.

---

## Estructura del repo

```
MIA-RetoKaggle-Lrojas-Dartunduaga/
├── pyproject.toml               # setuptools + ruff + pytest + mypy config
├── src/drowsy_cnn/              # PAQUETE PYTHON — código modularizado y testeado
│   ├── config.py                # ExpConfig, constantes, detección Kaggle vs local
│   ├── dataset.py               # DrowsyDataset, CropDataset, conversiones bbox
│   ├── augmentation.py          # build_transforms (none/basic/strong/cabin_realistic)
│   ├── models.py                # CustomCNN, MultitaskWrapper, CropClassifier
│   ├── losses.py                # MultitaskLoss (SmoothL1+GIoU), IoU, Dice, EarlyStopping
│   ├── splits.py                # make_split, make_kfold, make_group_split
│   ├── training.py              # train_one_config, train_epoch, eval_epoch
│   ├── inference.py             # predict_with_tta, ensemble_predict, stage1_predict_bboxes
│   └── robustness.py            # matriz condition × val_acc (túneles, gafas, etc.)
├── tests/
│   └── test_smoke.py            # 19 tests: shapes, roundtrips, forward, losses
├── reto_cnn.ipynb               # notebook oficial del taller (autocontenido, Kaggle-ready)
├── reto_cnn.py                  # fuente jupytext del notebook
├── tools/
│   ├── setup_data.sh            # descarga automatizada del dataset Kaggle
│   ├── search_refs.py           # búsqueda full-text sobre libros de referencia
│   ├── example_usage.py         # ejemplo mínimo de uso del paquete
│   ├── two_stage.py             # standalone del two-stage (§18) — usa src/drowsy_cnn
│   └── (otros scripts históricos)
├── docs/
│   ├── pipeline_design.md       # diseño de alto nivel con citas y decisiones
│   └── Rúbrica ... .xlsx        # rúbrica oficial
├── references/                  # libros clonados (git-ignored) — Raschka, Chollet, Géron, Lakshmanan
├── data/                        # dataset Kaggle (git-ignored)
├── checkpoints/                 # modelos entrenados (git-ignored)
└── outputs/                     # submissions + logs (git-tracked)
```

---

## Quickstart

### Instalación (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"                    # paquete + tests + linter
pytest tests/ -v                            # 19 smoke tests
```

### Descargar dataset

```bash
export KAGGLE_API_TOKEN="tu_token_aqui"     # o kaggle.json en ~/.kaggle/
bash tools/setup_data.sh
```

### Correr el notebook

```bash
jupyter lab reto_cnn.ipynb
```

Kernel: **Python (Reto Kaggle CNN)** (registrado con `python -m ipykernel install --user --name reto-kaggle-cnn`).

### Kaggle

El notebook detecta automáticamente si corre en Kaggle (`config.IS_KAGGLE = True`) y usa `/kaggle/input/…` como `DATA_DIR`. Para usarlo allí:
1. Subir el notebook a Kaggle.
2. Attach el dataset de la competencia como input.
3. Correr — el paquete `drowsy_cnn` se instala inline en la primera celda vía `pip install /kaggle/input/...` o el código se define en las celdas.

---

## Uso del paquete `drowsy_cnn`

```python
from drowsy_cnn.config import ExpConfig, DATA_DIR
from drowsy_cnn.splits import make_split
from drowsy_cnn.training import train_one_config
import pandas as pd

df_full = pd.read_csv(DATA_DIR / "train.csv")
df_tr, df_val = make_split(df_full, test_size=0.2)

cfg = ExpConfig(
    name="mi_experimento",
    model_kind="efficientnet_b0",
    aug_level="cabin_realistic",     # aug para robustez a iluminación + oclusiones
    label_smoothing=0.1,
    num_workers=4,                    # aprovecha CPUs locales
)
result = train_one_config(cfg, df_tr, df_val)
print(f"val_acc={result['best']['val_acc']:.4f}, dice={result['best']['val_dice']:.4f}")
```

Ver [`tools/example_usage.py`](tools/example_usage.py) para el patrón end-to-end.

---

## Estrategias de augmentation disponibles

| Nivel | Transforms | Uso |
|---|---|---|
| `none` | solo Resize + Normalize | val/test |
| `basic` | + HorizontalFlip + BrightnessContrast | dataset chico simple |
| `strong` | + Affine + HueSat + GaussNoise + MotionBlur | robustez general |
| `cabin_realistic` | strong + RandomShadow + RandomSunFlare + RandomGamma + CoarseDropout + CLAHE + ColorJitter + ImageCompression + Downscale | **para producción con variabilidad de iluminación (túneles, hora del día) + oclusiones (gafas, gorras, mascarillas)** |

Referencias: Yu et al. 2019 [arxiv:1910.09722](https://arxiv.org/abs/1910.09722), DrowsyDetectNet 2024, YOLO-FDCL 2025.

---

## Rúbrica y evolución de resultados

| # | Punto rúbrica | Peso | Mejor val_acc |
|---|---|---|---|
| 1 | CNN personalizado desde cero | 1.0 | 0.464 |
| 2+3 | Cabezas cls + bbox (≥3 hiperparámetros) | 2.0 | — |
| 4 | Data augmentation con albumentations | 0.5 | — |
| 5 | Transfer learning (≥2 backbones) | 1.0 | 0.845 (ResNet18 solo) |
| 6 | Elección del mejor modelo + interpretabilidad | 0.5 | — |
| — | Ensemble simple | — | 0.845 (dice=0.874) |
| — | Mega ensemble multi-scale + snapshots | — | 0.869 (dice=0.879) |
| — | **Two-stage (bbox→crop→classify)** | — | **0.988 (83/84)** |
| — | Bonificación por posición en Kaggle | +1.0 | Pendiente |

---

## Testing + linting

```bash
pytest tests/ -v         # 19 smoke tests: shapes, roundtrips, forward, losses
ruff check --fix .       # auto-fix estilo (line-length 100, imports ordenados)
ruff format .            # formato
mypy src/drowsy_cnn/     # type checking
```

---

## Stack técnico

- **Python:** 3.10+
- **DL:** PyTorch 2.13 (CPU) + torchvision + timm
- **Data Aug:** albumentations v2
- **Kaggle:** kaggle CLI 2.2
- **Optim:** optuna (opcional, para hyperparameter search)
- **Linter:** ruff (reemplaza flake8+black+isort)
- **Tests:** pytest
