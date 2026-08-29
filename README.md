# MIA-RetoKaggle-Lrojas-Dartunduaga

Taller CNN — **AAIV 2026-II · MIAA-MCD** (Icesi).
Competencia Kaggle: [aaiv-2026-ii-taller-cnn-miaa-mcd](https://www.kaggle.com/competitions/aaiv-2026-ii-taller-cnn-miaa-mcd).

**Objetivo:** clasificar el estado del conductor (`awake` / `drowsy`) y predecir la bounding box que lo localiza en la imagen.

**Métricas:** Accuracy (clasificación) · Dice coefficient (regresión bbox).

**Fechas clave:**
- Entrega notebook: **17-sep-2026 · 23:59** (intu)
- Sustentación: **21–26 sep 2026**

**Integrantes (2):** Luis Manuel Rojas Correa · Dartunduaga.

---

## Quickstart

```bash
# 1. Activar el entorno
source .venv/bin/activate

# 2. Levantar Jupyter
jupyter lab
```

Abrir `reto_cnn.ipynb` y seleccionar el kernel **Python (Reto Kaggle CNN)**.

### Descargar los datos

1. Configurar credenciales de Kaggle (una sola vez):
   - Ir a https://www.kaggle.com/settings/api → *Create New Token* → descarga `kaggle.json`.
   - Colocarlo en `~/.kaggle/kaggle.json` con permisos `600`.

2. Bajar el dataset:
   ```bash
   kaggle competitions download -c aaiv-2026-ii-taller-cnn-miaa-mcd -p data/
   unzip data/aaiv-2026-ii-taller-cnn-miaa-mcd.zip -d data/
   ```

---

## Estructura del repo

```
.
├── reto_cnn.ipynb              # notebook principal (siguiendo rúbrica)
├── requirements.txt            # dependencias exactas del venv
├── docs/
│   └── Rúbrica ... .xlsx       # rúbrica oficial del taller
├── references/                 # libros de referencia (git-ignored)
│   ├── machine-learning-book/          # Raschka (PyTorch)
│   ├── deep-learning-with-python-notebooks/   # Chollet
│   ├── handson-ml3/            # Géron
│   └── practical-ml-vision-book/  # Lakshmanan (bbox regression)
├── data/                       # dataset de Kaggle (git-ignored)
└── .venv/                      # entorno virtual (git-ignored)
```

---

## Rúbrica (resumen — ver [docs/](docs/) para detalle)

| # | Punto | Peso |
|---|-------|------|
| 1 | CNN personalizado desde cero (backbone) | 1.0 |
| 2 | Cabeza de regresión bbox + ≥3 hiperparámetros | 1.0 |
| 3 | Cabeza de clasificación + ≥3 hiperparámetros | 1.0 |
| 4 | Data Augmentation con **albumentations** | 0.5 |
| 5 | Transfer learning con **≥2** preentrenados (≠ código base) | 1.0 |
| 6 | Elección del mejor modelo con análisis crítico | 0.5 |
| — | Bonificación por posición en Kaggle (1º=+1, 5º=+0.2) | +1.0 |

**Sanción:** −0.5 pt por cada error de protocolo (fuga de datos train/val/test).

---

## Stack técnico

- **Python:** 3.14
- **DL:** PyTorch 2.13 (CPU) + torchvision
- **Data Aug:** albumentations (con `bbox_params`)
- **Transfer learning:** `timm` (EfficientNet, ResNet, ConvNeXt, etc.)
- **Kaggle:** `kaggle` CLI 2.2.4
- **Notebook:** JupyterLab + kernel `reto-kaggle-cnn`
