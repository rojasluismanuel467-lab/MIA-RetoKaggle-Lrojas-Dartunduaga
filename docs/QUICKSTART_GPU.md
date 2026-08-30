# Quickstart en máquina con GPU (o CPU)

Para correr el proyecto desde CERO en cualquier laptop/PC con Linux (Ubuntu/Debian/Fedora)
o macOS. **Windows: usar WSL2**.

## Requisitos previos

- `git` instalado
- Python 3.10+ (`python3 --version`)
- (Opcional pero recomendado) GPU NVIDIA con driver 470+ y CUDA compatible
- ~2 GB de espacio libre

## Instalación en 3 comandos

```bash
git clone https://github.com/rojasluismanuel467-lab/MIA-RetoKaggle-Lrojas-Dartunduaga.git
cd MIA-RetoKaggle-Lrojas-Dartunduaga
bash setup.sh
```

El `setup.sh` hace todo automáticamente:

1. Verifica Python 3.10+ y `git`
2. Detecta GPU NVIDIA y decide qué PyTorch instalar (CUDA 11.8 / 12.1 / 12.4 / CPU)
3. Crea `.venv/` y activa el entorno virtual
4. Instala PyTorch + `drowsy_cnn` (paquete local) + tests + linter
5. Configura credenciales Kaggle (interactivo si no existen)
6. Descarga el dataset (~100 MB)
7. Corre 19 smoke tests
8. Registra el kernel Jupyter

**Duración:** ~5 min con banda ancha (descarga PyTorch ~2 GB + dataset ~100 MB).

## Correr el pipeline completo

```bash
bash run.sh --tmux    # recomendado si estás vía SSH
```

Otras opciones:

```bash
bash run.sh           # foreground (ves el output en pantalla)
bash run.sh --bg      # background con nohup (cerrás terminal, sigue)
```

**Tiempo estimado:**
- Con GPU (RTX 3060+): **15-30 min**
- Sin GPU (CPU-only): **2-4 horas**

## Ver resultados

Al terminar quedan en `outputs/`:

- `submission_two_stage.csv` — el mejor modelo (val_acc = 0.988)
- `robustness_matrix.csv` — matriz de robustez sobre 10 condiciones (túnel, gafas, sombras, ...)
- `experiments_log.json` — registro de todos los experimentos con sus métricas

Y en el notebook:

- `reto_cnn.ipynb` con todas las output cells + gráficos + Grad-CAM + t-SNE

## Sesiones de trabajo (dev iterativo)

Después del setup inicial:

```bash
source .venv/bin/activate

# Notebook interactivo
jupyter lab reto_cnn.ipynb

# O correr un experimento específico
python tools/example_usage.py
python tools/run_robustness.py

# Correr tests después de editar código
pytest tests/ -v

# Linter (auto-fix estilo)
ruff check --fix src/ tests/
ruff format src/ tests/
```

## Troubleshooting

### "GPU no detectada" pero tengo NVIDIA

```bash
nvidia-smi                           # verificar driver
python -c "import torch; print(torch.cuda.is_available())"
```

Si `nvidia-smi` funciona pero PyTorch no ve la GPU, reinstalar con la versión CUDA correcta:

```bash
source .venv/bin/activate
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### "Kaggle 403 Forbidden" al descargar dataset

Debés **aceptar las reglas de la competencia** primero:
https://www.kaggle.com/competitions/aaiv-2026-ii-taller-cnn-miaa-mcd/rules → click "I Understand and Accept"

### "Out of memory" (CUDA OOM)

Editar `src/drowsy_cnn/config.py` y bajar `batch_size` en `ExpConfig`:

```python
batch_size: int = 16  # antes 32
```

### Corrida se corta al cerrar terminal

Usar `--tmux` en vez de foreground:

```bash
bash run.sh --tmux
# ...trabaja normal, cierra terminal cuando quieras
# reconectar: tmux attach -t reto-train
```

## Estructura del proyecto (referencia rápida)

```
├── setup.sh                # este script — corre 1 vez al clonar
├── run.sh                  # ejecuta el notebook end-to-end
├── reto_cnn.ipynb          # notebook oficial del taller
├── src/drowsy_cnn/         # paquete Python (importable, testeable)
├── tests/                  # 19 smoke tests
├── tools/                  # scripts standalone
└── outputs/                # submissions + logs (se genera al correr)
```
