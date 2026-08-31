#!/usr/bin/env bash
# Setup completo del proyecto Taller CNN drowsiness detection.
#
# Uso:
#   git clone https://github.com/rojasluismanuel467-lab/MIA-RetoKaggle-Lrojas-Dartunduaga.git
#   cd MIA-RetoKaggle-Lrojas-Dartunduaga
#   bash setup.sh
#
# El script:
#   1. Verifica requisitos (Python 3.10+, git)
#   2. Crea venv en .venv/
#   3. Detecta CUDA disponible y instala PyTorch con soporte GPU si aplica
#   4. Instala el paquete drowsy_cnn + deps + dev tools
#   5. Configura credenciales Kaggle (interactivo si no existen)
#   6. Descarga el dataset de la competencia (~100 MB)
#   7. Corre 19 smoke tests para verificar que todo funciona
#   8. Registra kernel Jupyter y muestra cómo lanzar
#
# Testeado en: Ubuntu 22.04+, Debian 12+, Fedora 39+, macOS.
# Windows: usar WSL2 (Ubuntu) — no se prueba en PowerShell/cmd.
set -euo pipefail

# ============================================================================
# Utilidades
# ============================================================================
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

log() { echo -e "${BLUE}[setup]${NC} $*"; }
ok()  { echo -e "${GREEN}✓${NC} $*"; }
warn(){ echo -e "${YELLOW}⚠${NC} $*"; }
err() { echo -e "${RED}✗${NC} $*" >&2; }

fail() { err "$1"; exit 1; }

# ============================================================================
# 1. Verificar requisitos
# ============================================================================
log "Verificando requisitos previos…"

command -v git >/dev/null 2>&1 || fail "git no está instalado. Instala: sudo apt install git"

PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PY_VER=$("$candidate" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
        MAJOR=$(echo "$PY_VER" | cut -d. -f1)
        MINOR=$(echo "$PY_VER" | cut -d. -f2)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

[ -n "$PYTHON_BIN" ] || fail "Python 3.10+ requerido. Instala: sudo apt install python3.11 python3.11-venv"
ok "Python detectado: $PYTHON_BIN ($($PYTHON_BIN --version))"

# Verificar módulo venv
"$PYTHON_BIN" -c 'import venv' 2>/dev/null || {
    fail "Módulo 'venv' no disponible. Instala: sudo apt install ${PYTHON_BIN}-venv"
}

# ============================================================================
# 2. Detectar GPU / CUDA
# ============================================================================
log "Detectando aceleración de hardware…"
CUDA_VERSION=""
HAS_GPU=false
if command -v nvidia-smi >/dev/null 2>&1; then
    HAS_GPU=true
    CUDA_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 || echo "")
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo "")
    ok "GPU NVIDIA detectada: $GPU_NAME (driver $CUDA_DRIVER)"
    # Determinar CUDA version compatible
    DRIVER_MAJOR=$(echo "$CUDA_DRIVER" | cut -d. -f1)
    if [ "$DRIVER_MAJOR" -ge 545 ]; then
        CUDA_VERSION="cu124"; log "  → PyTorch cu124 (CUDA 12.4)"
    elif [ "$DRIVER_MAJOR" -ge 525 ]; then
        CUDA_VERSION="cu121"; log "  → PyTorch cu121 (CUDA 12.1)"
    elif [ "$DRIVER_MAJOR" -ge 470 ]; then
        CUDA_VERSION="cu118"; log "  → PyTorch cu118 (CUDA 11.8)"
    else
        warn "Driver antiguo — instalando PyTorch CPU"
        CUDA_VERSION=""
    fi
else
    if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        ok "Apple Silicon detectado — se instalará PyTorch con soporte MPS"
    else
        warn "Sin GPU NVIDIA detectada. Se usará CPU (o el acelerador disponible)."
    fi
fi

# ============================================================================
# 3. Crear venv
# ============================================================================
log "Creando entorno virtual .venv/…"
if [ -d ".venv" ]; then
    warn ".venv ya existe. ¿Recrear? [y/N]"
    read -r RECREATE
    if [ "$RECREATE" = "y" ] || [ "$RECREATE" = "Y" ]; then
        rm -rf .venv
    fi
fi
[ -d ".venv" ] || "$PYTHON_BIN" -m venv .venv
ok ".venv creado"

# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --quiet --upgrade pip

# ============================================================================
# 4. Instalar PyTorch (con o sin CUDA) + paquete + deps
# ============================================================================
log "Instalando PyTorch…"
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    # En Apple Silicon PyTorch puede usar MPS; no forzar el índice CPU de Linux.
    pip install --quiet torch torchvision
elif [ -n "$CUDA_VERSION" ]; then
    pip install --quiet torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
else
    pip install --quiet torch torchvision --index-url "https://download.pytorch.org/whl/cpu"
fi
ok "PyTorch instalado"

log "Instalando paquete drowsy_cnn + dependencias (dev + extra)…"
pip install --quiet -e ".[dev,extra]"
ok "Paquete instalado en modo editable"

# Registrar kernel Jupyter
python -m ipykernel install --user --name reto-kaggle-cnn \
    --display-name "Python (Reto Kaggle CNN)" >/dev/null 2>&1 || true

# ============================================================================
# 5. Configurar Kaggle
# ============================================================================
log "Configurando credenciales Kaggle…"
KAGGLE_CFG_DIR="$HOME/.kaggle"
KAGGLE_JSON="$KAGGLE_CFG_DIR/kaggle.json"
KAGGLE_TOKEN_FILE="$KAGGLE_CFG_DIR/access_token"

mkdir -p "$KAGGLE_CFG_DIR"

if [ -n "${KAGGLE_API_TOKEN:-}" ]; then
    ok "Credenciales Kaggle detectadas en KAGGLE_API_TOKEN"
elif [ -f "$KAGGLE_JSON" ] || [ -f "$KAGGLE_TOKEN_FILE" ]; then
    ok "Credenciales Kaggle ya configuradas"
else
    warn "No hay credenciales Kaggle. Opciones:"
    echo "  A) Pegar el contenido de kaggle.json (formato: {\"username\":\"...\",\"key\":\"...\"})"
    echo "  B) Pegar el access_token estilo KGAT_..."
    echo "  C) Saltar (podrás correr el pipeline con datos que ya tengas en data/)"
    echo -n "Elige [A/B/C]: "
    read -r CHOICE
    CHOICE_UPPER=$(printf '%s' "$CHOICE" | tr '[:lower:]' '[:upper:]')
    case "$CHOICE_UPPER" in
        A)
            echo "Pega el JSON completo (una línea) y presiona Enter:"
            read -r JSON_CONTENT
            echo "$JSON_CONTENT" > "$KAGGLE_JSON"
            chmod 600 "$KAGGLE_JSON"
            ok "kaggle.json guardado"
            ;;
        B)
            echo "Pega el token (empieza con KGAT_):"
            read -r TOKEN
            echo -n "$TOKEN" > "$KAGGLE_TOKEN_FILE"
            chmod 600 "$KAGGLE_TOKEN_FILE"
            ok "access_token guardado"
            ;;
        *)
            warn "Saltando configuración Kaggle. Deberás bajar el dataset manualmente."
            SKIP_DATASET=1
            ;;
    esac
fi

# ============================================================================
# 6. Descargar dataset
# ============================================================================
if [ -z "${SKIP_DATASET:-}" ]; then
    if [ -f "data/train.csv" ] && [ -d "data/images" ]; then
        ok "Dataset ya presente en data/ — saltando descarga"
    else
        log "Descargando dataset de Kaggle (~100 MB)…"
        mkdir -p data
        if kaggle competitions download -c aaiv-2026-ii-taller-cnn-miaa-mcd -p data/ 2>&1; then
            log "Descomprimiendo…"
            cd data
            # La competencia puede entregar un ZIP externo que contiene images.zip.
            # Repetir hasta eliminar todos los niveles de compresión.
            while find . -maxdepth 1 -type f -name '*.zip' -print -quit | grep -q .; do
                for z in ./*.zip; do
                    [ -f "$z" ] || continue
                    if [ "$(basename "$z")" = "images.zip" ]; then
                        mkdir -p images
                        unzip -q -o "$z" -d images
                    else
                        unzip -q -o "$z"
                    fi
                    rm "$z"
                done
            done
            cd ..
            if [ -f "data/train.csv" ] && [ -f "data/test.csv" ] \
                && [ -f "data/sample_submission.csv" ] && [ -d "data/images" ]; then
                ok "Dataset listo en data/"
            else
                warn "La descarga terminó, pero faltan archivos esperados en data/."
            fi
        else
            warn "Falló la descarga del dataset. Verifica: "
            warn "  - Credenciales correctas"
            warn "  - Aceptaste las reglas en https://www.kaggle.com/competitions/aaiv-2026-ii-taller-cnn-miaa-mcd/rules"
        fi
    fi
fi

# ============================================================================
# 7. Smoke tests
# ============================================================================
log "Corriendo 19 smoke tests…"
if python -m pytest tests/ -q 2>&1 | tail -3; then
    ok "Todos los tests pasan"
else
    warn "Algunos tests fallaron — revisa la salida arriba"
fi

# ============================================================================
# 8. Verificar CUDA en PyTorch
# ============================================================================
log "Verificando setup PyTorch…"
python <<'PYEOF'
import torch
print(f"  PyTorch      : {torch.__version__}")
print(f"  CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU count    : {torch.cuda.device_count()}")
    print(f"  GPU name     : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM total   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    import os
    print(f"  CPU threads  : {torch.get_num_threads()}")
    print(f"  Cores disponibles: {os.cpu_count()}")
    print(f"  MPS disponible: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
PYEOF

# ============================================================================
# 9. Mensaje final
# ============================================================================
echo ""
echo "════════════════════════════════════════════════════════════════════"
ok "Setup completo. Para arrancar:"
echo ""
echo "  Opción A — Notebook interactivo:"
echo "    source .venv/bin/activate"
echo "    jupyter lab reto_cnn.ipynb"
echo ""
echo "  Opción B — Correr notebook end-to-end en background:"
echo "    source .venv/bin/activate"
echo "    nohup jupyter nbconvert --to notebook --execute reto_cnn.ipynb \\"
echo "          --output reto_cnn.ipynb --ExecutePreprocessor.timeout=7200 \\"
echo "          > run.log 2>&1 &"
echo "    tail -f run.log     # ver progreso"
echo ""
echo "  Opción C — Ejemplo mínimo del paquete (~5 min):"
echo "    source .venv/bin/activate"
echo "    python tools/example_usage.py"
echo ""
echo "  Los outputs quedan en:"
echo "    - outputs/submission_*.csv  (para Kaggle)"
echo "    - checkpoints/*.pt          (modelos entrenados)"
echo "    - outputs/experiments_log.json  (registro de experimentos)"
echo "════════════════════════════════════════════════════════════════════"
