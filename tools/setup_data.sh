#!/usr/bin/env bash
# Descarga y descomprime el dataset de la competencia una vez que el token de Kaggle esté en su sitio.
# Uso:  bash tools/setup_data.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
COMP="aaiv-2026-ii-taller-cnn-miaa-mcd"

# 1. Verificar credenciales
if [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "→ Usando KAGGLE_API_TOKEN del entorno"
elif [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
  chmod 600 "$HOME/.kaggle/kaggle.json"
elif [[ -f "$HOME/.kaggle/access_token" ]]; then
  echo "→ Usando ~/.kaggle/access_token"
else
  echo "❌ Falta ~/.kaggle/kaggle.json"
  echo "   Ve a https://www.kaggle.com/settings/api → Create New Token"
  echo "   y colócalo en ~/.kaggle/ con: chmod 600 ~/.kaggle/kaggle.json"
  echo "   Alternativa: export KAGGLE_API_TOKEN='KGAT_...'"
  exit 1
fi

# 2. Activar venv
source "$REPO_ROOT/.venv/bin/activate"

# 3. Verificar auth
echo "→ Verificando auth con Kaggle..."
kaggle competitions list --search "$COMP" | head -3

# 4. Descargar
mkdir -p "$DATA_DIR"
echo "→ Descargando dataset a $DATA_DIR ..."
kaggle competitions download -c "$COMP" -p "$DATA_DIR"

# 5. Descomprimir
echo "→ Descomprimiendo..."
cd "$DATA_DIR"
# La descarga puede traer un ZIP externo que contiene images.zip.
# Repetir hasta extraer todos los niveles.
while find . -maxdepth 1 -type f -name '*.zip' -print -quit | grep -q .; do
  for z in ./*.zip; do
    [[ -f "$z" ]] || continue
    if [[ "$(basename "$z")" == "images.zip" ]]; then
      mkdir -p images
      unzip -o "$z" -d images
    else
      unzip -o "$z"
    fi
    rm "$z"
  done
done

if [[ ! -f train.csv || ! -f test.csv || ! -f sample_submission.csv || ! -d images ]]; then
  echo "❌ Faltan archivos esperados: train.csv, test.csv, sample_submission.csv o images/"
  exit 1
fi

echo
echo "✓ Dataset listo en: $DATA_DIR"
ls -la "$DATA_DIR"
