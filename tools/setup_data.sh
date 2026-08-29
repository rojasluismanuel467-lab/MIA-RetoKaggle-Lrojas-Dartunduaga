#!/usr/bin/env bash
# Descarga y descomprime el dataset de la competencia una vez que el token de Kaggle esté en su sitio.
# Uso:  bash tools/setup_data.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
COMP="aaiv-2026-ii-taller-cnn-miaa-mcd"

# 1. Verificar credenciales
if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
  echo "❌ Falta ~/.kaggle/kaggle.json"
  echo "   Ve a https://www.kaggle.com/settings/api → Create New Token"
  echo "   y colócalo en ~/.kaggle/ con: chmod 600 ~/.kaggle/kaggle.json"
  exit 1
fi
chmod 600 "$HOME/.kaggle/kaggle.json"

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
for z in *.zip; do
  [[ -f "$z" ]] || continue
  unzip -o "$z" -d "${z%.zip}"
  rm "$z"
done

echo
echo "✓ Dataset listo en: $DATA_DIR"
ls -la "$DATA_DIR"
