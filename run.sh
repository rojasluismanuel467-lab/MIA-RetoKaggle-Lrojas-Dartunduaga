#!/usr/bin/env bash
# Ejecuta el notebook completo end-to-end en background.
#
# Uso:
#   bash run.sh              # ejecuta y sigue en foreground
#   bash run.sh --bg         # ejecuta en background (nohup + disown)
#   bash run.sh --tmux       # dentro de una sesión tmux (mejor para SSH)
#
# Al terminar (o si abortas), los resultados quedan en outputs/ y checkpoints/.
set -euo pipefail

readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

log() { echo -e "${BLUE}[run]${NC} $*"; }
ok()  { echo -e "${GREEN}✓${NC} $*"; }
warn(){ echo -e "${YELLOW}⚠${NC} $*"; }

# Verificar entorno
[ -f ".venv/bin/activate" ] || { warn "Falta .venv. Corre primero: bash setup.sh"; exit 1; }
[ -f "reto_cnn.ipynb" ]     || { warn "Falta reto_cnn.ipynb"; exit 1; }
[ -f "data/train.csv" ]     || { warn "Falta el dataset en data/. Corre setup.sh"; exit 1; }

# shellcheck source=/dev/null
source .venv/bin/activate

MODE="fg"
if [ "${1:-}" = "--bg" ]; then
    MODE="bg"
elif [ "${1:-}" = "--tmux" ]; then
    MODE="tmux"
fi

# Estimación de tiempo
GPU_OK=$(python -c 'import torch; print(int(torch.cuda.is_available()))')
if [ "$GPU_OK" = "1" ]; then
    log "GPU detectada — corrida estimada: ~15-30 min"
else
    warn "Sin GPU — corrida estimada: ~2-4 horas"
fi

RUN_STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="run_${RUN_STAMP}.log"

CMD="jupyter nbconvert --to notebook --execute reto_cnn.ipynb \
    --output reto_cnn.ipynb \
    --ExecutePreprocessor.timeout=14400"

case "$MODE" in
    fg)
        log "Ejecutando en foreground → log: $LOG_FILE"
        $CMD 2>&1 | tee "$LOG_FILE"
        ;;
    bg)
        log "Ejecutando en background → log: $LOG_FILE"
        nohup $CMD > "$LOG_FILE" 2>&1 &
        BG_PID=$!
        disown
        ok "PID: $BG_PID"
        ok "Ver progreso: tail -f $LOG_FILE"
        ok "Verificar activo: ps -p $BG_PID"
        ok "Cancelar: kill $BG_PID"
        ;;
    tmux)
        command -v tmux >/dev/null 2>&1 || { warn "tmux no instalado. Instala: sudo apt install tmux"; exit 1; }
        log "Lanzando en tmux session 'reto-train'…"
        tmux new-session -d -s reto-train "$CMD 2>&1 | tee $LOG_FILE"
        ok "Reconectar:  tmux attach -t reto-train"
        ok "Detach:      Ctrl+B luego D"
        ok "Matar:       tmux kill-session -t reto-train"
        ok "Ver log:     tail -f $LOG_FILE"
        ;;
esac

echo ""
echo "Cuando termine, los deliverables estarán en:"
echo "  outputs/submission_two_stage.csv    ← el mejor modelo"
echo "  outputs/robustness_matrix.csv       ← matriz de robustez"
echo "  outputs/experiments_log.json        ← registro de experimentos"
echo "  reto_cnn.ipynb                       ← notebook con outputs frescos"
