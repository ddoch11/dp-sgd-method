#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/sangjin/anaconda3/envs/pytorch/bin/python}"
METHOD="${1:?method required: non_dp|naive_dp|hooks_dp|vmap_dp|expanded_weights_dp|ghost_dp}"
EPSILON="${2:-2}"
GPU="${3:-0}"
EXPERIMENT_DATE="${4:-$(date +%F)}"
MAX_STEPS="${5:-2}"
LOG_DIR="$ROOT_DIR/experiments/$EXPERIMENT_DATE/launcher_logs"
mkdir -p "$LOG_DIR"

case "$METHOD" in
  non_dp|naive_dp|hooks_dp|vmap_dp|expanded_weights_dp|ghost_dp) ;;
  *) echo "unsupported method: $METHOD" >&2; exit 2 ;;
esac

LABEL="${METHOD}_eps${EPSILON}_$(date +%Y%m%d_%H%M%S)"
GPU_LOG="$LOG_DIR/${LABEL}_gpu.csv"
STDOUT_LOG="$LOG_DIR/${LABEL}.log"

nvidia-smi --id="$GPU" \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 1 >"$GPU_LOG" 2>&1 &
MONITOR_PID=$!
cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ARGS=(
  --config "${CONFIG_PATH:-$ROOT_DIR/configs/methods_bf16.yaml}"
  --method "$METHOD"
  --gpu "$GPU"
  --experiment-date "$EXPERIMENT_DATE"
  --max-steps "$MAX_STEPS"
)
if [[ "$EPSILON" != "none" ]]; then
  ARGS+=(--target-epsilon "$EPSILON")
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "$ROOT_DIR/src/train_methods.py" \
  "${ARGS[@]}" 2>&1 | tee "$STDOUT_LOG"
