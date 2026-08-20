#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/sangjin/anaconda3/envs/pytorch/bin/python}"
EPSILON="${1:-2}"
GPU="${2:-3}"
EXPERIMENT_DATE="${3:-$(date +%F)}"
MAX_STEPS="${4:-2}"
LOG_DIR="$ROOT_DIR/experiments/$EXPERIMENT_DATE/launcher_logs"
mkdir -p "$LOG_DIR"
LABEL="fastdp_bk_eps${EPSILON}_$(date +%Y%m%d_%H%M%S)"

nvidia-smi --id="$GPU" \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 1 >"$LOG_DIR/${LABEL}_gpu.csv" 2>&1 &
MONITOR_PID=$!
cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/vendor/fast-differential-privacy" \
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "$ROOT_DIR/src/train_fastdp.py" \
  --config "${CONFIG_PATH:-$ROOT_DIR/configs/methods_bf16.yaml}" \
  --target-epsilon "$EPSILON" \
  --gpu "$GPU" \
  --experiment-date "$EXPERIMENT_DATE" \
  --max-steps "$MAX_STEPS" 2>&1 | tee "$LOG_DIR/${LABEL}.log"
