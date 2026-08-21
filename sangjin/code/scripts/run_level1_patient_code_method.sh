#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/sangjin/anaconda3/envs/pytorch/bin/python}"
METHOD="${1:?method required: naive_dp|hooks_dp|vmap_dp|expanded_weights_dp|ghost_dp|fastdp_bk}"
GPU="${2:?gpu required}"
RUN_ID="${3:?run id required}"
MAX_STEPS="${4:-2}"
EPSILON="${EPSILON:-2}"
EPOCHS="${EPOCHS:-40}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/../configs/level1_patient_code_bf16.yaml}"
MANIFEST_PATH="${MANIFEST_PATH:-$ROOT_DIR/../results/level1_patient_code/level1_patient_codes_manifest.json}"
LOG_ROOT="$ROOT_DIR/../results/level1_patient_code/runs/launcher_logs/$RUN_ID"
mkdir -p "$LOG_ROOT"

case "$METHOD" in
  naive_dp|hooks_dp|vmap_dp|expanded_weights_dp|ghost_dp|fastdp_bk) ;;
  *) echo "unsupported method: $METHOD" >&2; exit 2 ;;
esac

CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
  "$ROOT_DIR/scripts/train_level1_patient_code_methods.py" \
  --config "$CONFIG_PATH" \
  --manifest "$MANIFEST_PATH" \
  --method "$METHOD" \
  --target-epsilon "$EPSILON" \
  --epochs "$EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --gpu "$GPU" \
  --run-id "$RUN_ID" \
  2>&1 | tee "$LOG_ROOT/${METHOD}_eps${EPSILON}.log"
