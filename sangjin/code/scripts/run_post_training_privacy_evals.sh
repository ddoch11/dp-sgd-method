#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <training-pid> <gpu> <model-label> <adapter-path> <training-summary>" >&2
  exit 2
fi

TRAINING_PID="$1"
GPU="$2"
MODEL_LABEL="$3"
ADAPTER_PATH="$4"
TRAINING_SUMMARY="$5"

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARTICIPANT_ROOT="$(cd "$CODE_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/sangjin/anaconda3/envs/pytorch/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$PARTICIPANT_ROOT/configs/privacy_evaluation.yaml}"
MANIFEST_PATH="${MANIFEST_PATH:-$PARTICIPANT_ROOT/results/privacy_eval/synthetic_canary_manifest.json}"
CANARY_RUN_ID="${CANARY_RUN_ID:-canary_full_20260820}"
PREFIX_RUN_ID="${PREFIX_RUN_ID:-canonical_canary_models_20260820}"

while kill -0 "$TRAINING_PID" 2>/dev/null; do
  sleep 15
done

if [[ ! -f "$TRAINING_SUMMARY" ]]; then
  echo "training process ended without summary: $TRAINING_SUMMARY" >&2
  exit 1
fi
if [[ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ]]; then
  echo "training process ended without adapter: $ADAPTER_PATH" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
  "$CODE_ROOT/scripts/evaluate_synthetic_canary.py" \
  --config "$CONFIG_PATH" \
  --manifest "$MANIFEST_PATH" \
  --model-label "$MODEL_LABEL" \
  --adapter "$ADAPTER_PATH" \
  --gpu "$GPU" \
  --run-id "$CANARY_RUN_ID"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
  "$CODE_ROOT/scripts/evaluate_prefix_suffix.py" \
  --config "$CONFIG_PATH" \
  --selection shuffled \
  --exclude-canary-manifest "$MANIFEST_PATH" \
  --model-label "$MODEL_LABEL" \
  --adapter "$ADAPTER_PATH" \
  --gpu "$GPU" \
  --run-id "$PREFIX_RUN_ID"
