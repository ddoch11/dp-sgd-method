#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
RELEASE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd -P)"
ENV_PYTHON="${VAULTGEMMA_PYTHON:-python3}"
if [[ "${ENV_PYTHON}" != /* ]]; then
  ENV_PYTHON="$(command -v -- "${ENV_PYTHON}" || true)"
fi
[[ -n "${ENV_PYTHON}" && -x "${ENV_PYTHON}" ]] || {
  echo "Error: set VAULTGEMMA_PYTHON to an executable Python environment" >&2
  exit 3
}
PYTHON="${ENV_PYTHON}"
NOTEBOOK="${PROJECT_ROOT}/workspaces/vaultgemma_vectorized/[VaultGemma]FineTuning_Vectorized.ipynb"
PATCH_MANIFEST="${PROJECT_ROOT}/workspaces/vaultgemma_vectorized/workspace_patch_manifest.json"
STORAGE_ENV="${VAULTGEMMA_STORAGE_ENV:-${RELEASE_ROOT}/configs/storage.env}"
SEED=42
TOTAL_OPTIMIZER_STEPS=342
DRY_RUN=0
SMOKE=0
BENCHMARK=0
FLAG_MAX_UPDATES=""

usage() {
  echo "Usage: $0 [--dry-run] [--smoke|--benchmark] [--max-updates N] <hooks|functorch|ew|ghost> <0.5|2|8> <run_id> [max_updates]" >&2
}

fail_usage() {
  echo "Error: $*" >&2
  usage
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --benchmark) BENCHMARK=1; shift ;;
    --max-updates)
      [[ $# -ge 2 ]] || fail_usage "--max-updates requires a value"
      [[ -z "${FLAG_MAX_UPDATES}" ]] || fail_usage "--max-updates may be supplied only once"
      FLAG_MAX_UPDATES="$2"
      shift 2
      ;;
    --) shift; break ;;
    -*) fail_usage "unknown option: $1" ;;
    *) break ;;
  esac
done

[[ $# -ge 3 && $# -le 4 ]] || fail_usage "expected mode, epsilon, run_id, and optional max_updates"
MODE="$1"
EPSILON="$2"
RUN_ID="$3"
POSITIONAL_MAX_UPDATES="${4:-}"

case "${MODE}" in hooks|functorch|ew|ghost) ;; *) fail_usage "invalid mode: ${MODE}" ;; esac
case "${EPSILON}" in 0.5|2|8) ;; *) fail_usage "invalid epsilon: ${EPSILON}" ;; esac
[[ "${RUN_ID}" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || \
  fail_usage "run_id must match [a-z0-9][a-z0-9_-]{0,63}"
if [[ -n "${FLAG_MAX_UPDATES}" && -n "${POSITIONAL_MAX_UPDATES}" ]]; then
  fail_usage "max_updates cannot be supplied both positionally and by flag"
fi
MAX_UPDATES="${FLAG_MAX_UPDATES:-${POSITIONAL_MAX_UPDATES:-0}}"
[[ "${MAX_UPDATES}" =~ ^(0|[1-9][0-9]*)$ ]] || \
  fail_usage "max_updates must be 0 or a canonical positive integer"
(( SMOKE + BENCHMARK <= 1 )) || fail_usage "--smoke and --benchmark are mutually exclusive"
if (( SMOKE )); then
  [[ "${MAX_UPDATES}" != 0 ]] || fail_usage "--smoke requires max_updates greater than 0"
  [[ ${#MAX_UPDATES} -le 3 ]] || fail_usage "smoke max_updates must be in 1..342"
  (( 10#${MAX_UPDATES} <= TOTAL_OPTIMIZER_STEPS )) || \
    fail_usage "smoke max_updates must be in 1..342"
  RUN_KIND="smoke"
  NETWORK_POLICY="default"
elif (( BENCHMARK )); then
  [[ "${EPSILON}" == 2 ]] || fail_usage "benchmark target epsilon must be 2"
  [[ "${MAX_UPDATES}" == 15 ]] || fail_usage "benchmark max_updates must be exactly 15"
  RUN_KIND="benchmark"
  NETWORK_POLICY="offline_cache_only"
else
  [[ "${MAX_UPDATES}" == 0 ]] || fail_usage "non-zero max_updates requires explicit --smoke"
  RUN_KIND="full"
  NETWORK_POLICY="default"
fi

CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
[[ "${CUDA_DEVICE}" =~ ^[0-9]+$ ]] || \
  fail_usage "CUDA_VISIBLE_DEVICES must name exactly one numeric GPU"

TEST_MODE="${VAULTGEMMA_RUNNER_TEST_MODE:-}"
TEST_ROOT="${VAULTGEMMA_RUNNER_TEST_ROOT:-}"
TEST_PYTHON="${VAULTGEMMA_RUNNER_TEST_PYTHON:-}"
if [[ -n "${TEST_MODE}" || -n "${TEST_ROOT}" || -n "${TEST_PYTHON}" ]]; then
  [[ "${TEST_MODE}" == "contract" && -n "${TEST_ROOT}" && -n "${TEST_PYTHON}" ]] || \
    fail_usage "test overrides require explicit contract mode, root, and Python"
  [[ "${TEST_ROOT}" =~ ^/tmp/vaultgemma-task7-test\.[A-Za-z0-9_-]+$ ]] || \
    fail_usage "test root is outside the contract-test namespace"
  [[ -d /tmp && ! -L /tmp ]] || fail_usage "test root parent is unsafe"
  [[ -d "${TEST_ROOT}" && ! -L "${TEST_ROOT}" ]] || fail_usage "test root is unsafe"
  CANONICAL_TEST_ROOT="$(readlink -f -- "${TEST_ROOT}")"
  [[ "${TEST_ROOT}" == "${CANONICAL_TEST_ROOT}" ]] || fail_usage "test root is not canonical"
  [[ "$(stat -c '%u' "${TEST_ROOT}")" == "$(id -u)" ]] || fail_usage "test root is not owned"
  [[ "$(stat -c '%a' "${TEST_ROOT}")" == 700 ]] || fail_usage "test root is not private"
  TEST_SENTINEL="${TEST_ROOT}/.vaultgemma-runner-contract-test-owned"
  [[ -f "${TEST_SENTINEL}" && ! -L "${TEST_SENTINEL}" ]] || \
    fail_usage "test root ownership sentinel is absent or unsafe"
  [[ "$(stat -c '%u' "${TEST_SENTINEL}")" == "$(id -u)" ]] || \
    fail_usage "test root ownership sentinel is not owned"
  [[ "$(stat -c '%a:%h' "${TEST_SENTINEL}")" == "600:1" ]] || \
    fail_usage "test root ownership sentinel is not private"
  [[ -f "${TEST_PYTHON}" && ! -L "${TEST_PYTHON}" && -x "${TEST_PYTHON}" ]] || \
    fail_usage "test Python must be a regular executable"
  CANONICAL_TEST_PYTHON="$(readlink -f -- "${TEST_PYTHON}")"
  [[ "${TEST_PYTHON}" == "${CANONICAL_TEST_PYTHON}" ]] || \
    fail_usage "test Python is not canonical"
  TEST_PYTHON="${CANONICAL_TEST_PYTHON}"
  [[ "${TEST_PYTHON}" == "${TEST_ROOT}"/* ]] || fail_usage "test Python escaped the test root"
  PYTHON="${TEST_PYTHON}"
fi

# shellcheck source=/dev/null
[[ -f "${STORAGE_ENV}" && ! -L "${STORAGE_ENV}" ]] || {
  echo "Error: copy configs/storage.env.example to configs/storage.env and customize it" >&2
  exit 3
}
export VAULTGEMMA_CODE_ROOT="${PROJECT_ROOT}"
source "${STORAGE_ENV}"
[[ "${AI_SAFETY_PROJECT_ROOT:-}" == "${PROJECT_ROOT}" ]] || {
  echo "Error: storage.env project root does not match this runner" >&2
  exit 3
}
[[ "${AI_SAFETY_RAID_ROOT:-}" == /* ]] || {
  echo "Error: AI_SAFETY_RAID_ROOT must be absolute" >&2
  exit 3
}
for storage_path in "${AI_SAFETY_CHECKPOINT_DIR:-}" "${AI_SAFETY_RUN_DIR:-}"; do
  [[ "${storage_path}" == "${AI_SAFETY_RAID_ROOT}"/* ]] || {
    echo "Error: large run paths must remain under the RAID root" >&2
    exit 3
  }
done
[[ -x "${PYTHON}" ]] || { echo "Error: selected Python is unavailable" >&2; exit 3; }
[[ -f "${NOTEBOOK}" && ! -L "${NOTEBOOK}" ]] || {
  echo "Error: workspace notebook must be a regular non-symlink file" >&2
  exit 3
}
[[ -f "${PATCH_MANIFEST}" && ! -L "${PATCH_MANIFEST}" ]] || {
  echo "Error: workspace patch manifest must be a regular non-symlink file" >&2
  exit 3
}
if ! MANIFEST_NOTEBOOK_SHA256="$(
  jq -er '.current_notebook_sha256 | select(type == "string" and test("^[0-9a-f]{64}$"))' \
    "${PATCH_MANIFEST}" 2>/dev/null
)"; then
  echo "Error: workspace patch manifest has no valid current notebook hash" >&2
  exit 3
fi
WORKSPACE_NOTEBOOK_SHA256="$(sha256sum -- "${NOTEBOOK}")"
WORKSPACE_NOTEBOOK_SHA256="${WORKSPACE_NOTEBOOK_SHA256%% *}"
[[ "${WORKSPACE_NOTEBOOK_SHA256}" == "${MANIFEST_NOTEBOOK_SHA256}" ]] || {
  echo "Error: workspace notebook hash does not match the patch manifest" >&2
  exit 3
}
PATCH_MANIFEST_SHA256="$(sha256sum -- "${PATCH_MANIFEST}")"
PATCH_MANIFEST_SHA256="${PATCH_MANIFEST_SHA256%% *}"

if [[ "${TEST_MODE}" == "contract" ]]; then
  OUTPUT_ROOT="${TEST_ROOT}"
  CHECKPOINT_ROOT="${TEST_ROOT}/checkpoints"
  RUN_ROOT="${TEST_ROOT}/runs"
else
  OUTPUT_ROOT="${PROJECT_ROOT}"
  CHECKPOINT_ROOT="${AI_SAFETY_CHECKPOINT_DIR}"
  RUN_ROOT="${AI_SAFETY_RUN_DIR}"
fi
LOG_FILE="${OUTPUT_ROOT}/logs/vaultgemma/${RUN_ID}.log"
METRICS_FILE="${OUTPUT_ROOT}/results/vaultgemma/runs/${RUN_ID}.json"
COMPATIBILITY_FILE="${OUTPUT_ROOT}/results/vaultgemma/compatibility/${MODE}-epsilon-${EPSILON}/${RUN_ID}.json"
ADAPTER_DIR="${CHECKPOINT_ROOT}/vaultgemma_vectorized/${RUN_ID}"
EXECUTION_DIR="${RUN_ROOT}/vaultgemma_vectorized/${RUN_ID}"
INPUT_NOTEBOOK="${EXECUTION_DIR}/input.ipynb"
EXECUTED_NOTEBOOK="${EXECUTION_DIR}/executed.ipynb"

validate_existing_directory_chain() {
  local path="$1" current="/" component
  local -a components
  IFS='/' read -r -a components <<<"${path#/}"
  for component in "${components[@]}"; do
    [[ -n "${component}" ]] || continue
    current="${current%/}/${component}"
    if [[ -L "${current}" ]]; then
      echo "Error: refusing symlink directory component: ${current}" >&2
      return 1
    fi
    if [[ -e "${current}" && ! -d "${current}" ]]; then
      echo "Error: refusing non-directory path component: ${current}" >&2
      return 1
    fi
  done
}

for parent in \
  "${OUTPUT_ROOT}/logs/vaultgemma" \
  "${OUTPUT_ROOT}/results/vaultgemma/runs" \
  "${OUTPUT_ROOT}/results/vaultgemma/compatibility/${MODE}-epsilon-${EPSILON}" \
  "${CHECKPOINT_ROOT}/vaultgemma_vectorized" \
  "${RUN_ROOT}/vaultgemma_vectorized"; do
  validate_existing_directory_chain "${parent}"
done
for target in \
  "${LOG_FILE}" "${METRICS_FILE}" "${COMPATIBILITY_FILE}" \
  "${ADAPTER_DIR}" "${EXECUTION_DIR}" "${INPUT_NOTEBOOK}" "${EXECUTED_NOTEBOOK}"; do
  if [[ -e "${target}" || -L "${target}" ]]; then
    echo "Error: refusing to overwrite existing target: ${target}" >&2
    exit 3
  fi
done

print_plan() {
  printf '%s\n' \
    "run_id=${RUN_ID}" \
    "run_kind=${RUN_KIND}" \
    "network_policy=${NETWORK_POLICY}" \
    "grad_sample_mode=${MODE}" \
    "target_epsilon=${EPSILON}" \
    "max_updates=${MAX_UPDATES}" \
    "seed=${SEED}" \
    "cuda_visible_devices=${CUDA_DEVICE}" \
    "compatibility_attempt_id=${RUN_ID}" \
    "python=${PYTHON}" \
    "source_notebook=${NOTEBOOK}" \
    "workspace_notebook_sha256=${WORKSPACE_NOTEBOOK_SHA256}" \
    "log_path=${LOG_FILE}" \
    "metrics_path=${METRICS_FILE}" \
    "compatibility_path=${COMPATIBILITY_FILE}" \
    "adapter_path=${ADAPTER_DIR}" \
    "executed_notebook=${EXECUTED_NOTEBOOK}"
}

if (( DRY_RUN )); then
  print_plan
  exit 0
fi

mkdir -p \
  "${OUTPUT_ROOT}/logs/vaultgemma" \
  "${OUTPUT_ROOT}/results/vaultgemma/runs" \
  "${CHECKPOINT_ROOT}/vaultgemma_vectorized" \
  "${RUN_ROOT}/vaultgemma_vectorized"
for parent in \
  "${OUTPUT_ROOT}/logs/vaultgemma" \
  "${OUTPUT_ROOT}/results/vaultgemma/runs" \
  "${CHECKPOINT_ROOT}/vaultgemma_vectorized" \
  "${RUN_ROOT}/vaultgemma_vectorized"; do
  validate_existing_directory_chain "${parent}"
done

OWNED_ADAPTER=0
OWNED_EXECUTION=0
OWNED_LOG=0
RUN_STARTED=0
INPUT_TEMP=""
cleanup_unstarted_claims() {
  if (( RUN_STARTED == 0 )); then
    (( OWNED_LOG == 0 )) || rm -f "${LOG_FILE}"
    [[ -z "${INPUT_TEMP}" ]] || rm -f -- "${INPUT_TEMP}"
    (( OWNED_EXECUTION == 0 )) || rm -f "${INPUT_NOTEBOOK}" "${EXECUTED_NOTEBOOK}"
    (( OWNED_EXECUTION == 0 )) || rmdir "${EXECUTION_DIR}" 2>/dev/null || true
    (( OWNED_ADAPTER == 0 )) || rmdir "${ADAPTER_DIR}" 2>/dev/null || true
  fi
}
trap cleanup_unstarted_claims EXIT

mkdir "${ADAPTER_DIR}"
OWNED_ADAPTER=1
mkdir "${EXECUTION_DIR}"
OWNED_EXECUTION=1
INPUT_TEMP="$(mktemp "${EXECUTION_DIR}/.input.ipynb.XXXXXX")"
cp --preserve=mode,timestamps -- "${NOTEBOOK}" "${INPUT_TEMP}"
COPIED_NOTEBOOK_SHA256="$(sha256sum -- "${INPUT_TEMP}")"
COPIED_NOTEBOOK_SHA256="${COPIED_NOTEBOOK_SHA256%% *}"
[[ "${COPIED_NOTEBOOK_SHA256}" == "${WORKSPACE_NOTEBOOK_SHA256}" ]] || {
  echo "Error: copied notebook hash differs from the validated workspace notebook" >&2
  exit 3
}
ln -- "${INPUT_TEMP}" "${INPUT_NOTEBOOK}"
rm -f -- "${INPUT_TEMP}"
INPUT_TEMP=""
EXECUTED_INPUT_SHA256="$(sha256sum -- "${INPUT_NOTEBOOK}")"
EXECUTED_INPUT_SHA256="${EXECUTED_INPUT_SHA256%% *}"
[[ "${EXECUTED_INPUT_SHA256}" == "${WORKSPACE_NOTEBOOK_SHA256}" ]] || {
  echo "Error: published input notebook hash differs from the validated workspace notebook" >&2
  exit 3
}
set -o noclobber
exec 9>"${LOG_FILE}"
set +o noclobber
OWNED_LOG=1

print_plan
RUN_STARTED=1
export PATH="$(dirname "${PYTHON}"):${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export NVIDIA_TF32_OVERRIDE=0
export TOKENIZERS_PARALLELISM=false
export AI_SAFETY_SEED="${SEED}"
export AI_SAFETY_GRAD_SAMPLE_MODE="${MODE}"
export VAULTGEMMA_TARGET_EPSILON="${EPSILON}"
export AI_SAFETY_COMPATIBILITY_ATTEMPT_ID="${RUN_ID}"
export AI_SAFETY_RUN_ID="${RUN_ID}"
export AI_SAFETY_METRICS_FILE="${METRICS_FILE}"
export AI_SAFETY_ADAPTER_DIR="${ADAPTER_DIR}"
export AI_SAFETY_MAX_UPDATES="${MAX_UPDATES}"
export AI_SAFETY_RUN_KIND="${RUN_KIND}"
export AI_SAFETY_EXECUTED_INPUT_SHA256="${EXECUTED_INPUT_SHA256}"
export VAULTGEMMA_GPU_INDEX=0
if [[ "${RUN_KIND}" == "benchmark" ]]; then
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

publish_failure_metrics() {
  [[ ! -e "${METRICS_FILE}" && ! -L "${METRICS_FILE}" ]] || return 0
  "${ENV_PYTHON}" - \
    "${METRICS_FILE}" "${RUN_ID}" "${RUN_KIND}" "${SEED}" "${EPSILON}" \
    "${MAX_UPDATES}" "${MODE}" "${WORKSPACE_NOTEBOOK_SHA256}" \
    "${PATCH_MANIFEST_SHA256}" "${EXECUTED_INPUT_SHA256}" \
    "${LOG_FILE}" "${COMPATIBILITY_FILE}" "${ADAPTER_DIR}" \
    "${INPUT_NOTEBOOK}" "${EXECUTED_NOTEBOOK}" "$1" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

(
    metrics_raw,
    run_id,
    run_kind,
    seed_raw,
    epsilon_raw,
    max_updates_raw,
    requested_mode,
    notebook_sha256,
    manifest_sha256,
    executed_input_sha256,
    log_path,
    compatibility_path,
    adapter_path,
    input_notebook,
    executed_notebook,
    exit_code_raw,
) = sys.argv[1:]
metrics_path = pathlib.Path(metrics_raw)
payload = {
    "schema_version": 1,
    "run_id": run_id,
    "status": "failed",
    "configuration": {
        "run_kind": run_kind,
        "seed": int(seed_raw),
        "target_epsilon": float(epsilon_raw),
        "max_updates": int(max_updates_raw),
    },
    "upstream_hashes": {
        "workspace_notebook_sha256": notebook_sha256,
        "workspace_patch_manifest_sha256": manifest_sha256,
        "executed_input_sha256": executed_input_sha256,
    },
    "requested_mode": requested_mode,
    "run_paths": {
        "log": log_path,
        "metrics": str(metrics_path),
        "compatibility": compatibility_path,
        "adapter": adapter_path,
        "input_notebook": input_notebook,
        "executed_notebook": executed_notebook,
    },
    "failure": {
        "stage": "notebook_execution",
        "type": "NotebookExecutionError",
        "message": "Notebook execution failed; inspect the run log.",
        "exit_code": int(exit_code_raw),
    },
}
encoded = (
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    + "\n"
).encode("utf-8")
temporary_fd, temporary_raw = tempfile.mkstemp(
    prefix=f".{metrics_path.name}.", dir=metrics_path.parent
)
temporary_path = pathlib.Path(temporary_raw)
try:
    with os.fdopen(temporary_fd, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary_path, metrics_path)
    except FileExistsError:
        pass
    else:
        directory_fd = os.open(metrics_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
finally:
    temporary_path.unlink(missing_ok=True)
PY
}

# Direct environment contract: python -m jupyter nbconvert; log contract: tee "${LOG_FILE}".
set +e
"${PYTHON}" -m jupyter nbconvert \
  --to notebook \
  --execute "${INPUT_NOTEBOOK}" \
  --output "$(basename "${EXECUTED_NOTEBOOK}")" \
  --output-dir "${EXECUTION_DIR}" \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=-1 \
  --TagRemovePreprocessor.enabled=True \
  --TagRemovePreprocessor.remove_cell_tags='["skip-runner-execution"]' \
  2>&1 | tee -a /dev/fd/9
pipeline_status=("${PIPESTATUS[@]}")
set -e
notebook_status="${pipeline_status[0]}"
tee_status="${pipeline_status[1]}"
if (( notebook_status != 0 )); then
  publish_failure_metrics "${notebook_status}" 2>/dev/null || \
    echo "Error: failed to publish redacted notebook failure metrics" >&2
  exit "${notebook_status}"
fi
if (( tee_status != 0 )); then
  exit "${tee_status}"
fi
