#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build/ubuntu-h100-cuda/solstice"
MANIFEST="${1:?Usage: evaluate_general_h100_audited.sh MANIFEST OUTPUT_DIR [CHECKPOINT]}"
OUTPUT_DIR="${2:?Usage: evaluate_general_h100_audited.sh MANIFEST OUTPUT_DIR [CHECKPOINT]}"
CHECKPOINT="${3:-${ROOT}/models/solstice_general_h100.rlfsp}"
TRAINING_MANIFEST="${4:-${ROOT}/results/codex_campaign/h100_training/training_artifact_manifest.tsv}"

[[ -x "${BIN}" ]] || { echo "Build the H100 target first." >&2; exit 1; }
[[ -f "${MANIFEST}" ]] || { echo "Evaluation manifest not found: ${MANIFEST}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "Checkpoint not found: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${TRAINING_MANIFEST}" ]] || { echo "Training manifest not found: ${TRAINING_MANIFEST}" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}"
export CUDA_MODULE_LOADING=LAZY
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"

"${ROOT}/scripts/authorize_serving_checkpoint.sh" \
  "${CHECKPOINT}" "${TRAINING_MANIFEST}" "${BIN}" \
  | tee "${OUTPUT_DIR}/evaluation_authorization.txt"
{
  printf 'kind\tsha256\tpath\n'
  printf 'checkpoint\t%s\t%s\n' \
    "$(sha256sum -- "${CHECKPOINT}" | awk '{print $1}')" "$(realpath "${CHECKPOINT}")"
  printf 'training_manifest\t%s\t%s\n' \
    "$(sha256sum -- "${TRAINING_MANIFEST}" | awk '{print $1}')" "$(realpath "${TRAINING_MANIFEST}")"
} >"${OUTPUT_DIR}/evaluation_identity.tsv"

EVALUATION_ARGUMENTS=(
  evaluate-batch
  --checkpoint "${CHECKPOINT}"
  --manifest "${MANIFEST}"
  --output "${OUTPUT_DIR}"
  --profile general-h100
  --backend cuda
  --max-tokens "${RLF_EVAL_MAX_TOKENS:-4096}"
  --top-k "${RLF_EVAL_TOP_K:-64}"
  --temperature "${RLF_EVAL_TEMPERATURE:-0.8}"
  --seed "${RLF_EVAL_SEED:-6003100749088244549}"
)
if [[ "${RLF_EVAL_ENABLE_TOOLS:-0}" == "1" ]]; then
  EVALUATION_ARGUMENTS+=(--enable-tools)
  if [[ -n "${RLF_EVAL_TOOL_ROOT:-}" ]]; then
    EVALUATION_ARGUMENTS+=(--tool-root "${RLF_EVAL_TOOL_ROOT}")
  fi
fi

"${ROOT}/scripts/run_h100_with_vram_guard.sh" \
  --trace "${OUTPUT_DIR}/evaluation_vram.csv" \
  --summary "${OUTPUT_DIR}/evaluation_resource_summary.json" -- \
  "${BIN}" "${EVALUATION_ARGUMENTS[@]}" \
  | tee "${OUTPUT_DIR}/evaluation_stdout.json"

find "${OUTPUT_DIR}" -maxdepth 1 -type f \
  ! -name artifact_manifest.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${OUTPUT_DIR}/artifact_manifest.sha256"
echo "Evaluation artifacts: ${OUTPUT_DIR}"
