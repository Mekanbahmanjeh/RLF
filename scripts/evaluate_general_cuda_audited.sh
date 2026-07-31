#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQUESTS="${1:?Usage: evaluate_general_cuda_audited.sh REQUESTS OUTPUT CHECKPOINT TRAINING_MANIFEST}"
OUTPUT="${2:?output directory required}"; CHECKPOINT="${3:?checkpoint required}"; TRAINING_MANIFEST="${4:?training manifest required}"
[[ -f "${REQUESTS}" ]] || { echo "request manifest missing" >&2; exit 1; }
"${ROOT}/scripts/create_training_artifact_manifest.sh" --verify "${TRAINING_MANIFEST}" >/dev/null
READINESS_REPORT="$(awk -F '\t' '$1 == "readiness_report" { print $4 }' "${TRAINING_MANIFEST}")"
TRAINING_ENVIRONMENT="$(awk -F '\t' '$1 == "environment" { print $4 }' "${TRAINING_MANIFEST}")"
json_string() { sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"([^\"]*)\".*/\1/p" "$1" | head -n 1; }
json_number() { sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p" "$1" | head -n 1; }
PROFILE="$(json_string "${READINESS_REPORT}" profile)"; GPU_UUID="$(json_string "${READINESS_REPORT}" gpu_uuid)"; GPU_INDEX="$(json_number "${READINESS_REPORT}" gpu_index)"
if [[ "${PROFILE}" == *rtx-pro-6000-96g* ]]; then BIN="${ROOT}/build/ubuntu-rtx-pro-6000-cuda/solstice"; else BIN="${ROOT}/build/ubuntu-general-cuda-compat/solstice"; fi
[[ -x "${BIN}" ]] || { echo "target executable missing for ${PROFILE}" >&2; exit 1; }
TRAINING_BINARY_SHA256="$(awk -F= '$1 == "solstice_binary_sha256" { print $2 }' "${TRAINING_ENVIRONMENT}")"
[[ "${TRAINING_BINARY_SHA256}" =~ ^[0-9a-f]{64}$ ]] || { echo "training binary identity is missing" >&2; exit 1; }
[[ "$(sha256sum -- "${BIN}" | awk '{print $1}')" == "${TRAINING_BINARY_SHA256}" ]] || { echo "evaluation binary differs from the training binary" >&2; exit 1; }
"${ROOT}/scripts/authorize_serving_checkpoint.sh" "${CHECKPOINT}" "${TRAINING_MANIFEST}" "${BIN}"
mkdir -p "${OUTPUT}"
ARGS=(evaluate-batch --checkpoint "${CHECKPOINT}" --manifest "${REQUESTS}" --output "${OUTPUT}" --profile "${PROFILE}" --enforce-profile --backend cuda --max-tokens "${RLF_EVAL_MAX_TOKENS:-4096}" --top-k "${RLF_EVAL_TOP_K:-64}" --temperature "${RLF_EVAL_TEMPERATURE:-0.8}" --seed "${RLF_EVAL_SEED:-6003100749088244549}")
if [[ "${RLF_EVAL_ENABLE_TOOLS:-0}" == 1 ]]; then ARGS+=(--enable-tools); [[ -n "${RLF_EVAL_TOOL_ROOT:-}" ]] && ARGS+=(--tool-root "${RLF_EVAL_TOOL_ROOT}"); fi
"${ROOT}/scripts/run_general_cuda_with_vram_guard.sh" --profile "${PROFILE}" --gpu-index "${GPU_INDEX}" --expected-uuid "${GPU_UUID}" --trace "${OUTPUT}/evaluation_vram.csv" --summary "${OUTPUT}/evaluation_resource_summary.json" -- "${BIN}" "${ARGS[@]}" | tee "${OUTPUT}/evaluation_stdout.json"
find "${OUTPUT}" -maxdepth 1 -type f ! -name artifact_manifest.sha256 -print0 | sort -z | xargs -0 sha256sum >"${OUTPUT}/artifact_manifest.sha256"
