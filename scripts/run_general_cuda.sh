#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:?Usage: run_general_cuda.sh CHECKPOINT TRAINING_MANIFEST [chat options]}"
TRAINING_MANIFEST="${2:?Usage: run_general_cuda.sh CHECKPOINT TRAINING_MANIFEST [chat options]}"
shift 2
"${ROOT}/scripts/create_training_artifact_manifest.sh" --verify "${TRAINING_MANIFEST}" >/dev/null
READINESS_REPORT="$(awk -F '\t' '$1 == "readiness_report" { print $4 }' "${TRAINING_MANIFEST}")"
TRAINING_ENVIRONMENT="$(awk -F '\t' '$1 == "environment" { print $4 }' "${TRAINING_MANIFEST}")"
[[ -f "${READINESS_REPORT}" ]] || { echo "manifest readiness report is missing" >&2; exit 1; }
json_string() { sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"([^\"]*)\".*/\1/p" "$1" | head -n 1; }
json_number() { sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p" "$1" | head -n 1; }
PROFILE="$(json_string "${READINESS_REPORT}" profile)"; GPU_UUID="$(json_string "${READINESS_REPORT}" gpu_uuid)"; GPU_INDEX="$(json_number "${READINESS_REPORT}" gpu_index)"
if [[ "${PROFILE}" == *rtx-pro-6000-96g* ]]; then BIN="${ROOT}/build/ubuntu-rtx-pro-6000-cuda/solstice"; else BIN="${ROOT}/build/ubuntu-general-cuda-compat/solstice"; fi
[[ -x "${BIN}" ]] || { echo "target executable missing for ${PROFILE}" >&2; exit 1; }
TRAINING_BINARY_SHA256="$(awk -F= '$1 == "solstice_binary_sha256" { print $2 }' "${TRAINING_ENVIRONMENT}")"
[[ "${TRAINING_BINARY_SHA256}" =~ ^[0-9a-f]{64}$ ]] || { echo "training binary identity is missing" >&2; exit 1; }
[[ "$(sha256sum -- "${BIN}" | awk '{print $1}')" == "${TRAINING_BINARY_SHA256}" ]] || { echo "serving binary differs from the training binary" >&2; exit 1; }
"${ROOT}/scripts/authorize_serving_checkpoint.sh" "${CHECKPOINT}" "${TRAINING_MANIFEST}" "${BIN}"
RESULT_DIR="${RLF_SERVE_RESULTS:-${ROOT}/results/codex_campaign/${PROFILE}_serving}"; mkdir -p "${RESULT_DIR}"
{
  printf 'kind\tsha256\tpath\ncheckpoint\t%s\t%s\ntraining_manifest\t%s\t%s\nprofile\t-\t%s\n' \
    "$(sha256sum -- "${CHECKPOINT}" | awk '{print $1}')" "$(realpath "${CHECKPOINT}")" \
    "$(sha256sum -- "${TRAINING_MANIFEST}" | awk '{print $1}')" "$(realpath "${TRAINING_MANIFEST}")" "${PROFILE}"
} >"${RESULT_DIR}/serving_identity.tsv"
exec "${ROOT}/scripts/run_general_cuda_with_vram_guard.sh" --profile "${PROFILE}" --gpu-index "${GPU_INDEX}" --expected-uuid "${GPU_UUID}" --trace "${RESULT_DIR}/serving_vram.csv" --summary "${RESULT_DIR}/serving_resource_summary.json" -- \
  "${BIN}" chat --checkpoint "${CHECKPOINT}" --profile "${PROFILE}" --enforce-profile --backend cuda --max-tokens 1024 --top-k 64 --temperature 0.75 "$@"
