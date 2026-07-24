#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT="${1:?Usage: generate_video_cuda_audited.sh PROMPT OUTPUT CHECKPOINT TRAINING_MANIFEST [FRAMES]}"
OUTPUT="${2:?output directory required}"
CHECKPOINT="${3:?checkpoint required}"
TRAINING_MANIFEST="${4:?training manifest required}"
FRAMES="${5:-16}"
[[ "${FRAMES}" =~ ^[1-9][0-9]*$ ]] || { echo "FRAMES must be a positive integer" >&2; exit 2; }
[[ ! -e "${OUTPUT}" ]] || { echo "output already exists" >&2; exit 1; }
EVIDENCE="${OUTPUT}.evidence"
[[ ! -e "${EVIDENCE}" ]] || { echo "evidence output already exists" >&2; exit 1; }

"${ROOT}/scripts/create_training_artifact_manifest.sh" --verify "${TRAINING_MANIFEST}" >/dev/null
READINESS_REPORT="$(awk -F '\t' '$1 == "readiness_report" { print $4 }' "${TRAINING_MANIFEST}")"
TRAINING_ENVIRONMENT="$(awk -F '\t' '$1 == "environment" { print $4 }' "${TRAINING_MANIFEST}")"
json_string() { sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"([^\"]*)\".*/\1/p" "$1" | head -n 1; }
json_number() { sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p" "$1" | head -n 1; }
PROFILE="$(json_string "${READINESS_REPORT}" profile)"
GPU_UUID="$(json_string "${READINESS_REPORT}" gpu_uuid)"
GPU_INDEX="$(json_number "${READINESS_REPORT}" gpu_index)"
[[ "${PROFILE}" == video-rtx-pro-6000-96g || "${PROFILE}" == video-v100-32g ]] || { echo "training artifact is not an approved video profile" >&2; exit 1; }
if [[ "${PROFILE}" == video-v100-32g ]]; then
  BIN="${ROOT}/build/ubuntu-general-cuda-compat/solstice"
else
  BIN="${ROOT}/build/ubuntu-rtx-pro-6000-cuda/solstice"
fi
[[ -x "${BIN}" ]] || { echo "profile-matched video executable missing" >&2; exit 1; }
TRAINING_BINARY_SHA256="$(awk -F= '$1 == "solstice_binary_sha256" { print $2 }' "${TRAINING_ENVIRONMENT}")"
[[ "${TRAINING_BINARY_SHA256}" =~ ^[0-9a-f]{64}$ ]] || { echo "training binary identity is missing" >&2; exit 1; }
[[ "$(sha256sum -- "${BIN}" | awk '{print $1}')" == "${TRAINING_BINARY_SHA256}" ]] || { echo "generation binary differs from training" >&2; exit 1; }
"${ROOT}/scripts/authorize_serving_checkpoint.sh" "${CHECKPOINT}" "${TRAINING_MANIFEST}" "${BIN}"

mkdir -p "${EVIDENCE}"
"${ROOT}/scripts/run_general_cuda_with_vram_guard.sh" \
  --profile "${PROFILE}" --gpu-index "${GPU_INDEX}" --expected-uuid "${GPU_UUID}" \
  --trace "${EVIDENCE}/generation_vram.csv" \
  --summary "${EVIDENCE}/generation_resource_summary.json" -- \
  "${BIN}" generate-video --checkpoint "${CHECKPOINT}" --profile "${PROFILE}" \
  --prompt "${PROMPT}" --frames "${FRAMES}" --output "${OUTPUT}" \
  | tee "${EVIDENCE}/generation_stdout.txt"
find "${EVIDENCE}" -maxdepth 1 -type f ! -name artifact_manifest.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${EVIDENCE}/artifact_manifest.sha256"
