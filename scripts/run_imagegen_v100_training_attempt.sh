#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?Usage: run_imagegen_v100_training_attempt.sh ROOT BIN CHECKPOINT PAIR_MANIFEST RESULT_DIR GPU_INDEX GPU_UUID SEED}"
BIN="${2:?binary required}"
CHECKPOINT="${3:?checkpoint required}"
PAIR_MANIFEST="${4:?pair manifest required}"
RESULT_DIR="${5:?result directory required}"
GPU_INDEX="${6:?GPU index required}"
GPU_UUID="${7:?GPU UUID required}"
SEED="${8:?seed required}"
PROFILE=imagegen-v100-32g
MANIFEST="${RESULT_DIR}/artifact_manifest.tsv"
SIDECAR="${MANIFEST}.sha256"

cleanup_failed_finalization() {
  local status=$?
  if ((status != 0)); then
    rm -f -- "${MANIFEST}" "${SIDECAR}"
  fi
  trap - EXIT
  exit "${status}"
}
trap cleanup_failed_finalization EXIT

if [[ ! -f "${CHECKPOINT}" ]]; then
  "${BIN}" imagegen-bootstrap --profile "${PROFILE}" \
    --checkpoint "${CHECKPOINT}" --seed "${SEED}"
fi

"${ROOT}/scripts/run_general_cuda_with_vram_guard.sh" \
  --profile "${PROFILE}" --gpu-index "${GPU_INDEX}" \
  --expected-uuid "${GPU_UUID}" --trace "${RESULT_DIR}/raw_gpu_trace.csv" \
  --summary "${RESULT_DIR}/resource_summary.json" -- \
  "${BIN}" imagegen-train-manifest --profile "${PROFILE}" \
    --backend cuda --checkpoint "${CHECKPOINT}" --manifest "${PAIR_MANIFEST}" \
  >"${RESULT_DIR}/training_telemetry.txt" 2>&1

grep -Eq '^backend_device_local_update_calls=[1-9][0-9]*$' \
  "${RESULT_DIR}/training_telemetry.txt" || {
  echo "physical image-generation shard did not execute CUDA local updates" >&2
  exit 1
}
"${BIN}" imagegen-verify --profile "${PROFILE}" --checkpoint "${CHECKPOINT}" >/dev/null
"${BIN}" imagegen-inspect --checkpoint "${CHECKPOINT}" \
  >"${RESULT_DIR}/checkpoint_inspection.txt"
"${ROOT}/scripts/create_imagegen_artifact_manifest.sh" "${RESULT_DIR}" >/dev/null
"${BIN}" imagegen-verify-artifacts --manifest "${MANIFEST}" >/dev/null

trap - EXIT
