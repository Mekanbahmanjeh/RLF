#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEDGER="${1:?Usage: train_general_h200_audited.sh LEDGER CHECKPOINT RESULT_DIR READINESS_REPORT TARGET_TOKENS}"
CHECKPOINT="${2:?checkpoint required}"
RESULT_DIR="${3:?result directory required}"
READINESS_REPORT="${4:?readiness report required}"
TARGET_TOKENS="${5:?exact cumulative tokenizer-piece target required}"
BIN="${ROOT}/build/ubuntu-h200-cuda/solstice"

[[ "${TARGET_TOKENS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "TARGET_TOKENS must be a positive integer" >&2; exit 2;
}
case "${TARGET_TOKENS}" in
  1000000000000|5000000000000|15000000000000|30000000000000) ;;
  *) echo "unsupported H200 campaign token target" >&2; exit 2 ;;
esac
[[ -x "${BIN}" && -f "${LEDGER}" && -f "${READINESS_REPORT}" ]] || {
  echo "H200 binary, ledger, and readiness report are required" >&2; exit 1;
}
grep -Eq '"schema"[[:space:]]*:[[:space:]]*"rlf-general-h200-readiness-v1"' \
  "${READINESS_REPORT}" &&
grep -Eq '"ready"[[:space:]]*:[[:space:]]*true' "${READINESS_REPORT}" &&
grep -Eq '"test_doubles"[[:space:]]*:[[:space:]]*false' "${READINESS_REPORT}" &&
grep -Eq '"profile"[[:space:]]*:[[:space:]]*"general-h200-141g-30t"' \
  "${READINESS_REPORT}" || {
    echo "physical H200 readiness is required" >&2; exit 1;
  }

mkdir -p "${RESULT_DIR}" "$(dirname "${CHECKPOINT}")"
MAX_AUDIT_RECORDS="${RLF_MAX_AUDIT_RECORDS:-10000000}"
MAX_TEXT_SHARD_BYTES="${RLF_MAX_TEXT_SHARD_BYTES:-2147483648}"
MAX_TRAIN_SHARD_BYTES="${RLF_MAX_TRAIN_SHARD_BYTES:-4294967296}"
export CUDA_MODULE_LOADING=LAZY
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"

"${BIN}" audit-data --ledger "${LEDGER}" \
  --output "${RESULT_DIR}/data_audit.json" --require-media-hashes \
  --max-audit-records "${MAX_AUDIT_RECORDS}" \
  --max-text-shard-bytes "${MAX_TEXT_SHARD_BYTES}" \
  --max-train-shard-bytes "${MAX_TRAIN_SHARD_BYTES}"
bash "${ROOT}/scripts/create_source_manifest.sh" "${RESULT_DIR}/source_manifest.tsv"

bash "${ROOT}/scripts/run_h200_with_vram_guard.sh" \
  --trace "${RESULT_DIR}/training_vram.csv" \
  --summary "${RESULT_DIR}/training_resource_summary.json" -- \
  "${BIN}" train-data --ledger "${LEDGER}" --checkpoint "${CHECKPOINT}" \
    --profile general-h200-141g-30t --backend cuda --blank --enforce-profile \
    --require-media-hashes --target-training-tokens "${TARGET_TOKENS}" \
    --max-audit-records "${MAX_AUDIT_RECORDS}" \
    --max-text-shard-bytes "${MAX_TEXT_SHARD_BYTES}" \
    --max-train-shard-bytes "${MAX_TRAIN_SHARD_BYTES}" \
    --telemetry "${RESULT_DIR}/training_pipeline_telemetry.json" \
    --seed 6003100749088244549

"${BIN}" verify-checkpoint --checkpoint "${CHECKPOINT}" \
  --profile general-h200-141g-30t --enforce-profile
"${BIN}" inspect-checkpoint --checkpoint "${CHECKPOINT}" \
  --profile general-h200-141g-30t --enforce-profile \
  >"${RESULT_DIR}/checkpoint_inspection.txt"
grep -Eq "^language_tokens_seen=${TARGET_TOKENS}$" \
  "${RESULT_DIR}/checkpoint_inspection.txt" || {
    echo "checkpoint did not reach the exact token target" >&2; exit 1;
  }

{
  echo schema=rlf-h200-training-environment-v1
  echo profile=general-h200-141g-30t
  echo target_training_tokens=${TARGET_TOKENS}
  echo timestamp_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uname -a
  nvcc --version | tail -n 1
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader,nounits
} >"${RESULT_DIR}/training_environment.txt"

bash "${ROOT}/scripts/create_training_artifact_manifest.sh" \
  --checkpoint "${CHECKPOINT}" --ledger "${LEDGER}" \
  --source-manifest "${RESULT_DIR}/source_manifest.tsv" \
  --data-audit "${RESULT_DIR}/data_audit.json" \
  --resource-summary "${RESULT_DIR}/training_resource_summary.json" \
  --vram-trace "${RESULT_DIR}/training_vram.csv" \
  --environment "${RESULT_DIR}/training_environment.txt" \
  --checkpoint-inspection "${RESULT_DIR}/checkpoint_inspection.txt" \
  --readiness-report "${READINESS_REPORT}" \
  --output "${RESULT_DIR}/training_artifact_manifest.tsv"
echo "Created exact-token H200 checkpoint ${CHECKPOINT}"
