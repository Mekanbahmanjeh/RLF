#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build/ubuntu-h100-cuda/solstice"
LEDGER="${1:?Usage: train_general_h100_audited.sh LEDGER [CHECKPOINT] [RESULT_DIR] [READINESS_REPORT]}"
CHECKPOINT="${2:-${ROOT}/models/solstice_general_h100.rlfsp}"
RESULT_DIR="${3:-${ROOT}/results/codex_campaign/h100_training}"
READINESS_REPORT="${4:-${ROOT}/results/codex_campaign/h100_readiness/readiness.json}"

[[ -x "${BIN}" ]] || { echo "Build the H100 target first." >&2; exit 1; }
[[ -f "${LEDGER}" ]] || { echo "Data ledger not found: ${LEDGER}" >&2; exit 1; }
[[ -f "${READINESS_REPORT}" ]] || { echo "Readiness report not found: ${READINESS_REPORT}" >&2; exit 1; }
grep -Eq '"schema"[[:space:]]*:[[:space:]]*"rlf-general-h100-readiness-v1"' "${READINESS_REPORT}" || {
  echo "The readiness report has the wrong schema." >&2; exit 1;
}
grep -Eq '"ready"[[:space:]]*:[[:space:]]*true' "${READINESS_REPORT}" || {
  echo "A real ready=true H100 report is required before training." >&2; exit 1;
}
grep -Eq '"test_doubles"[[:space:]]*:[[:space:]]*false' "${READINESS_REPORT}" || {
  echo "A test-double readiness report cannot authorize training." >&2; exit 1;
}
grep -Eq '"profile"[[:space:]]*:[[:space:]]*"general-h100"' "${READINESS_REPORT}" || {
  echo "The readiness report does not authorize the general-h100 profile." >&2; exit 1;
}
mkdir -p "${RESULT_DIR}" "$(dirname "${CHECKPOINT}")"
export CUDA_MODULE_LOADING=LAZY
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
MAX_AUDIT_RECORDS="${RLF_MAX_AUDIT_RECORDS:-10000000}"
MAX_TEXT_SHARD_BYTES="${RLF_MAX_TEXT_SHARD_BYTES:-2147483648}"
MAX_TRAIN_SHARD_BYTES="${RLF_MAX_TRAIN_SHARD_BYTES:-4294967296}"
LEDGER_SHA256="$(sha256sum -- "${LEDGER}" | awk '{print $1}')"
for expected in \
  "ledger_sha256:\"${LEDGER_SHA256}\"" \
  "maximum_audit_records:${MAX_AUDIT_RECORDS}" \
  "maximum_text_shard_bytes:${MAX_TEXT_SHARD_BYTES}" \
  "maximum_train_shard_bytes:${MAX_TRAIN_SHARD_BYTES}"; do
  key="${expected%%:*}"
  value="${expected#*:}"
  grep -Eq "\"${key}\"[[:space:]]*:[[:space:]]*${value}([,[:space:]]|$)" \
    "${READINESS_REPORT}" || {
      echo "Readiness report does not authorize ${key}=${value}." >&2
      exit 1
    }
done

{
  printf 'schema=rlf-h100-training-environment-v1\n'
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uname -a
  cmake --version | head -n 1
  g++ --version | head -n 1
  nvcc --version | tail -n 1
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader,nounits
} >"${RESULT_DIR}/training_environment.txt"
"${ROOT}/scripts/create_source_manifest.sh" \
  "${RESULT_DIR}/source_manifest.tsv"

"${BIN}" audit-data --ledger "${LEDGER}" \
  --output "${RESULT_DIR}/data_audit.json" \
  --max-audit-records "${MAX_AUDIT_RECORDS}" \
  --max-text-shard-bytes "${MAX_TEXT_SHARD_BYTES}" \
  --max-train-shard-bytes "${MAX_TRAIN_SHARD_BYTES}"

"${ROOT}/scripts/run_h100_with_vram_guard.sh" \
  --trace "${RESULT_DIR}/training_vram.csv" \
  --summary "${RESULT_DIR}/training_resource_summary.json" -- \
  "${BIN}" train-data --ledger "${LEDGER}" --checkpoint "${CHECKPOINT}" \
    --profile general-h100 --backend cuda --blank --enforce-profile \
    --max-audit-records "${MAX_AUDIT_RECORDS}" \
    --max-text-shard-bytes "${MAX_TEXT_SHARD_BYTES}" \
    --max-train-shard-bytes "${MAX_TRAIN_SHARD_BYTES}" \
    --seed 6003100749088244549

"${BIN}" verify-checkpoint --checkpoint "${CHECKPOINT}" \
  --profile general-h100 --enforce-profile
"${BIN}" inspect-checkpoint --checkpoint "${CHECKPOINT}" \
  --profile general-h100 --enforce-profile \
  >"${RESULT_DIR}/checkpoint_inspection.txt"
sha256sum "${CHECKPOINT}" >"${RESULT_DIR}/checkpoint.sha256"
"${ROOT}/scripts/create_training_artifact_manifest.sh" \
  --checkpoint "${CHECKPOINT}" \
  --ledger "${LEDGER}" \
  --source-manifest "${RESULT_DIR}/source_manifest.tsv" \
  --data-audit "${RESULT_DIR}/data_audit.json" \
  --resource-summary "${RESULT_DIR}/training_resource_summary.json" \
  --vram-trace "${RESULT_DIR}/training_vram.csv" \
  --environment "${RESULT_DIR}/training_environment.txt" \
  --checkpoint-inspection "${RESULT_DIR}/checkpoint_inspection.txt" \
  --readiness-report "${READINESS_REPORT}" \
  --output "${RESULT_DIR}/training_artifact_manifest.tsv"
echo "Created and verified ${CHECKPOINT}"
