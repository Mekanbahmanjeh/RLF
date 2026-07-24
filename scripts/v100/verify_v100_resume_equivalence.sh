#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: verify_v100_resume_equivalence.sh --ledger FILE --target-records N \
  --readiness-report FILE --output DIR [--gpu-index N]

Run only through the controller's physical-throughput-probe stage. The ledger
must contain at least two immutable training shards and exactly N training
records. This performs bounded physical CUDA training twice: uninterrupted,
then an intentional post-first-shard stop plus resume. Success requires byte-
identical final checkpoints and <=30 GiB in every raw VRAM trace. It is
resumability evidence, never frontier or efficiency evidence.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE=general-v100-32g-500m
LEDGER=""
TARGET_RECORDS=""
READINESS_REPORT=""
OUTPUT=""
GPU_INDEX=""
while (($# > 0)); do
  case "$1" in
    --ledger) LEDGER="${2:?--ledger requires a file}"; shift 2 ;;
    --target-records) TARGET_RECORDS="${2:?--target-records requires a value}"; shift 2 ;;
    --readiness-report) READINESS_REPORT="${2:?--readiness-report requires a file}"; shift 2 ;;
    --output) OUTPUT="${2:?--output requires a directory}"; shift 2 ;;
    --gpu-index) GPU_INDEX="${2:?--gpu-index requires an integer}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "${LEDGER}" && ! -L "${LEDGER}" && -f "${READINESS_REPORT}" && ! -L "${READINESS_REPORT}" ]] || {
  echo "regular ledger and readiness report are required" >&2; exit 2;
}
[[ "${TARGET_RECORDS}" =~ ^[1-9][0-9]*$ && -n "${OUTPUT}" ]] || {
  echo "positive --target-records and --output are required" >&2; exit 2;
}
[[ "${RLF_CAMPAIGN_STAGE:-}" == physical-throughput-probe && \
   "${RLF_CAMPAIGN_PROFILE:-}" == "${PROFILE}" && \
   "${RLF_TRAINING_AUTHORIZED:-0}" == 0 && \
   "${RLF_CAMPAIGN_BINDING_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "resume drill must run inside the bound physical-throughput-probe stage" >&2
  exit 1
}
[[ "${RLF_CAMPAIGN_STATE_DIR:-}" == /* && \
   -f "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" && \
   ! -L "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" ]] || {
  echo "regular absolute V100 campaign state is required" >&2; exit 1;
}
grep -Fqx 'schema=rlf-v100-1586h-campaign-v1' "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" || { echo "wrong campaign schema" >&2; exit 1; }
grep -Fqx "profile=${PROFILE}" "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" || { echo "campaign profile mismatch" >&2; exit 1; }
grep -Fqx "binding_sha256=${RLF_CAMPAIGN_BINDING_SHA256}" "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" || { echo "campaign binding mismatch" >&2; exit 1; }

json_string() { sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"([^\"]*)\".*/\1/p" "$1" | head -n 1; }
json_number() { sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p" "$1" | head -n 1; }
grep -Eq '"schema"[[:space:]]*:[[:space:]]*"rlf-general-cuda-readiness-v1"' "${READINESS_REPORT}" || { echo "wrong readiness schema" >&2; exit 1; }
grep -Eq '"ready"[[:space:]]*:[[:space:]]*true' "${READINESS_REPORT}" || { echo "physical ready=true report required" >&2; exit 1; }
grep -Eq '"test_doubles"[[:space:]]*:[[:space:]]*false' "${READINESS_REPORT}" || { echo "test-double readiness is forbidden" >&2; exit 1; }
[[ "$(json_string "${READINESS_REPORT}" profile)" == "${PROFILE}" ]] || { echo "readiness profile mismatch" >&2; exit 1; }
LEDGER="$(realpath "${LEDGER}")"
READINESS_REPORT="$(realpath "${READINESS_REPORT}")"
LEDGER_SHA256="$(sha256sum -- "${LEDGER}" | awk '{print $1}')"
[[ "$(json_string "${READINESS_REPORT}" ledger_sha256)" == "${LEDGER_SHA256}" ]] || { echo "readiness ledger mismatch" >&2; exit 1; }
MAX_AUDIT_RECORDS="$(json_number "${READINESS_REPORT}" maximum_audit_records)"
MAX_TEXT_SHARD_BYTES="$(json_number "${READINESS_REPORT}" maximum_text_shard_bytes)"
MAX_TRAIN_SHARD_BYTES="$(json_number "${READINESS_REPORT}" maximum_train_shard_bytes)"
[[ "${MAX_AUDIT_RECORDS}" =~ ^[1-9][0-9]*$ && \
   "${MAX_TEXT_SHARD_BYTES}" =~ ^[1-9][0-9]*$ && \
   "${MAX_TRAIN_SHARD_BYTES}" =~ ^[1-9][0-9]*$ && \
   "${MAX_AUDIT_RECORDS}" -ge "${TARGET_RECORDS}" ]] || {
  echo "readiness audit limits do not cover the resume-drill target" >&2; exit 1;
}

BIN="${ROOT}/build/ubuntu-general-cuda-compat/solstice"
[[ -x "${BIN}" ]] || { echo "compatible CUDA binary is missing" >&2; exit 1; }
EXPECTED_BINARY_SHA256="$(json_string "${READINESS_REPORT}" solstice_binary_sha256)"
[[ "${EXPECTED_BINARY_SHA256}" =~ ^[0-9a-f]{64}$ && \
   "$(sha256sum -- "${BIN}" | awk '{print $1}')" == "${EXPECTED_BINARY_SHA256}" ]] || {
  echo "CUDA binary changed after readiness" >&2; exit 1;
}
READINESS_GPU_INDEX="$(json_number "${READINESS_REPORT}" gpu_index)"
[[ -n "${GPU_INDEX}" ]] || GPU_INDEX="${READINESS_GPU_INDEX}"
[[ "${GPU_INDEX}" =~ ^[0-9]+$ && "${GPU_INDEX}" == "${READINESS_GPU_INDEX}" ]] || {
  echo "GPU index differs from readiness" >&2; exit 1;
}
GPU_UUID="$(json_string "${READINESS_REPORT}" gpu_uuid)"
[[ -n "${GPU_UUID}" ]] || { echo "readiness GPU UUID is missing" >&2; exit 1; }

OUTPUT="$(realpath -m "${OUTPUT}")"
[[ ! -e "${OUTPUT}" ]] || { echo "resume-equivalence output already exists" >&2; exit 2; }
mkdir -p "${OUTPUT}"
UNINTERRUPTED="${OUTPUT}/uninterrupted.rlfsp"
RESUMED="${OUTPUT}/resumed.rlfsp"
COMMON=(train-data --ledger "${LEDGER}" --profile "${PROFILE}" --backend cuda
  --blank --enforce-profile --require-media-hashes
  --max-audit-records "${MAX_AUDIT_RECORDS}"
  --max-text-shard-bytes "${MAX_TEXT_SHARD_BYTES}"
  --max-train-shard-bytes "${MAX_TRAIN_SHARD_BYTES}"
  --target-training-records "${TARGET_RECORDS}"
  --seed 6003100749088244549)

run_guarded() {
  local name="$1"; shift
  "${ROOT}/scripts/run_general_cuda_with_vram_guard.sh" \
    --profile "${PROFILE}" --gpu-index "${GPU_INDEX}" --expected-uuid "${GPU_UUID}" \
    --trace "${OUTPUT}/${name}_vram.csv" --summary "${OUTPUT}/${name}_resource.json" -- "$@"
}

run_guarded uninterrupted "${BIN}" "${COMMON[@]}" --checkpoint "${UNINTERRUPTED}" \
  --output "${OUTPUT}/uninterrupted_audit.json" \
  --telemetry "${OUTPUT}/uninterrupted_telemetry.json" >"${OUTPUT}/uninterrupted_stdout.txt"

set +e
run_guarded interrupted "${BIN}" "${COMMON[@]}" --checkpoint "${RESUMED}" \
  --maximum-new-shards 1 --output "${OUTPUT}/interrupted_audit.json" \
  --telemetry "${OUTPUT}/interrupted_telemetry.json" >"${OUTPUT}/interrupted_stdout.txt" 2>"${OUTPUT}/interrupted_stderr.txt"
INTERRUPTED_STATUS=$?
set -e
[[ "${INTERRUPTED_STATUS}" -eq 4 ]] || { echo "intentional shard stop did not exit 4" >&2; exit 1; }
grep -Eq '"within_limit"[[:space:]]*:[[:space:]]*true' "${OUTPUT}/interrupted_resource.json" || { echo "partial run exceeded the VRAM limit" >&2; exit 1; }
grep -Eq '"sampler_ok"[[:space:]]*:[[:space:]]*true' "${OUTPUT}/interrupted_resource.json" || { echo "partial-run VRAM telemetry failed" >&2; exit 1; }
grep -Eq '"command_exit_code"[[:space:]]*:[[:space:]]*4([,[:space:]]|$)' "${OUTPUT}/interrupted_resource.json" || { echo "exit 4 did not originate from the intentional-stop command" >&2; exit 1; }
grep -Eq '"training_target_reached"[[:space:]]*:[[:space:]]*false' "${OUTPUT}/interrupted_telemetry.json" || { echo "partial telemetry did not record an incomplete target" >&2; exit 1; }
grep -Eq '"intentional_shard_stop"[[:space:]]*:[[:space:]]*true' "${OUTPUT}/interrupted_telemetry.json" || { echo "partial telemetry lacks intentional-stop evidence" >&2; exit 1; }
"${BIN}" verify-checkpoint --checkpoint "${RESUMED}" --profile "${PROFILE}" --enforce-profile >/dev/null

run_guarded resumed "${BIN}" "${COMMON[@]}" --checkpoint "${RESUMED}" \
  --output "${OUTPUT}/resumed_audit.json" \
  --telemetry "${OUTPUT}/resumed_telemetry.json" >"${OUTPUT}/resumed_stdout.txt"
"${BIN}" verify-checkpoint --checkpoint "${UNINTERRUPTED}" --profile "${PROFILE}" --enforce-profile >/dev/null
"${BIN}" verify-checkpoint --checkpoint "${RESUMED}" --profile "${PROFILE}" --enforce-profile >/dev/null
UNINTERRUPTED_SHA256="$(sha256sum -- "${UNINTERRUPTED}" | awk '{print $1}')"
RESUMED_SHA256="$(sha256sum -- "${RESUMED}" | awk '{print $1}')"
[[ "${UNINTERRUPTED_SHA256}" == "${RESUMED_SHA256}" ]] || {
  echo "resumed checkpoint differs from uninterrupted checkpoint" >&2; exit 1;
}
grep -Eq "\"cumulative_training_records_after\"[[:space:]]*:[[:space:]]*${TARGET_RECORDS}([,[:space:]]|$)" "${OUTPUT}/resumed_telemetry.json" || { echo "resumed target count mismatch" >&2; exit 1; }

TEMPORARY="$(mktemp "${OUTPUT}/.resume-equivalence.XXXXXX")"
cat >"${TEMPORARY}" <<EOF
{
  "schema": "rlf-v100-resume-equivalence-v1",
  "passed": true,
  "physical_training_performed": true,
  "test_doubles": false,
  "claim_eligible": false,
  "profile": "${PROFILE}",
  "campaign_binding_sha256": "${RLF_CAMPAIGN_BINDING_SHA256}",
  "ledger_sha256": "${LEDGER_SHA256}",
  "target_training_records": ${TARGET_RECORDS},
  "intentional_stop_exit_code": 4,
  "uninterrupted_checkpoint_sha256": "${UNINTERRUPTED_SHA256}",
  "resumed_checkpoint_sha256": "${RESUMED_SHA256}",
  "byte_identical": true,
  "frontier_claim_authorized": false,
  "efficiency_claim_authorized": false
}
EOF
mv -f -- "${TEMPORARY}" "${OUTPUT}/resume_equivalence.json"
find "${OUTPUT}" -maxdepth 1 -type f ! -name artifact_manifest.sha256 -print0 | \
  sort -z | xargs -0 sha256sum >"${OUTPUT}/artifact_manifest.sha256"
printf 'resume_equivalence_passed=true\ncheckpoint_sha256=%s\nclaim_eligible=false\n' "${RESUMED_SHA256}"
