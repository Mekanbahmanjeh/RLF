#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

# This gate authorizes bounded training work; it never establishes a frontier
# or efficiency claim. All scientific inputs are read from an immutable,
# SHA-256-manifested evidence directory. There are deliberately no CLI flags
# that can assert pass/fail measurements.

usage() {
  cat <<'EOF'
Usage:
  v100_scale_promotion_gate.sh authorize --campaign-state DIR \
    --evidence-dir DIR --manifest FILE --target-records N --output FILE
  v100_scale_promotion_gate.sh verify --campaign-state DIR \
    --evidence-dir DIR --manifest FILE --target-records N --ticket FILE

N must be exactly 50000000, 200000000, or 500000000. The evidence manifest
must have been created with scripts/vast/artifact_manifest.sh and must cover
the seven fixed *.env evidence files consumed by this gate.
EOF
}

die() { echo "$*" >&2; exit 1; }
bad_usage() { echo "$*" >&2; usage >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE=general-v100-32g-500m
TOTAL_SECONDS=5709600
MAX_PEAK_VRAM_BYTES=32212254720
ACTION=""
CAMPAIGN_STATE=""
EVIDENCE_DIR=""
MANIFEST=""
TARGET_RECORDS=""
OUTPUT=""
TICKET=""

while (($# > 0)); do
  case "$1" in
    authorize|verify)
      [[ -z "${ACTION}" ]] || bad_usage "only one action may be selected"
      ACTION="$1"; shift
      ;;
    --campaign-state) CAMPAIGN_STATE="${2:?--campaign-state requires a directory}"; shift 2 ;;
    --evidence-dir) EVIDENCE_DIR="${2:?--evidence-dir requires a directory}"; shift 2 ;;
    --manifest) MANIFEST="${2:?--manifest requires a file}"; shift 2 ;;
    --target-records) TARGET_RECORDS="${2:?--target-records requires a value}"; shift 2 ;;
    --output) OUTPUT="${2:?--output requires a file}"; shift 2 ;;
    --ticket) TICKET="${2:?--ticket requires a file}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) bad_usage "unknown option: $1" ;;
  esac
done

[[ "${ACTION}" == authorize || "${ACTION}" == verify ]] || bad_usage "an action is required"
[[ -d "${CAMPAIGN_STATE}" && -d "${EVIDENCE_DIR}" && -f "${MANIFEST}" ]] || \
  bad_usage "campaign state, evidence directory, and manifest must exist"
case "${TARGET_RECORDS}" in
  50000000|200000000|500000000) ;;
  *) bad_usage "target records must be exactly 50000000, 200000000, or 500000000" ;;
esac
if [[ "${ACTION}" == authorize ]]; then
  [[ -n "${OUTPUT}" && -z "${TICKET}" ]] || bad_usage "authorize requires --output only"
else
  [[ -n "${TICKET}" && -z "${OUTPUT}" ]] || bad_usage "verify requires --ticket only"
fi

for required_tool in awk date realpath sha256sum; do
  command -v "${required_tool}" >/dev/null 2>&1 || bad_usage "missing command: ${required_tool}"
done

kv() {
  local file="$1" key="$2" count value
  [[ -f "${file}" && ! -L "${file}" ]] || die "missing regular evidence file: ${file}"
  count="$(awk -F= -v wanted="${key}" '$1 == wanted { count++ } END { print count + 0 }' "${file}")"
  [[ "${count}" == 1 ]] || die "${file}: expected exactly one ${key} field"
  value="$(awk -F= -v wanted="${key}" '$1 == wanted { print substr($0, length(wanted) + 2) }' "${file}")"
  [[ -n "${value}" && "${value}" != *$'\r'* && "${value}" != *$'\n'* ]] || \
    die "${file}: invalid ${key} value"
  printf '%s' "${value}"
}

require_eq() {
  local file="$1" key="$2" expected="$3" actual
  actual="$(kv "${file}" "${key}")"
  [[ "${actual}" == "${expected}" ]] || die "${file}: ${key} must be ${expected}, got ${actual}"
}

require_uint() {
  local value="$1" label="$2"
  [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]] || die "${label} must be an unsigned canonical integer"
}

require_decimal() {
  local value="$1" label="$2"
  [[ "${value}" =~ ^(0|[1-9][0-9]*)(\.[0-9]+)?$ ]] || die "${label} must be a non-negative decimal"
}

json_string() {
  local file="$1" key="$2" count value
  count="$(grep -Ec "\"${key}\"[[:space:]]*:" "${file}")"
  [[ "${count}" == 1 ]] || die "${file}: expected exactly one JSON ${key} field"
  value="$(sed -nE "s/.*\"${key}\"[[:space:]]*:[[:space:]]*\"([^\"]*)\".*/\1/p" "${file}" | head -n 1)"
  [[ -n "${value}" ]] || die "${file}: invalid JSON string ${key}"
  printf '%s' "${value}"
}

json_bool() {
  local file="$1" key="$2" count value
  count="$(grep -Ec "\"${key}\"[[:space:]]*:" "${file}")"
  [[ "${count}" == 1 ]] || die "${file}: expected exactly one JSON ${key} field"
  value="$(sed -nE "s/.*\"${key}\"[[:space:]]*:[[:space:]]*(true|false).*/\1/p" "${file}" | head -n 1)"
  [[ "${value}" == true || "${value}" == false ]] || die "${file}: invalid JSON boolean ${key}"
  printf '%s' "${value}"
}

verify_bound_artifact() {
  local evidence_file="$1" path_key="$2" expected_hash="$3" relative absolute actual
  relative="$(kv "${evidence_file}" "${path_key}")"
  [[ "${relative}" != /* && "${relative}" != . && "${relative}" != .. && \
     "${relative}" != ../* && "${relative}" != */../* && "${relative}" != */.. && \
     "${relative}" != *$'\t'* && "${relative}" != *$'\r'* ]] || \
    die "unsafe bound artifact path in ${path_key}"
  absolute="$(realpath "${EVIDENCE_DIR}/${relative}")" || die "missing bound artifact: ${relative}"
  case "${absolute}" in
    "${EVIDENCE_DIR}"/*) ;;
    *) die "bound artifact escapes evidence directory: ${relative}" ;;
  esac
  [[ -f "${absolute}" && ! -L "${absolute}" ]] || die "bound artifact is not a regular file: ${relative}"
  actual="$(sha256sum -- "${absolute}" | awk '{ print $1 }')"
  [[ "${actual}" == "${expected_hash}" ]] || die "bound artifact SHA-256 mismatch: ${relative}"
}

META="$(realpath "${CAMPAIGN_STATE}")/campaign.meta"
[[ -f "${META}" && ! -L "${META}" ]] || die "campaign was not started"
require_eq "${META}" schema rlf-v100-1586h-campaign-v1
require_eq "${META}" profile "${PROFILE}"
require_eq "${META}" total_budget_seconds "${TOTAL_SECONDS}"
CAMPAIGN_BINDING="$(kv "${META}" binding_sha256)"
[[ "${CAMPAIGN_BINDING}" =~ ^[0-9a-f]{64}$ ]] || die "invalid campaign binding"

EVIDENCE_DIR="$(realpath "${EVIDENCE_DIR}")"
MANIFEST="$(realpath "${MANIFEST}")"
case "${MANIFEST}" in
  "${EVIDENCE_DIR}"|"${EVIDENCE_DIR}"/*) die "manifest must be outside the evidence directory" ;;
esac
"${ROOT}/scripts/vast/artifact_manifest.sh" verify \
  --source "${EVIDENCE_DIR}" --manifest "${MANIFEST}" >/dev/null
MANIFEST_SHA256="$(sha256sum -- "${MANIFEST}" | awk '{ print $1 }')"
[[ "${MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] || die "invalid evidence manifest hash"

PROVENANCE="${EVIDENCE_DIR}/provenance.env"
HARDWARE="${EVIDENCE_DIR}/hardware.env"
STAGE="${EVIDENCE_DIR}/stage.env"
THROUGHPUT="${EVIDENCE_DIR}/throughput.env"
CHECKPOINT="${EVIDENCE_DIR}/checkpoint.env"
FROZEN_EVAL="${EVIDENCE_DIR}/frozen_eval.env"
RESOURCES="${EVIDENCE_DIR}/resources.env"

for evidence_file in "${PROVENANCE}" "${HARDWARE}" "${STAGE}" \
  "${THROUGHPUT}" "${CHECKPOINT}" "${FROZEN_EVAL}" "${RESOURCES}"; do
  require_eq "${evidence_file}" campaign_binding_sha256 "${CAMPAIGN_BINDING}"
done
require_eq "${PROVENANCE}" schema rlf-v100-provenance-evidence-v1
require_eq "${HARDWARE}" schema rlf-v100-hardware-evidence-v1
require_eq "${STAGE}" schema rlf-v100-stage-evidence-v1
require_eq "${THROUGHPUT}" schema rlf-v100-throughput-evidence-v1
require_eq "${CHECKPOINT}" schema rlf-v100-checkpoint-evidence-v1
require_eq "${FROZEN_EVAL}" schema rlf-v100-frozen-eval-evidence-v1
require_eq "${RESOURCES}" schema rlf-v100-resource-evidence-v1

# Claim-ineligible, simulated, or incompletely governed data cannot authorize
# paid scale training. These are evidence fields covered by the manifest, not
# caller-provided booleans.
require_eq "${PROVENANCE}" claim_eligible true
require_eq "${PROVENANCE}" test_doubles false
require_eq "${PROVENANCE}" provenance_complete true
require_eq "${PROVENANCE}" dedup_complete true
require_eq "${PROVENANCE}" contamination_controls_passed true
SOURCE_MANIFEST_SHA256="$(kv "${PROVENANCE}" source_manifest_sha256)"
[[ "${SOURCE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] || die "invalid source manifest hash"
verify_bound_artifact "${PROVENANCE}" source_manifest_relative_path "${SOURCE_MANIFEST_SHA256}"
AUDITED_SOURCE_RECORDS="$(kv "${PROVENANCE}" audited_source_records)"
require_uint "${AUDITED_SOURCE_RECORDS}" audited_source_records
((AUDITED_SOURCE_RECORDS >= TARGET_RECORDS)) || die "audited source has fewer records than target"

require_eq "${HARDWARE}" physical_measurement true
require_eq "${HARDWARE}" test_doubles false
require_eq "${HARDWARE}" gpu_count 1
GPU_NAME="$(kv "${HARDWARE}" gpu_name)"
[[ "${GPU_NAME}" =~ ^(NVIDIA[[:space:]])?Tesla[[:space:]]V100.*32GB$ ]] || \
  die "physical GPU is not a Tesla V100 32GB"
require_eq "${HARDWARE}" compute_capability 7.0
require_eq "${HARDWARE}" cuda_major 12
VRAM_TOTAL="$(kv "${HARDWARE}" vram_total_bytes)"
PEAK_VRAM="$(kv "${HARDWARE}" peak_vram_bytes)"
require_uint "${VRAM_TOTAL}" vram_total_bytes
require_uint "${PEAK_VRAM}" peak_vram_bytes
((VRAM_TOTAL >= 32000000000 && VRAM_TOTAL <= 36000000000)) || die "unexpected V100 VRAM total"
((PEAK_VRAM <= MAX_PEAK_VRAM_BYTES)) || die "peak VRAM exceeds the 30 GiB ceiling"

COMPLETED_RECORDS="$(kv "${STAGE}" completed_stage_records)"
CAPACITY_SKIPS="$(kv "${STAGE}" capacity_skips)"
require_uint "${COMPLETED_RECORDS}" completed_stage_records
require_uint "${CAPACITY_SKIPS}" capacity_skips
[[ "${CAPACITY_SKIPS}" == 0 ]] || die "capacity skips must be zero"
case "${TARGET_RECORDS}:${COMPLETED_RECORDS}" in
  50000000:0) require_eq "${STAGE}" stage_completed false ;;
  200000000:50000000) require_eq "${STAGE}" stage_completed true ;;
  500000000:50000000|500000000:200000000) require_eq "${STAGE}" stage_completed true ;;
  *) die "target ${TARGET_RECORDS} is not promotable from completed stage ${COMPLETED_RECORDS}" ;;
esac

require_eq "${THROUGHPUT}" physical_measurement true
require_eq "${THROUGHPUT}" test_doubles false
MEASURED_RECORDS="$(kv "${THROUGHPUT}" measured_records)"
MEASURED_RPS="$(kv "${THROUGHPUT}" end_to_end_records_per_second)"
require_uint "${MEASURED_RECORDS}" measured_records
require_decimal "${MEASURED_RPS}" end_to_end_records_per_second
((MEASURED_RECORDS >= 100000)) || die "throughput sample must cover at least 100000 physical records"
awk -v value="${MEASURED_RPS}" 'BEGIN { exit !(value > 0) }' || die "measured throughput must be positive"

require_eq "${CHECKPOINT}" checkpoint_verified true
require_eq "${CHECKPOINT}" resume_recovery_passed true
RESUME_REPORT_SHA256="$(kv "${CHECKPOINT}" resume_equivalence_report_sha256)"
[[ "${RESUME_REPORT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || die "invalid resume-equivalence report hash"
verify_bound_artifact "${CHECKPOINT}" resume_equivalence_report_relative_path "${RESUME_REPORT_SHA256}"
RESUME_REPORT_RELATIVE="$(kv "${CHECKPOINT}" resume_equivalence_report_relative_path)"
RESUME_REPORT="$(realpath "${EVIDENCE_DIR}/${RESUME_REPORT_RELATIVE}")"
[[ "$(json_string "${RESUME_REPORT}" schema)" == rlf-v100-resume-equivalence-v1 ]] || die "wrong resume-equivalence schema"
[[ "$(json_bool "${RESUME_REPORT}" passed)" == true ]] || die "resume equivalence did not pass"
[[ "$(json_bool "${RESUME_REPORT}" physical_training_performed)" == true ]] || die "resume equivalence lacks physical training"
[[ "$(json_bool "${RESUME_REPORT}" test_doubles)" == false ]] || die "resume equivalence used test doubles"
[[ "$(json_bool "${RESUME_REPORT}" byte_identical)" == true ]] || die "resume checkpoint is not byte-identical"
[[ "$(json_bool "${RESUME_REPORT}" claim_eligible)" == false ]] || die "resume drill must remain claim-ineligible"
[[ "$(json_string "${RESUME_REPORT}" profile)" == "${PROFILE}" ]] || die "resume-equivalence profile mismatch"
[[ "$(json_string "${RESUME_REPORT}" campaign_binding_sha256)" == "${CAMPAIGN_BINDING}" ]] || die "resume-equivalence campaign binding mismatch"
CHECKPOINT_SHA256="$(kv "${CHECKPOINT}" checkpoint_sha256)"
[[ "${CHECKPOINT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || die "invalid checkpoint hash"
verify_bound_artifact "${CHECKPOINT}" checkpoint_relative_path "${CHECKPOINT_SHA256}"
require_eq "${CHECKPOINT}" checkpoint_profile "${PROFILE}"
CHECKPOINT_RECORDS="$(kv "${CHECKPOINT}" checkpoint_training_records)"
RECOVERED_RECORDS="$(kv "${CHECKPOINT}" resume_recovered_training_records)"
require_uint "${CHECKPOINT_RECORDS}" checkpoint_training_records
require_uint "${RECOVERED_RECORDS}" resume_recovered_training_records
if ((COMPLETED_RECORDS == 0)); then
  EXPECTED_CHECKPOINT_RECORDS="${MEASURED_RECORDS}"
else
  EXPECTED_CHECKPOINT_RECORDS="${COMPLETED_RECORDS}"
fi
[[ "${CHECKPOINT_RECORDS}" == "${EXPECTED_CHECKPOINT_RECORDS}" && \
   "${RECOVERED_RECORDS}" == "${EXPECTED_CHECKPOINT_RECORDS}" ]] || \
  die "checkpoint/resume records do not bind to the measured completed stage"
require_eq "${STAGE}" completed_checkpoint_sha256 "${CHECKPOINT_SHA256}"
require_eq "${THROUGHPUT}" measured_checkpoint_sha256 "${CHECKPOINT_SHA256}"

require_eq "${FROZEN_EVAL}" contamination_audit_passed true
require_eq "${FROZEN_EVAL}" evaluated_checkpoint_sha256 "${CHECKPOINT_SHA256}"
FROZEN_SPLIT_SHA256="$(kv "${FROZEN_EVAL}" frozen_split_sha256)"
[[ "${FROZEN_SPLIT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || die "invalid frozen split hash"
verify_bound_artifact "${FROZEN_EVAL}" frozen_split_relative_path "${FROZEN_SPLIT_SHA256}"
FROZEN_SAMPLES="$(kv "${FROZEN_EVAL}" sample_count)"
QUALITY_BEFORE="$(kv "${FROZEN_EVAL}" quality_before)"
QUALITY_AFTER="$(kv "${FROZEN_EVAL}" quality_after)"
QUALITY_DIRECTION="$(kv "${FROZEN_EVAL}" quality_direction)"
require_uint "${FROZEN_SAMPLES}" sample_count
require_decimal "${QUALITY_BEFORE}" quality_before
require_decimal "${QUALITY_AFTER}" quality_after
((FROZEN_SAMPLES > 0)) || die "frozen evaluation must contain samples"
case "${QUALITY_DIRECTION}" in
  higher_is_better)
    awk -v before="${QUALITY_BEFORE}" -v after="${QUALITY_AFTER}" \
      'BEGIN { exit !(after >= before) }' || die "frozen held-out quality regressed"
    ;;
  lower_is_better)
    awk -v before="${QUALITY_BEFORE}" -v after="${QUALITY_AFTER}" \
      'BEGIN { exit !(after <= before) }' || die "frozen held-out quality regressed"
    ;;
  *) die "unsupported quality direction" ;;
esac

for resource_name in host_ram disk checkpoint_space; do
  projected="$(kv "${RESOURCES}" "projected_${resource_name}_bytes")"
  available="$(kv "${RESOURCES}" "available_${resource_name}_bytes")"
  require_uint "${projected}" "projected_${resource_name}_bytes"
  require_uint "${available}" "available_${resource_name}_bytes"
  ((projected > 0 && available > 0)) || die "${resource_name} projection and availability must be positive"
  awk -v projected="${projected}" -v available="${available}" \
    'BEGIN { exit !(available >= projected * 1.2) }' || \
    die "${resource_name} lacks 20% projected headroom"
done
require_eq "${RESOURCES}" projection_target_records "${TARGET_RECORDS}"

case "${TARGET_RECORDS}" in
  50000000) TARGET_STAGE=train-50m; TARGET_STAGE_SECONDS=$((250 * 3600)) ;;
  200000000) TARGET_STAGE=train-200m-primary; TARGET_STAGE_SECONDS=$((600 * 3600)) ;;
  500000000) TARGET_STAGE=train-500m-promoted; TARGET_STAGE_SECONDS=$((300 * 3600)) ;;
esac

state_consumed() {
  local state_file="$1" expected_budget="$2" consumed=0 active
  if [[ -e "${state_file}" ]]; then
    [[ -f "${state_file}" && ! -L "${state_file}" ]] || die "invalid budget state: ${state_file}"
    require_eq "${state_file}" schema rlf-wall-budget-v2
    require_eq "${state_file}" budget_seconds "${expected_budget}"
    active="$(kv "${state_file}" active)"
    [[ "${active}" == false ]] || die "cannot promote while a campaign command is active"
    consumed="$(kv "${state_file}" consumed_seconds)"
    require_uint "${consumed}" consumed_seconds
    ((consumed <= expected_budget)) || die "budget state exceeds its ceiling"
  fi
  printf '%s' "${consumed}"
}

OVERALL_CONSUMED="$(state_consumed "${CAMPAIGN_STATE}/overall.wall.state" "${TOTAL_SECONDS}")"
STAGE_CONSUMED="$(state_consumed "${CAMPAIGN_STATE}/${TARGET_STAGE}.wall.state" "${TARGET_STAGE_SECONDS}")"
OVERALL_REMAINING=$((TOTAL_SECONDS - OVERALL_CONSUMED))
STAGE_REMAINING=$((TARGET_STAGE_SECONDS - STAGE_CONSUMED))
AVAILABLE_SECONDS="${OVERALL_REMAINING}"
((STAGE_REMAINING < AVAILABLE_SECONDS)) && AVAILABLE_SECONDS="${STAGE_REMAINING}"
((AVAILABLE_SECONDS > 0)) || die "no training budget remains for ${TARGET_STAGE}"
REMAINING_RECORDS=$((TARGET_RECORDS - COMPLETED_RECORDS))
((REMAINING_RECORDS > 0)) || die "target must exceed completed stage records"
REQUIRED_RPS="$(awk -v records="${REMAINING_RECORDS}" -v seconds="${AVAILABLE_SECONDS}" \
  'BEGIN { printf "%.9f", (records * 1.2) / seconds }')"
awk -v measured="${MEASURED_RPS}" -v required="${REQUIRED_RPS}" \
  'BEGIN { exit !(measured >= required) }' || \
  die "measured throughput ${MEASURED_RPS} lacks 20% budget headroom; need ${REQUIRED_RPS} records/s"

write_ticket() {
  local destination="$1" temporary issued
  [[ ! -e "${destination}" && ! -e "${destination}.sha256" ]] || \
    bad_usage "authorization output already exists"
  mkdir -p "$(dirname "${destination}")"
  issued="$(date +%s)"
  temporary="$(mktemp "$(dirname "${destination}")/.v100-authorization.XXXXXX")"
  {
    printf 'schema=rlf-v100-scale-authorization-v1\n'
    printf 'profile=%s\n' "${PROFILE}"
    printf 'authorization=training_only_not_claim_evidence\n'
    printf 'frontier_claim_authorized=false\n'
    printf 'campaign_binding_sha256=%s\n' "${CAMPAIGN_BINDING}"
    printf 'target_records=%s\n' "${TARGET_RECORDS}"
    printf 'completed_stage_records=%s\n' "${COMPLETED_RECORDS}"
    printf 'evidence_manifest_sha256=%s\n' "${MANIFEST_SHA256}"
    printf 'source_manifest_sha256=%s\n' "${SOURCE_MANIFEST_SHA256}"
    printf 'checkpoint_sha256=%s\n' "${CHECKPOINT_SHA256}"
    printf 'frozen_split_sha256=%s\n' "${FROZEN_SPLIT_SHA256}"
    printf 'measured_end_to_end_records_per_second=%s\n' "${MEASURED_RPS}"
    printf 'required_records_per_second_at_issue=%s\n' "${REQUIRED_RPS}"
    printf 'available_training_seconds_at_issue=%s\n' "${AVAILABLE_SECONDS}"
    printf 'issued_epoch=%s\n' "${issued}"
  } >"${temporary}"
  mv -f -- "${temporary}" "${destination}"
  ticket_hash="$(sha256sum -- "${destination}" | awk '{ print $1 }')"
  printf '%s  %s\n' "${ticket_hash}" "$(basename "${destination}")" >"${destination}.sha256"
  printf 'authorization_ticket=%s\nauthorization_ticket_sha256=%s\n' "${destination}" "${ticket_hash}"
}

verify_ticket() {
  local ticket_file="$1" expected_hash actual_hash
  [[ -f "${ticket_file}" && ! -L "${ticket_file}" && -f "${ticket_file}.sha256" ]] || \
    die "authorization ticket and SHA-256 sidecar are required"
  expected_hash="$(awk 'NR == 1 { print $1 }' "${ticket_file}.sha256")"
  [[ "${expected_hash}" =~ ^[0-9a-f]{64}$ ]] || die "invalid authorization ticket sidecar"
  actual_hash="$(sha256sum -- "${ticket_file}" | awk '{ print $1 }')"
  [[ "${actual_hash}" == "${expected_hash}" ]] || die "authorization ticket SHA-256 mismatch"
  require_eq "${ticket_file}" schema rlf-v100-scale-authorization-v1
  require_eq "${ticket_file}" profile "${PROFILE}"
  require_eq "${ticket_file}" authorization training_only_not_claim_evidence
  require_eq "${ticket_file}" frontier_claim_authorized false
  require_eq "${ticket_file}" campaign_binding_sha256 "${CAMPAIGN_BINDING}"
  require_eq "${ticket_file}" target_records "${TARGET_RECORDS}"
  require_eq "${ticket_file}" completed_stage_records "${COMPLETED_RECORDS}"
  require_eq "${ticket_file}" evidence_manifest_sha256 "${MANIFEST_SHA256}"
  require_eq "${ticket_file}" source_manifest_sha256 "${SOURCE_MANIFEST_SHA256}"
  require_eq "${ticket_file}" checkpoint_sha256 "${CHECKPOINT_SHA256}"
  require_eq "${ticket_file}" frozen_split_sha256 "${FROZEN_SPLIT_SHA256}"
  ticket_rps="$(kv "${ticket_file}" measured_end_to_end_records_per_second)"
  [[ "${ticket_rps}" == "${MEASURED_RPS}" ]] || die "ticket throughput is not bound to current evidence"
  issued="$(kv "${ticket_file}" issued_epoch)"
  require_uint "${issued}" issued_epoch
  printf 'authorization_verified=true\ntarget_records=%s\nevidence_manifest_sha256=%s\n' \
    "${TARGET_RECORDS}" "${MANIFEST_SHA256}"
}

if [[ "${ACTION}" == authorize ]]; then
  write_ticket "$(realpath -m "${OUTPUT}")"
else
  verify_ticket "$(realpath "${TICKET}")"
fi
