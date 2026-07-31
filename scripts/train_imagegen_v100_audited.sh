#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAIR_MANIFEST="${1:?Usage: train_imagegen_v100_audited.sh PAIR_MANIFEST RESULT_DIR READINESS_REPORT AUDIT_DIR RESUME_REPORT}"
RESULT_DIR="${2:?result directory required}"
READINESS="${3:?readiness report required}"
AUDIT_DIR="${4:?audit directory required}"
RESUME_REPORT="${5:?resume-equivalence report required}"
[[ -f "${PAIR_MANIFEST}" && -f "${READINESS}" && -f "${RESUME_REPORT}" ]] || {
  echo "manifest/readiness/resume report missing" >&2; exit 1;
}
[[ -d "${AUDIT_DIR}" ]] || { echo "audit directory missing" >&2; exit 1; }
[[ ! -e "${RESULT_DIR}/artifact_manifest.tsv" ]] || {
  echo "result bundle already finalized" >&2; exit 1;
}

json_string() {
  sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"([^\"]*)\".*/\1/p" "$1" | head -n 1
}
json_number() {
  sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p" "$1" | head -n 1
}
require_literal_once() {
  local file="$1" key="$2" value="$3"
  local matches
  matches="$( (grep -Eo "\"${key}\"[[:space:]]*:[[:space:]]*${value}([,}[:space:]]|$)" \
    "${file}" || true) | wc -l | tr -d '[:space:]')"
  [[ "${matches}" -eq 1 ]] || {
    echo "${file} must contain exactly one ${key}=${value}" >&2
    exit 1
  }
}
require_literal_once "${READINESS}" schema '"rlf-imagegen-v100-readiness-v1"'
require_literal_once "${READINESS}" ready true
require_literal_once "${READINESS}" test_doubles false
require_literal_once "${READINESS}" training_performed false
PROFILE="$(json_string "${READINESS}" profile)"
GPU_INDEX="$(json_number "${READINESS}" gpu_index)"
GPU_UUID="$(json_string "${READINESS}" gpu_uuid)"
SOURCE_SHA="$(json_string "${READINESS}" source_manifest_sha256)"
BINARY_SHA="$(json_string "${READINESS}" solstice_binary_sha256)"
[[ "${PROFILE}" == imagegen-v100-32g && -n "${GPU_UUID}" && \
   "${SOURCE_SHA}" =~ ^[0-9a-f]{64}$ && "${BINARY_SHA}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "readiness identity is incomplete" >&2; exit 1;
}
BIN="${ROOT}/build/ubuntu-general-cuda-compat/solstice"
[[ -x "${BIN}" && "$(sha256sum -- "${BIN}" | awk '{print $1}')" == "${BINARY_SHA}" ]] || {
  echo "Solstice binary changed after image-generation readiness" >&2; exit 1;
}

CAMPAIGN_PROFILE="${RLF_CAMPAIGN_PROFILE:-}"
CAMPAIGN_SCHEMA=""
EXPECTED_STAGE=""
EXPECTED_AUTHORIZATION=""
case "${CAMPAIGN_PROFILE}" in
  general-v100-32g-500m)
    CAMPAIGN_SCHEMA=rlf-v100-1586h-campaign-v1
    EXPECTED_STAGE=physical-throughput-probe
    EXPECTED_AUTHORIZATION=0
    ;;
  imagegen-v100-32g)
    CAMPAIGN_SCHEMA=rlf-imagegen-v100-40h-campaign-v1
    EXPECTED_STAGE=controlled-training
    EXPECTED_AUTHORIZATION=1
    ;;
  *) echo "unsupported imagegen campaign profile" >&2; exit 1 ;;
esac
[[ "${RLF_CAMPAIGN_STAGE:-}" == "${EXPECTED_STAGE}" && \
   "${RLF_TRAINING_AUTHORIZED:-0}" == "${EXPECTED_AUTHORIZATION}" && \
   "${RLF_CAMPAIGN_BINDING_SHA256:-}" =~ ^[0-9a-f]{64}$ && \
   "${RLF_CAMPAIGN_STATE_DIR:-}" == /* && \
   -f "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" && \
   ! -L "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" ]] || {
  echo "imagegen training is outside its bound campaign stage" >&2
  exit 1
}
grep -Fqx "schema=${CAMPAIGN_SCHEMA}" \
  "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" || { echo "wrong campaign schema" >&2; exit 1; }
grep -Fqx "profile=${CAMPAIGN_PROFILE}" \
  "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" || { echo "campaign profile mismatch" >&2; exit 1; }
grep -Fqx "binding_sha256=${RLF_CAMPAIGN_BINDING_SHA256}" \
  "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" || { echo "campaign binding mismatch" >&2; exit 1; }

require_literal_once "${RESUME_REPORT}" schema '"rlf-imagegen-v100-resume-equivalence-v1"'
require_literal_once "${RESUME_REPORT}" passed true
require_literal_once "${RESUME_REPORT}" physical_training_performed true
require_literal_once "${RESUME_REPORT}" physical_cuda_local_updates_verified true
require_literal_once "${RESUME_REPORT}" test_doubles false
require_literal_once "${RESUME_REPORT}" claim_eligible false
require_literal_once "${RESUME_REPORT}" checkpoint_reloaded_in_new_process true
require_literal_once "${RESUME_REPORT}" byte_identical true
require_literal_once "${RESUME_REPORT}" frontier_claim_authorized false
require_literal_once "${RESUME_REPORT}" state_of_art_claim_authorized false
require_literal_once "${RESUME_REPORT}" efficiency_claim_authorized false
[[ "$(json_string "${RESUME_REPORT}" profile)" == imagegen-v100-32g && \
   "$(json_string "${RESUME_REPORT}" campaign_profile)" == "${CAMPAIGN_PROFILE}" && \
   "$(json_string "${RESUME_REPORT}" campaign_binding_sha256)" == \
     "${RLF_CAMPAIGN_BINDING_SHA256}" && \
   "$(json_string "${RESUME_REPORT}" source_manifest_sha256)" == "${SOURCE_SHA}" && \
   "$(json_string "${RESUME_REPORT}" solstice_binary_sha256)" == "${BINARY_SHA}" && \
   "$(json_string "${RESUME_REPORT}" gpu_uuid)" == "${GPU_UUID}" ]] || {
  echo "imagegen resume evidence identity mismatch" >&2; exit 1;
}
READINESS_SHA="$(sha256sum -- "${READINESS}" | awk '{print $1}')"
[[ "$(json_string "${RESUME_REPORT}" readiness_report_sha256)" == "${READINESS_SHA}" ]] || {
  echo "imagegen resume evidence does not bind this readiness report" >&2; exit 1;
}
UNINTERRUPTED_SHA="$(json_string "${RESUME_REPORT}" uninterrupted_checkpoint_sha256)"
RESUMED_SHA="$(json_string "${RESUME_REPORT}" resumed_checkpoint_sha256)"
TOTAL_RESUME_RECORDS="$(json_number "${RESUME_REPORT}" total_training_records)"
[[ "${UNINTERRUPTED_SHA}" =~ ^[0-9a-f]{64}$ && \
   "${UNINTERRUPTED_SHA}" == "${RESUMED_SHA}" && \
   "${TOTAL_RESUME_RECORDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "imagegen resume evidence is incomplete or non-identical" >&2; exit 1;
}
RESUME_DIR="$(cd "$(dirname "${RESUME_REPORT}")" && pwd)"
RESUME_ARTIFACT_MANIFEST="${RESUME_DIR}/artifact_manifest.sha256"
[[ -f "${RESUME_ARTIFACT_MANIFEST}" && ! -L "${RESUME_ARTIFACT_MANIFEST}" ]] || {
  echo "imagegen resume raw-artifact hash manifest is missing" >&2; exit 1;
}
(
  cd "${RESUME_DIR}"
  sha256sum -c -- "$(basename "${RESUME_ARTIFACT_MANIFEST}")" >/dev/null
) || { echo "imagegen resume raw artifacts failed hash verification" >&2; exit 1; }

PAIR_MANIFEST="$(realpath "${PAIR_MANIFEST}")"
AUDIT_VALIDATION="$(bash "${ROOT}/scripts/verify_imagegen_data_audits.sh" \
  "${PAIR_MANIFEST}" "${AUDIT_DIR}")"
grep -Fqx 'audit_validation=pass' <<<"${AUDIT_VALIDATION}" || {
  echo "imagegen data-audit validation did not pass" >&2; exit 1;
}
PAIR_SHA="$(awk -F= '$1 == "pair_manifest_sha256" {print $2}' <<<"${AUDIT_VALIDATION}")"
PAIR_RECORDS="$(awk -F= '$1 == "records_audited" {print $2}' <<<"${AUDIT_VALIDATION}")"
LICENSE_POLICY_SHA="$(awk -F= '$1 == "license_policy_sha256" {print $2}' <<<"${AUDIT_VALIDATION}")"
EVALUATION_MANIFEST_SHA="$(awk -F= '$1 == "evaluation_manifest_sha256" {print $2}' <<<"${AUDIT_VALIDATION}")"

mkdir -p "${RESULT_DIR}"
RESULT_DIR="$(realpath "${RESULT_DIR}")"
CHECKPOINT="${RESULT_DIR}/model.rlfimg"
PROMPT_COMPLETION_SHA=""
if [[ "${CAMPAIGN_PROFILE}" == imagegen-v100-32g ]]; then
  PROMPT_COMPLETION="${RESULT_DIR}/prompt_stage_completion.env"
  for prompt_artifact in prompt_language_ledger.tsv prompt_language_audit.json \
    prompt_language_telemetry.txt prompt_checkpoint_inspection.txt \
    prompt_source_manifest.tsv prompt_readiness.json \
    prompt_stage_completion.env model.rlfimg; do
    [[ -f "${RESULT_DIR}/${prompt_artifact}" &&
       ! -L "${RESULT_DIR}/${prompt_artifact}" &&
       -s "${RESULT_DIR}/${prompt_artifact}" ]] || {
      echo "missing prompt-stage artifact: ${prompt_artifact}" >&2; exit 1;
    }
  done
  prompt_value() {
    local count value
    count="$(awk -F= -v key="$2" '$1 == key {n++} END {print n+0}' "$1")"
    value="$(awk -F= -v key="$2" '$1 == key {print substr($0,length(key)+2)}' "$1")"
    [[ "${count}" -eq 1 && -n "${value}" && "${value}" != *$'\n'* ]] || {
      echo "invalid prompt-stage completion key: $2" >&2; exit 1;
    }
    printf '%s' "${value}"
  }
  [[ "$(prompt_value "${PROMPT_COMPLETION}" schema)" == \
       rlf-imagegen-prompt-stage-completion-v1 &&
     "$(prompt_value "${PROMPT_COMPLETION}" campaign_binding_sha256)" == \
       "${RLF_CAMPAIGN_BINDING_SHA256}" &&
     "$(prompt_value "${PROMPT_COMPLETION}" frontier_claim_authorized)" == false &&
     "$(prompt_value "${PROMPT_COMPLETION}" target_records)" =~ ^[1-9][0-9]*$ &&
     "$(prompt_value "${PROMPT_COMPLETION}" prompt_language_ledger_sha256)" == \
       "$(awk -F= '$1 == "prompt_language_ledger_sha256" {print $2}' "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta")" &&
     "$(prompt_value "${PROMPT_COMPLETION}" checkpoint_sha256)" == \
       "$(sha256sum -- "${CHECKPOINT}" | awk '{print $1}')" ]] || {
    echo "prompt-stage completion identity mismatch" >&2; exit 1;
  }
  while read -r key relative; do
    [[ "$(prompt_value "${PROMPT_COMPLETION}" "${key}")" == \
       "$(sha256sum -- "${RESULT_DIR}/${relative}" | awk '{print $1}')" ]] || {
      echo "prompt-stage artifact hash mismatch: ${relative}" >&2; exit 1;
    }
  done <<'EOF'
prompt_language_ledger_sha256 prompt_language_ledger.tsv
prompt_language_audit_sha256 prompt_language_audit.json
prompt_language_telemetry_sha256 prompt_language_telemetry.txt
prompt_checkpoint_inspection_sha256 prompt_checkpoint_inspection.txt
prompt_source_manifest_sha256 prompt_source_manifest.tsv
prompt_readiness_sha256 prompt_readiness.json
EOF
  PROMPT_COMPLETION_SHA="$(sha256sum -- "${PROMPT_COMPLETION}" | awk '{print $1}')"
fi
cp -- "${PAIR_MANIFEST}" "${RESULT_DIR}/training_pairs.tsv"
cp -- "${READINESS}" "${RESULT_DIR}/readiness.json"
cp -- "${RESUME_REPORT}" "${RESULT_DIR}/resume_equivalence.json"
for report in data_audit.json license_report.json exact_dedup_report.json \
  near_dedup_report.json perceptual_dedup_report.json contamination_report.json; do
  cp -- "${AUDIT_DIR}/${report}" "${RESULT_DIR}/${report}"
done
mkdir -p "${ROOT}/results"
SOURCE_MANIFEST_TEMP="$(mktemp "${ROOT}/results/.imagegen-source-manifest.XXXXXX")"
"${ROOT}/scripts/create_source_manifest.sh" "${SOURCE_MANIFEST_TEMP}" >/dev/null
cp -- "${SOURCE_MANIFEST_TEMP}" "${RESULT_DIR}/source_manifest.tsv"
rm -f -- "${SOURCE_MANIFEST_TEMP}" "${SOURCE_MANIFEST_TEMP}.sha256"
[[ "$(sha256sum -- "${RESULT_DIR}/source_manifest.tsv" | awk '{print $1}')" == "${SOURCE_SHA}" ]] || {
  echo "source tree changed after image-generation readiness" >&2; exit 1;
}

{
  printf 'schema=rlf-imagegen-v100-training-environment-v1\n'
  printf 'profile=imagegen-v100-32g\ngpu_uuid=%s\ngpu_index=%s\n' "${GPU_UUID}" "${GPU_INDEX}"
  printf 'source_manifest_sha256=%s\nsolstice_binary_sha256=%s\n' "${SOURCE_SHA}" "${BINARY_SHA}"
  printf 'pair_manifest_sha256=%s\n' "${PAIR_SHA}"
  printf 'pair_manifest_records=%s\n' "${PAIR_RECORDS}"
  printf 'license_policy_sha256=%s\n' "${LICENSE_POLICY_SHA}"
  printf 'evaluation_manifest_sha256=%s\n' "${EVALUATION_MANIFEST_SHA}"
  if [[ -n "${PROMPT_COMPLETION_SHA}" ]]; then
    printf 'prompt_stage_completion_sha256=%s\n' "${PROMPT_COMPLETION_SHA}"
    printf 'prompt_language_ledger_sha256=%s\n' \
      "$(sha256sum -- "${RESULT_DIR}/prompt_language_ledger.tsv" | awk '{print $1}')"
  fi
  printf 'cuda_local_update_policy=%s\nfrontier_claim_authorized=false\n' "${RLF_CUDA_LOCAL_UPDATE_POLICY:-device}"
  uname -a; nvcc --version | tail -n 1
  nvidia-smi --id="${GPU_INDEX}" --query-gpu=index,name,uuid,memory.total,compute_cap,driver_version --format=csv,noheader,nounits
} >"${RESULT_DIR}/environment.txt"

export RLF_CUDA_LOCAL_UPDATE_POLICY="${RLF_CUDA_LOCAL_UPDATE_POLICY:-device}"
"${ROOT}/scripts/run_checkpoint_transaction.sh" \
  --checkpoint "${CHECKPOINT}" \
  --failure-root "${RESULT_DIR}/failed_training_attempts" \
  --preserve "${RESULT_DIR}/training_telemetry.txt" \
  --preserve "${RESULT_DIR}/resource_summary.json" \
  --preserve "${RESULT_DIR}/raw_gpu_trace.csv" \
  --preserve "${RESULT_DIR}/checkpoint_inspection.txt" -- \
  bash "${ROOT}/scripts/run_imagegen_v100_training_attempt.sh" \
    "${ROOT}" "${BIN}" "${CHECKPOINT}" "${PAIR_MANIFEST}" "${RESULT_DIR}" \
    "${GPU_INDEX}" "${GPU_UUID}" "${RLF_IMAGEGEN_SEED:-77}"
printf 'training_bundle=%s\nfrontier_claim_authorized=false\n' "${RESULT_DIR}"
