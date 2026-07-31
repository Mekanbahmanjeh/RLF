#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEDGER="${1:?Usage: train_imagegen_prompt_semantics.sh LEDGER TARGET_RECORDS RESULT_DIR READINESS_REPORT}"
TARGET_RECORDS="${2:?exact target records required}"
RESULT_DIR="${3:?result directory required}"
READINESS="${4:?readiness report required}"
PROFILE=imagegen-v100-32g

[[ -f "${LEDGER}" && ! -L "${LEDGER}" && -f "${READINESS}" && ! -L "${READINESS}" ]] || {
  echo "language ledger/readiness report must be regular non-symlink files" >&2
  exit 1
}
[[ "${TARGET_RECORDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "target records must be a positive integer" >&2; exit 2;
}
[[ "${RLF_CAMPAIGN_PROFILE:-}" == "${PROFILE}" &&
   "${RLF_CAMPAIGN_STAGE:-}" == prompt-semantic-training &&
   "${RLF_TRAINING_AUTHORIZED:-0}" == 1 &&
   "${RLF_CAMPAIGN_BINDING_SHA256:-}" =~ ^[0-9a-f]{64}$ &&
   "${RLF_CAMPAIGN_STATE_DIR:-}" == /* &&
   -f "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" &&
   ! -L "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" ]] || {
  echo "prompt-semantic training is outside its bound campaign stage" >&2
  exit 1
}
grep -Fqx 'schema=rlf-imagegen-v100-40h-campaign-v1' \
  "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta"
grep -Fqx "profile=${PROFILE}" "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta"
grep -Fqx "binding_sha256=${RLF_CAMPAIGN_BINDING_SHA256}" \
  "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta"
grep -Fqx "prompt_language_ledger_sha256=$(sha256sum -- "${LEDGER}" | awk '{print $1}')" \
  "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" || {
  echo "prompt-language ledger is not campaign-bound" >&2; exit 1;
}

json_string() {
  sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"([^\"]*)\".*/\1/p" "$1" | head -n 1
}
require_literal_once() {
  local count
  count="$( (grep -Eo "\"$2\"[[:space:]]*:[[:space:]]*$3([,}[:space:]]|$)" "$1" || true) | wc -l | tr -d '[:space:]')"
  [[ "${count}" -eq 1 ]] || {
    echo "$1 requires exactly one $2=$3" >&2; exit 1;
  }
}
require_string_once() {
  local count value
  count="$( (grep -Eo "\"$2\"[[:space:]]*:" "$1" || true) | wc -l | tr -d '[:space:]')"
  value="$(json_string "$1" "$2")"
  [[ "${count}" -eq 1 && -n "${value}" ]] || {
    echo "$1 requires exactly one nonempty string $2" >&2; exit 1;
  }
  printf '%s' "${value}"
}
[[ "$(require_string_once "${READINESS}" schema)" == \
   rlf-imagegen-v100-readiness-v1 ]]
require_literal_once "${READINESS}" ready true
require_literal_once "${READINESS}" test_doubles false
require_literal_once "${READINESS}" training_performed false
[[ "$(require_string_once "${READINESS}" profile)" == "${PROFILE}" ]]
BIN="${ROOT}/build/ubuntu-general-cuda-compat/solstice"
BINARY_SHA="$(require_string_once "${READINESS}" solstice_binary_sha256)"
SOURCE_SHA="$(require_string_once "${READINESS}" source_manifest_sha256)"
[[ -x "${BIN}" && "${BINARY_SHA}" =~ ^[0-9a-f]{64}$ &&
   "${SOURCE_SHA}" =~ ^[0-9a-f]{64}$ &&
   "$(sha256sum -- "${BIN}" | awk '{print $1}')" == "${BINARY_SHA}" ]] || {
  echo "Solstice binary changed after readiness" >&2; exit 1;
}

mkdir -p "${RESULT_DIR}"
RESULT_DIR="$(realpath "${RESULT_DIR}")"
LEDGER="$(realpath "${LEDGER}")"
CHECKPOINT="${RESULT_DIR}/model.rlfimg"
if [[ ! -f "${CHECKPOINT}" ]]; then
  "${BIN}" imagegen-bootstrap --profile "${PROFILE}" \
    --checkpoint "${CHECKPOINT}" --seed "${RLF_IMAGEGEN_SEED:-77}"
fi
cp -- "${LEDGER}" "${RESULT_DIR}/prompt_language_ledger.tsv"
cp -- "${READINESS}" "${RESULT_DIR}/prompt_readiness.json"
mkdir -p "${ROOT}/results"
PROMPT_SOURCE_TEMP="$(mktemp "${ROOT}/results/.prompt-source-manifest.XXXXXX")"
"${ROOT}/scripts/create_source_manifest.sh" "${PROMPT_SOURCE_TEMP}" >/dev/null
cp -- "${PROMPT_SOURCE_TEMP}" "${RESULT_DIR}/prompt_source_manifest.tsv"
rm -f -- "${PROMPT_SOURCE_TEMP}" "${PROMPT_SOURCE_TEMP}.sha256"
[[ "$(sha256sum -- "${RESULT_DIR}/prompt_source_manifest.tsv" | awk '{print $1}')" == \
   "${SOURCE_SHA}" ]] || {
  echo "source tree changed after prompt readiness" >&2; exit 1;
}
"${BIN}" imagegen-train-language-ledger --profile "${PROFILE}" \
  --checkpoint "${CHECKPOINT}" --ledger "${LEDGER}" \
  --target-training-records "${TARGET_RECORDS}" \
  --max-audit-records "${TARGET_RECORDS}" \
  --max-text-shard-bytes 4294967296 --max-train-shard-bytes 4294967296 \
  --output "${RESULT_DIR}/prompt_language_audit.json" \
  >"${RESULT_DIR}/prompt_language_telemetry.txt"
"${BIN}" imagegen-verify --profile "${PROFILE}" \
  --checkpoint "${CHECKPOINT}" >/dev/null
"${BIN}" imagegen-inspect --checkpoint "${CHECKPOINT}" \
  >"${RESULT_DIR}/prompt_checkpoint_inspection.txt"
grep -Fqx "prompt_language_records=${TARGET_RECORDS}" \
  "${RESULT_DIR}/prompt_checkpoint_inspection.txt"
COMPLETION_TEMP="$(mktemp "${RESULT_DIR}/.prompt-stage-completion.XXXXXX")"
{
  printf 'schema=rlf-imagegen-prompt-stage-completion-v1\n'
  printf 'campaign_binding_sha256=%s\n' "${RLF_CAMPAIGN_BINDING_SHA256}"
  printf 'target_records=%s\n' "${TARGET_RECORDS}"
  printf 'prompt_language_ledger_sha256=%s\n' \
    "$(sha256sum -- "${RESULT_DIR}/prompt_language_ledger.tsv" | awk '{print $1}')"
  printf 'prompt_language_audit_sha256=%s\n' \
    "$(sha256sum -- "${RESULT_DIR}/prompt_language_audit.json" | awk '{print $1}')"
  printf 'prompt_language_telemetry_sha256=%s\n' \
    "$(sha256sum -- "${RESULT_DIR}/prompt_language_telemetry.txt" | awk '{print $1}')"
  printf 'prompt_checkpoint_inspection_sha256=%s\n' \
    "$(sha256sum -- "${RESULT_DIR}/prompt_checkpoint_inspection.txt" | awk '{print $1}')"
  printf 'prompt_source_manifest_sha256=%s\n' \
    "$(sha256sum -- "${RESULT_DIR}/prompt_source_manifest.tsv" | awk '{print $1}')"
  printf 'prompt_readiness_sha256=%s\n' \
    "$(sha256sum -- "${RESULT_DIR}/prompt_readiness.json" | awk '{print $1}')"
  printf 'checkpoint_sha256=%s\n' \
    "$(sha256sum -- "${CHECKPOINT}" | awk '{print $1}')"
  printf 'frontier_claim_authorized=false\n'
} >"${COMPLETION_TEMP}"
mv -f -- "${COMPLETION_TEMP}" "${RESULT_DIR}/prompt_stage_completion.env"
printf 'prompt_semantic_training_complete=true\ntarget_records=%s\ncheckpoint=%s\n' \
  "${TARGET_RECORDS}" "${CHECKPOINT}"
