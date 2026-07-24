#!/usr/bin/env bash
set -euo pipefail

INTERNAL="${1:?Usage: verify_imagegen_external_diffusion_baseline.sh INTERNAL_SUMMARY EXTERNAL_EVIDENCE ARTIFACT_MANIFEST OUTPUT}"
EXTERNAL="${2:?external evidence required}"
MANIFEST="${3:?artifact manifest required}"
OUTPUT="${4:?output required}"
for file in "${INTERNAL}" "${EXTERNAL}" "${MANIFEST}"; do
  [[ -f "${file}" && ! -L "${file}" ]] || {
    echo "comparison inputs must be regular non-symlink files" >&2; exit 2;
  }
done
[[ ! -e "${OUTPUT}" ]] || { echo "comparison output already exists" >&2; exit 2; }

env_value() {
  local matches
  matches="$(awk -F= -v key="$2" '$1 == key {print substr($0,length(key)+2)}' "$1")"
  [[ -n "${matches}" && "${matches}" != *$'\n'* ]] || {
    echo "$1 must contain exactly one $2" >&2; exit 1;
  }
  printf '%s' "${matches}"
}
json_string() {
  local values count
  count="$( (grep -Eo "\"$2\"[[:space:]]*:" "$1" || true) | wc -l | tr -d '[:space:]')"
  values="$(sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"([^\"]*)\".*/\1/p" "$1")"
  [[ "${count}" == 1 && -n "${values}" && "${values}" != *$'\n'* ]] || {
    echo "$1 must contain exactly one string $2" >&2; exit 1;
  }
  printf '%s' "${values}"
}
json_number() {
  local values count
  count="$( (grep -Eo "\"$2\"[[:space:]]*:" "$1" || true) | wc -l | tr -d '[:space:]')"
  values="$(sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*([-+0-9.eE]+).*/\1/p" "$1")"
  [[ "${count}" == 1 && -n "${values}" && "${values}" != *$'\n'* ]] || {
    echo "$1 must contain exactly one number $2" >&2; exit 1;
  }
  printf '%s' "${values}"
}
require_json_literal() {
  local count
  count="$( (grep -Eo "\"$2\"[[:space:]]*:[[:space:]]*$3([,}[:space:]]|$)" "$1" || true) | wc -l | tr -d '[:space:]')"
  [[ "${count}" == 1 ]] || { echo "$1 requires exactly one $2=$3" >&2; exit 1; }
}
valid_number() { [[ "$1" =~ ^-?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]; }
unit_interval() { valid_number "$1" && awk -v x="$1" 'BEGIN {exit !(x >= 0 && x <= 1)}'; }

[[ "$(json_string "${INTERNAL}" schema)" == rlf-imagegen-frozen-evaluation-v1 ]] || {
  echo "wrong internal evaluation schema" >&2; exit 1;
}
require_json_literal "${INTERNAL}" internal_baselines_present true
require_json_literal "${INTERNAL}" external_diffusion_baseline_present false
require_json_literal "${INTERNAL}" frontier_claim_authorized false
INTERNAL_EVAL_SHA="$(json_string "${INTERNAL}" evaluation_manifest_sha256)"
INTERNAL_RECORDS="$(json_number "${INTERNAL}" records)"
RLF_MAE="$(json_number "${INTERNAL}" mean_absolute_error)"
RLF_SSIM="$(json_number "${INTERNAL}" mean_ssim)"
RLF_DIVERSITY="$(json_number "${INTERNAL}" unique_output_fraction)"
RLF_COPY="$(json_number "${INTERNAL}" exact_training_copy_rate)"
INTERNAL_MANIFEST_SHA="$(json_string "${INTERNAL}" raw_artifact_manifest_sha256)"
[[ "${INTERNAL_EVAL_SHA}" =~ ^[0-9a-f]{64}$ && "${INTERNAL_RECORDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "internal evaluation identity is incomplete" >&2; exit 1;
}
for value in "${RLF_MAE}" "${RLF_SSIM}" "${RLF_DIVERSITY}" "${RLF_COPY}"; do
  valid_number "${value}" || { echo "invalid internal metric" >&2; exit 1; }
done
INTERNAL_BASE="$(cd "$(dirname "${INTERNAL}")" && pwd)"
INTERNAL_MANIFEST="${INTERNAL_BASE}/raw_artifacts.tsv"
[[ -f "${INTERNAL_MANIFEST}" && ! -L "${INTERNAL_MANIFEST}" &&
   "${INTERNAL_MANIFEST_SHA}" =~ ^[0-9a-f]{64}$ &&
   "$(sha256sum -- "${INTERNAL_MANIFEST}" | awk '{print $1}')" == "${INTERNAL_MANIFEST_SHA}" ]] || {
  echo "internal raw-artifact manifest binding mismatch" >&2; exit 1;
}
INTERNAL_ROWS=0
declare -A INTERNAL_SEEN=()
while IFS=$'\t' read -r relative sha bytes extra; do
  [[ -z "${relative}" ]] && continue
  [[ "${relative}" != path || "${sha}" != sha256 || "${bytes}" != bytes ]] || continue
  [[ -z "${extra:-}" && "${relative}" != /* && "${relative}" != *..* &&
     "${sha}" =~ ^[0-9a-f]{64}$ && "${bytes}" =~ ^[1-9][0-9]*$ &&
     -z "${INTERNAL_SEEN[${relative}]:-}" ]] || {
    echo "invalid internal raw-artifact row" >&2; exit 1;
  }
  INTERNAL_SEEN["${relative}"]=1
  resolved="$(realpath -m "${INTERNAL_BASE}/${relative}")"
  [[ "${resolved}" == "${INTERNAL_BASE}/"* && -f "${resolved}" &&
     ! -L "${resolved}" && "$(stat -c %s -- "${resolved}")" == "${bytes}" &&
     "$(sha256sum -- "${resolved}" | awk '{print $1}')" == "${sha}" ]] || {
    echo "internal image artifact verification failed: ${relative}" >&2; exit 1;
  }
  INTERNAL_ROWS=$((INTERNAL_ROWS + 1))
done <"${INTERNAL_MANIFEST}"
((INTERNAL_ROWS >= INTERNAL_RECORDS * 3 + 1)) || {
  echo "internal artifact manifest is incomplete" >&2; exit 1;
}

[[ "$(env_value "${EXTERNAL}" schema)" == rlf-imagegen-external-diffusion-baseline-v1 &&
   "$(env_value "${EXTERNAL}" baseline_family)" == diffusion &&
   "$(env_value "${EXTERNAL}" independent_evaluation)" == true &&
   "$(env_value "${EXTERNAL}" test_doubles)" == false &&
   "$(env_value "${EXTERNAL}" frontier_claim_authorized)" == false ]] || {
  echo "external baseline is not independent physical diffusion evidence" >&2; exit 1;
}
PROVIDER="$(env_value "${EXTERNAL}" provider)"
MODEL_ID="$(env_value "${EXTERNAL}" model_id)"
MODEL_REVISION="$(env_value "${EXTERNAL}" model_revision)"
EVALUATOR_ID="$(env_value "${EXTERNAL}" evaluator_id)"
EVALUATOR_REVISION="$(env_value "${EXTERNAL}" evaluator_revision)"
EXTERNAL_EVAL_SHA="$(env_value "${EXTERNAL}" evaluation_manifest_sha256)"
EXTERNAL_RECORDS="$(env_value "${EXTERNAL}" records)"
EXT_MAE="$(env_value "${EXTERNAL}" mean_absolute_error)"
EXT_SSIM="$(env_value "${EXTERNAL}" mean_ssim)"
EXT_DIVERSITY="$(env_value "${EXTERNAL}" unique_output_fraction)"
EXT_COPY="$(env_value "${EXTERNAL}" exact_training_copy_rate)"
RLF_FID="$(env_value "${EXTERNAL}" rlf_fid)"
EXT_FID="$(env_value "${EXTERNAL}" diffusion_fid)"
RLF_ALIGNMENT="$(env_value "${EXTERNAL}" rlf_prompt_alignment)"
EXT_ALIGNMENT="$(env_value "${EXTERNAL}" diffusion_prompt_alignment)"
RLF_LPIPS="$(env_value "${EXTERNAL}" rlf_lpips_to_reference)"
EXT_LPIPS="$(env_value "${EXTERNAL}" diffusion_lpips_to_reference)"
BOUND_INTERNAL_MANIFEST_SHA="$(env_value "${EXTERNAL}" internal_artifact_manifest_sha256)"
MANIFEST_SHA="$(env_value "${EXTERNAL}" artifact_manifest_sha256)"
SAFE_ID='^[A-Za-z0-9._:/+@-]+$'
[[ "${PROVIDER}" =~ ${SAFE_ID} && "${MODEL_ID}" =~ ${SAFE_ID} &&
   "${MODEL_REVISION}" =~ ${SAFE_ID} && "${EVALUATOR_ID}" =~ ${SAFE_ID} &&
   "${EVALUATOR_REVISION}" =~ ${SAFE_ID} &&
   "${EXTERNAL_EVAL_SHA}" == "${INTERNAL_EVAL_SHA}" &&
   "${EXTERNAL_RECORDS}" == "${INTERNAL_RECORDS}" &&
   "${BOUND_INTERNAL_MANIFEST_SHA}" == "${INTERNAL_MANIFEST_SHA}" &&
   "${MANIFEST_SHA}" =~ ^[0-9a-f]{64}$ &&
   "$(sha256sum -- "${MANIFEST}" | awk '{print $1}')" == "${MANIFEST_SHA}" ]] || {
  echo "external baseline identity/split/manifest binding mismatch" >&2; exit 1;
}
valid_number "${EXT_MAE}" && unit_interval "${EXT_SSIM}" &&
  unit_interval "${EXT_DIVERSITY}" && unit_interval "${EXT_COPY}" || {
  echo "invalid external baseline metric" >&2; exit 1;
}
for value in "${RLF_FID}" "${EXT_FID}" "${RLF_LPIPS}" "${EXT_LPIPS}"; do
  valid_number "${value}" && ge_zero="$(awk -v x="${value}" 'BEGIN {print x >= 0 ? "true" : "false"}')" &&
    [[ "${ge_zero}" == true ]] || { echo "invalid external perceptual metric" >&2; exit 1; }
done
unit_interval "${RLF_ALIGNMENT}" && unit_interval "${EXT_ALIGNMENT}" || {
  echo "invalid external prompt-alignment metric" >&2; exit 1;
}

BASE="$(cd "$(dirname "${MANIFEST}")" && pwd)"
ROWS=0
declare -A SEEN=()
while IFS=$'\t' read -r relative sha bytes extra; do
  [[ -z "${relative}" ]] && continue
  [[ "${relative}" != path || "${sha}" != sha256 || "${bytes}" != bytes ]] || continue
  [[ -z "${extra:-}" && "${relative}" != /* && "${relative}" != *..* &&
     "${sha}" =~ ^[0-9a-f]{64}$ && "${bytes}" =~ ^[1-9][0-9]*$ &&
     -z "${SEEN[${relative}]:-}" ]] || {
    echo "invalid or duplicate external artifact row" >&2; exit 1;
  }
  SEEN["${relative}"]=1
  resolved="$(realpath -m "${BASE}/${relative}")"
  [[ "${resolved}" == "${BASE}/"* && -f "${resolved}" && ! -L "${resolved}" &&
     "$(stat -c %s -- "${resolved}")" == "${bytes}" &&
     "$(sha256sum -- "${resolved}" | awk '{print $1}')" == "${sha}" ]] || {
    echo "external diffusion artifact verification failed: ${relative}" >&2; exit 1;
  }
  ROWS=$((ROWS + 1))
done <"${MANIFEST}"
((ROWS >= EXTERNAL_RECORDS)) || {
  echo "external artifact manifest has fewer files than evaluation records" >&2; exit 1;
}

TEMP="$(mktemp "$(dirname "${OUTPUT}")/.imagegen-comparison.XXXXXX")"
trap 'rm -f -- "${TEMP}"' EXIT
cat >"${TEMP}" <<EOF
{
  "schema": "rlf-imagegen-diffusion-comparison-v1",
  "comparable": true,
  "same_frozen_evaluation_split": true,
  "evaluation_manifest_sha256": "${INTERNAL_EVAL_SHA}",
  "records": ${INTERNAL_RECORDS},
  "external_provider": "${PROVIDER}",
  "external_model_id": "${MODEL_ID}",
  "external_model_revision": "${MODEL_REVISION}",
  "external_evaluator_id": "${EVALUATOR_ID}",
  "external_evaluator_revision": "${EVALUATOR_REVISION}",
  "rlf_mean_absolute_error": ${RLF_MAE},
  "diffusion_mean_absolute_error": ${EXT_MAE},
  "rlf_mean_ssim": ${RLF_SSIM},
  "diffusion_mean_ssim": ${EXT_SSIM},
  "rlf_unique_output_fraction": ${RLF_DIVERSITY},
  "diffusion_unique_output_fraction": ${EXT_DIVERSITY},
  "rlf_exact_training_copy_rate": ${RLF_COPY},
  "diffusion_exact_training_copy_rate": ${EXT_COPY},
  "rlf_fid": ${RLF_FID},
  "diffusion_fid": ${EXT_FID},
  "rlf_prompt_alignment": ${RLF_ALIGNMENT},
  "diffusion_prompt_alignment": ${EXT_ALIGNMENT},
  "rlf_lpips_to_reference": ${RLF_LPIPS},
  "diffusion_lpips_to_reference": ${EXT_LPIPS},
  "external_artifacts_verified": true,
  "test_doubles": false,
  "frontier_claim_authorized": false
}
EOF
mv -- "${TEMP}" "${OUTPUT}"
trap - EXIT
printf 'comparison_ready=true\nevaluation_manifest_sha256=%s\nrecords=%s\nfrontier_claim_authorized=false\n' \
  "${INTERNAL_EVAL_SHA}" "${INTERNAL_RECORDS}"
