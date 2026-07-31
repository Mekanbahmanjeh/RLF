#!/usr/bin/env bash
set -euo pipefail

SUMMARY="${1:?Usage: imagegen_40h_result_gate.sh INTERNAL_SUMMARY DIFFUSION_COMPARISON RESOURCE_SUMMARY RESUME_REPORT OUTPUT}"
COMPARISON="${2:?diffusion comparison required}"
RESOURCE="${3:?resource summary required}"
RESUME="${4:?resume report required}"
OUTPUT="${5:?output required}"
for file in "${SUMMARY}" "${COMPARISON}" "${RESOURCE}" "${RESUME}"; do
  [[ -f "${file}" && ! -L "${file}" ]] || {
    echo "result-gate inputs must be regular non-symlink files" >&2; exit 2;
  }
done
[[ ! -e "${OUTPUT}" ]] || { echo "result-gate output already exists" >&2; exit 2; }

string() {
  local values count
  count="$( (grep -Eo "\"$2\"[[:space:]]*:" "$1" || true) | wc -l | tr -d '[:space:]')"
  values="$(sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"([^\"]*)\".*/\1/p" "$1")"
  [[ "${count}" == 1 && -n "${values}" && "${values}" != *$'\n'* ]] || {
    echo "$1 requires exactly one string $2" >&2; exit 1;
  }
  printf '%s' "${values}"
}
number() {
  local values count
  count="$( (grep -Eo "\"$2\"[[:space:]]*:" "$1" || true) | wc -l | tr -d '[:space:]')"
  values="$(sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*([-+0-9.eE]+).*/\1/p" "$1")"
  [[ "${count}" == 1 && -n "${values}" && "${values}" != *$'\n'* ]] || {
    echo "$1 requires exactly one number $2" >&2; exit 1;
  }
  printf '%s' "${values}"
}
literal() {
  local count
  count="$( (grep -Eo "\"$2\"[[:space:]]*:[[:space:]]*$3([,}[:space:]]|$)" "$1" || true) | wc -l | tr -d '[:space:]')"
  [[ "${count}" == 1 ]] || { echo "$1 requires exactly one $2=$3" >&2; exit 1; }
}
ge() { awk -v a="$1" -v b="$2" 'BEGIN {exit !(a >= b)}'; }
le() { awk -v a="$1" -v b="$2" 'BEGIN {exit !(a <= b)}'; }
boolean() { "$@" && echo true || echo false; }

[[ "$(string "${SUMMARY}" schema)" == rlf-imagegen-frozen-evaluation-v1 &&
   "$(string "${COMPARISON}" schema)" == rlf-imagegen-diffusion-comparison-v1 &&
   "$(string "${RESOURCE}" schema)" == rlf-general-cuda-vram-v1 &&
   "$(string "${RESUME}" schema)" == rlf-imagegen-v100-resume-equivalence-v1 ]] || {
  echo "result-gate schema mismatch" >&2; exit 1;
}
literal "${SUMMARY}" internal_baselines_present true
literal "${SUMMARY}" external_diffusion_baseline_present false
literal "${SUMMARY}" frontier_claim_authorized false
literal "${COMPARISON}" comparable true
literal "${COMPARISON}" same_frozen_evaluation_split true
literal "${COMPARISON}" external_artifacts_verified true
literal "${COMPARISON}" test_doubles false
literal "${COMPARISON}" frontier_claim_authorized false
literal "${RESOURCE}" within_limit true
literal "${RESOURCE}" sampler_ok true
literal "${RESUME}" passed true
literal "${RESUME}" physical_training_performed true
literal "${RESUME}" physical_cuda_local_updates_verified true
literal "${RESUME}" byte_identical true
literal "${RESUME}" test_doubles false
literal "${RESUME}" frontier_claim_authorized false

EVAL_SHA="$(string "${SUMMARY}" evaluation_manifest_sha256)"
[[ "$(string "${COMPARISON}" evaluation_manifest_sha256)" == "${EVAL_SHA}" &&
   "$(string "${RESOURCE}" profile)" == imagegen-v100-32g &&
   "$(string "${RESUME}" profile)" == imagegen-v100-32g ]] || {
  echo "result-gate split/profile binding mismatch" >&2; exit 1;
}
RECORDS="$(number "${SUMMARY}" records)"
[[ "$(number "${COMPARISON}" records)" == "${RECORDS}" ]] || {
  echo "comparison record count mismatch" >&2; exit 1;
}
UNSEEN="$(number "${SUMMARY}" unseen_prompt_records)"
COMPOSITIONS="$(number "${SUMMARY}" composition_records)"
PARAPHRASES="$(number "${SUMMARY}" paraphrase_records)"
NATURAL_IMAGES="$(number "${SUMMARY}" natural_image_records)"
MULTILINGUAL="$(number "${SUMMARY}" multilingual_records)"
RLF_SSIM="$(number "${SUMMARY}" mean_ssim)"
RLF_MAE="$(number "${SUMMARY}" mean_absolute_error)"
NEAREST_SSIM="$(number "${SUMMARY}" nearest_example_mean_ssim)"
QUILT_SSIM="$(number "${SUMMARY}" patch_quilt_mean_ssim)"
UNSEEN_SSIM="$(number "${SUMMARY}" unseen_prompt_mean_ssim)"
COMPOSITION_SSIM="$(number "${SUMMARY}" composition_mean_ssim)"
PARAPHRASE_SSIM="$(number "${SUMMARY}" paraphrase_mean_ssim)"
NATURAL_IMAGE_SSIM="$(number "${SUMMARY}" natural_image_mean_ssim)"
MULTILINGUAL_SSIM="$(number "${SUMMARY}" multilingual_mean_ssim)"
DIVERSITY="$(number "${SUMMARY}" unique_output_fraction)"
PERCEPTUAL_DIVERSITY="$(number "${SUMMARY}" perceptually_unique_output_fraction)"
COPY_RATE="$(number "${SUMMARY}" exact_training_copy_rate)"
PERCEPTUAL_COPY_RATE="$(number "${SUMMARY}" perceptual_training_copy_rate)"
UNRESOLVED="$(number "${SUMMARY}" unresolved_record_rate)"
DIFFUSION_SSIM="$(number "${COMPARISON}" diffusion_mean_ssim)"
DIFFUSION_MAE="$(number "${COMPARISON}" diffusion_mean_absolute_error)"
RLF_FID="$(number "${COMPARISON}" rlf_fid)"
DIFFUSION_FID="$(number "${COMPARISON}" diffusion_fid)"
RLF_ALIGNMENT="$(number "${COMPARISON}" rlf_prompt_alignment)"
DIFFUSION_ALIGNMENT="$(number "${COMPARISON}" diffusion_prompt_alignment)"
RLF_LPIPS="$(number "${COMPARISON}" rlf_lpips_to_reference)"
DIFFUSION_LPIPS="$(number "${COMPARISON}" diffusion_lpips_to_reference)"
PEAK_MIB="$(number "${RESOURCE}" peak_memory_mib)"

ENOUGH_RECORDS="$(boolean ge "${RECORDS}" 500)"
ENOUGH_UNSEEN="$(boolean ge "${UNSEEN}" 100)"
ENOUGH_COMPOSITIONS="$(boolean ge "${COMPOSITIONS}" 100)"
ENOUGH_PARAPHRASES="$(boolean ge "${PARAPHRASES}" 100)"
ENOUGH_NATURAL_IMAGES="$(boolean ge "${NATURAL_IMAGES}" 500)"
ENOUGH_MULTILINGUAL="$(boolean ge "${MULTILINGUAL}" 100)"
UNSEEN_QUALITY="$(boolean ge "${UNSEEN_SSIM}" 0.5)"
COMPOSITION_QUALITY="$(boolean ge "${COMPOSITION_SSIM}" 0.5)"
PARAPHRASE_QUALITY="$(boolean ge "${PARAPHRASE_SSIM}" 0.5)"
NATURAL_IMAGE_QUALITY="$(boolean ge "${NATURAL_IMAGE_SSIM}" 0.5)"
MULTILINGUAL_QUALITY="$(boolean ge "${MULTILINGUAL_SSIM}" 0.5)"
INTERNAL_BASELINES="$(boolean awk -v r="${RLF_SSIM}" -v n="${NEAREST_SSIM}" -v q="${QUILT_SSIM}" 'BEGIN {exit !(r >= n && r >= q)}')"
DIFFUSION_BAND="$(boolean awk -v rs="${RLF_SSIM}" -v rm="${RLF_MAE}" -v ds="${DIFFUSION_SSIM}" -v dm="${DIFFUSION_MAE}" -v rf="${RLF_FID}" -v df="${DIFFUSION_FID}" -v ra="${RLF_ALIGNMENT}" -v da="${DIFFUSION_ALIGNMENT}" -v rp="${RLF_LPIPS}" -v dp="${DIFFUSION_LPIPS}" 'BEGIN {exit !(rs >= ds * 0.8 && rm <= dm * 1.25 && rf <= df * 1.25 && ra >= da * 0.8 && rp <= dp * 1.25)}')"
DIVERSITY_PASS="$(boolean ge "${DIVERSITY}" 0.9)"
PERCEPTUAL_DIVERSITY_PASS="$(boolean ge "${PERCEPTUAL_DIVERSITY}" 0.9)"
COPY_PASS="$(boolean le "${COPY_RATE}" 0.02)"
PERCEPTUAL_COPY_PASS="$(boolean le "${PERCEPTUAL_COPY_RATE}" 0.02)"
UNRESOLVED_PASS="$(boolean le "${UNRESOLVED}" 0.05)"
VRAM_PASS="$(boolean le "${PEAK_MIB}" 30720)"
AUTHORIZED=true
for result in "${ENOUGH_RECORDS}" "${ENOUGH_UNSEEN}" "${ENOUGH_COMPOSITIONS}" \
  "${ENOUGH_PARAPHRASES}" "${ENOUGH_NATURAL_IMAGES}" \
  "${ENOUGH_MULTILINGUAL}" "${UNSEEN_QUALITY}" "${COMPOSITION_QUALITY}" \
  "${PARAPHRASE_QUALITY}" "${NATURAL_IMAGE_QUALITY}" \
  "${MULTILINGUAL_QUALITY}" "${INTERNAL_BASELINES}" \
  "${DIFFUSION_BAND}" "${DIVERSITY_PASS}" "${PERCEPTUAL_DIVERSITY_PASS}" \
  "${COPY_PASS}" "${PERCEPTUAL_COPY_PASS}" \
  "${UNRESOLVED_PASS}" "${VRAM_PASS}"; do
  [[ "${result}" == true ]] || AUTHORIZED=false
done

TEMP="$(mktemp "$(dirname "${OUTPUT}")/.imagegen-result-gate.XXXXXX")"
trap 'rm -f -- "${TEMP}"' EXIT
cat >"${TEMP}" <<EOF
{
  "schema": "rlf-imagegen-v100-40h-result-gate-v1",
  "continuation_authorized": ${AUTHORIZED},
  "evaluation_manifest_sha256": "${EVAL_SHA}",
  "records": ${RECORDS},
  "minimum_records_passed": ${ENOUGH_RECORDS},
  "minimum_unseen_prompts_passed": ${ENOUGH_UNSEEN},
  "minimum_compositions_passed": ${ENOUGH_COMPOSITIONS},
  "minimum_paraphrases_passed": ${ENOUGH_PARAPHRASES},
  "minimum_natural_images_passed": ${ENOUGH_NATURAL_IMAGES},
  "minimum_multilingual_prompts_passed": ${ENOUGH_MULTILINGUAL},
  "unseen_prompt_quality_passed": ${UNSEEN_QUALITY},
  "composition_quality_passed": ${COMPOSITION_QUALITY},
  "paraphrase_quality_passed": ${PARAPHRASE_QUALITY},
  "natural_image_quality_passed": ${NATURAL_IMAGE_QUALITY},
  "multilingual_quality_passed": ${MULTILINGUAL_QUALITY},
  "internal_baselines_passed": ${INTERNAL_BASELINES},
  "diffusion_comparison_band_passed": ${DIFFUSION_BAND},
  "diversity_passed": ${DIVERSITY_PASS},
  "perceptual_diversity_passed": ${PERCEPTUAL_DIVERSITY_PASS},
  "memorization_passed": ${COPY_PASS},
  "perceptual_memorization_passed": ${PERCEPTUAL_COPY_PASS},
  "unresolved_rate_passed": ${UNRESOLVED_PASS},
  "vram_passed": ${VRAM_PASS},
  "resume_equivalence_passed": true,
  "thresholds": {"minimum_records":500,"minimum_unseen_prompts":100,"minimum_paraphrases":100,"minimum_compositions":100,"minimum_natural_images":500,"minimum_multilingual_prompts":100,"minimum_unseen_ssim":0.5,"minimum_paraphrase_ssim":0.5,"minimum_composition_ssim":0.5,"minimum_natural_image_ssim":0.5,"minimum_multilingual_ssim":0.5,"minimum_unique_output_fraction":0.9,"minimum_perceptually_unique_output_fraction":0.9,"maximum_exact_training_copy_rate":0.02,"maximum_perceptual_training_copy_rate":0.02,"maximum_unresolved_record_rate":0.05,"maximum_peak_memory_mib":30720,"minimum_diffusion_ssim_ratio":0.8,"maximum_diffusion_mae_ratio":1.25,"maximum_diffusion_fid_ratio":1.25,"minimum_diffusion_prompt_alignment_ratio":0.8,"maximum_diffusion_lpips_ratio":1.25},
  "state_of_art_claim_authorized": false,
  "frontier_claim_authorized": false,
  "efficiency_claim_authorized": false
}
EOF
mv -- "${TEMP}" "${OUTPUT}"; trap - EXIT
printf 'continuation_authorized=%s\nfrontier_claim_authorized=false\n' "${AUTHORIZED}"
[[ "${AUTHORIZED}" == true ]] || exit 3
