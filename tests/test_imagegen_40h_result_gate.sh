#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="${ROOT}/scripts/v100/imagegen_40h_result_gate.sh"
WORK="$(mktemp -d)"; trap 'rm -rf -- "${WORK}"' EXIT
SHA="$(printf eval | sha256sum | awk '{print $1}')"
cat >"${WORK}/summary.json" <<EOF
{"schema":"rlf-imagegen-frozen-evaluation-v1","evaluation_manifest_sha256":"${SHA}","records":500,"mean_absolute_error":10,"mean_ssim":0.80,"nearest_example_mean_ssim":0.70,"patch_quilt_mean_ssim":0.75,"unseen_prompt_records":150,"unseen_prompt_mean_ssim":0.70,"paraphrase_records":100,"paraphrase_mean_ssim":0.68,"composition_records":100,"composition_mean_ssim":0.65,"natural_image_records":500,"natural_image_mean_ssim":0.72,"multilingual_records":100,"multilingual_mean_ssim":0.64,"unique_output_fraction":0.95,"perceptually_unique_output_fraction":0.93,"exact_training_copy_rate":0.01,"perceptual_training_copy_rate":0.01,"unresolved_record_rate":0.01,"internal_baselines_present":true,"external_diffusion_baseline_present":false,"frontier_claim_authorized":false}
EOF
cat >"${WORK}/comparison.json" <<EOF
{"schema":"rlf-imagegen-diffusion-comparison-v1","comparable":true,"same_frozen_evaluation_split":true,"evaluation_manifest_sha256":"${SHA}","records":500,"diffusion_mean_absolute_error":9,"diffusion_mean_ssim":0.85,"rlf_fid":20,"diffusion_fid":18,"rlf_prompt_alignment":0.70,"diffusion_prompt_alignment":0.75,"rlf_lpips_to_reference":0.30,"diffusion_lpips_to_reference":0.28,"external_artifacts_verified":true,"test_doubles":false,"frontier_claim_authorized":false}
EOF
cat >"${WORK}/resource.json" <<'EOF'
{"schema":"rlf-general-cuda-vram-v1","profile":"imagegen-v100-32g","peak_memory_mib":30000,"within_limit":true,"sampler_ok":true}
EOF
cat >"${WORK}/resume.json" <<'EOF'
{"schema":"rlf-imagegen-v100-resume-equivalence-v1","profile":"imagegen-v100-32g","passed":true,"physical_training_performed":true,"physical_cuda_local_updates_verified":true,"byte_identical":true,"test_doubles":false,"frontier_claim_authorized":false}
EOF
bash -n "${GATE}"
bash "${GATE}" "${WORK}/summary.json" "${WORK}/comparison.json" \
  "${WORK}/resource.json" "${WORK}/resume.json" "${WORK}/pass.json" \
  >"${WORK}/pass.txt"
grep -Fqx 'continuation_authorized=true' "${WORK}/pass.txt"
grep -q '"frontier_claim_authorized": false' "${WORK}/pass.json"

sed -i 's/"exact_training_copy_rate":0.01/"exact_training_copy_rate":0.20/' \
  "${WORK}/summary.json"
set +e
bash "${GATE}" "${WORK}/summary.json" "${WORK}/comparison.json" \
  "${WORK}/resource.json" "${WORK}/resume.json" "${WORK}/fail.json" \
  >"${WORK}/fail.txt"
STATUS=$?
set -e
[[ "${STATUS}" -eq 3 ]]
grep -Fqx 'continuation_authorized=false' "${WORK}/fail.txt"
grep -q '"memorization_passed": false' "${WORK}/fail.json"
printf 'imagegen_40h_result_gate_test=pass\n'
