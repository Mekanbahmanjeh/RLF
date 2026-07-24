#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERIFY="${ROOT}/scripts/v100/verify_imagegen_external_diffusion_baseline.sh"
WORK="$(mktemp -d)"; trap 'rm -rf -- "${WORK}"' EXIT
SHA="$(printf frozen-split | sha256sum | awk '{print $1}')"
for name in predictions.tsv generated.ppm nearest.ppm quilt.ppm; do
  printf 'internal %s\n' "${name}" >"${WORK}/${name}"
done
printf 'path\tsha256\tbytes\n' >"${WORK}/raw_artifacts.tsv"
for name in predictions.tsv generated.ppm nearest.ppm quilt.ppm; do
  printf '%s\t%s\t%s\n' "${name}" \
    "$(sha256sum -- "${WORK}/${name}" | awk '{print $1}')" \
    "$(stat -c %s -- "${WORK}/${name}")" >>"${WORK}/raw_artifacts.tsv"
done
RAW_SHA="$(sha256sum -- "${WORK}/raw_artifacts.tsv" | awk '{print $1}')"
cat >"${WORK}/internal.json" <<EOF
{"schema":"rlf-imagegen-frozen-evaluation-v1","evaluation_manifest_sha256":"${SHA}","raw_artifact_manifest_sha256":"${RAW_SHA}","records":1,"mean_absolute_error":4.0,"mean_ssim":0.8,"unique_output_fraction":1.0,"exact_training_copy_rate":0.0,"internal_baselines_present":true,"external_diffusion_baseline_present":false,"frontier_claim_authorized":false}
EOF
printf 'external generated image\n' >"${WORK}/sample.ppm"
ART_SHA="$(sha256sum -- "${WORK}/sample.ppm" | awk '{print $1}')"
ART_BYTES="$(stat -c %s -- "${WORK}/sample.ppm")"
printf 'path\tsha256\tbytes\nsample.ppm\t%s\t%s\n' "${ART_SHA}" "${ART_BYTES}" \
  >"${WORK}/artifacts.tsv"
MANIFEST_SHA="$(sha256sum -- "${WORK}/artifacts.tsv" | awk '{print $1}')"
cat >"${WORK}/external.env" <<EOF
schema=rlf-imagegen-external-diffusion-baseline-v1
baseline_family=diffusion
provider=independent-fixture
model_id=diffusion-fixture
model_revision=immutable-revision
evaluator_id=independent-fid-clip-lpips-suite
evaluator_revision=immutable-evaluator-revision
evaluation_manifest_sha256=${SHA}
records=1
mean_absolute_error=3.0
mean_ssim=0.85
unique_output_fraction=1.0
exact_training_copy_rate=0.0
rlf_fid=20.0
diffusion_fid=18.0
rlf_prompt_alignment=0.70
diffusion_prompt_alignment=0.75
rlf_lpips_to_reference=0.30
diffusion_lpips_to_reference=0.28
internal_artifact_manifest_sha256=${RAW_SHA}
artifact_manifest_sha256=${MANIFEST_SHA}
independent_evaluation=true
test_doubles=false
frontier_claim_authorized=false
EOF
bash -n "${VERIFY}"
bash "${VERIFY}" "${WORK}/internal.json" "${WORK}/external.env" \
  "${WORK}/artifacts.tsv" "${WORK}/comparison.json" >"${WORK}/stdout.txt"
grep -Fqx 'comparison_ready=true' "${WORK}/stdout.txt"
grep -q '"comparable": true' "${WORK}/comparison.json"
grep -q '"frontier_claim_authorized": false' "${WORK}/comparison.json"

sed -i "s/evaluation_manifest_sha256=${SHA}/evaluation_manifest_sha256=$(printf other | sha256sum | awk '{print $1}')/" \
  "${WORK}/external.env"
set +e
bash "${VERIFY}" "${WORK}/internal.json" "${WORK}/external.env" \
  "${WORK}/artifacts.tsv" "${WORK}/bad.json" >/dev/null 2>"${WORK}/bad.err"
STATUS=$?
set -e
[[ "${STATUS}" -eq 1 && ! -e "${WORK}/bad.json" ]]
grep -q 'binding mismatch' "${WORK}/bad.err"
printf 'imagegen_external_diffusion_baseline_test=pass\n'
