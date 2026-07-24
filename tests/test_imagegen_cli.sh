#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${1:?solstice executable is required}"
WORK="${ROOT}/results/imagegen_cli_shell_test"
[[ "$(realpath -m "${WORK}")" == "$(realpath "${ROOT}")/results/imagegen_cli_shell_test" ]]
rm -rf -- "${WORK}"
mkdir -p "${WORK}"
trap 'rm -rf -- "${WORK}"' EXIT

{
  printf 'P6\n8 8\n255\n'
  for _ in $(seq 1 64); do printf '\000\000\000'; done
} >"${WORK}/black.ppm"
{
  printf 'P6\n8 8\n255\n'
  for _ in $(seq 1 64); do printf '\040\000\000'; done
} >"${WORK}/red.ppm"

"${BIN}" imagegen-profile-info --profile imagegen-v100-32g \
  >"${WORK}/profile.txt"
grep -q '^gpu_working_set_gib=30$' "${WORK}/profile.txt"
grep -q '^maximum_modes=48000000$' "${WORK}/profile.txt"

"${BIN}" imagegen-bootstrap --profile imagegen-v100-32g \
  --checkpoint "${WORK}/model.rlfimg" --seed 77 \
  >"${WORK}/bootstrap.txt"

# Prompt-language ingestion is provenance-bound, exact-target, shard-resumable,
# and keeps its accounting separate from image training_step.
printf 'crimson pigment appears on a brightly illuminated canvas\n' \
  >"${WORK}/language-0001.txt"
printf 'orbital mechanics predicts a distant satellite trajectory\n' \
  >"${WORK}/language-0002.txt"
LANGUAGE_SHA_1="$(sha256sum -- "${WORK}/language-0001.txt" | awk '{print $1}')"
LANGUAGE_SHA_2="$(sha256sum -- "${WORK}/language-0002.txt" | awk '{print $1}')"
printf 'language-0001\ttext\ttrain\ttext\ten\tgeneral\ttext_lines\tlanguage-0001.txt\tlocal:test-fixture\tCC0-1.0\t2026-07-23\t%s\ttest-v1\tnone\tnone\ttrue\n' \
  "${LANGUAGE_SHA_1}" >"${WORK}/language-ledger.tsv"
printf 'language-0002\ttext\ttrain\ttext\ten\tgeneral\ttext_lines\tlanguage-0002.txt\tlocal:test-fixture-2\tCC0-1.0\t2026-07-23\t%s\ttest-v1\tnone\tnone\ttrue\n' \
  "${LANGUAGE_SHA_2}" >>"${WORK}/language-ledger.tsv"
"${BIN}" imagegen-bootstrap --profile imagegen-v100-32g \
  --checkpoint "${WORK}/language-model.rlfimg" --seed 79 >/dev/null
set +e
"${BIN}" imagegen-train-language-ledger --profile imagegen-v100-32g \
  --checkpoint "${WORK}/language-model.rlfimg" \
  --ledger "${WORK}/language-ledger.tsv" --target-training-records 2 \
  --maximum-new-shards 1 --output "${WORK}/language-audit.json" \
  >"${WORK}/language-partial.txt"
LANGUAGE_PARTIAL_STATUS=$?
set -e
[[ "${LANGUAGE_PARTIAL_STATUS}" -eq 4 ]]
grep -q '^prompt_language_records=1$' "${WORK}/language-partial.txt"
grep -q '^training_target_reached=false$' "${WORK}/language-partial.txt"
"${BIN}" imagegen-train-language-ledger --profile imagegen-v100-32g \
  --checkpoint "${WORK}/language-model.rlfimg" \
  --ledger "${WORK}/language-ledger.tsv" --target-training-records 2 \
  --output "${WORK}/language-audit.json" >"${WORK}/language-complete.txt"
grep -q '^prompt_language_records=2$' "${WORK}/language-complete.txt"
grep -q '^training_target_reached=true$' "${WORK}/language-complete.txt"
"${BIN}" imagegen-inspect --checkpoint "${WORK}/language-model.rlfimg" \
  >"${WORK}/language-inspect.txt"
grep -q '^training_step=0$' "${WORK}/language-inspect.txt"
grep -q '^prompt_language_records=2$' "${WORK}/language-inspect.txt"
grep -Eq '^prompt_semantic_modes=[1-9][0-9]*$' \
  "${WORK}/language-inspect.txt"
LANGUAGE_MODEL_SHA="$(sha256sum -- "${WORK}/language-model.rlfimg" | awk '{print $1}')"
"${BIN}" imagegen-train-language-ledger --profile imagegen-v100-32g \
  --checkpoint "${WORK}/language-model.rlfimg" \
  --ledger "${WORK}/language-ledger.tsv" --target-training-records 2 \
  --output "${WORK}/language-audit.json" >"${WORK}/language-resume.txt"
grep -q '^trained_prompt_language_shards=0$' "${WORK}/language-resume.txt"
grep -q '^resumed_prompt_language_shards=2$' "${WORK}/language-resume.txt"
[[ "$(sha256sum -- "${WORK}/language-model.rlfimg" | awk '{print $1}')" == \
   "${LANGUAGE_MODEL_SHA}" ]]

"${BIN}" imagegen-train-pair --profile imagegen-v100-32g \
  --checkpoint "${WORK}/model.rlfimg" --image "${WORK}/black.ppm" \
  --target-image "${WORK}/red.ppm" --prompt red-shift \
  >"${WORK}/train-1.txt"
grep -q '^training_step=1$' "${WORK}/train-1.txt"
grep -q '^modes_created=1$' "${WORK}/train-1.txt"

"${BIN}" imagegen-verify --profile imagegen-v100-32g \
  --checkpoint "${WORK}/model.rlfimg" >/dev/null
"${BIN}" imagegen-generate --profile imagegen-v100-32g \
  --checkpoint "${WORK}/model.rlfimg" --image "${WORK}/black.ppm" \
  --transform red-shift --output "${WORK}/generated.ppm" \
  >"${WORK}/generate.txt"
cmp "${WORK}/red.ppm" "${WORK}/generated.ppm"
grep -q 'not diffusion parity or frontier evidence' \
  "${WORK}/generated.ppm.json"

# Prompt-only generation uses a canonical neutral phase seed. It is still an
# canonical neutral-seed mapping; no caller-supplied base image is needed.
"${BIN}" imagegen-bootstrap --profile imagegen-v100-32g \
  --checkpoint "${WORK}/prompt-model.rlfimg" --seed 78 >/dev/null
"${BIN}" imagegen-train-pair --profile imagegen-v100-32g \
  --checkpoint "${WORK}/prompt-model.rlfimg" \
  --target-image "${WORK}/red.ppm" --prompt solid-red \
  >"${WORK}/prompt-train.txt"
grep -q '^source_kind=neutral-prompt$' "${WORK}/prompt-train.txt"
"${BIN}" imagegen-generate --profile imagegen-v100-32g \
  --checkpoint "${WORK}/prompt-model.rlfimg" --prompt solid-red \
  --width 8 --height 8 --output "${WORK}/prompt-generated.ppm" \
  >"${WORK}/prompt-generate.txt"
cmp "${WORK}/red.ppm" "${WORK}/prompt-generated.ppm"
grep -q '"generation_mode": "prompt-only-neutral-seed"' \
  "${WORK}/prompt-generated.ppm.json"
grep -q 'not diffusion parity or frontier evidence' \
  "${WORK}/prompt-generated.ppm.json"
set +e
"${BIN}" imagegen-generate --profile imagegen-v100-32g \
  --checkpoint "${WORK}/prompt-model.rlfimg" --prompt solid-red \
  --output "${WORK}/missing-size.ppm" >/dev/null 2>&1
MISSING_SIZE_STATUS=$?
set -e
[[ "${MISSING_SIZE_STATUS}" -ne 0 && ! -e "${WORK}/missing-size.ppm" ]]

"${BIN}" imagegen-inspect --checkpoint "${WORK}/model.rlfimg" \
  >"${WORK}/inspect.txt"
grep -q '^format_version=4$' "${WORK}/inspect.txt"
grep -q '^architecture=resonant-fabric$' "${WORK}/inspect.txt"
grep -q '^training_step=1$' "${WORK}/inspect.txt"
grep -q '^learned_modes=1$' "${WORK}/inspect.txt"

"${BIN}" imagegen-train-pair --profile imagegen-v100-32g \
  --checkpoint "${WORK}/model.rlfimg" --image "${WORK}/black.ppm" \
  --target-image "${WORK}/red.ppm" --prompt red-shift \
  >"${WORK}/train-2.txt"
grep -q '^training_step=2$' "${WORK}/train-2.txt"
grep -q '^modes_updated=1$' "${WORK}/train-2.txt"

"${BIN}" imagegen-bootstrap --profile imagegen-reference \
  --architecture patch-quilt-baseline \
  --checkpoint "${WORK}/baseline.rlfimg" >/dev/null
set +e
"${BIN}" imagegen-train-pair --checkpoint "${WORK}/baseline.rlfimg" \
  --image "${WORK}/black.ppm" --target-image "${WORK}/red.ppm" \
  --prompt forbidden-relabel >/dev/null 2>&1
BASELINE_STATUS=$?
set -e
[[ "${BASELINE_STATUS}" -ne 0 ]]

"${BIN}" imagegen-bootstrap --profile imagegen-v100-32g \
  --checkpoint "${WORK}/manifest-model.rlfimg" --seed 99 >/dev/null
BLACK_SHA="$(sha256sum -- "${WORK}/black.ppm" | awk '{print $1}')"
RED_SHA="$(sha256sum -- "${WORK}/red.ppm" | awk '{print $1}')"
printf 'pair-1\tblack.ppm\t%s\tred.ppm\t%s\tred-shift\tlocal:test-fixture\tCC0-1.0\n' \
  "${BLACK_SHA}" "${RED_SHA}" >"${WORK}/pairs.tsv"
"${BIN}" imagegen-train-manifest --profile imagegen-v100-32g \
  --checkpoint "${WORK}/manifest-model.rlfimg" \
  --manifest "${WORK}/pairs.tsv" >"${WORK}/manifest-train.txt"
grep -q '^records=1$' "${WORK}/manifest-train.txt"
grep -q '^completed_shards=1$' "${WORK}/manifest-train.txt"
MANIFEST_MODEL_SHA="$(sha256sum -- "${WORK}/manifest-model.rlfimg" | awk '{print $1}')"
"${BIN}" imagegen-train-manifest --profile imagegen-v100-32g \
  --checkpoint "${WORK}/manifest-model.rlfimg" \
  --manifest "${WORK}/pairs.tsv" >"${WORK}/manifest-resume.txt"
grep -q '^shard_already_completed=true$' "${WORK}/manifest-resume.txt"
[[ "$(sha256sum -- "${WORK}/manifest-model.rlfimg" | awk '{print $1}')" == \
   "${MANIFEST_MODEL_SHA}" ]]

# The audited manifest supports the canonical neutral source marker. Its hash
# authenticates the deterministic seed algorithm rather than pretending that
# an absent source file was observed.
"${BIN}" imagegen-bootstrap --profile imagegen-v100-32g \
  --checkpoint "${WORK}/prompt-manifest-model.rlfimg" --seed 100 >/dev/null
NEUTRAL_MARKER='@neutral-gray128-target-size-v1'
NEUTRAL_SHA="$(printf '%s' "${NEUTRAL_MARKER}" | sha256sum | awk '{print $1}')"
printf 'prompt-1\t%s\t%s\tred.ppm\t%s\tsolid-red\tlocal:test-fixture\tCC0-1.0\n' \
  "${NEUTRAL_MARKER}" "${NEUTRAL_SHA}" "${RED_SHA}" \
  >"${WORK}/prompt-pairs.tsv"
"${BIN}" imagegen-train-manifest --profile imagegen-v100-32g \
  --checkpoint "${WORK}/prompt-manifest-model.rlfimg" \
  --manifest "${WORK}/prompt-pairs.tsv" >"${WORK}/prompt-manifest-train.txt"
grep -q '^records=1$' "${WORK}/prompt-manifest-train.txt"
"${BIN}" imagegen-generate --profile imagegen-v100-32g \
  --checkpoint "${WORK}/prompt-manifest-model.rlfimg" --prompt solid-red \
  --width 8 --height 8 --output "${WORK}/prompt-manifest-generated.ppm" >/dev/null
cmp "${WORK}/red.ppm" "${WORK}/prompt-manifest-generated.ppm"
printf 'eval-prompt-1\t%s\t%s\tred.ppm\t%s\tplease make solid red\tlocal:test-fixture\tCC0-1.0\tunseen_prompt,paraphrase,natural_image,multilingual\n' \
  "${NEUTRAL_MARKER}" "${NEUTRAL_SHA}" "${RED_SHA}" \
  >"${WORK}/prompt-evaluation.tsv"
"${BIN}" imagegen-evaluate-manifest --profile imagegen-v100-32g \
  --checkpoint "${WORK}/prompt-manifest-model.rlfimg" \
  --manifest "${WORK}/prompt-pairs.tsv" \
  --evaluation-manifest "${WORK}/prompt-evaluation.tsv" \
  --output "${WORK}/frozen-evaluation" >"${WORK}/evaluate.txt"
grep -q '^evaluation_records=1$' "${WORK}/evaluate.txt"
grep -q '"mean_ssim": 1' "${WORK}/frozen-evaluation/summary.json"
grep -q '"exact_training_copy_rate": 1' "${WORK}/frozen-evaluation/summary.json"
grep -q '"perceptual_training_copy_rate": 1' \
  "${WORK}/frozen-evaluation/summary.json"
grep -q '"perceptually_unique_output_fraction": 1' \
  "${WORK}/frozen-evaluation/summary.json"
RAW_MANIFEST_SHA="$(sha256sum -- "${WORK}/frozen-evaluation/raw_artifacts.tsv" | awk '{print $1}')"
grep -q "\"raw_artifact_manifest_sha256\": \"${RAW_MANIFEST_SHA}\"" \
  "${WORK}/frozen-evaluation/summary.json"
grep -q '"nearest_example_mean_ssim": 1' \
  "${WORK}/frozen-evaluation/summary.json"
grep -q '"patch_quilt_mean_ssim": 1' \
  "${WORK}/frozen-evaluation/summary.json"
grep -q '"internal_baselines_present": true' \
  "${WORK}/frozen-evaluation/summary.json"
grep -q '"unseen_prompt_records": 1' \
  "${WORK}/frozen-evaluation/summary.json"
grep -q '"unseen_prompt_mean_ssim": 1' \
  "${WORK}/frozen-evaluation/summary.json"
grep -q '"paraphrase_records": 1' \
  "${WORK}/frozen-evaluation/summary.json"
grep -q '"paraphrase_mean_ssim": 1' \
  "${WORK}/frozen-evaluation/summary.json"
grep -q '"natural_image_records": 1' \
  "${WORK}/frozen-evaluation/summary.json"
grep -q '"multilingual_records": 1' \
  "${WORK}/frozen-evaluation/summary.json"
grep -q '"external_diffusion_baseline_present": false' \
  "${WORK}/frozen-evaluation/summary.json"
cmp "${WORK}/red.ppm" "${WORK}/frozen-evaluation/generated/sample-0.ppm"
cmp "${WORK}/red.ppm" \
  "${WORK}/frozen-evaluation/nearest-example/sample-0.ppm"
cmp "${WORK}/red.ppm" "${WORK}/frozen-evaluation/patch-quilt/sample-0.ppm"
sed "s/${RED_SHA}/$(printf wrong-target | sha256sum | awk '{print $1}')/" \
  "${WORK}/prompt-evaluation.tsv" >"${WORK}/bad-evaluation.tsv"
set +e
"${BIN}" imagegen-evaluate-manifest --profile imagegen-v100-32g \
  --checkpoint "${WORK}/prompt-manifest-model.rlfimg" \
  --manifest "${WORK}/prompt-pairs.tsv" \
  --evaluation-manifest "${WORK}/bad-evaluation.tsv" \
  --output "${WORK}/failed-evaluation" >/dev/null 2>&1
FAILED_EVALUATION_STATUS=$?
set -e
[[ "${FAILED_EVALUATION_STATUS}" -ne 0 && \
   ! -e "${WORK}/failed-evaluation" && \
   ! -e "${WORK}/failed-evaluation.partial" ]]
sed 's/unseen_prompt,paraphrase,natural_image,multilingual/unseen_prompt,composition,natural_image/' \
  "${WORK}/prompt-evaluation.tsv" >"${WORK}/contradictory-tags.tsv"
set +e
"${BIN}" imagegen-evaluate-manifest --profile imagegen-v100-32g \
  --checkpoint "${WORK}/prompt-manifest-model.rlfimg" \
  --manifest "${WORK}/prompt-pairs.tsv" \
  --evaluation-manifest "${WORK}/contradictory-tags.tsv" \
  --output "${WORK}/contradictory-evaluation" >/dev/null 2>&1
CONTRADICTORY_TAG_STATUS=$?
set -e
[[ "${CONTRADICTORY_TAG_STATUS}" -ne 0 && \
   ! -e "${WORK}/contradictory-evaluation" && \
   ! -e "${WORK}/contradictory-evaluation.partial" ]]
PROMPT_MANIFEST_MODEL_SHA="$(sha256sum -- "${WORK}/prompt-manifest-model.rlfimg" | awk '{print $1}')"
printf 'prompt-2\t%s\t%s\tred.ppm\t%s\tbad-neutral\tlocal:test-fixture\tCC0-1.0\n' \
  "${NEUTRAL_MARKER}" "${RED_SHA}" "${RED_SHA}" >"${WORK}/bad-neutral.tsv"
set +e
"${BIN}" imagegen-train-manifest --profile imagegen-v100-32g \
  --checkpoint "${WORK}/prompt-manifest-model.rlfimg" \
  --manifest "${WORK}/bad-neutral.tsv" >/dev/null 2>&1
BAD_NEUTRAL_STATUS=$?
set -e
[[ "${BAD_NEUTRAL_STATUS}" -ne 0 ]]
[[ "$(sha256sum -- "${WORK}/prompt-manifest-model.rlfimg" | awk '{print $1}')" == \
   "${PROMPT_MANIFEST_MODEL_SHA}" ]]

printf 'pair-1\tblack.ppm\t%s\tred.ppm\t%s\tred-shift\tlocal:test-fixture\tCC0-1.0\n' \
  "${RED_SHA}" "${RED_SHA}" >"${WORK}/pairs.tsv"
set +e
"${BIN}" imagegen-train-manifest --profile imagegen-v100-32g \
  --checkpoint "${WORK}/manifest-model.rlfimg" \
  --manifest "${WORK}/pairs.tsv" >/dev/null 2>&1
REUSED_SHARD_STATUS=$?
set -e
[[ "${REUSED_SHARD_STATUS}" -ne 0 ]]
[[ "$(sha256sum -- "${WORK}/manifest-model.rlfimg" | awk '{print $1}')" == \
   "${MANIFEST_MODEL_SHA}" ]]

mkdir -p "${WORK}/bundle"
cp -- "${WORK}/manifest-model.rlfimg" "${WORK}/bundle/model.rlfimg"
cp -- "${WORK}/pairs.tsv" "${WORK}/bundle/training_pairs.tsv"
for artifact in source_manifest.tsv data_audit.json license_report.json \
  exact_dedup_report.json near_dedup_report.json perceptual_dedup_report.json \
  contamination_report.json training_telemetry.txt resource_summary.json \
  raw_gpu_trace.csv environment.txt checkpoint_inspection.txt readiness.json \
  resume_equivalence.json; do
  printf 'synthetic fixture for integrity-only verification\n' \
    >"${WORK}/bundle/${artifact}"
done
"${ROOT}/scripts/create_imagegen_artifact_manifest.sh" "${WORK}/bundle" \
  >"${WORK}/bundle-create.txt"
"${BIN}" imagegen-verify-artifacts \
  --manifest "${WORK}/bundle/artifact_manifest.tsv" \
  >"${WORK}/bundle-verify.txt"
grep -q '^bundle_integrity_verified=true$' "${WORK}/bundle-verify.txt"
grep -q '^origin_authenticated=false$' "${WORK}/bundle-verify.txt"
grep -q '^state_of_art_claim_proven=false$' "${WORK}/bundle-verify.txt"

printf 'imagegen_cli_shell_test=pass\nphysical_training_performed=false\nclaim_eligible=false\n'
