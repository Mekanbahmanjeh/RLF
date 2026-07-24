#!/usr/bin/env bash
set -euo pipefail
BIN="${1:?solstice executable required}"
WORK="$(mktemp -d)"; trap 'rm -rf -- "${WORK}"' EXIT
sha() { sha256sum -- "$1" | awk '{print $1}'; }
row() {
  printf '%s\t%s\ttrain\t%s\ten\ttest\t%s\t%s\tlocal:test\tCC0-1.0\t2026-07-20\t%s\ttest-v1\tnone\tnone\ttrue\n' \
    "$1" "$2" "$3" "$4" "$5" "$6"
}
row_eval() {
  printf '%s\t%s\tevaluation\t%s\ten\ttest\t%s\t%s\tlocal:test\tCC0-1.0\t2026-07-20\t%s\ttest-v1\tnone\tvideo-prototype\ttrue\n' \
    "$1" "$2" "$3" "$4" "$5" "$6"
}
printf 'alpha beta gamma\n' >"${WORK}/text.txt"
row text-1 text text text_lines text.txt "$(sha "${WORK}/text.txt")" >"${WORK}/text-ledger.tsv"
"${BIN}" train-data --ledger "${WORK}/text-ledger.tsv" --checkpoint "${WORK}/multi.rlfsp" \
  --profile general-v100-32g --backend optimized_cpu --blank --enforce-profile --require-media-hashes \
  --telemetry "${WORK}/first-telemetry.json" >/dev/null
grep -q '"schema": "rlf-train-data-telemetry-v1"' "${WORK}/first-telemetry.json"
grep -q '"training_performed": true' "${WORK}/first-telemetry.json"
grep -Eq '"trained_records": 1' "${WORK}/first-telemetry.json"
grep -q '"gpu_active_seconds": null' "${WORK}/first-telemetry.json"
grep -Eq '"language_tokens_processed": [1-9][0-9]*' "${WORK}/first-telemetry.json"
grep -Eq '"tokens_per_second": [0-9]' "${WORK}/first-telemetry.json"
grep -Eq '"images_processed": 0' "${WORK}/first-telemetry.json"
grep -Eq '"images_per_second": 0' "${WORK}/first-telemetry.json"
grep -Eq '"checkpoint_bytes": [1-9][0-9]*' "${WORK}/first-telemetry.json"
grep -Eq '"peak_ram_bytes": ([1-9][0-9]*|null)' "${WORK}/first-telemetry.json"
grep -q '"backend_operations": {' "${WORK}/first-telemetry.json"
grep -Eq '"host_to_device_bytes": 0' "${WORK}/first-telemetry.json"
FIRST_CHECKPOINT_SHA="$(sha256sum -- "${WORK}/multi.rlfsp" | awk '{print $1}')"
"${BIN}" train-data --ledger "${WORK}/text-ledger.tsv" --checkpoint "${WORK}/multi.rlfsp" \
  --profile general-v100-32g --backend optimized_cpu --blank --enforce-profile --require-media-hashes \
  --telemetry "${WORK}/resume-telemetry.json" >"${WORK}/resume.txt"
grep -Eq '^trained_shards=0$' "${WORK}/resume.txt"
grep -Eq '^resumed_shards=1$' "${WORK}/resume.txt"
grep -q '"training_performed": false' "${WORK}/resume-telemetry.json"
grep -Eq '"language_tokens_processed": 0' "${WORK}/resume-telemetry.json"
grep -Eq '"images_processed": 0' "${WORK}/resume-telemetry.json"
[[ "$(sha256sum -- "${WORK}/multi.rlfsp" | awk '{print $1}')" == "${FIRST_CHECKPOINT_SHA}" ]]

# The 1,586-hour profile is cumulative and fail-closed: stage names are not
# enough. The immutable ledger must contain the exact authorized record total,
# and every shard already represented by the checkpoint must remain present
# with identical content as the ledger expands from one stage to the next.
set +e
"${BIN}" train-data --ledger "${WORK}/text-ledger.tsv" \
  --checkpoint "${WORK}/staged.rlfsp" --profile general-v100-32g-500m \
  --backend optimized_cpu --blank --enforce-profile --require-media-hashes \
  >/dev/null 2>"${WORK}/missing-target.err"
MISSING_TARGET_STATUS=$?
set -e
[[ "${MISSING_TARGET_STATUS}" -ne 0 && ! -e "${WORK}/staged.rlfsp" ]]
grep -q 'requires --target-training-records' "${WORK}/missing-target.err"

# The H200 frontier profile binds scale to cumulative tokenizer pieces rather
# than assuming a fixed tokens-per-record ratio.
set +e
"${BIN}" train-data --ledger "${WORK}/text-ledger.tsv" \
  --checkpoint "${WORK}/h200-missing.rlfsp" --profile general-h200-141g-30t \
  --backend optimized_cpu --blank --enforce-profile --require-media-hashes \
  >/dev/null 2>"${WORK}/h200-missing-target.err"
H200_MISSING_STATUS=$?
set -e
[[ "${H200_MISSING_STATUS}" -ne 0 && ! -e "${WORK}/h200-missing.rlfsp" ]]
grep -q 'requires --target-training-tokens' "${WORK}/h200-missing-target.err"

"${BIN}" train-data --ledger "${WORK}/text-ledger.tsv" \
  --checkpoint "${WORK}/h200-probe.rlfsp" --profile general-h100 \
  --backend optimized_cpu --blank --enforce-profile --require-media-hashes \
  --telemetry "${WORK}/h200-probe.json" >/dev/null
H200_TARGET_TOKENS="$(sed -nE \
  's/.*"cumulative_tokens_after": ([0-9]+).*/\1/p' \
  "${WORK}/h200-probe.json")"
[[ "${H200_TARGET_TOKENS}" =~ ^[1-9][0-9]*$ ]]
"${BIN}" train-data --ledger "${WORK}/text-ledger.tsv" \
  --checkpoint "${WORK}/h200.rlfsp" --profile general-h200-141g-30t \
  --backend optimized_cpu --blank --enforce-profile --require-media-hashes \
  --target-training-tokens "${H200_TARGET_TOKENS}" \
  --telemetry "${WORK}/h200.json" >/dev/null
grep -Eq "\"target_training_tokens\": ${H200_TARGET_TOKENS}" "${WORK}/h200.json"
grep -Eq "\"cumulative_tokens_after\": ${H200_TARGET_TOKENS}" "${WORK}/h200.json"
grep -Fq '"token_target_reached": true' "${WORK}/h200.json"
"${BIN}" verify-checkpoint --checkpoint "${WORK}/h200.rlfsp" \
  --profile general-h200-141g-30t --enforce-profile >/dev/null

set +e
"${BIN}" train-data --ledger "${WORK}/text-ledger.tsv" \
  --checkpoint "${WORK}/staged.rlfsp" --profile general-v100-32g-500m \
  --backend optimized_cpu --blank --enforce-profile --require-media-hashes \
  --target-training-records 2 >/dev/null 2>"${WORK}/wrong-target.err"
WRONG_TARGET_STATUS=$?
set -e
[[ "${WRONG_TARGET_STATUS}" -ne 0 && ! -e "${WORK}/staged.rlfsp" ]]
grep -q 'exact target is 2' "${WORK}/wrong-target.err"

"${BIN}" train-data --ledger "${WORK}/text-ledger.tsv" \
  --checkpoint "${WORK}/staged.rlfsp" --profile general-v100-32g-500m \
  --backend optimized_cpu --blank --enforce-profile --require-media-hashes \
  --target-training-records 1 --telemetry "${WORK}/staged-1.json" >/dev/null
grep -Eq '"target_training_records": 1' "${WORK}/staged-1.json"
grep -Eq '"cumulative_training_records_before": 0' "${WORK}/staged-1.json"
grep -Eq '"cumulative_training_records_after": 1' "${WORK}/staged-1.json"

printf 'delta epsilon zeta\n' >"${WORK}/text-2.txt"
cp "${WORK}/text-ledger.tsv" "${WORK}/text-ledger-2.tsv"
row text-2 text text text_lines text-2.txt "$(sha "${WORK}/text-2.txt")" >>"${WORK}/text-ledger-2.tsv"
"${BIN}" train-data --ledger "${WORK}/text-ledger-2.tsv" \
  --checkpoint "${WORK}/staged.rlfsp" --profile general-v100-32g-500m \
  --backend optimized_cpu --blank --enforce-profile --require-media-hashes \
  --target-training-records 2 --telemetry "${WORK}/staged-2.json" >/dev/null
grep -Eq '"target_training_records": 2' "${WORK}/staged-2.json"
grep -Eq '"cumulative_training_records_before": 1' "${WORK}/staged-2.json"
grep -Eq '"cumulative_training_records_after": 2' "${WORK}/staged-2.json"
"${BIN}" inspect-checkpoint --checkpoint "${WORK}/staged.rlfsp" \
  --profile general-v100-32g-500m --enforce-profile >"${WORK}/staged-inspection.txt"
grep -Eq '^audited_training_records=2$' "${WORK}/staged-inspection.txt"
"${BIN}" verify-checkpoint --checkpoint "${WORK}/staged.rlfsp" \
  --profile general-v100-32g-500m --enforce-profile >/dev/null
set +e
"${BIN}" verify-checkpoint --checkpoint "${WORK}/staged.rlfsp" \
  --profile general-v100-32g --enforce-profile >/dev/null 2>"${WORK}/wrong-verify-profile.err"
WRONG_VERIFY_PROFILE_STATUS=$?
"${BIN}" inspect-checkpoint --checkpoint "${WORK}/staged.rlfsp" \
  --profile general-h100 --enforce-profile >/dev/null 2>"${WORK}/wrong-inspect-profile.err"
WRONG_INSPECT_PROFILE_STATUS=$?
set -e
[[ "${WRONG_VERIFY_PROFILE_STATUS}" -ne 0 && "${WRONG_INSPECT_PROFILE_STATUS}" -ne 0 ]]
grep -q 'does not match selected profile' "${WORK}/wrong-verify-profile.err"
grep -q 'does not match selected profile' "${WORK}/wrong-inspect-profile.err"

# Simulate a process interruption immediately after the first transactional
# shard checkpoint while retaining the same full ledger identity. Resuming must
# produce a byte-identical checkpoint to uninterrupted training.
"${BIN}" train-data --ledger "${WORK}/text-ledger-2.tsv" \
  --checkpoint "${WORK}/uninterrupted.rlfsp" --profile general-v100-32g-500m \
  --backend optimized_cpu --blank --enforce-profile --require-media-hashes \
  --target-training-records 2 >/dev/null
set +e
"${BIN}" train-data --ledger "${WORK}/text-ledger-2.tsv" \
  --checkpoint "${WORK}/interrupted.rlfsp" --profile general-v100-32g-500m \
  --backend optimized_cpu --blank --enforce-profile --require-media-hashes \
  --target-training-records 2 --maximum-new-shards 1 \
  --telemetry "${WORK}/interrupted.json" >/dev/null
INTENTIONAL_STOP_STATUS=$?
set -e
[[ "${INTENTIONAL_STOP_STATUS}" -eq 4 ]]
grep -Eq '"training_target_reached": false' "${WORK}/interrupted.json"
grep -Eq '"intentional_shard_stop": true' "${WORK}/interrupted.json"
"${BIN}" inspect-checkpoint --checkpoint "${WORK}/interrupted.rlfsp" \
  --profile general-v100-32g-500m --enforce-profile >"${WORK}/interrupted-inspection.txt"
grep -Eq '^audited_training_records=1$' "${WORK}/interrupted-inspection.txt"
"${BIN}" train-data --ledger "${WORK}/text-ledger-2.tsv" \
  --checkpoint "${WORK}/interrupted.rlfsp" --profile general-v100-32g-500m \
  --backend optimized_cpu --blank --enforce-profile --require-media-hashes \
  --target-training-records 2 --telemetry "${WORK}/resumed-equivalence.json" >/dev/null
grep -Eq '"cumulative_training_records_before": 1' "${WORK}/resumed-equivalence.json"
grep -Eq '"cumulative_training_records_after": 2' "${WORK}/resumed-equivalence.json"
[[ "$(sha "${WORK}/interrupted.rlfsp")" == "$(sha "${WORK}/uninterrupted.rlfsp")" ]]

row text-2 text text text_lines text-2.txt "$(sha "${WORK}/text-2.txt")" >"${WORK}/omits-old-ledger.tsv"
STAGED_SHA="$(sha "${WORK}/staged.rlfsp")"
set +e
"${BIN}" train-data --ledger "${WORK}/omits-old-ledger.tsv" \
  --checkpoint "${WORK}/staged.rlfsp" --profile general-v100-32g-500m \
  --backend optimized_cpu --blank --enforce-profile --require-media-hashes \
  --target-training-records 1 >/dev/null 2>"${WORK}/omits-old.err"
OMITS_OLD_STATUS=$?
set -e
[[ "${OMITS_OLD_STATUS}" -ne 0 && "$(sha "${WORK}/staged.rlfsp")" == "${STAGED_SHA}" ]]
grep -Eq 'checkpoint already exceeds|omits or changes completed shard' "${WORK}/omits-old.err"

"${BIN}" train-data --ledger "${WORK}/text-ledger.tsv" --checkpoint "${WORK}/pro-multi.rlfsp" \
  --profile general-rtx-pro-6000-96g --backend optimized_cpu --blank --enforce-profile --require-media-hashes >/dev/null

# The RTX 50M profiles default to non-owning TSV fields. The allocation-heavy
# reference path remains an explicit ablation and must learn byte-identical
# state. H100 retains the established copied-field path.
printf 'task-a\tdomain-a\tprompt alpha\trationale alpha\tresponse alpha\t1\n' >"${WORK}/instructions.tsv"
printf 'task-b\tdomain-b\tprompt beta\trationale beta\tresponse beta\t0.75\n' >>"${WORK}/instructions.tsv"
row instruction-1 instruction text tsv instructions.tsv "$(sha "${WORK}/instructions.tsv")" >"${WORK}/instruction-ledger.tsv"
"${BIN}" train-data --ledger "${WORK}/instruction-ledger.tsv" \
  --checkpoint "${WORK}/rtx-views.rlfsp" --profile rtx-pro-6000-96g \
  --backend optimized_cpu --blank --enforce-profile --require-media-hashes \
  --telemetry "${WORK}/rtx-views-telemetry.json" >/dev/null
"${BIN}" train-data --ledger "${WORK}/instruction-ledger.tsv" \
  --checkpoint "${WORK}/rtx-copied.rlfsp" --profile rtx-pro-6000-96g \
  --backend optimized_cpu --blank --enforce-profile --require-media-hashes \
  --copy-tsv-fields --telemetry "${WORK}/rtx-copied-telemetry.json" >/dev/null
[[ "$(sha "${WORK}/rtx-views.rlfsp")" == "$(sha "${WORK}/rtx-copied.rlfsp")" ]]
grep -Fq '"tsv_field_policy": "views"' "${WORK}/rtx-views-telemetry.json"
grep -Fq '"tsv_field_policy": "copied"' "${WORK}/rtx-copied-telemetry.json"
"${BIN}" train-data --ledger "${WORK}/instruction-ledger.tsv" \
  --checkpoint "${WORK}/h100-copied.rlfsp" --profile general-h100 \
  --backend optimized_cpu --blank --enforce-profile --require-media-hashes \
  --telemetry "${WORK}/h100-copied-telemetry.json" >/dev/null
grep -Fq '"tsv_field_policy": "copied"' "${WORK}/h100-copied-telemetry.json"
set +e
"${BIN}" train-data --ledger "${WORK}/text-ledger.tsv" --checkpoint "${WORK}/multi.rlfsp" \
  --profile general-v100-32g-text --backend optimized_cpu --blank --enforce-profile --require-media-hashes >/dev/null 2>&1
STATUS=$?
set -e
[[ "${STATUS}" -ne 0 ]]

printf 'P6\n1 1\n255\n\x10\x20\x30' >"${WORK}/frame.ppm"
printf 'frame.ppm\t%s\ta test image\n' "$(sha "${WORK}/frame.ppm")" >"${WORK}/vision.tsv"
row vision-1 vision image-text vision_tsv vision.tsv "$(sha "${WORK}/vision.tsv")" >"${WORK}/vision-ledger.tsv"
"${BIN}" train-data --ledger "${WORK}/vision-ledger.tsv" --checkpoint "${WORK}/vision-metrics.rlfsp" \
  --profile general-v100-32g --backend optimized_cpu --blank --enforce-profile --require-media-hashes \
  --telemetry "${WORK}/vision-telemetry.json" >/dev/null
grep -Eq '"images_processed": 1' "${WORK}/vision-telemetry.json"
grep -Eq '"images_per_second": [1-9][0-9.eE+-]*' "${WORK}/vision-telemetry.json"
grep -Eq '"image_decode_seconds": [0-9]' "${WORK}/vision-telemetry.json"
grep -Eq '"local_update_calls": [0-9]+' "${WORK}/vision-telemetry.json"
set +e
"${BIN}" train-data --ledger "${WORK}/vision-ledger.tsv" --checkpoint "${WORK}/text.rlfsp" \
  --profile general-v100-32g-text --backend optimized_cpu --blank --enforce-profile --require-media-hashes >/dev/null 2>&1
STATUS=$?
set -e
[[ "${STATUS}" -ne 0 && ! -e "${WORK}/text.rlfsp" ]]
set +e
"${BIN}" train-data --ledger "${WORK}/vision-ledger.tsv" --checkpoint "${WORK}/pro-text.rlfsp" \
  --profile general-rtx-pro-6000-96g-text --backend optimized_cpu --blank --enforce-profile --require-media-hashes >/dev/null 2>&1
STATUS=$?
set -e
[[ "${STATUS}" -ne 0 && ! -e "${WORK}/pro-text.rlfsp" ]]

printf 'P6\n4 2\n255\n\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00' >"${WORK}/video0.ppm"
printf 'P6\n4 2\n255\n\x00\x00\x00\x00\x00\x00\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x00\x00\x00\x00\x00' >"${WORK}/video1.ppm"
printf 'clip-1\t0\t24\tvideo0.ppm\t%s\ta red cube moves right\tthe first frame\n' "$(sha "${WORK}/video0.ppm")" >"${WORK}/video.tsv"
printf 'clip-1\t1\t24\tvideo1.ppm\t%s\ta red cube moves right\tthe second frame\n' "$(sha "${WORK}/video1.ppm")" >>"${WORK}/video.tsv"
printf 'P6\n4 2\n255\n\x00\x00\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00' >"${WORK}/eval0.ppm"
printf 'P6\n4 2\n255\n\x00\x00\x00\x00\x00\x00\x00\x00\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x00\x00\x00' >"${WORK}/eval1.ppm"
printf 'clip-eval\t0\t24\teval0.ppm\t%s\ta blue cube travels right\theld-out first frame\n' "$(sha "${WORK}/eval0.ppm")" >"${WORK}/video-eval.tsv"
printf 'clip-eval\t1\t24\teval1.ppm\t%s\ta blue cube travels right\theld-out second frame\n' "$(sha "${WORK}/eval1.ppm")" >>"${WORK}/video-eval.tsv"
row video-1 video video-text video_frames_tsv video.tsv "$(sha "${WORK}/video.tsv")" >"${WORK}/video-ledger.tsv"
row_eval video-eval video video-text video_frames_tsv video-eval.tsv "$(sha "${WORK}/video-eval.tsv")" >>"${WORK}/video-ledger.tsv"
set +e
"${BIN}" train-data --ledger "${WORK}/video-ledger.tsv" --checkpoint "${WORK}/wrong-video.rlfsp" \
  --profile general-v100-32g --backend optimized_cpu --blank --enforce-profile --require-media-hashes >/dev/null 2>&1
STATUS=$?
set -e
[[ "${STATUS}" -ne 0 && ! -e "${WORK}/wrong-video.rlfsp" ]]
"${BIN}" train-data --ledger "${WORK}/video-ledger.tsv" --checkpoint "${WORK}/video.rlfsp" \
  --profile video-rtx-pro-6000-96g --backend optimized_cpu --blank --enforce-profile --require-media-hashes >/dev/null
"${BIN}" inspect-checkpoint --checkpoint "${WORK}/video.rlfsp" >"${WORK}/inspection.txt"
grep -Eq '^images_seen=2$' "${WORK}/inspection.txt"
grep -Eq '^video_prototypes=1$' "${WORK}/inspection.txt"
grep -Eq '^video_sequences_seen=1$' "${WORK}/inspection.txt"
grep -Eq '^video_frames_seen=2$' "${WORK}/inspection.txt"
grep -Eq '^completed_training_shards=1$' "${WORK}/inspection.txt"
"${BIN}" train-data --ledger "${WORK}/video-ledger.tsv" --checkpoint "${WORK}/v100-video.rlfsp" \
  --profile video-v100-32g --backend optimized_cpu --blank --enforce-profile --require-media-hashes >/dev/null
"${BIN}" verify-checkpoint --checkpoint "${WORK}/v100-video.rlfsp" \
  --profile video-v100-32g --enforce-profile >/dev/null
"${BIN}" generate-video --checkpoint "${WORK}/v100-video.rlfsp" \
  --profile video-v100-32g --prompt 'a red cube moves right' \
  --frames 2 --output "${WORK}/v100-generated" >/dev/null
grep -q '"profile": "video-v100-32g"' \
  "${WORK}/v100-generated/generation_manifest.json"
"${BIN}" generate-video --checkpoint "${WORK}/video.rlfsp" \
  --profile video-rtx-pro-6000-96g --prompt 'a red cube moves right' \
  --frames 3 --output "${WORK}/generated" >/dev/null
[[ -f "${WORK}/generated/generation_manifest.json" ]]
[[ "$(find "${WORK}/generated/frames" -type f -name '*.ppm' | wc -l)" -eq 3 ]]
grep -q 'not photorealistic video synthesis' "${WORK}/generated/generation_manifest.json"
"${BIN}" evaluate-video --checkpoint "${WORK}/video.rlfsp" \
  --profile video-rtx-pro-6000-96g --ledger "${WORK}/video-ledger.tsv" \
  --output "${WORK}/video-evaluation" --require-media-hashes >/dev/null
grep -q '"evaluated_sequences": 1' "${WORK}/video-evaluation/summary.json"
grep -q '"mean_held_frame_motion_error"' "${WORK}/video-evaluation/summary.json"
grep -q '"mean_held_frame_pixel_mae"' "${WORK}/video-evaluation/summary.json"
[[ -f "${WORK}/video-evaluation/raw_predictions.tsv" ]]
[[ -f "${WORK}/video-evaluation/data_audit.json" ]]
