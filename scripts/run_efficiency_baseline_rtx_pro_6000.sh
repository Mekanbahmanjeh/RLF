#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
  cat <<'EOF'
Usage: run_efficiency_baseline_rtx_pro_6000.sh \
  LEDGER EVALUATION_REQUESTS EVALUATION_EXPECTED READINESS_REPORT OUTPUT_ROOT

Runs three clean, fixed-seed, end-to-end baseline repetitions on the exact
rtx-pro-6000-96g profile. It refuses existing output/checkpoints and delegates
training, VRAM/power sampling, checkpoint transactions, artifact binding, and
evaluation to the audited wrappers. GNU time captures whole-run CPU and peak
resident-memory evidence around every train/evaluate/score repetition. This
script trains; do not run it on the preparation-only WSL host.
EOF
}

[[ $# -eq 5 ]] || { usage >&2; exit 2; }
CAMPAIGN_VARIANT="${RLF_EFFICIENCY_CAMPAIGN_VARIANT:-baseline}"
case "${CAMPAIGN_VARIANT}" in
  baseline) CUDA_LOCAL_UPDATE_POLICY=device; CUDA_CACHED_COSINE_POLICY=inline_norms; CUDA_SMALL_COSINE_POLICY=device; SPARSE_ROUTER_UPDATE_POLICY=rebuild; SPARSE_RERANK_POLICY=per_query; CONCEPT_UPDATE_POLICY=linear; EXAMPLE_DUPLICATE_POLICY=linear; MODE_ID_INDEX_POLICY=rebuild; GROUNDING_INDEX_POLICY=rebuild; LANGUAGE_OUTCOME_POLICY=linear; LANGUAGE_DIALOGUE_ENCODING_POLICY=redundant; INSTRUCTION_DUPLICATE_POLICY=retrieval; TOOL_KEYWORD_UPDATE_POLICY=linear; PREFERENCE_DUPLICATE_POLICY=linear; ACTIVE_LEARNING_DUPLICATE_POLICY=linear; SEPARATE_VISION_ANALYSIS=1 ;;
  optimized) CUDA_LOCAL_UPDATE_POLICY=hybrid; CUDA_CACHED_COSINE_POLICY=precomputed_norms; CUDA_SMALL_COSINE_POLICY=hybrid; SPARSE_ROUTER_UPDATE_POLICY=incremental; SPARSE_RERANK_POLICY=batched; CONCEPT_UPDATE_POLICY=indexed; EXAMPLE_DUPLICATE_POLICY=indexed; MODE_ID_INDEX_POLICY=persistent; GROUNDING_INDEX_POLICY=persistent; LANGUAGE_OUTCOME_POLICY=indexed; LANGUAGE_DIALOGUE_ENCODING_POLICY=fused; INSTRUCTION_DUPLICATE_POLICY=indexed; TOOL_KEYWORD_UPDATE_POLICY=indexed; PREFERENCE_DUPLICATE_POLICY=indexed; ACTIVE_LEARNING_DUPLICATE_POLICY=indexed; SEPARATE_VISION_ANALYSIS=0 ;;
  *) echo "campaign variant must be baseline or optimized" >&2; exit 2 ;;
esac
LEDGER="$(realpath "${1}")"
EVALUATION_REQUESTS="$(realpath "${2}")"
EVALUATION_EXPECTED="$(realpath "${3}")"
READINESS_REPORT="$(realpath "${4}")"
OUTPUT_ROOT="$(realpath -m "${5}")"
SCORER="${ROOT}/build/ubuntu-rtx-pro-6000-cuda/rlf_efficiency_campaign"
[[ -f "${LEDGER}" && -f "${EVALUATION_REQUESTS}" && -f "${EVALUATION_EXPECTED}" && \
   -f "${READINESS_REPORT}" && -x "${SCORER}" ]] || {
  echo "ledger, evaluation inputs, readiness report, and campaign scorer must exist" >&2; exit 2
}
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "output root already exists" >&2; exit 2; }
[[ -x /usr/bin/time ]] || { echo "GNU /usr/bin/time is required" >&2; exit 2; }
grep -Eq '"profile"[[:space:]]*:[[:space:]]*"rtx-pro-6000-96g"' "${READINESS_REPORT}" || {
  echo "baseline requires the exact rtx-pro-6000-96g readiness profile" >&2; exit 2
}
grep -Eq '"ready"[[:space:]]*:[[:space:]]*true' "${READINESS_REPORT}" || {
  echo "baseline requires a real ready=true report" >&2; exit 2
}
grep -Eq '"test_doubles"[[:space:]]*:[[:space:]]*false' "${READINESS_REPORT}" || {
  echo "test-double readiness is not baseline evidence" >&2; exit 2
}

mkdir -p "${OUTPUT_ROOT}/runs" "${OUTPUT_ROOT}/checkpoints"
SUMMARY="${OUTPUT_ROOT}/runs.tsv"
printf 'variant\trun\tstart_utc\tend_utc\tend_to_end_wall_seconds\tcheckpoint\ttraining_result\tevaluation_result\twhole_run_resource_time\n' >"${SUMMARY}"
for run in 1 2 3; do
  CHECKPOINT="${OUTPUT_ROOT}/checkpoints/${CAMPAIGN_VARIANT}_run_${run}.rlfsp"
  TRAINING_RESULT="${OUTPUT_ROOT}/runs/run_${run}/training"
  EVALUATION_RESULT="${OUTPUT_ROOT}/runs/run_${run}/evaluation"
  WHOLE_RUN_RESOURCE_TIME="${OUTPUT_ROOT}/runs/run_${run}/whole_run_resource_time.txt"
  [[ ! -e "${CHECKPOINT}" && ! -e "${TRAINING_RESULT}" && ! -e "${EVALUATION_RESULT}" ]] || {
    echo "run ${run} output already exists" >&2; exit 2
  }
  mkdir -p "$(dirname "${WHOLE_RUN_RESOURCE_TIME}")"
  START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  START_NS="$(date +%s%N)"
  LC_ALL=C RLF_CUDA_LOCAL_UPDATE_POLICY="${CUDA_LOCAL_UPDATE_POLICY}" \
    RLF_CUDA_CACHED_COSINE_POLICY="${CUDA_CACHED_COSINE_POLICY}" \
    RLF_CUDA_SMALL_COSINE_POLICY="${CUDA_SMALL_COSINE_POLICY}" \
    RLF_SPARSE_ROUTER_UPDATE_POLICY="${SPARSE_ROUTER_UPDATE_POLICY}" \
    RLF_SPARSE_RERANK_POLICY="${SPARSE_RERANK_POLICY}" \
    RLF_CONCEPT_UPDATE_POLICY="${CONCEPT_UPDATE_POLICY}" \
    RLF_EXAMPLE_DUPLICATE_POLICY="${EXAMPLE_DUPLICATE_POLICY}" \
    RLF_MODE_ID_INDEX_POLICY="${MODE_ID_INDEX_POLICY}" \
    RLF_GROUNDING_INDEX_POLICY="${GROUNDING_INDEX_POLICY}" \
    RLF_LANGUAGE_OUTCOME_POLICY="${LANGUAGE_OUTCOME_POLICY}" \
    RLF_LANGUAGE_DIALOGUE_ENCODING_POLICY="${LANGUAGE_DIALOGUE_ENCODING_POLICY}" \
    RLF_INSTRUCTION_DUPLICATE_POLICY="${INSTRUCTION_DUPLICATE_POLICY}" \
    RLF_TOOL_KEYWORD_UPDATE_POLICY="${TOOL_KEYWORD_UPDATE_POLICY}" \
    RLF_PREFERENCE_DUPLICATE_POLICY="${PREFERENCE_DUPLICATE_POLICY}" \
    RLF_ACTIVE_LEARNING_DUPLICATE_POLICY="${ACTIVE_LEARNING_DUPLICATE_POLICY}" \
    RLF_SEPARATE_VISION_ANALYSIS="${SEPARATE_VISION_ANALYSIS}" \
    /usr/bin/time --verbose --output="${WHOLE_RUN_RESOURCE_TIME}" -- \
    "${ROOT}/scripts/run_efficiency_single_rtx_pro_6000.sh" \
    "${LEDGER}" "${EVALUATION_REQUESTS}" "${EVALUATION_EXPECTED}" \
    "${READINESS_REPORT}" "${CHECKPOINT}" "${TRAINING_RESULT}" \
    "${EVALUATION_RESULT}" "${SCORER}"
  END_NS="$(date +%s%N)"
  END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  WALL_SECONDS="$(awk -v start="${START_NS}" -v end="${END_NS}" \
    'BEGIN { printf "%.9f", (end - start) / 1000000000.0 }')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${CAMPAIGN_VARIANT}" "${run}" "${START_UTC}" "${END_UTC}" "${WALL_SECONDS}" \
    "$(realpath "${CHECKPOINT}")" "$(realpath "${TRAINING_RESULT}")" \
    "$(realpath "${EVALUATION_RESULT}")" \
    "$(realpath "${WHOLE_RUN_RESOURCE_TIME}")" >>"${SUMMARY}"
done
sha256sum -- "${LEDGER}" "${EVALUATION_REQUESTS}" "${EVALUATION_EXPECTED}" \
  "${READINESS_REPORT}" "${SCORER}" \
  "${SUMMARY}" >"${OUTPUT_ROOT}/input_and_summary.sha256"
find "${OUTPUT_ROOT}" -type f ! -name complete_campaign.sha256 -print0 |
  sort -z | xargs -0 sha256sum >"${OUTPUT_ROOT}/complete_campaign.sha256"
echo "Completed three physical ${CAMPAIGN_VARIANT} runs: ${OUTPUT_ROOT}"
