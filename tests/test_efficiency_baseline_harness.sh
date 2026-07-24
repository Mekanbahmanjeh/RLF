#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash -n "${ROOT}/scripts/run_efficiency_single_rtx_pro_6000.sh" \
  "${ROOT}/scripts/run_efficiency_baseline_rtx_pro_6000.sh" \
  "${ROOT}/scripts/run_efficiency_optimized_rtx_pro_6000.sh" \
  "${ROOT}/scripts/compare_efficiency_campaigns_rtx_pro_6000.sh"
grep -q 'baseline) CUDA_LOCAL_UPDATE_POLICY=device; CUDA_CACHED_COSINE_POLICY=inline_norms; CUDA_SMALL_COSINE_POLICY=device; SPARSE_ROUTER_UPDATE_POLICY=rebuild; SPARSE_RERANK_POLICY=per_query; CONCEPT_UPDATE_POLICY=linear; EXAMPLE_DUPLICATE_POLICY=linear; MODE_ID_INDEX_POLICY=rebuild; GROUNDING_INDEX_POLICY=rebuild; LANGUAGE_OUTCOME_POLICY=linear; LANGUAGE_DIALOGUE_ENCODING_POLICY=redundant; INSTRUCTION_DUPLICATE_POLICY=retrieval; TOOL_KEYWORD_UPDATE_POLICY=linear; PREFERENCE_DUPLICATE_POLICY=linear; ACTIVE_LEARNING_DUPLICATE_POLICY=linear; SEPARATE_VISION_ANALYSIS=1' \
  "${ROOT}/scripts/run_efficiency_baseline_rtx_pro_6000.sh"
grep -q 'optimized) CUDA_LOCAL_UPDATE_POLICY=hybrid; CUDA_CACHED_COSINE_POLICY=precomputed_norms; CUDA_SMALL_COSINE_POLICY=hybrid; SPARSE_ROUTER_UPDATE_POLICY=incremental; SPARSE_RERANK_POLICY=batched; CONCEPT_UPDATE_POLICY=indexed; EXAMPLE_DUPLICATE_POLICY=indexed; MODE_ID_INDEX_POLICY=persistent; GROUNDING_INDEX_POLICY=persistent; LANGUAGE_OUTCOME_POLICY=indexed; LANGUAGE_DIALOGUE_ENCODING_POLICY=fused; INSTRUCTION_DUPLICATE_POLICY=indexed; TOOL_KEYWORD_UPDATE_POLICY=indexed; PREFERENCE_DUPLICATE_POLICY=indexed; ACTIVE_LEARNING_DUPLICATE_POLICY=indexed; SEPARATE_VISION_ANALYSIS=0' \
  "${ROOT}/scripts/run_efficiency_baseline_rtx_pro_6000.sh"
grep -q 'RLF_EFFICIENCY_CAMPAIGN_VARIANT=optimized' \
  "${ROOT}/scripts/run_efficiency_optimized_rtx_pro_6000.sh"

WORK="$(mktemp -d)"
trap 'rm -rf -- "${WORK}"' EXIT
for file in ledger.tsv requests.tsv expected.tsv readiness.json; do
  printf 'fixture\n' >"${WORK}/${file}"
done
printf '#!/usr/bin/env bash\nexit 0\n' >"${WORK}/scorer"
chmod +x "${WORK}/scorer"
printf 'existing checkpoint\n' >"${WORK}/checkpoint.rlfsp"

set +e
"${ROOT}/scripts/run_efficiency_single_rtx_pro_6000.sh" \
  "${WORK}/ledger.tsv" "${WORK}/requests.tsv" "${WORK}/expected.tsv" \
  "${WORK}/readiness.json" "${WORK}/checkpoint.rlfsp" \
  "${WORK}/training" "${WORK}/evaluation" "${WORK}/scorer" \
  >"${WORK}/stdout.txt" 2>"${WORK}/stderr.txt"
STATUS=$?
set -e
[[ "${STATUS}" -eq 2 ]]
grep -q 'refuses existing outputs' "${WORK}/stderr.txt"
[[ ! -e "${WORK}/training" && ! -e "${WORK}/evaluation" ]]

# The top-level campaign must fail before creating output when its exact CUDA
# scorer/build is unavailable on a preparation host.
set +e
"${ROOT}/scripts/run_efficiency_baseline_rtx_pro_6000.sh" \
  "${WORK}/ledger.tsv" "${WORK}/requests.tsv" "${WORK}/expected.tsv" \
  "${WORK}/readiness.json" "${WORK}/campaign" \
  >"${WORK}/campaign-stdout.txt" 2>"${WORK}/campaign-stderr.txt"
STATUS=$?
set -e
[[ "${STATUS}" -ne 0 ]]
[[ ! -e "${WORK}/campaign" ]]

COMMON_LEDGER="$(printf ledger | sha256sum | awk '{print $1}')"
COMMON_SOURCE="$(printf source | sha256sum | awk '{print $1}')"
COMMON_BINARY="$(printf binary | sha256sum | awk '{print $1}')"
COMMON_EXPECTED="$(printf expected | sha256sum | awk '{print $1}')"
make_campaign() {
  local root="$1" variant="$2" wall="$3" gpu="$4" train_wall="$5" avoided="$6" norm_fmas="$7" host_norm_fmas="$8" host_cosine="$9" cosine_fmas="${10}"
  mkdir -p "${root}/runs" "${root}/checkpoints"
  printf 'variant\trun\tstart_utc\tend_utc\tend_to_end_wall_seconds\tcheckpoint\ttraining_result\tevaluation_result\twhole_run_resource_time\n' >"${root}/runs.tsv"
  for run in 1 2 3; do
    local training="${root}/runs/run_${run}/training"
    local evaluation="${root}/runs/run_${run}/evaluation"
    local checkpoint="${root}/checkpoints/${variant}_run_${run}.rlfsp" router
    mkdir -p "${training}" "${evaluation}/scoring"
    printf 'identical checkpoint\n' >"${checkpoint}"
    if [[ "${variant}" == baseline ]]; then
      printf 'schema=rlf-general-cuda-training-environment-v1\nsource_manifest_sha256=%s\nsolstice_binary_sha256=%s\ncuda_local_update_policy=device\ncuda_cached_cosine_policy=inline_norms\ncuda_small_cosine_policy=device\nsparse_router_update_policy=rebuild\nsparse_rerank_policy=per_query\nconcept_update_policy=linear\nexample_duplicate_policy=linear\nmode_id_index_policy=rebuild\ngrounding_index_policy=rebuild\nlanguage_outcome_policy=linear\nlanguage_dialogue_encoding_policy=redundant\ninstruction_duplicate_policy=retrieval\ntool_keyword_update_policy=linear\npreference_duplicate_policy=linear\nactive_learning_duplicate_policy=linear\nseparate_vision_analysis=1\n' "${COMMON_SOURCE}" "${COMMON_BINARY}" >"${training}/training_environment.txt"
    else
      printf 'schema=rlf-general-cuda-training-environment-v1\nsource_manifest_sha256=%s\nsolstice_binary_sha256=%s\ncuda_local_update_policy=hybrid\ncuda_cached_cosine_policy=precomputed_norms\ncuda_small_cosine_policy=hybrid\nsparse_router_update_policy=incremental\nsparse_rerank_policy=batched\nconcept_update_policy=indexed\nexample_duplicate_policy=indexed\nmode_id_index_policy=persistent\ngrounding_index_policy=persistent\nlanguage_outcome_policy=indexed\nlanguage_dialogue_encoding_policy=fused\ninstruction_duplicate_policy=indexed\ntool_keyword_update_policy=indexed\npreference_duplicate_policy=indexed\nactive_learning_duplicate_policy=indexed\nseparate_vision_analysis=0\n' "${COMMON_SOURCE}" "${COMMON_BINARY}" >"${training}/training_environment.txt"
    fi
    if [[ "${variant}" == baseline ]]; then
      router='"full_rebuilds":5,"vectors_rebuilt":40000,"incremental_updates":0,"vectors_incrementally_updated":0,"vectors_appended":0'
    else
      router='"full_rebuilds":1,"vectors_rebuilt":8000,"incremental_updates":45,"vectors_incrementally_updated":40,"vectors_appended":5'
    fi
    local indexed_calls=0 indexed_pairs=0
    local linear_concepts=6000 indexed_concepts=0
    local linear_examples=4498501 indexed_examples=0
    local mode_rebuilt=3000000
    local linear_regions=6000000 indexed_regions=0
    local ground_rebuilt=3000000 ground_candidates=0 ground_inserts=0
    local ground_confidence=3000000 ground_sweeps=3000000 ground_sorts=6000000
    local language_linear=500000 language_indexed=0 outcome_builds=0 outcome_entries=0 outcome_inserts=0
    local preference_linear=500000 preference_indexed=0 preference_rebuilds=0 preference_entries=0 preference_inserts=0
    local active_linear=500000 active_indexed=0 active_rebuilds=0 active_entries=0 active_inserts=0
    if [[ "${variant}" == optimized ]]; then indexed_calls=20; indexed_pairs=3000; fi
    if [[ "${variant}" == optimized ]]; then linear_concepts=0; indexed_concepts=3000; fi
    if [[ "${variant}" == optimized ]]; then linear_examples=0; indexed_examples=1; fi
    if [[ "${variant}" == optimized ]]; then mode_rebuilt=0; fi
    if [[ "${variant}" == optimized ]]; then linear_regions=0; indexed_regions=3000; fi
    if [[ "${variant}" == optimized ]]; then ground_rebuilt=0; ground_candidates=6000; ground_inserts=20; ground_confidence=3000; ground_sweeps=0; ground_sorts=0; fi
    if [[ "${variant}" == optimized ]]; then language_linear=1000; language_indexed=9000; outcome_builds=5; outcome_entries=160; outcome_inserts=100; fi
    if [[ "${variant}" == optimized ]]; then preference_linear=0; preference_indexed=10000; preference_rebuilds=1; preference_entries=1000; preference_inserts=100; fi
    if [[ "${variant}" == optimized ]]; then active_linear=0; active_indexed=10000; active_rebuilds=1; active_entries=1000; active_inserts=100; fi
    printf '{"schema":"rlf-train-data-telemetry-v1","ledger_sha256":"%s","total_wall_seconds":%s,"local_update_calls":10,"avoided_kernel_launches":%s,"avoided_pairwise_norm_fma_operations":%s,"host_precomputed_norm_fma_operations":%s,"host_batch_cosine_calls":%s,"avoided_device_cosine_fma_operations":%s,"indexed_batch_cosine_calls":%s,"indexed_cosine_pairs":%s,"sparse_router_operations":{%s},"visual_training_operations":{"concept_update_lookups":3000,"linear_concept_comparisons":%s,"indexed_concept_lookups":%s,"concept_index_rebuilds":1,"concept_index_entries_built":0,"example_duplicate_lookups":3001,"linear_example_comparisons":%s,"indexed_example_candidates":%s,"example_index_rebuilds":1,"example_index_entries_built":0,"mode_id_lookups":3000,"mode_id_index_full_rebuilds":1,"mode_id_index_entries_rebuilt":%s,"mode_id_index_incremental_inserts":10,"region_mode_id_lookups":3000,"linear_region_mode_comparisons":%s,"indexed_region_mode_lookups":%s},"grounding_operations":{"link_lookups":3000,"full_lookup_entries_rebuilt":%s,"indexed_link_candidates_examined":%s,"incremental_posting_inserts":%s,"confidence_recomputations":%s,"full_confidence_sweep_entries":%s,"derived_sort_entries":%s,"mode_query_full_scan_entries":0,"mode_query_indexed_candidates":0,"concept_query_full_scan_entries":0,"concept_query_indexed_candidates":0},"language_training_operations":{"outcome_update_lookups":10000,"linear_outcome_comparisons":%s,"indexed_outcome_lookups":%s,"outcome_index_builds":%s,"outcome_index_entries_built":%s,"outcome_index_incremental_inserts":%s},"general_training_operations":{"preference_duplicate_lookups":10000,"linear_preference_comparisons":%s,"indexed_preference_candidates":%s,"preference_index_rebuilds":%s,"preference_index_entries_built":%s,"preference_index_incremental_inserts":%s,"active_learning_duplicate_lookups":10000,"linear_active_learning_comparisons":%s,"indexed_active_learning_candidates":%s,"active_learning_index_rebuilds":%s,"active_learning_index_entries_built":%s,"active_learning_index_incremental_inserts":%s}}\n' "${COMMON_LEDGER}" "${train_wall}" "${avoided}" "${norm_fmas}" "${host_norm_fmas}" "${host_cosine}" "${cosine_fmas}" "${indexed_calls}" "${indexed_pairs}" "${router}" "${linear_concepts}" "${indexed_concepts}" "${linear_examples}" "${indexed_examples}" "${mode_rebuilt}" "${linear_regions}" "${indexed_regions}" "${ground_rebuilt}" "${ground_candidates}" "${ground_inserts}" "${ground_confidence}" "${ground_sweeps}" "${ground_sorts}" "${language_linear}" "${language_indexed}" "${outcome_builds}" "${outcome_entries}" "${outcome_inserts}" "${preference_linear}" "${preference_indexed}" "${preference_rebuilds}" "${preference_entries}" "${preference_inserts}" "${active_linear}" "${active_indexed}" "${active_rebuilds}" "${active_entries}" "${active_inserts}" >"${training}/training_pipeline_telemetry.json"
    printf '{"gpu_active_seconds_estimate":%s}\n' "${gpu}" >"${training}/training_resource_summary.json"
    printf '{"expected_sha256":"%s","accuracy":1}\n' "${COMMON_EXPECTED}" >"${evaluation}/scoring/summary.json"
    for index in 1 2 3 4 5 6; do printf 'artifact %s\n' "${index}" >"${training}/dummy${index}"; done
    local manifest="${training}/training_artifact_manifest.tsv"
    printf '# rlf-training-artifact-manifest-v2\n' >"${manifest}"
    for assignment in \
      "checkpoint:${checkpoint}" "ledger:${training}/dummy1" \
      "source_manifest:${training}/dummy2" "data_audit:${training}/dummy3" \
      "telemetry:${training}/training_pipeline_telemetry.json" \
      "resource_summary:${training}/training_resource_summary.json" \
      "vram_trace:${training}/dummy4" \
      "environment:${training}/training_environment.txt" \
      "checkpoint_inspection:${training}/dummy5" \
      "readiness_report:${training}/dummy6"; do
      local kind="${assignment%%:*}" file="${assignment#*:}"
      printf '%s\t%s\t%s\t%s\n' "${kind}" \
        "$(sha256sum -- "${file}" | awk '{print $1}')" \
        "$(stat -c '%s' -- "${file}")" "$(realpath "${file}")" >>"${manifest}"
    done
    (cd "${training}" && sha256sum training_artifact_manifest.tsv >training_artifact_manifest.tsv.sha256)
    printf '%s\t%s\tstart\tend\t%s\t%s\t%s\t%s\tresource\n' \
      "${variant}" "${run}" "${wall}" "$(realpath "${checkpoint}")" \
      "$(realpath "${training}")" "$(realpath "${evaluation}")" >>"${root}/runs.tsv"
  done
}
make_campaign "${WORK}/baseline" baseline 100 10 80 0 0 0 0 0
make_campaign "${WORK}/optimized" optimized 50 5 40 10 1000 100 20 3000
"${ROOT}/scripts/compare_efficiency_campaigns_rtx_pro_6000.sh" \
  "${WORK}/baseline" "${WORK}/optimized" "${WORK}/comparison.json"
grep -Eq '"end_to_end_speedup": 2([.]0*)?,' "${WORK}/comparison.json"
grep -q '"matched_controlled_quality": true' "${WORK}/comparison.json"
grep -q '"matched_update_counts": true' "${WORK}/comparison.json"
grep -q '"optimized_avoided_pairwise_norm_fma_operations": \[1000, 1000, 1000\]' "${WORK}/comparison.json"
grep -q '"optimized_host_precomputed_norm_fma_operations": \[100, 100, 100\]' "${WORK}/comparison.json"
grep -q '"optimized_host_batch_cosine_calls": \[20, 20, 20\]' "${WORK}/comparison.json"
grep -q '"optimized_avoided_device_cosine_fma_operations": \[3000, 3000, 3000\]' "${WORK}/comparison.json"
grep -q '"baseline_sparse_router_vectors_rebuilt": \[40000, 40000, 40000\]' "${WORK}/comparison.json"
grep -q '"optimized_sparse_router_vectors_rebuilt": \[8000, 8000, 8000\]' "${WORK}/comparison.json"
grep -q '"optimized_sparse_router_vectors_incrementally_updated": \[40, 40, 40\]' "${WORK}/comparison.json"
grep -q '"optimized_sparse_router_vectors_appended": \[5, 5, 5\]' "${WORK}/comparison.json"
grep -q '"optimized_indexed_sparse_rerank_calls": \[20, 20, 20\]' "${WORK}/comparison.json"
grep -q '"optimized_indexed_sparse_rerank_pairs": \[3000, 3000, 3000\]' "${WORK}/comparison.json"
grep -q '"concept_update_lookups": \[3000, 3000, 3000\]' "${WORK}/comparison.json"
grep -q '"baseline_linear_concept_comparisons": \[6000, 6000, 6000\]' "${WORK}/comparison.json"
grep -q '"optimized_indexed_concept_lookups": \[3000, 3000, 3000\]' "${WORK}/comparison.json"
grep -q '"example_duplicate_lookups": \[3001, 3001, 3001\]' "${WORK}/comparison.json"
grep -q '"baseline_linear_example_comparisons": \[4498501, 4498501, 4498501\]' "${WORK}/comparison.json"
grep -q '"optimized_indexed_example_candidates": \[1, 1, 1\]' "${WORK}/comparison.json"
grep -q '"mode_id_lookups": \[3000, 3000, 3000\]' "${WORK}/comparison.json"
grep -q '"baseline_mode_id_index_entries_rebuilt": \[3000000, 3000000, 3000000\]' "${WORK}/comparison.json"
grep -q '"optimized_mode_id_index_entries_rebuilt": \[0, 0, 0\]' "${WORK}/comparison.json"
grep -q '"mode_id_index_incremental_inserts": \[10, 10, 10\]' "${WORK}/comparison.json"
grep -q '"region_mode_id_lookups": \[3000, 3000, 3000\]' "${WORK}/comparison.json"
grep -q '"baseline_linear_region_mode_comparisons": \[6000000, 6000000, 6000000\]' "${WORK}/comparison.json"
grep -q '"optimized_indexed_region_mode_lookups": \[3000, 3000, 3000\]' "${WORK}/comparison.json"
grep -q '"baseline_grounding_lookup_entries_rebuilt": \[3000000, 3000000, 3000000\]' "${WORK}/comparison.json"
grep -q '"optimized_grounding_confidence_recomputations": \[3000, 3000, 3000\]' "${WORK}/comparison.json"
grep -q '"baseline_linear_outcome_comparisons": \[500000, 500000, 500000\]' "${WORK}/comparison.json"
grep -q '"optimized_indexed_outcome_lookups": \[9000, 9000, 9000\]' "${WORK}/comparison.json"
grep -q '"baseline_linear_preference_comparisons": \[500000, 500000, 500000\]' "${WORK}/comparison.json"
grep -q '"optimized_indexed_preference_candidates": \[10000, 10000, 10000\]' "${WORK}/comparison.json"
grep -q '"baseline_linear_active_learning_comparisons": \[500000, 500000, 500000\]' "${WORK}/comparison.json"
grep -q '"optimized_indexed_active_learning_candidates": \[10000, 10000, 10000\]' "${WORK}/comparison.json"
grep -q '"general_100000x_compute_efficiency_proven": false' "${WORK}/comparison.json"
sed -i 's/"accuracy":1/"accuracy":0.5/' \
  "${WORK}/optimized/runs/run_3/evaluation/scoring/summary.json"
set +e
"${ROOT}/scripts/compare_efficiency_campaigns_rtx_pro_6000.sh" \
  "${WORK}/baseline" "${WORK}/optimized" "${WORK}/quality-mismatch.json" >/dev/null
STATUS=$?
set -e
[[ "${STATUS}" -eq 3 ]]
grep -q '"matched_controlled_quality": false' "${WORK}/quality-mismatch.json"
grep -q '"general_100000x_compute_efficiency_proven": false' \
  "${WORK}/quality-mismatch.json"
