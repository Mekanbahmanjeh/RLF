#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
  echo "Usage: compare_efficiency_campaigns_rtx_pro_6000.sh BASELINE_ROOT OPTIMIZED_ROOT NEW_REPORT.json" >&2
}
[[ $# -eq 3 ]] || { usage; exit 2; }
BASELINE_ROOT="$(realpath "${1}")"
OPTIMIZED_ROOT="$(realpath "${2}")"
REPORT="$(realpath -m "${3}")"
[[ -f "${BASELINE_ROOT}/runs.tsv" && -f "${OPTIMIZED_ROOT}/runs.tsv" ]] || {
  echo "both complete campaign roots are required" >&2; exit 2
}
[[ ! -e "${REPORT}" ]] || { echo "comparison report already exists" >&2; exit 2; }

json_number() {
  sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*([-+0-9.eE]+).*/\1/p" "$1" | head -n 1
}
json_string() {
  sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"([^\"]*)\".*/\1/p" "$1" | head -n 1
}
environment_value() {
  awk -F= -v key="$2" '$1 == key { sub(/^[^=]*=/, ""); print; exit }' "$1"
}
mean_three() {
  awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN { printf "%.17g", (a+b+c)/3.0 }'
}
ratio_or_null() {
  awk -v numerator="$1" -v denominator="$2" 'BEGIN { if (denominator > 0) printf "%.17g", numerator/denominator; else print "null" }'
}
json_array_three() { printf '[%s, %s, %s]' "$1" "$2" "$3"; }
json_string_array_three() { printf '["%s", "%s", "%s"]' "$1" "$2" "$3"; }

declare -a BASE_WALL BASE_GPU BASE_TRAIN_WALL BASE_ACCURACY BASE_UPDATES BASE_AVOIDED BASE_NORM_FMAS BASE_HOST_NORM_FMAS BASE_HOST_COSINE BASE_COSINE_FMAS BASE_ROUTER_REBUILT BASE_ROUTER_UPDATED BASE_ROUTER_APPENDED BASE_INDEXED_RERANK BASE_INDEXED_PAIRS BASE_CONCEPT_LOOKUPS BASE_LINEAR_CONCEPT BASE_INDEXED_CONCEPT BASE_EXAMPLE_LOOKUPS BASE_LINEAR_EXAMPLE BASE_INDEXED_EXAMPLE BASE_MODE_LOOKUPS BASE_MODE_REBUILT BASE_MODE_INSERTS BASE_REGION_LOOKUPS BASE_LINEAR_REGION BASE_INDEXED_REGION BASE_GROUND_LOOKUPS BASE_GROUND_REBUILT BASE_GROUND_CANDIDATES BASE_GROUND_INSERTS BASE_GROUND_CONFIDENCE BASE_GROUND_SWEEPS BASE_GROUND_SORTS BASE_GROUND_MODE_SCAN BASE_GROUND_MODE_INDEXED BASE_GROUND_CONCEPT_SCAN BASE_GROUND_CONCEPT_INDEXED BASE_LANGUAGE_LOOKUPS BASE_LINEAR_OUTCOMES BASE_INDEXED_OUTCOMES BASE_OUTCOME_BUILDS BASE_OUTCOME_ENTRIES BASE_OUTCOME_INSERTS BASE_PREFERENCE_LOOKUPS BASE_LINEAR_PREFERENCES BASE_INDEXED_PREFERENCES BASE_PREFERENCE_REBUILDS BASE_PREFERENCE_ENTRIES BASE_PREFERENCE_INSERTS BASE_ACTIVE_LOOKUPS BASE_LINEAR_ACTIVE BASE_INDEXED_ACTIVE BASE_ACTIVE_REBUILDS BASE_ACTIVE_ENTRIES BASE_ACTIVE_INSERTS BASE_CHECKPOINT
declare -a OPT_WALL OPT_GPU OPT_TRAIN_WALL OPT_ACCURACY OPT_UPDATES OPT_AVOIDED OPT_NORM_FMAS OPT_HOST_NORM_FMAS OPT_HOST_COSINE OPT_COSINE_FMAS OPT_ROUTER_REBUILT OPT_ROUTER_UPDATED OPT_ROUTER_APPENDED OPT_INDEXED_RERANK OPT_INDEXED_PAIRS OPT_CONCEPT_LOOKUPS OPT_LINEAR_CONCEPT OPT_INDEXED_CONCEPT OPT_EXAMPLE_LOOKUPS OPT_LINEAR_EXAMPLE OPT_INDEXED_EXAMPLE OPT_MODE_LOOKUPS OPT_MODE_REBUILT OPT_MODE_INSERTS OPT_REGION_LOOKUPS OPT_LINEAR_REGION OPT_INDEXED_REGION OPT_GROUND_LOOKUPS OPT_GROUND_REBUILT OPT_GROUND_CANDIDATES OPT_GROUND_INSERTS OPT_GROUND_CONFIDENCE OPT_GROUND_SWEEPS OPT_GROUND_SORTS OPT_GROUND_MODE_SCAN OPT_GROUND_MODE_INDEXED OPT_GROUND_CONCEPT_SCAN OPT_GROUND_CONCEPT_INDEXED OPT_LANGUAGE_LOOKUPS OPT_LINEAR_OUTCOMES OPT_INDEXED_OUTCOMES OPT_OUTCOME_BUILDS OPT_OUTCOME_ENTRIES OPT_OUTCOME_INSERTS OPT_PREFERENCE_LOOKUPS OPT_LINEAR_PREFERENCES OPT_INDEXED_PREFERENCES OPT_PREFERENCE_REBUILDS OPT_PREFERENCE_ENTRIES OPT_PREFERENCE_INSERTS OPT_ACTIVE_LOOKUPS OPT_LINEAR_ACTIVE OPT_INDEXED_ACTIVE OPT_ACTIVE_REBUILDS OPT_ACTIVE_ENTRIES OPT_ACTIVE_INSERTS OPT_CHECKPOINT
COMMON_LEDGER=""; COMMON_SOURCE=""; COMMON_BINARY=""; COMMON_EXPECTED=""
collect() {
  local campaign_root="$1" expected_variant="$2" array_prefix="$3"
  local -n wall_array="${array_prefix}_WALL"
  local -n gpu_array="${array_prefix}_GPU"
  local -n train_wall_array="${array_prefix}_TRAIN_WALL"
  local -n accuracy_array="${array_prefix}_ACCURACY"
  local -n updates_array="${array_prefix}_UPDATES"
  local -n avoided_array="${array_prefix}_AVOIDED"
  local -n norm_fmas_array="${array_prefix}_NORM_FMAS"
  local -n host_norm_fmas_array="${array_prefix}_HOST_NORM_FMAS"
  local -n host_cosine_array="${array_prefix}_HOST_COSINE"
  local -n cosine_fmas_array="${array_prefix}_COSINE_FMAS"
  local -n router_rebuilt_array="${array_prefix}_ROUTER_REBUILT"
  local -n router_updated_array="${array_prefix}_ROUTER_UPDATED"
  local -n router_appended_array="${array_prefix}_ROUTER_APPENDED"
  local -n indexed_rerank_array="${array_prefix}_INDEXED_RERANK"
  local -n indexed_pairs_array="${array_prefix}_INDEXED_PAIRS"
  local -n concept_lookups_array="${array_prefix}_CONCEPT_LOOKUPS"
  local -n linear_concept_array="${array_prefix}_LINEAR_CONCEPT"
  local -n indexed_concept_array="${array_prefix}_INDEXED_CONCEPT"
  local -n example_lookups_array="${array_prefix}_EXAMPLE_LOOKUPS"
  local -n linear_example_array="${array_prefix}_LINEAR_EXAMPLE"
  local -n indexed_example_array="${array_prefix}_INDEXED_EXAMPLE"
  local -n mode_lookups_array="${array_prefix}_MODE_LOOKUPS"
  local -n mode_rebuilt_array="${array_prefix}_MODE_REBUILT"
  local -n mode_inserts_array="${array_prefix}_MODE_INSERTS"
  local -n region_lookups_array="${array_prefix}_REGION_LOOKUPS"
  local -n linear_region_array="${array_prefix}_LINEAR_REGION"
  local -n indexed_region_array="${array_prefix}_INDEXED_REGION"
  local -n ground_lookups_array="${array_prefix}_GROUND_LOOKUPS"
  local -n ground_rebuilt_array="${array_prefix}_GROUND_REBUILT"
  local -n ground_candidates_array="${array_prefix}_GROUND_CANDIDATES"
  local -n ground_inserts_array="${array_prefix}_GROUND_INSERTS"
  local -n ground_confidence_array="${array_prefix}_GROUND_CONFIDENCE"
  local -n ground_sweeps_array="${array_prefix}_GROUND_SWEEPS"
  local -n ground_sorts_array="${array_prefix}_GROUND_SORTS"
  local -n ground_mode_scan_array="${array_prefix}_GROUND_MODE_SCAN"
  local -n ground_mode_indexed_array="${array_prefix}_GROUND_MODE_INDEXED"
  local -n ground_concept_scan_array="${array_prefix}_GROUND_CONCEPT_SCAN"
  local -n ground_concept_indexed_array="${array_prefix}_GROUND_CONCEPT_INDEXED"
  local -n language_lookups_array="${array_prefix}_LANGUAGE_LOOKUPS"
  local -n linear_outcomes_array="${array_prefix}_LINEAR_OUTCOMES"
  local -n indexed_outcomes_array="${array_prefix}_INDEXED_OUTCOMES"
  local -n outcome_builds_array="${array_prefix}_OUTCOME_BUILDS"
  local -n outcome_entries_array="${array_prefix}_OUTCOME_ENTRIES"
  local -n outcome_inserts_array="${array_prefix}_OUTCOME_INSERTS"
  local -n preference_lookups_array="${array_prefix}_PREFERENCE_LOOKUPS"
  local -n linear_preferences_array="${array_prefix}_LINEAR_PREFERENCES"
  local -n indexed_preferences_array="${array_prefix}_INDEXED_PREFERENCES"
  local -n preference_rebuilds_array="${array_prefix}_PREFERENCE_REBUILDS"
  local -n preference_entries_array="${array_prefix}_PREFERENCE_ENTRIES"
  local -n preference_inserts_array="${array_prefix}_PREFERENCE_INSERTS"
  local -n active_lookups_array="${array_prefix}_ACTIVE_LOOKUPS"
  local -n linear_active_array="${array_prefix}_LINEAR_ACTIVE"
  local -n indexed_active_array="${array_prefix}_INDEXED_ACTIVE"
  local -n active_rebuilds_array="${array_prefix}_ACTIVE_REBUILDS"
  local -n active_entries_array="${array_prefix}_ACTIVE_ENTRIES"
  local -n active_inserts_array="${array_prefix}_ACTIVE_INSERTS"
  local -n checkpoint_array="${array_prefix}_CHECKPOINT"
  local expected_update_policy expected_cosine_policy expected_small_cosine_policy expected_router_policy expected_rerank_policy expected_concept_policy expected_example_policy expected_mode_policy expected_grounding_policy expected_language_policy expected_dialogue_policy expected_instruction_policy expected_tool_keyword_policy expected_preference_policy expected_active_policy expected_separate
  if [[ "${expected_variant}" == baseline ]]; then
    expected_update_policy=device; expected_cosine_policy=inline_norms; expected_small_cosine_policy=device; expected_router_policy=rebuild; expected_rerank_policy=per_query; expected_concept_policy=linear; expected_example_policy=linear; expected_mode_policy=rebuild; expected_grounding_policy=rebuild; expected_language_policy=linear; expected_dialogue_policy=redundant; expected_instruction_policy=retrieval; expected_tool_keyword_policy=linear; expected_preference_policy=linear; expected_active_policy=linear; expected_separate=1
  else
    expected_update_policy=hybrid; expected_cosine_policy=precomputed_norms; expected_small_cosine_policy=hybrid; expected_router_policy=incremental; expected_rerank_policy=batched; expected_concept_policy=indexed; expected_example_policy=indexed; expected_mode_policy=persistent; expected_grounding_policy=persistent; expected_language_policy=indexed; expected_dialogue_policy=fused; expected_instruction_policy=indexed; expected_tool_keyword_policy=indexed; expected_preference_policy=indexed; expected_active_policy=indexed; expected_separate=0
  fi
  [[ "$(awk -F'\t' 'NR == 1 { print $1 }' "${campaign_root}/runs.tsv")" == variant ]] || {
    echo "runs.tsv lacks a variant column" >&2; exit 1
  }
  for run in 1 2 3; do
    local row variant wall checkpoint training evaluation environment telemetry resource score manifest
    row="$(awk -F'\t' -v run="${run}" 'NR > 1 && $2 == run { print; found += 1 } END { if (found != 1) exit 1 }' "${campaign_root}/runs.tsv")" || {
      echo "campaign must contain exactly one row for run ${run}" >&2; exit 1
    }
    IFS=$'\t' read -r variant _ _ _ wall checkpoint training evaluation _ <<<"${row}"
    [[ "${variant}" == "${expected_variant}" ]] || { echo "campaign variant mismatch" >&2; exit 1; }
    environment="${training}/training_environment.txt"
    telemetry="${training}/training_pipeline_telemetry.json"
    resource="${training}/training_resource_summary.json"
    score="${evaluation}/scoring/summary.json"
    manifest="${training}/training_artifact_manifest.tsv"
    for file in "${checkpoint}" "${environment}" "${telemetry}" "${resource}" "${score}" "${manifest}"; do
      [[ -f "${file}" ]] || { echo "missing campaign artifact: ${file}" >&2; exit 1; }
    done
    "${ROOT}/scripts/create_training_artifact_manifest.sh" --verify "${manifest}" >/dev/null
    [[ "$(environment_value "${environment}" cuda_local_update_policy)" == "${expected_update_policy}" ]] || { echo "local-update policy mismatch" >&2; exit 1; }
    [[ "$(environment_value "${environment}" cuda_cached_cosine_policy)" == "${expected_cosine_policy}" ]] || { echo "cached-cosine policy mismatch" >&2; exit 1; }
    [[ "$(environment_value "${environment}" cuda_small_cosine_policy)" == "${expected_small_cosine_policy}" ]] || { echo "small-cosine policy mismatch" >&2; exit 1; }
    [[ "$(environment_value "${environment}" sparse_router_update_policy)" == "${expected_router_policy}" ]] || { echo "sparse-router policy mismatch" >&2; exit 1; }
    [[ "$(environment_value "${environment}" sparse_rerank_policy)" == "${expected_rerank_policy}" ]] || { echo "sparse-rerank policy mismatch" >&2; exit 1; }
    [[ "$(environment_value "${environment}" concept_update_policy)" == "${expected_concept_policy}" ]] || { echo "concept-update policy mismatch" >&2; exit 1; }
    [[ "$(environment_value "${environment}" example_duplicate_policy)" == "${expected_example_policy}" ]] || { echo "example-duplicate policy mismatch" >&2; exit 1; }
    [[ "$(environment_value "${environment}" mode_id_index_policy)" == "${expected_mode_policy}" ]] || { echo "mode-ID index policy mismatch" >&2; exit 1; }
    [[ "$(environment_value "${environment}" grounding_index_policy)" == "${expected_grounding_policy}" ]] || { echo "grounding index policy mismatch" >&2; exit 1; }
    [[ "$(environment_value "${environment}" language_outcome_policy)" == "${expected_language_policy}" ]] || { echo "language outcome policy mismatch" >&2; exit 1; }
    [[ "$(environment_value "${environment}" language_dialogue_encoding_policy)" == "${expected_dialogue_policy}" ]] || { echo "language dialogue encoding policy mismatch" >&2; exit 1; }
    [[ "$(environment_value "${environment}" instruction_duplicate_policy)" == "${expected_instruction_policy}" ]] || { echo "instruction duplicate policy mismatch" >&2; exit 1; }
    [[ "$(environment_value "${environment}" tool_keyword_update_policy)" == "${expected_tool_keyword_policy}" ]] || { echo "tool keyword policy mismatch" >&2; exit 1; }
    [[ "$(environment_value "${environment}" preference_duplicate_policy)" == "${expected_preference_policy}" ]] || { echo "preference duplicate policy mismatch" >&2; exit 1; }
    [[ "$(environment_value "${environment}" active_learning_duplicate_policy)" == "${expected_active_policy}" ]] || { echo "active-learning duplicate policy mismatch" >&2; exit 1; }
    [[ "$(environment_value "${environment}" separate_vision_analysis)" == "${expected_separate}" ]] || { echo "visual-analysis policy mismatch" >&2; exit 1; }
    local ledger source binary expected_sha gpu train_wall accuracy updates avoided norm_fmas host_norm_fmas host_cosine cosine_fmas router_rebuilt router_updated router_appended indexed_rerank indexed_pairs concept_lookups linear_concept indexed_concept example_lookups linear_example indexed_example mode_lookups mode_rebuilt mode_inserts region_lookups linear_region indexed_region ground_lookups ground_rebuilt ground_candidates ground_inserts ground_confidence ground_sweeps ground_sorts ground_mode_scan ground_mode_indexed ground_concept_scan ground_concept_indexed language_lookups linear_outcomes indexed_outcomes outcome_builds outcome_entries outcome_inserts preference_lookups linear_preferences indexed_preferences preference_rebuilds preference_entries preference_inserts active_lookups linear_active indexed_active active_rebuilds active_entries active_inserts checkpoint_sha
    ledger="$(json_string "${telemetry}" ledger_sha256)"
    source="$(environment_value "${environment}" source_manifest_sha256)"
    binary="$(environment_value "${environment}" solstice_binary_sha256)"
    expected_sha="$(json_string "${score}" expected_sha256)"
    gpu="$(json_number "${resource}" gpu_active_seconds_estimate)"
    train_wall="$(json_number "${telemetry}" total_wall_seconds)"
    accuracy="$(json_number "${score}" accuracy)"
    updates="$(json_number "${telemetry}" local_update_calls)"
    avoided="$(json_number "${telemetry}" avoided_kernel_launches)"
    norm_fmas="$(json_number "${telemetry}" avoided_pairwise_norm_fma_operations)"
    host_norm_fmas="$(json_number "${telemetry}" host_precomputed_norm_fma_operations)"
    host_cosine="$(json_number "${telemetry}" host_batch_cosine_calls)"
    cosine_fmas="$(json_number "${telemetry}" avoided_device_cosine_fma_operations)"
    router_rebuilt="$(json_number "${telemetry}" vectors_rebuilt)"
    router_updated="$(json_number "${telemetry}" vectors_incrementally_updated)"
    router_appended="$(json_number "${telemetry}" vectors_appended)"
    indexed_rerank="$(json_number "${telemetry}" indexed_batch_cosine_calls)"
    indexed_pairs="$(json_number "${telemetry}" indexed_cosine_pairs)"
    concept_lookups="$(json_number "${telemetry}" concept_update_lookups)"
    linear_concept="$(json_number "${telemetry}" linear_concept_comparisons)"
    indexed_concept="$(json_number "${telemetry}" indexed_concept_lookups)"
    example_lookups="$(json_number "${telemetry}" example_duplicate_lookups)"
    linear_example="$(json_number "${telemetry}" linear_example_comparisons)"
    indexed_example="$(json_number "${telemetry}" indexed_example_candidates)"
    mode_lookups="$(json_number "${telemetry}" mode_id_lookups)"
    mode_rebuilt="$(json_number "${telemetry}" mode_id_index_entries_rebuilt)"
    mode_inserts="$(json_number "${telemetry}" mode_id_index_incremental_inserts)"
    region_lookups="$(json_number "${telemetry}" region_mode_id_lookups)"
    linear_region="$(json_number "${telemetry}" linear_region_mode_comparisons)"
    indexed_region="$(json_number "${telemetry}" indexed_region_mode_lookups)"
    ground_lookups="$(json_number "${telemetry}" link_lookups)"
    ground_rebuilt="$(json_number "${telemetry}" full_lookup_entries_rebuilt)"
    ground_candidates="$(json_number "${telemetry}" indexed_link_candidates_examined)"
    ground_inserts="$(json_number "${telemetry}" incremental_posting_inserts)"
    ground_confidence="$(json_number "${telemetry}" confidence_recomputations)"
    ground_sweeps="$(json_number "${telemetry}" full_confidence_sweep_entries)"
    ground_sorts="$(json_number "${telemetry}" derived_sort_entries)"
    ground_mode_scan="$(json_number "${telemetry}" mode_query_full_scan_entries)"
    ground_mode_indexed="$(json_number "${telemetry}" mode_query_indexed_candidates)"
    ground_concept_scan="$(json_number "${telemetry}" concept_query_full_scan_entries)"
    ground_concept_indexed="$(json_number "${telemetry}" concept_query_indexed_candidates)"
    language_lookups="$(json_number "${telemetry}" outcome_update_lookups)"
    linear_outcomes="$(json_number "${telemetry}" linear_outcome_comparisons)"
    indexed_outcomes="$(json_number "${telemetry}" indexed_outcome_lookups)"
    outcome_builds="$(json_number "${telemetry}" outcome_index_builds)"
    outcome_entries="$(json_number "${telemetry}" outcome_index_entries_built)"
    outcome_inserts="$(json_number "${telemetry}" outcome_index_incremental_inserts)"
    preference_lookups="$(json_number "${telemetry}" preference_duplicate_lookups)"
    linear_preferences="$(json_number "${telemetry}" linear_preference_comparisons)"
    indexed_preferences="$(json_number "${telemetry}" indexed_preference_candidates)"
    preference_rebuilds="$(json_number "${telemetry}" preference_index_rebuilds)"
    preference_entries="$(json_number "${telemetry}" preference_index_entries_built)"
    preference_inserts="$(json_number "${telemetry}" preference_index_incremental_inserts)"
    active_lookups="$(json_number "${telemetry}" active_learning_duplicate_lookups)"
    linear_active="$(json_number "${telemetry}" linear_active_learning_comparisons)"
    indexed_active="$(json_number "${telemetry}" indexed_active_learning_candidates)"
    active_rebuilds="$(json_number "${telemetry}" active_learning_index_rebuilds)"
    active_entries="$(json_number "${telemetry}" active_learning_index_entries_built)"
    active_inserts="$(json_number "${telemetry}" active_learning_index_incremental_inserts)"
    checkpoint_sha="$(sha256sum -- "${checkpoint}" | awk '{print $1}')"
    [[ "${ledger}" =~ ^[0-9a-f]{64}$ && "${source}" =~ ^[0-9a-f]{64}$ &&
       "${binary}" =~ ^[0-9a-f]{64}$ && "${expected_sha}" =~ ^[0-9a-f]{64}$ &&
       -n "${gpu}" && -n "${train_wall}" && -n "${accuracy}" &&
       -n "${updates}" && -n "${avoided}" && -n "${norm_fmas}" &&
       -n "${host_norm_fmas}" && -n "${host_cosine}" && -n "${cosine_fmas}" &&
       -n "${router_rebuilt}" && -n "${router_updated}" &&
       -n "${router_appended}" && -n "${indexed_rerank}" &&
       -n "${indexed_pairs}" && -n "${concept_lookups}" &&
       -n "${linear_concept}" && -n "${indexed_concept}" &&
       -n "${example_lookups}" && -n "${linear_example}" &&
       -n "${indexed_example}" && -n "${mode_lookups}" &&
       -n "${mode_rebuilt}" && -n "${mode_inserts}" &&
       -n "${region_lookups}" && -n "${linear_region}" &&
       -n "${indexed_region}" && -n "${ground_lookups}" &&
       -n "${ground_rebuilt}" && -n "${ground_candidates}" &&
       -n "${ground_inserts}" && -n "${ground_confidence}" &&
       -n "${ground_sweeps}" && -n "${ground_sorts}" &&
       -n "${ground_mode_scan}" && -n "${ground_mode_indexed}" &&
       -n "${ground_concept_scan}" && -n "${ground_concept_indexed}" &&
       -n "${language_lookups}" && -n "${linear_outcomes}" &&
       -n "${indexed_outcomes}" && -n "${outcome_builds}" &&
       -n "${outcome_entries}" && -n "${outcome_inserts}" &&
       -n "${preference_lookups}" && -n "${linear_preferences}" &&
       -n "${indexed_preferences}" && -n "${preference_rebuilds}" &&
       -n "${preference_entries}" && -n "${preference_inserts}" &&
       -n "${active_lookups}" && -n "${linear_active}" &&
       -n "${indexed_active}" && -n "${active_rebuilds}" &&
       -n "${active_entries}" && -n "${active_inserts}" ]] || {
      echo "campaign telemetry is incomplete" >&2; exit 1
    }
    if [[ -z "${COMMON_LEDGER}" ]]; then
      COMMON_LEDGER="${ledger}"; COMMON_SOURCE="${source}"
      COMMON_BINARY="${binary}"; COMMON_EXPECTED="${expected_sha}"
    fi
    [[ "${ledger}" == "${COMMON_LEDGER}" && "${source}" == "${COMMON_SOURCE}" &&
       "${binary}" == "${COMMON_BINARY}" && "${expected_sha}" == "${COMMON_EXPECTED}" ]] || {
      echo "baseline and optimized input/binary identities differ" >&2; exit 1
    }
    wall_array[run]="${wall}"
    gpu_array[run]="${gpu}"
    train_wall_array[run]="${train_wall}"
    accuracy_array[run]="${accuracy}"
    updates_array[run]="${updates}"
    avoided_array[run]="${avoided}"
    norm_fmas_array[run]="${norm_fmas}"
    host_norm_fmas_array[run]="${host_norm_fmas}"
    host_cosine_array[run]="${host_cosine}"
    cosine_fmas_array[run]="${cosine_fmas}"
    router_rebuilt_array[run]="${router_rebuilt}"
    router_updated_array[run]="${router_updated}"
    router_appended_array[run]="${router_appended}"
    indexed_rerank_array[run]="${indexed_rerank}"
    indexed_pairs_array[run]="${indexed_pairs}"
    concept_lookups_array[run]="${concept_lookups}"
    linear_concept_array[run]="${linear_concept}"
    indexed_concept_array[run]="${indexed_concept}"
    example_lookups_array[run]="${example_lookups}"
    linear_example_array[run]="${linear_example}"
    indexed_example_array[run]="${indexed_example}"
    mode_lookups_array[run]="${mode_lookups}"
    mode_rebuilt_array[run]="${mode_rebuilt}"
    mode_inserts_array[run]="${mode_inserts}"
    region_lookups_array[run]="${region_lookups}"
    linear_region_array[run]="${linear_region}"
    indexed_region_array[run]="${indexed_region}"
    ground_lookups_array[run]="${ground_lookups}"
    ground_rebuilt_array[run]="${ground_rebuilt}"
    ground_candidates_array[run]="${ground_candidates}"
    ground_inserts_array[run]="${ground_inserts}"
    ground_confidence_array[run]="${ground_confidence}"
    ground_sweeps_array[run]="${ground_sweeps}"
    ground_sorts_array[run]="${ground_sorts}"
    ground_mode_scan_array[run]="${ground_mode_scan}"
    ground_mode_indexed_array[run]="${ground_mode_indexed}"
    ground_concept_scan_array[run]="${ground_concept_scan}"
    ground_concept_indexed_array[run]="${ground_concept_indexed}"
    language_lookups_array[run]="${language_lookups}"
    linear_outcomes_array[run]="${linear_outcomes}"
    indexed_outcomes_array[run]="${indexed_outcomes}"
    outcome_builds_array[run]="${outcome_builds}"
    outcome_entries_array[run]="${outcome_entries}"
    outcome_inserts_array[run]="${outcome_inserts}"
    preference_lookups_array[run]="${preference_lookups}"
    linear_preferences_array[run]="${linear_preferences}"
    indexed_preferences_array[run]="${indexed_preferences}"
    preference_rebuilds_array[run]="${preference_rebuilds}"
    preference_entries_array[run]="${preference_entries}"
    preference_inserts_array[run]="${preference_inserts}"
    active_lookups_array[run]="${active_lookups}"
    linear_active_array[run]="${linear_active}"
    indexed_active_array[run]="${indexed_active}"
    active_rebuilds_array[run]="${active_rebuilds}"
    active_entries_array[run]="${active_entries}"
    active_inserts_array[run]="${active_inserts}"
    checkpoint_array[run]="${checkpoint_sha}"
  done
}

collect "${BASELINE_ROOT}" baseline BASE
collect "${OPTIMIZED_ROOT}" optimized OPT

MATCHED_CONTROLLED_QUALITY=true
MATCHED_UPDATE_COUNTS=true
for run in 1 2 3; do
  [[ "${BASE_ACCURACY[run]}" == "${OPT_ACCURACY[run]}" ]] || MATCHED_CONTROLLED_QUALITY=false
  [[ "${BASE_UPDATES[run]}" == "${OPT_UPDATES[run]}" ]] || MATCHED_UPDATE_COUNTS=false
  [[ "${BASE_AVOIDED[run]}" == 0 ]] || { echo "device baseline reports avoided launches" >&2; exit 1; }
  [[ "${BASE_NORM_FMAS[run]}" == 0 ]] || { echo "inline-norm baseline reports avoided norm FMAs" >&2; exit 1; }
  [[ "${BASE_HOST_NORM_FMAS[run]}" == 0 ]] || { echo "inline-norm baseline reports host norm FMAs" >&2; exit 1; }
  [[ "${BASE_HOST_COSINE[run]}" == 0 ]] || { echo "device baseline reports host cosine calls" >&2; exit 1; }
  [[ "${BASE_COSINE_FMAS[run]}" == 0 ]] || { echo "device baseline reports avoided cosine FMAs" >&2; exit 1; }
  [[ "${BASE_ROUTER_UPDATED[run]}" == 0 ]] || { echo "rebuild baseline reports incremental router updates" >&2; exit 1; }
  [[ "${BASE_ROUTER_APPENDED[run]}" == 0 ]] || { echo "rebuild baseline reports incremental router appends" >&2; exit 1; }
  [[ "${BASE_INDEXED_RERANK[run]}" == 0 ]] || { echo "per-query baseline reports indexed sparse reranking" >&2; exit 1; }
  [[ "${OPT_INDEXED_RERANK[run]}" -gt 0 ]] || { echo "optimized run did not exercise batched sparse reranking" >&2; exit 1; }
  [[ "${OPT_INDEXED_PAIRS[run]}" -gt 0 ]] || { echo "optimized run reports no indexed sparse-rerank work" >&2; exit 1; }
  [[ "${BASE_INDEXED_CONCEPT[run]}" == 0 ]] || { echo "linear baseline reports indexed concept updates" >&2; exit 1; }
  [[ "${OPT_INDEXED_CONCEPT[run]}" -gt 0 ]] || { echo "optimized run did not exercise indexed concept updates" >&2; exit 1; }
  [[ "${BASE_CONCEPT_LOOKUPS[run]}" == "${OPT_CONCEPT_LOOKUPS[run]}" ]] || { echo "concept-update workloads differ" >&2; exit 1; }
  [[ "${BASE_INDEXED_EXAMPLE[run]}" == 0 ]] || { echo "linear baseline reports indexed example candidates" >&2; exit 1; }
  [[ "${BASE_EXAMPLE_LOOKUPS[run]}" == "${OPT_EXAMPLE_LOOKUPS[run]}" ]] || { echo "example-duplicate workloads differ" >&2; exit 1; }
  if [[ "${OPT_EXAMPLE_LOOKUPS[run]}" -gt 1 ]]; then
    [[ "${OPT_LINEAR_EXAMPLE[run]}" == 0 ]] || { echo "optimized run reports linear example comparisons" >&2; exit 1; }
    [[ "${BASE_LINEAR_EXAMPLE[run]}" -gt "${OPT_INDEXED_EXAMPLE[run]}" ]] || { echo "optimized example index did not reduce examined candidates" >&2; exit 1; }
  fi
  [[ "${BASE_MODE_LOOKUPS[run]}" == "${OPT_MODE_LOOKUPS[run]}" ]] || { echo "mode-ID lookup workloads differ" >&2; exit 1; }
  [[ "${BASE_MODE_INSERTS[run]}" == "${OPT_MODE_INSERTS[run]}" ]] || { echo "mode-ID insertion workloads differ" >&2; exit 1; }
  if [[ "${OPT_MODE_LOOKUPS[run]}" -gt 1 ]]; then
    [[ "${BASE_MODE_REBUILT[run]}" -gt "${OPT_MODE_REBUILT[run]}" ]] || { echo "persistent mode-ID index did not reduce rebuilt entries" >&2; exit 1; }
  fi
  [[ "${BASE_REGION_LOOKUPS[run]}" == "${OPT_REGION_LOOKUPS[run]}" ]] || { echo "region mode-ID lookup workloads differ" >&2; exit 1; }
  [[ "${BASE_INDEXED_REGION[run]}" == 0 ]] || { echo "rebuild baseline reports indexed region mode-ID lookups" >&2; exit 1; }
  [[ "${OPT_LINEAR_REGION[run]}" == 0 ]] || { echo "persistent optimized run reports linear region mode-ID comparisons" >&2; exit 1; }
  [[ "${OPT_INDEXED_REGION[run]}" == "${OPT_REGION_LOOKUPS[run]}" ]] || { echo "persistent optimized run did not index every region mode-ID lookup" >&2; exit 1; }
  [[ "${BASE_GROUND_LOOKUPS[run]}" == "${OPT_GROUND_LOOKUPS[run]}" ]] || { echo "grounding link lookup workloads differ" >&2; exit 1; }
  [[ "${BASE_GROUND_CANDIDATES[run]}" == 0 ]] || { echo "rebuild baseline reports indexed grounding candidates" >&2; exit 1; }
  [[ "${OPT_GROUND_REBUILT[run]}" == 0 ]] || { echo "persistent grounding run rebuilt lookup entries" >&2; exit 1; }
  [[ "${OPT_GROUND_SWEEPS[run]}" == 0 ]] || { echo "persistent grounding run swept historical confidence entries" >&2; exit 1; }
  [[ "${OPT_GROUND_SORTS[run]}" == 0 ]] || { echo "persistent grounding run sorted historical derived entries" >&2; exit 1; }
  if [[ "${OPT_GROUND_LOOKUPS[run]}" -gt 3 ]]; then
    [[ "${BASE_GROUND_REBUILT[run]}" -gt "${OPT_GROUND_REBUILT[run]}" ]] || { echo "persistent grounding index did not reduce rebuild work" >&2; exit 1; }
    [[ "${BASE_GROUND_SWEEPS[run]}" -gt "${OPT_GROUND_CONFIDENCE[run]}" ]] || { echo "sparse grounding confidence updates did not reduce work" >&2; exit 1; }
    [[ "${BASE_GROUND_SORTS[run]}" -gt "${OPT_GROUND_SORTS[run]}" ]] || { echo "persistent grounding index did not remove repeated sorts" >&2; exit 1; }
  fi
  [[ "${BASE_LANGUAGE_LOOKUPS[run]}" == "${OPT_LANGUAGE_LOOKUPS[run]}" ]] || { echo "language outcome update workloads differ" >&2; exit 1; }
  [[ "${BASE_INDEXED_OUTCOMES[run]}" == 0 ]] || { echo "linear baseline reports indexed language outcomes" >&2; exit 1; }
  [[ "${OPT_INDEXED_OUTCOMES[run]}" -gt 0 ]] || { echo "optimized run did not exercise indexed language outcomes" >&2; exit 1; }
  [[ "${BASE_LINEAR_OUTCOMES[run]}" -gt "${OPT_LINEAR_OUTCOMES[run]}" ]] || { echo "adaptive language outcome index did not reduce linear comparisons" >&2; exit 1; }
  [[ "${BASE_PREFERENCE_LOOKUPS[run]}" == "${OPT_PREFERENCE_LOOKUPS[run]}" ]] || { echo "preference duplicate workloads differ" >&2; exit 1; }
  [[ "${BASE_INDEXED_PREFERENCES[run]}" == 0 ]] || { echo "linear baseline reports indexed preference candidates" >&2; exit 1; }
  if [[ "${OPT_PREFERENCE_LOOKUPS[run]}" -gt 1 ]]; then
    [[ "${OPT_LINEAR_PREFERENCES[run]}" == 0 ]] || { echo "optimized run reports linear preference comparisons" >&2; exit 1; }
    [[ "${BASE_LINEAR_PREFERENCES[run]}" -gt "${OPT_INDEXED_PREFERENCES[run]}" ]] || { echo "preference duplicate index did not reduce examined candidates" >&2; exit 1; }
  fi
  [[ "${BASE_ACTIVE_LOOKUPS[run]}" == "${OPT_ACTIVE_LOOKUPS[run]}" ]] || { echo "active-learning duplicate workloads differ" >&2; exit 1; }
  [[ "${BASE_INDEXED_ACTIVE[run]}" == 0 ]] || { echo "linear baseline reports indexed active-learning candidates" >&2; exit 1; }
  if [[ "${OPT_ACTIVE_LOOKUPS[run]}" -gt 1 ]]; then
    [[ "${OPT_LINEAR_ACTIVE[run]}" == 0 ]] || { echo "optimized run reports linear active-learning comparisons" >&2; exit 1; }
    [[ "${BASE_LINEAR_ACTIVE[run]}" -gt "${OPT_INDEXED_ACTIVE[run]}" ]] || { echo "active-learning duplicate index did not reduce examined candidates" >&2; exit 1; }
  fi
done
BASE_WALL_MEAN="$(mean_three "${BASE_WALL[1]}" "${BASE_WALL[2]}" "${BASE_WALL[3]}")"
OPT_WALL_MEAN="$(mean_three "${OPT_WALL[1]}" "${OPT_WALL[2]}" "${OPT_WALL[3]}")"
BASE_GPU_MEAN="$(mean_three "${BASE_GPU[1]}" "${BASE_GPU[2]}" "${BASE_GPU[3]}")"
OPT_GPU_MEAN="$(mean_three "${OPT_GPU[1]}" "${OPT_GPU[2]}" "${OPT_GPU[3]}")"
BASE_TRAIN_MEAN="$(mean_three "${BASE_TRAIN_WALL[1]}" "${BASE_TRAIN_WALL[2]}" "${BASE_TRAIN_WALL[3]}")"
OPT_TRAIN_MEAN="$(mean_three "${OPT_TRAIN_WALL[1]}" "${OPT_TRAIN_WALL[2]}" "${OPT_TRAIN_WALL[3]}")"
END_TO_END_SPEEDUP="$(ratio_or_null "${BASE_WALL_MEAN}" "${OPT_WALL_MEAN}")"
TRAIN_WALL_SPEEDUP="$(ratio_or_null "${BASE_TRAIN_MEAN}" "${OPT_TRAIN_MEAN}")"
GPU_ACTIVE_SPEEDUP="$(ratio_or_null "${BASE_GPU_MEAN}" "${OPT_GPU_MEAN}")"
CHECKPOINTS_IDENTICAL=true
FIRST_CHECKPOINT="${BASE_CHECKPOINT[1]}"
for hash in "${BASE_CHECKPOINT[1]}" "${BASE_CHECKPOINT[2]}" \
  "${BASE_CHECKPOINT[3]}" "${OPT_CHECKPOINT[1]}" \
  "${OPT_CHECKPOINT[2]}" "${OPT_CHECKPOINT[3]}"; do
  [[ "${hash}" == "${FIRST_CHECKPOINT}" ]] || CHECKPOINTS_IDENTICAL=false
done

mkdir -p "$(dirname "${REPORT}")"
TEMPORARY="$(mktemp "$(dirname "${REPORT}")/.efficiency-comparison.XXXXXX")"
trap 'rm -f -- "${TEMPORARY}"' EXIT
cat >"${TEMPORARY}" <<EOF
{
  "schema": "rlf-physical-efficiency-comparison-v1",
  "hardware_profile": "rtx-pro-6000-96g",
  "ledger_sha256": "${COMMON_LEDGER}",
  "source_manifest_sha256": "${COMMON_SOURCE}",
  "solstice_binary_sha256": "${COMMON_BINARY}",
  "expected_answers_sha256": "${COMMON_EXPECTED}",
  "baseline_policy": {"cuda_local_update_policy": "device", "cuda_cached_cosine_policy": "inline_norms", "cuda_small_cosine_policy": "device", "sparse_router_update_policy": "rebuild", "sparse_rerank_policy": "per_query", "concept_update_policy": "linear", "example_duplicate_policy": "linear", "mode_id_index_policy": "rebuild", "grounding_index_policy": "rebuild", "language_outcome_policy": "linear", "language_dialogue_encoding_policy": "redundant", "instruction_duplicate_policy": "retrieval", "tool_keyword_update_policy": "linear", "preference_duplicate_policy": "linear", "active_learning_duplicate_policy": "linear", "separate_vision_analysis": true},
  "optimized_policy": {"cuda_local_update_policy": "hybrid", "cuda_cached_cosine_policy": "precomputed_norms", "cuda_small_cosine_policy": "hybrid", "sparse_router_update_policy": "incremental", "sparse_rerank_policy": "batched", "concept_update_policy": "indexed", "example_duplicate_policy": "indexed", "mode_id_index_policy": "persistent", "grounding_index_policy": "persistent", "language_outcome_policy": "indexed", "language_dialogue_encoding_policy": "fused", "instruction_duplicate_policy": "indexed", "tool_keyword_update_policy": "indexed", "preference_duplicate_policy": "indexed", "active_learning_duplicate_policy": "indexed", "separate_vision_analysis": false},
  "baseline_end_to_end_seconds": $(json_array_three "${BASE_WALL[1]}" "${BASE_WALL[2]}" "${BASE_WALL[3]}"),
  "optimized_end_to_end_seconds": $(json_array_three "${OPT_WALL[1]}" "${OPT_WALL[2]}" "${OPT_WALL[3]}"),
  "end_to_end_speedup": ${END_TO_END_SPEEDUP},
  "training_wall_speedup": ${TRAIN_WALL_SPEEDUP},
  "training_gpu_active_speedup": ${GPU_ACTIVE_SPEEDUP},
  "baseline_accuracy": $(json_array_three "${BASE_ACCURACY[1]}" "${BASE_ACCURACY[2]}" "${BASE_ACCURACY[3]}"),
  "optimized_accuracy": $(json_array_three "${OPT_ACCURACY[1]}" "${OPT_ACCURACY[2]}" "${OPT_ACCURACY[3]}"),
  "baseline_local_updates": $(json_array_three "${BASE_UPDATES[1]}" "${BASE_UPDATES[2]}" "${BASE_UPDATES[3]}"),
  "optimized_local_updates": $(json_array_three "${OPT_UPDATES[1]}" "${OPT_UPDATES[2]}" "${OPT_UPDATES[3]}"),
  "optimized_avoided_kernel_launches": $(json_array_three "${OPT_AVOIDED[1]}" "${OPT_AVOIDED[2]}" "${OPT_AVOIDED[3]}"),
  "optimized_avoided_pairwise_norm_fma_operations": $(json_array_three "${OPT_NORM_FMAS[1]}" "${OPT_NORM_FMAS[2]}" "${OPT_NORM_FMAS[3]}"),
  "optimized_host_precomputed_norm_fma_operations": $(json_array_three "${OPT_HOST_NORM_FMAS[1]}" "${OPT_HOST_NORM_FMAS[2]}" "${OPT_HOST_NORM_FMAS[3]}"),
  "optimized_host_batch_cosine_calls": $(json_array_three "${OPT_HOST_COSINE[1]}" "${OPT_HOST_COSINE[2]}" "${OPT_HOST_COSINE[3]}"),
  "optimized_avoided_device_cosine_fma_operations": $(json_array_three "${OPT_COSINE_FMAS[1]}" "${OPT_COSINE_FMAS[2]}" "${OPT_COSINE_FMAS[3]}"),
  "baseline_sparse_router_vectors_rebuilt": $(json_array_three "${BASE_ROUTER_REBUILT[1]}" "${BASE_ROUTER_REBUILT[2]}" "${BASE_ROUTER_REBUILT[3]}"),
  "optimized_sparse_router_vectors_rebuilt": $(json_array_three "${OPT_ROUTER_REBUILT[1]}" "${OPT_ROUTER_REBUILT[2]}" "${OPT_ROUTER_REBUILT[3]}"),
  "optimized_sparse_router_vectors_incrementally_updated": $(json_array_three "${OPT_ROUTER_UPDATED[1]}" "${OPT_ROUTER_UPDATED[2]}" "${OPT_ROUTER_UPDATED[3]}"),
  "optimized_sparse_router_vectors_appended": $(json_array_three "${OPT_ROUTER_APPENDED[1]}" "${OPT_ROUTER_APPENDED[2]}" "${OPT_ROUTER_APPENDED[3]}"),
  "optimized_indexed_sparse_rerank_calls": $(json_array_three "${OPT_INDEXED_RERANK[1]}" "${OPT_INDEXED_RERANK[2]}" "${OPT_INDEXED_RERANK[3]}"),
  "optimized_indexed_sparse_rerank_pairs": $(json_array_three "${OPT_INDEXED_PAIRS[1]}" "${OPT_INDEXED_PAIRS[2]}" "${OPT_INDEXED_PAIRS[3]}"),
  "concept_update_lookups": $(json_array_three "${OPT_CONCEPT_LOOKUPS[1]}" "${OPT_CONCEPT_LOOKUPS[2]}" "${OPT_CONCEPT_LOOKUPS[3]}"),
  "baseline_linear_concept_comparisons": $(json_array_three "${BASE_LINEAR_CONCEPT[1]}" "${BASE_LINEAR_CONCEPT[2]}" "${BASE_LINEAR_CONCEPT[3]}"),
  "optimized_indexed_concept_lookups": $(json_array_three "${OPT_INDEXED_CONCEPT[1]}" "${OPT_INDEXED_CONCEPT[2]}" "${OPT_INDEXED_CONCEPT[3]}"),
  "example_duplicate_lookups": $(json_array_three "${OPT_EXAMPLE_LOOKUPS[1]}" "${OPT_EXAMPLE_LOOKUPS[2]}" "${OPT_EXAMPLE_LOOKUPS[3]}"),
  "baseline_linear_example_comparisons": $(json_array_three "${BASE_LINEAR_EXAMPLE[1]}" "${BASE_LINEAR_EXAMPLE[2]}" "${BASE_LINEAR_EXAMPLE[3]}"),
  "optimized_indexed_example_candidates": $(json_array_three "${OPT_INDEXED_EXAMPLE[1]}" "${OPT_INDEXED_EXAMPLE[2]}" "${OPT_INDEXED_EXAMPLE[3]}"),
  "mode_id_lookups": $(json_array_three "${OPT_MODE_LOOKUPS[1]}" "${OPT_MODE_LOOKUPS[2]}" "${OPT_MODE_LOOKUPS[3]}"),
  "baseline_mode_id_index_entries_rebuilt": $(json_array_three "${BASE_MODE_REBUILT[1]}" "${BASE_MODE_REBUILT[2]}" "${BASE_MODE_REBUILT[3]}"),
  "optimized_mode_id_index_entries_rebuilt": $(json_array_three "${OPT_MODE_REBUILT[1]}" "${OPT_MODE_REBUILT[2]}" "${OPT_MODE_REBUILT[3]}"),
  "mode_id_index_incremental_inserts": $(json_array_three "${OPT_MODE_INSERTS[1]}" "${OPT_MODE_INSERTS[2]}" "${OPT_MODE_INSERTS[3]}"),
  "region_mode_id_lookups": $(json_array_three "${OPT_REGION_LOOKUPS[1]}" "${OPT_REGION_LOOKUPS[2]}" "${OPT_REGION_LOOKUPS[3]}"),
  "baseline_linear_region_mode_comparisons": $(json_array_three "${BASE_LINEAR_REGION[1]}" "${BASE_LINEAR_REGION[2]}" "${BASE_LINEAR_REGION[3]}"),
  "optimized_indexed_region_mode_lookups": $(json_array_three "${OPT_INDEXED_REGION[1]}" "${OPT_INDEXED_REGION[2]}" "${OPT_INDEXED_REGION[3]}"),
  "grounding_link_lookups": $(json_array_three "${OPT_GROUND_LOOKUPS[1]}" "${OPT_GROUND_LOOKUPS[2]}" "${OPT_GROUND_LOOKUPS[3]}"),
  "baseline_grounding_lookup_entries_rebuilt": $(json_array_three "${BASE_GROUND_REBUILT[1]}" "${BASE_GROUND_REBUILT[2]}" "${BASE_GROUND_REBUILT[3]}"),
  "optimized_grounding_indexed_candidates_examined": $(json_array_three "${OPT_GROUND_CANDIDATES[1]}" "${OPT_GROUND_CANDIDATES[2]}" "${OPT_GROUND_CANDIDATES[3]}"),
  "optimized_grounding_incremental_posting_inserts": $(json_array_three "${OPT_GROUND_INSERTS[1]}" "${OPT_GROUND_INSERTS[2]}" "${OPT_GROUND_INSERTS[3]}"),
  "baseline_grounding_full_confidence_sweep_entries": $(json_array_three "${BASE_GROUND_SWEEPS[1]}" "${BASE_GROUND_SWEEPS[2]}" "${BASE_GROUND_SWEEPS[3]}"),
  "optimized_grounding_confidence_recomputations": $(json_array_three "${OPT_GROUND_CONFIDENCE[1]}" "${OPT_GROUND_CONFIDENCE[2]}" "${OPT_GROUND_CONFIDENCE[3]}"),
  "baseline_grounding_derived_sort_entries": $(json_array_three "${BASE_GROUND_SORTS[1]}" "${BASE_GROUND_SORTS[2]}" "${BASE_GROUND_SORTS[3]}"),
  "baseline_grounding_mode_query_full_scan_entries": $(json_array_three "${BASE_GROUND_MODE_SCAN[1]}" "${BASE_GROUND_MODE_SCAN[2]}" "${BASE_GROUND_MODE_SCAN[3]}"),
  "optimized_grounding_mode_query_indexed_candidates": $(json_array_three "${OPT_GROUND_MODE_INDEXED[1]}" "${OPT_GROUND_MODE_INDEXED[2]}" "${OPT_GROUND_MODE_INDEXED[3]}"),
  "baseline_grounding_concept_query_full_scan_entries": $(json_array_three "${BASE_GROUND_CONCEPT_SCAN[1]}" "${BASE_GROUND_CONCEPT_SCAN[2]}" "${BASE_GROUND_CONCEPT_SCAN[3]}"),
  "optimized_grounding_concept_query_indexed_candidates": $(json_array_three "${OPT_GROUND_CONCEPT_INDEXED[1]}" "${OPT_GROUND_CONCEPT_INDEXED[2]}" "${OPT_GROUND_CONCEPT_INDEXED[3]}"),
  "language_outcome_update_lookups": $(json_array_three "${OPT_LANGUAGE_LOOKUPS[1]}" "${OPT_LANGUAGE_LOOKUPS[2]}" "${OPT_LANGUAGE_LOOKUPS[3]}"),
  "baseline_linear_outcome_comparisons": $(json_array_three "${BASE_LINEAR_OUTCOMES[1]}" "${BASE_LINEAR_OUTCOMES[2]}" "${BASE_LINEAR_OUTCOMES[3]}"),
  "optimized_linear_outcome_comparisons": $(json_array_three "${OPT_LINEAR_OUTCOMES[1]}" "${OPT_LINEAR_OUTCOMES[2]}" "${OPT_LINEAR_OUTCOMES[3]}"),
  "optimized_indexed_outcome_lookups": $(json_array_three "${OPT_INDEXED_OUTCOMES[1]}" "${OPT_INDEXED_OUTCOMES[2]}" "${OPT_INDEXED_OUTCOMES[3]}"),
  "optimized_outcome_index_builds": $(json_array_three "${OPT_OUTCOME_BUILDS[1]}" "${OPT_OUTCOME_BUILDS[2]}" "${OPT_OUTCOME_BUILDS[3]}"),
  "optimized_outcome_index_entries_built": $(json_array_three "${OPT_OUTCOME_ENTRIES[1]}" "${OPT_OUTCOME_ENTRIES[2]}" "${OPT_OUTCOME_ENTRIES[3]}"),
  "optimized_outcome_index_incremental_inserts": $(json_array_three "${OPT_OUTCOME_INSERTS[1]}" "${OPT_OUTCOME_INSERTS[2]}" "${OPT_OUTCOME_INSERTS[3]}"),
  "preference_duplicate_lookups": $(json_array_three "${OPT_PREFERENCE_LOOKUPS[1]}" "${OPT_PREFERENCE_LOOKUPS[2]}" "${OPT_PREFERENCE_LOOKUPS[3]}"),
  "baseline_linear_preference_comparisons": $(json_array_three "${BASE_LINEAR_PREFERENCES[1]}" "${BASE_LINEAR_PREFERENCES[2]}" "${BASE_LINEAR_PREFERENCES[3]}"),
  "optimized_indexed_preference_candidates": $(json_array_three "${OPT_INDEXED_PREFERENCES[1]}" "${OPT_INDEXED_PREFERENCES[2]}" "${OPT_INDEXED_PREFERENCES[3]}"),
  "optimized_preference_index_rebuilds": $(json_array_three "${OPT_PREFERENCE_REBUILDS[1]}" "${OPT_PREFERENCE_REBUILDS[2]}" "${OPT_PREFERENCE_REBUILDS[3]}"),
  "optimized_preference_index_entries_built": $(json_array_three "${OPT_PREFERENCE_ENTRIES[1]}" "${OPT_PREFERENCE_ENTRIES[2]}" "${OPT_PREFERENCE_ENTRIES[3]}"),
  "optimized_preference_index_incremental_inserts": $(json_array_three "${OPT_PREFERENCE_INSERTS[1]}" "${OPT_PREFERENCE_INSERTS[2]}" "${OPT_PREFERENCE_INSERTS[3]}"),
  "active_learning_duplicate_lookups": $(json_array_three "${OPT_ACTIVE_LOOKUPS[1]}" "${OPT_ACTIVE_LOOKUPS[2]}" "${OPT_ACTIVE_LOOKUPS[3]}"),
  "baseline_linear_active_learning_comparisons": $(json_array_three "${BASE_LINEAR_ACTIVE[1]}" "${BASE_LINEAR_ACTIVE[2]}" "${BASE_LINEAR_ACTIVE[3]}"),
  "optimized_indexed_active_learning_candidates": $(json_array_three "${OPT_INDEXED_ACTIVE[1]}" "${OPT_INDEXED_ACTIVE[2]}" "${OPT_INDEXED_ACTIVE[3]}"),
  "optimized_active_learning_index_rebuilds": $(json_array_three "${OPT_ACTIVE_REBUILDS[1]}" "${OPT_ACTIVE_REBUILDS[2]}" "${OPT_ACTIVE_REBUILDS[3]}"),
  "optimized_active_learning_index_entries_built": $(json_array_three "${OPT_ACTIVE_ENTRIES[1]}" "${OPT_ACTIVE_ENTRIES[2]}" "${OPT_ACTIVE_ENTRIES[3]}"),
  "optimized_active_learning_index_incremental_inserts": $(json_array_three "${OPT_ACTIVE_INSERTS[1]}" "${OPT_ACTIVE_INSERTS[2]}" "${OPT_ACTIVE_INSERTS[3]}"),
  "baseline_checkpoint_sha256": $(json_string_array_three "${BASE_CHECKPOINT[1]}" "${BASE_CHECKPOINT[2]}" "${BASE_CHECKPOINT[3]}"),
  "optimized_checkpoint_sha256": $(json_string_array_three "${OPT_CHECKPOINT[1]}" "${OPT_CHECKPOINT[2]}" "${OPT_CHECKPOINT[3]}"),
  "matched_controlled_quality": ${MATCHED_CONTROLLED_QUALITY},
  "matched_update_counts": ${MATCHED_UPDATE_COUNTS},
  "all_checkpoint_bytes_identical": ${CHECKPOINTS_IDENTICAL},
  "external_quality_evidence_eligible": false,
  "general_100000x_compute_efficiency_proven": false,
  "claim_boundary": "This reducer authenticates same-binary physical campaign artifacts and controlled quality only; external matched quality and general 100,000x efficiency remain separate fail-closed gates."
}
EOF
mv -- "${TEMPORARY}" "${REPORT}"
trap - EXIT
[[ "${MATCHED_CONTROLLED_QUALITY}" == true && "${MATCHED_UPDATE_COUNTS}" == true ]] || exit 3
echo "Created scoped physical comparison: ${REPORT}"
