#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
  echo "Usage: create_general_cuda_training_artifact_manifest.sh --checkpoint FILE --ledger FILE --source-manifest FILE --data-audit FILE --telemetry FILE --resource-summary FILE --vram-trace FILE --environment FILE --checkpoint-inspection FILE --readiness-report FILE [--wall-budget-state FILE] --output FILE" >&2
}
CHECKPOINT=""; LEDGER=""; SOURCE_MANIFEST=""; DATA_AUDIT=""; TELEMETRY=""; RESOURCE_SUMMARY=""; VRAM_TRACE=""; ENVIRONMENT=""; CHECKPOINT_INSPECTION=""; READINESS_REPORT=""; WALL_BUDGET_STATE=""; OUTPUT=""
while (($# > 0)); do
  case "$1" in
    --checkpoint) CHECKPOINT="${2:?}"; shift 2 ;; --ledger) LEDGER="${2:?}"; shift 2 ;;
    --source-manifest) SOURCE_MANIFEST="${2:?}"; shift 2 ;; --data-audit) DATA_AUDIT="${2:?}"; shift 2 ;;
    --telemetry) TELEMETRY="${2:?}"; shift 2 ;;
    --resource-summary) RESOURCE_SUMMARY="${2:?}"; shift 2 ;; --vram-trace) VRAM_TRACE="${2:?}"; shift 2 ;;
    --environment) ENVIRONMENT="${2:?}"; shift 2 ;; --checkpoint-inspection) CHECKPOINT_INSPECTION="${2:?}"; shift 2 ;;
    --readiness-report) READINESS_REPORT="${2:?}"; shift 2 ;; --output) OUTPUT="${2:?}"; shift 2 ;;
    --wall-budget-state) WALL_BUDGET_STATE="${2:?}"; shift 2 ;;
    --help|-h) usage; exit 0 ;; *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done
for assignment in "checkpoint:${CHECKPOINT}" "ledger:${LEDGER}" "source_manifest:${SOURCE_MANIFEST}" "data_audit:${DATA_AUDIT}" "telemetry:${TELEMETRY}" "resource_summary:${RESOURCE_SUMMARY}" "vram_trace:${VRAM_TRACE}" "environment:${ENVIRONMENT}" "checkpoint_inspection:${CHECKPOINT_INSPECTION}" "readiness_report:${READINESS_REPORT}"; do
  kind="${assignment%%:*}"; path="${assignment#*:}"; [[ -n "${path}" && -f "${path}" ]] || { echo "${kind} file not found: ${path}" >&2; exit 2; }
done
[[ -z "${WALL_BUDGET_STATE}" || -f "${WALL_BUDGET_STATE}" ]] || { echo "wall_budget_state file not found: ${WALL_BUDGET_STATE}" >&2; exit 2; }
[[ -n "${OUTPUT}" ]] || { echo "--output is required" >&2; exit 2; }
grep -Eq '"schema"[[:space:]]*:[[:space:]]*"rlf-general-cuda-vram-v1"' "${RESOURCE_SUMMARY}" || { echo "resource summary has the wrong schema" >&2; exit 1; }
grep -Eq '"within_limit"[[:space:]]*:[[:space:]]*true' "${RESOURCE_SUMMARY}" || { echo "resource limit failed" >&2; exit 1; }
grep -Eq '"sampler_ok"[[:space:]]*:[[:space:]]*true' "${RESOURCE_SUMMARY}" || { echo "resource telemetry failed" >&2; exit 1; }
grep -Eq '"command_exit_code"[[:space:]]*:[[:space:]]*0' "${RESOURCE_SUMMARY}" || { echo "training command failed" >&2; exit 1; }
grep -Eq '"schema"[[:space:]]*:[[:space:]]*"rlf-general-cuda-readiness-v1"' "${READINESS_REPORT}" || { echo "readiness report has the wrong schema" >&2; exit 1; }
grep -Eq '"ready"[[:space:]]*:[[:space:]]*true' "${READINESS_REPORT}" || { echo "readiness did not pass" >&2; exit 1; }
grep -Eq '"test_doubles"[[:space:]]*:[[:space:]]*false' "${READINESS_REPORT}" || { echo "test doubles cannot authorize artifacts" >&2; exit 1; }
grep -Eq '"require_media_hashes"[[:space:]]*:[[:space:]]*true' "${READINESS_REPORT}" || { echo "readiness did not require media hashes" >&2; exit 1; }
grep -Eq '"valid"[[:space:]]*:[[:space:]]*true' "${DATA_AUDIT}" || { echo "data audit did not pass" >&2; exit 1; }
grep -Eq '"schema"[[:space:]]*:[[:space:]]*"rlf-train-data-telemetry-v1"' "${TELEMETRY}" || { echo "training telemetry has the wrong schema" >&2; exit 1; }
grep -Eq '"training_performed"[[:space:]]*:[[:space:]]*true' "${TELEMETRY}" || { echo "training telemetry does not record model updates" >&2; exit 1; }
grep -Eq '"backend_operations"[[:space:]]*:[[:space:]]*\{' "${TELEMETRY}" || { echo "training telemetry lacks backend operation accounting" >&2; exit 1; }
grep -Eq '"precomputed_norm_cosine_calls"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks cached-cosine call accounting" >&2; exit 1; }
grep -Eq '"avoided_pairwise_norm_fma_operations"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks cached-cosine operation accounting" >&2; exit 1; }
grep -Eq '"host_precomputed_norm_fma_operations"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks host norm-precompute accounting" >&2; exit 1; }
grep -Eq '"precomputed_cached_cosine_norms"[[:space:]]*:[[:space:]]*(true|false)' "${TELEMETRY}" || { echo "training telemetry lacks cached-cosine policy accounting" >&2; exit 1; }
grep -Eq '"host_batch_cosine_calls"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks small-cosine dispatch accounting" >&2; exit 1; }
grep -Eq '"avoided_device_cosine_fma_operations"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks shifted cosine accounting" >&2; exit 1; }
grep -Eq '"hybrid_small_batch_cosine"[[:space:]]*:[[:space:]]*(true|false)' "${TELEMETRY}" || { echo "training telemetry lacks small-cosine policy accounting" >&2; exit 1; }
grep -Eq '"indexed_batch_cosine_calls"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks indexed-rerank call accounting" >&2; exit 1; }
grep -Eq '"indexed_cosine_pairs"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks indexed-rerank work accounting" >&2; exit 1; }
grep -Eq '"sparse_router_operations"[[:space:]]*:[[:space:]]*\{' "${TELEMETRY}" || { echo "training telemetry lacks sparse-router accounting" >&2; exit 1; }
grep -Eq '"vectors_rebuilt"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks sparse-router rebuild accounting" >&2; exit 1; }
grep -Eq '"vectors_incrementally_updated"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks sparse-router incremental accounting" >&2; exit 1; }
grep -Eq '"vectors_appended"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks sparse-router growth accounting" >&2; exit 1; }
grep -Eq '"visual_training_operations"[[:space:]]*:[[:space:]]*\{' "${TELEMETRY}" || { echo "training telemetry lacks visual operation accounting" >&2; exit 1; }
grep -Eq '"concept_update_lookups"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks concept-update accounting" >&2; exit 1; }
grep -Eq '"linear_concept_comparisons"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks linear concept accounting" >&2; exit 1; }
grep -Eq '"indexed_concept_lookups"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks indexed concept accounting" >&2; exit 1; }
grep -Eq '"example_duplicate_lookups"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks example-duplicate accounting" >&2; exit 1; }
grep -Eq '"linear_example_comparisons"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks linear example accounting" >&2; exit 1; }
grep -Eq '"indexed_example_candidates"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks indexed example accounting" >&2; exit 1; }
grep -Eq '"mode_id_lookups"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks mode-ID lookup accounting" >&2; exit 1; }
grep -Eq '"mode_id_index_entries_rebuilt"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks mode-ID rebuild accounting" >&2; exit 1; }
grep -Eq '"mode_id_index_incremental_inserts"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks mode-ID incremental accounting" >&2; exit 1; }
grep -Eq '"region_mode_id_lookups"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks region mode-ID lookup accounting" >&2; exit 1; }
grep -Eq '"linear_region_mode_comparisons"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks linear region mode-ID accounting" >&2; exit 1; }
grep -Eq '"indexed_region_mode_lookups"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks indexed region mode-ID accounting" >&2; exit 1; }
grep -Eq '"grounding_operations"[[:space:]]*:[[:space:]]*\{' "${TELEMETRY}" || { echo "training telemetry lacks grounding operation accounting" >&2; exit 1; }
grep -Eq '"full_lookup_entries_rebuilt"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks grounding lookup rebuild accounting" >&2; exit 1; }
grep -Eq '"indexed_link_candidates_examined"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks indexed grounding lookup accounting" >&2; exit 1; }
grep -Eq '"full_confidence_sweep_entries"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks grounding confidence sweep accounting" >&2; exit 1; }
grep -Eq '"derived_sort_entries"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks grounding sort accounting" >&2; exit 1; }
grep -Eq '"language_training_operations"[[:space:]]*:[[:space:]]*\{' "${TELEMETRY}" || { echo "training telemetry lacks language operation accounting" >&2; exit 1; }
grep -Eq '"dialogue_training_calls"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks dialogue accounting" >&2; exit 1; }
grep -Eq '"dialogue_tokenizer_encode_calls"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks dialogue encoding accounting" >&2; exit 1; }
grep -Eq '"redundant_dialogue_encode_calls_avoided"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks dialogue encoding avoidance accounting" >&2; exit 1; }
grep -Eq '"episode_capacity_skips"[[:space:]]*:[[:space:]]*0([},[:space:]]|$)' "${TELEMETRY}" || { echo "episode capacity saturated during training" >&2; exit 1; }
grep -Eq '"context_capacity_skips"[[:space:]]*:[[:space:]]*0([},[:space:]]|$)' "${TELEMETRY}" || { echo "language context capacity saturated during training" >&2; exit 1; }
grep -Eq '"outcome_update_lookups"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks language outcome lookup accounting" >&2; exit 1; }
grep -Eq '"linear_outcome_comparisons"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks linear language outcome accounting" >&2; exit 1; }
grep -Eq '"indexed_outcome_lookups"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks indexed language outcome accounting" >&2; exit 1; }
grep -Eq '"tool_router_training_operations"[[:space:]]*:[[:space:]]*\{' "${TELEMETRY}" || { echo "training telemetry lacks tool-router operation accounting" >&2; exit 1; }
grep -Eq '"keyword_update_lookups"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks tool-keyword lookup accounting" >&2; exit 1; }
grep -Eq '"keyword_capacity_skips"[[:space:]]*:[[:space:]]*0([},[:space:]]|$)' "${TELEMETRY}" || { echo "tool keyword capacity saturated during training" >&2; exit 1; }
grep -Eq '"general_training_operations"[[:space:]]*:[[:space:]]*\{' "${TELEMETRY}" || { echo "training telemetry lacks general operation accounting" >&2; exit 1; }
grep -Eq '"instruction_duplicate_prefilter_lookups"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks instruction duplicate accounting" >&2; exit 1; }
grep -Eq '"instruction_duplicate_retrievals_avoided"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks instruction retrieval-avoidance accounting" >&2; exit 1; }
grep -Eq '"preference_duplicate_lookups"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks preference duplicate lookup accounting" >&2; exit 1; }
grep -Eq '"linear_preference_comparisons"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks linear preference accounting" >&2; exit 1; }
grep -Eq '"indexed_preference_candidates"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks indexed preference accounting" >&2; exit 1; }
grep -Eq '"active_learning_duplicate_lookups"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks active-learning duplicate lookup accounting" >&2; exit 1; }
grep -Eq '"linear_active_learning_comparisons"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks linear active-learning accounting" >&2; exit 1; }
grep -Eq '"indexed_active_learning_candidates"[[:space:]]*:[[:space:]]*[0-9]+' "${TELEMETRY}" || { echo "training telemetry lacks indexed active-learning accounting" >&2; exit 1; }
grep -Eq '^format_version=6$' "${CHECKPOINT_INSPECTION}" || { echo "checkpoint inspection is not format 6" >&2; exit 1; }
grep -Fqx 'schema=rlf-general-cuda-training-environment-v1' "${ENVIRONMENT}" || { echo "training environment has the wrong schema" >&2; exit 1; }
grep -Fqx 'preflight_only=false' "${ENVIRONMENT}" || { echo "preflight-only environment cannot authorize training artifacts" >&2; exit 1; }
grep -Eq '^cuda_local_update_policy=(hybrid|device)$' "${ENVIRONMENT}" || { echo "CUDA local-update policy is missing" >&2; exit 1; }
grep -Eq '^cuda_cached_cosine_policy=(precomputed_norms|inline_norms)$' "${ENVIRONMENT}" || { echo "CUDA cached-cosine policy is missing" >&2; exit 1; }
grep -Eq '^cuda_small_cosine_policy=(hybrid|device)$' "${ENVIRONMENT}" || { echo "CUDA small-cosine policy is missing" >&2; exit 1; }
grep -Eq '^sparse_router_update_policy=(incremental|rebuild)$' "${ENVIRONMENT}" || { echo "sparse-router update policy is missing" >&2; exit 1; }
grep -Eq '^sparse_rerank_policy=(batched|per_query)$' "${ENVIRONMENT}" || { echo "sparse-rerank policy is missing" >&2; exit 1; }
grep -Eq '^concept_update_policy=(indexed|linear)$' "${ENVIRONMENT}" || { echo "concept-update policy is missing" >&2; exit 1; }
grep -Eq '^example_duplicate_policy=(indexed|linear)$' "${ENVIRONMENT}" || { echo "example-duplicate policy is missing" >&2; exit 1; }
grep -Eq '^mode_id_index_policy=(persistent|rebuild)$' "${ENVIRONMENT}" || { echo "mode-ID index policy is missing" >&2; exit 1; }
grep -Eq '^grounding_index_policy=(persistent|rebuild)$' "${ENVIRONMENT}" || { echo "grounding index policy is missing" >&2; exit 1; }
grep -Eq '^language_outcome_policy=(indexed|linear)$' "${ENVIRONMENT}" || { echo "language outcome policy is missing" >&2; exit 1; }
grep -Eq '^language_dialogue_encoding_policy=(fused|redundant)$' "${ENVIRONMENT}" || { echo "language dialogue encoding policy is missing" >&2; exit 1; }
grep -Eq '^instruction_duplicate_policy=(indexed|retrieval)$' "${ENVIRONMENT}" || { echo "instruction duplicate policy is missing" >&2; exit 1; }
grep -Eq '^tool_keyword_update_policy=(indexed|linear)$' "${ENVIRONMENT}" || { echo "tool keyword update policy is missing" >&2; exit 1; }
grep -Eq '^preference_duplicate_policy=(indexed|linear)$' "${ENVIRONMENT}" || { echo "preference duplicate policy is missing" >&2; exit 1; }
grep -Eq '^active_learning_duplicate_policy=(indexed|linear)$' "${ENVIRONMENT}" || { echo "active-learning duplicate policy is missing" >&2; exit 1; }
grep -Eq '^separate_vision_analysis=(0|1)$' "${ENVIRONMENT}" || { echo "visual-analysis policy is missing" >&2; exit 1; }
json_string() { sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"([^\"]*)\".*/\1/p" "$1" | head -n 1; }
json_number() { sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p" "$1" | head -n 1; }
READINESS_PROFILE="$(json_string "${READINESS_REPORT}" profile)"; RESOURCE_PROFILE="$(json_string "${RESOURCE_SUMMARY}" profile)"
READINESS_UUID="$(json_string "${READINESS_REPORT}" gpu_uuid)"; RESOURCE_UUID="$(json_string "${RESOURCE_SUMMARY}" gpu_uuid)"
READINESS_SOURCE_SHA256="$(json_string "${READINESS_REPORT}" source_manifest_sha256)"
READINESS_BINARY_SHA256="$(json_string "${READINESS_REPORT}" solstice_binary_sha256)"
[[ "${READINESS_PROFILE}" == general-40g || "${READINESS_PROFILE}" == general-v100-32g || "${READINESS_PROFILE}" == general-v100-32g-text || "${READINESS_PROFILE}" == general-v100-32g-500m || "${READINESS_PROFILE}" == video-v100-32g || "${READINESS_PROFILE}" == rtx-pro-6000-96g || "${READINESS_PROFILE}" == general-rtx-pro-6000-96g || "${READINESS_PROFILE}" == general-rtx-pro-6000-96g-text || "${READINESS_PROFILE}" == video-rtx-pro-6000-96g ]] || { echo "unapproved profile" >&2; exit 1; }
[[ "${RESOURCE_PROFILE}" == "${READINESS_PROFILE}" ]] || { echo "profile identity mismatch" >&2; exit 1; }
[[ -n "${READINESS_UUID}" && "${RESOURCE_UUID}" == "${READINESS_UUID}" ]] || { echo "GPU UUID identity mismatch" >&2; exit 1; }
case "${READINESS_PROFILE}" in
  general-40g) EXPECTED_LIMIT=38912 ;;
  general-v100-32g|general-v100-32g-text|general-v100-32g-500m|video-v100-32g) EXPECTED_LIMIT=30720 ;;
  rtx-pro-6000-96g) EXPECTED_LIMIT=90112 ;;
  general-rtx-pro-6000-96g|general-rtx-pro-6000-96g-text|video-rtx-pro-6000-96g) EXPECTED_LIMIT=92160 ;;
esac
[[ "$(json_number "${RESOURCE_SUMMARY}" limit_memory_mib)" == "${EXPECTED_LIMIT}" ]] || { echo "resource limit does not match profile" >&2; exit 1; }
grep -Fqx "profile=${READINESS_PROFILE}" "${ENVIRONMENT}" || { echo "training environment profile mismatch" >&2; exit 1; }
[[ "${READINESS_SOURCE_SHA256}" =~ ^[0-9a-f]{64}$ && "${READINESS_BINARY_SHA256}" =~ ^[0-9a-f]{64}$ ]] || { echo "readiness source/binary identity missing" >&2; exit 1; }
[[ "$(sha256sum -- "${SOURCE_MANIFEST}" | awk '{print $1}')" == "${READINESS_SOURCE_SHA256}" ]] || { echo "source manifest changed after readiness" >&2; exit 1; }
grep -Fqx "source_manifest_sha256=${READINESS_SOURCE_SHA256}" "${ENVIRONMENT}" || { echo "training source identity mismatch" >&2; exit 1; }
grep -Fqx "solstice_binary_sha256=${READINESS_BINARY_SHA256}" "${ENVIRONMENT}" || { echo "training binary identity mismatch" >&2; exit 1; }
if [[ "${READINESS_PROFILE}" == general-v100-32g || "${READINESS_PROFILE}" == general-v100-32g-text || "${READINESS_PROFILE}" == video-v100-32g ]]; then
  [[ "$(json_number "${READINESS_REPORT}" training_wall_budget_seconds)" == 900000 ]] || { echo "V100 readiness budget mismatch" >&2; exit 1; }
  grep -Fqx 'training_wall_budget_seconds=900000' "${ENVIRONMENT}" || { echo "V100 training budget missing" >&2; exit 1; }
  grep -Eq '^training_wall_consumed_seconds=[0-9]+$' "${ENVIRONMENT}" || { echo "V100 consumed-time evidence missing" >&2; exit 1; }
  CONSUMED_SECONDS="$(sed -nE 's/^training_wall_consumed_seconds=([0-9]+)$/\1/p' "${ENVIRONMENT}" | head -n 1)"
  ((CONSUMED_SECONDS <= 900000)) || { echo "V100 consumed time exceeds budget" >&2; exit 1; }
  EXPECTED_WALL_BINDING="$(printf '%s\n%s\n%s\n' "${READINESS_PROFILE}" "$(sha256sum -- "${LEDGER}" | awk '{print $1}')" "$(realpath "${CHECKPOINT}")" | sha256sum | awk '{print $1}')"
  grep -Fqx "training_wall_binding_sha256=${EXPECTED_WALL_BINDING}" "${ENVIRONMENT}" || { echo "V100 campaign binding mismatch" >&2; exit 1; }
elif [[ "${READINESS_PROFILE}" == general-v100-32g-500m ]]; then
  EXPECTED_BUDGET=5709600
  [[ "$(json_number "${READINESS_REPORT}" training_wall_budget_seconds)" == "${EXPECTED_BUDGET}" ]] || { echo "500M V100 readiness budget mismatch" >&2; exit 1; }
  grep -Fqx "training_wall_budget_seconds=${EXPECTED_BUDGET}" "${ENVIRONMENT}" || { echo "500M V100 training budget missing" >&2; exit 1; }
  grep -Eq '^training_wall_consumed_seconds=[0-9]+$' "${ENVIRONMENT}" || { echo "500M V100 consumed-time evidence missing" >&2; exit 1; }
  CONSUMED_SECONDS="$(sed -nE 's/^training_wall_consumed_seconds=([0-9]+)$/\1/p' "${ENVIRONMENT}" | head -n 1)"
  ((CONSUMED_SECONDS <= EXPECTED_BUDGET)) || { echo "500M V100 consumed time exceeds budget" >&2; exit 1; }
  EXPECTED_WALL_BINDING="$(printf '%s\n%s\n%s\n' "${READINESS_PROFILE}" "$(realpath "${CHECKPOINT}")" "${EXPECTED_BUDGET}" | sha256sum | awk '{print $1}')"
  [[ "$(json_string "${READINESS_REPORT}" training_wall_binding_sha256)" == "${EXPECTED_WALL_BINDING}" ]] || { echo "500M V100 readiness campaign binding mismatch" >&2; exit 1; }
  grep -Fqx "training_wall_binding_sha256=${EXPECTED_WALL_BINDING}" "${ENVIRONMENT}" || { echo "500M V100 training campaign binding mismatch" >&2; exit 1; }
  [[ -n "${WALL_BUDGET_STATE}" && -f "${WALL_BUDGET_STATE}" ]] || { echo "500M V100 wall-budget state is required" >&2; exit 1; }
  grep -Fqx 'schema=rlf-wall-budget-v2' "${WALL_BUDGET_STATE}" || { echo "500M V100 wall-budget state schema mismatch" >&2; exit 1; }
  grep -Fqx "budget_seconds=${EXPECTED_BUDGET}" "${WALL_BUDGET_STATE}" || { echo "500M V100 wall-budget state total mismatch" >&2; exit 1; }
  grep -Fqx "binding_sha256=${EXPECTED_WALL_BINDING}" "${WALL_BUDGET_STATE}" || { echo "500M V100 wall-budget state binding mismatch" >&2; exit 1; }
  grep -Fqx "consumed_seconds=${CONSUMED_SECONDS}" "${WALL_BUDGET_STATE}" || { echo "500M V100 wall-budget consumed-time mismatch" >&2; exit 1; }
  grep -Fqx 'active=false' "${WALL_BUDGET_STATE}" || { echo "500M V100 wall-budget state is active" >&2; exit 1; }
  WALL_BUDGET_STATE_SHA256="$(sha256sum -- "${WALL_BUDGET_STATE}" | awk '{print $1}')"
  grep -Fqx "training_wall_budget_state_sha256=${WALL_BUDGET_STATE_SHA256}" "${ENVIRONMENT}" || { echo "500M V100 wall-budget state hash mismatch" >&2; exit 1; }
  AUTHORIZED_RECORDS="$(sed -nE 's/^authorized_training_records=([0-9]+)$/\1/p' "${ENVIRONMENT}" | head -n 1)"
  case "${AUTHORIZED_RECORDS}" in 50000000|200000000|500000000) ;; *) echo "staged V100 authorized record target is missing or invalid" >&2; exit 1 ;; esac
  grep -Fqx "campaign_binding_sha256=${EXPECTED_WALL_BINDING}" "${ENVIRONMENT}" || { echo "staged V100 controller binding mismatch" >&2; exit 1; }
  case "$(sed -nE 's/^campaign_stage=(.*)$/\1/p' "${ENVIRONMENT}" | head -n 1):${AUTHORIZED_RECORDS}" in
    train-50m:50000000|train-200m-primary:200000000|train-500m-promoted:500000000) ;;
    *) echo "staged V100 stage/record target mismatch" >&2; exit 1 ;;
  esac
  grep -Eq "\"target_training_records\"[[:space:]]*:[[:space:]]*${AUTHORIZED_RECORDS}([,[:space:]]|$)" "${TELEMETRY}" || { echo "training telemetry target mismatch" >&2; exit 1; }
  grep -Eq "\"cumulative_training_records_after\"[[:space:]]*:[[:space:]]*${AUTHORIZED_RECORDS}([,[:space:]]|$)" "${TELEMETRY}" || { echo "training telemetry cumulative count mismatch" >&2; exit 1; }
  grep -Fqx "audited_training_records=${AUTHORIZED_RECORDS}" "${CHECKPOINT_INSPECTION}" || { echo "checkpoint exact training count mismatch" >&2; exit 1; }
else
  [[ "$(json_number "${READINESS_REPORT}" training_wall_budget_seconds)" == 0 ]] || { echo "unexpected readiness training budget" >&2; exit 1; }
  grep -Fqx 'training_wall_budget_seconds=0' "${ENVIRONMENT}" || { echo "training environment budget mismatch" >&2; exit 1; }
fi
LEDGER_SHA="$(sha256sum -- "${LEDGER}" | awk '{print $1}')"
[[ "$(json_string "${READINESS_REPORT}" ledger_sha256)" == "${LEDGER_SHA}" && "$(json_string "${DATA_AUDIT}" ledger_sha256)" == "${LEDGER_SHA}" ]] || { echo "ledger identity mismatch" >&2; exit 1; }

OUTPUT="$(realpath -m "${OUTPUT}")"; mkdir -p "$(dirname "${OUTPUT}")"
WORK="$(mktemp "$(dirname "${OUTPUT}")/.cuda-training-manifest.XXXXXX")"; trap 'rm -f -- "${WORK}"' EXIT
printf '# rlf-training-artifact-manifest-v2\n# kind\tsha256\tbytes\tabsolute_path\n' >"${WORK}"
if [[ -n "${WALL_BUDGET_STATE}" ]]; then
  WALL_BUDGET_STATE_PATH="$(realpath "${WALL_BUDGET_STATE}")"
  printf '# wall_budget_state\t%s\t%s\t%s\n' "$(sha256sum -- "${WALL_BUDGET_STATE_PATH}" | awk '{print $1}')" "$(stat -c '%s' -- "${WALL_BUDGET_STATE_PATH}")" "${WALL_BUDGET_STATE_PATH}" >>"${WORK}"
fi
ASSIGNMENTS=("checkpoint:${CHECKPOINT}" "ledger:${LEDGER}" "source_manifest:${SOURCE_MANIFEST}" "data_audit:${DATA_AUDIT}" "telemetry:${TELEMETRY}" "resource_summary:${RESOURCE_SUMMARY}" "vram_trace:${VRAM_TRACE}" "environment:${ENVIRONMENT}" "checkpoint_inspection:${CHECKPOINT_INSPECTION}" "readiness_report:${READINESS_REPORT}")
for assignment in "${ASSIGNMENTS[@]}"; do
  kind="${assignment%%:*}"; path="$(realpath "${assignment#*:}")"
  printf '%s\t%s\t%s\t%s\n' "${kind}" "$(sha256sum -- "${path}" | awk '{print $1}')" "$(stat -c '%s' -- "${path}")" "${path}" >>"${WORK}"
done
mv -f -- "${WORK}" "${OUTPUT}"
(cd "$(dirname "${OUTPUT}")" && sha256sum -- "$(basename "${OUTPUT}")" >"$(basename "${OUTPUT}").sha256")
"${ROOT}/scripts/create_training_artifact_manifest.sh" --verify "${OUTPUT}"
