#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"; trap 'rm -rf -- "${WORK}"' EXIT
for file in source vram; do printf '%s\n' "${file}" >"${WORK}/${file}"; done
SOURCE_SHA="$(sha256sum "${WORK}/source" | awk '{print $1}')"
BINARY_SHA="$(printf 'fixture-binary' | sha256sum | awk '{print $1}')"
cp "${ROOT}/results/codex_campaign/resume_smoke.rlfsp" "${WORK}/checkpoint"
printf 'ledger\n' >"${WORK}/ledger"; LEDGER_SHA="$(sha256sum "${WORK}/ledger" | awk '{print $1}')"
WALL_BINDING_SHA="$(printf '%s\n%s\n%s\n' general-v100-32g "${LEDGER_SHA}" "$(realpath "${WORK}/checkpoint")" | sha256sum | awk '{print $1}')"
printf '{"valid":true,"ledger_sha256":"%s"}\n' "${LEDGER_SHA}" >"${WORK}/audit.json"
printf '%s\n' '{"schema":"rlf-train-data-telemetry-v1","training_performed":true,"backend_operations":{"precomputed_norm_cosine_calls":1,"avoided_pairwise_norm_fma_operations":64,"host_precomputed_norm_fma_operations":16,"precomputed_cached_cosine_norms":true,"host_batch_cosine_calls":2,"avoided_device_cosine_fma_operations":96,"hybrid_small_batch_cosine":true,"indexed_batch_cosine_calls":3,"indexed_cosine_pairs":48},"sparse_router_operations":{"full_rebuilds":1,"vectors_rebuilt":8192,"incremental_updates":2,"vectors_incrementally_updated":2,"vectors_appended":0},"visual_training_operations":{"concept_update_lookups":12,"linear_concept_comparisons":0,"indexed_concept_lookups":12,"concept_index_rebuilds":1,"concept_index_entries_built":0,"example_duplicate_lookups":4,"linear_example_comparisons":0,"indexed_example_candidates":1,"example_index_rebuilds":1,"example_index_entries_built":0,"mode_id_lookups":12,"mode_id_index_full_rebuilds":0,"mode_id_index_entries_rebuilt":0,"mode_id_index_incremental_inserts":4,"region_mode_id_lookups":4,"linear_region_mode_comparisons":0,"indexed_region_mode_lookups":4},"grounding_operations":{"link_lookups":12,"full_lookup_entries_rebuilt":0,"indexed_link_candidates_examined":8,"incremental_posting_inserts":4,"confidence_recomputations":12,"full_confidence_sweep_entries":0,"derived_sort_entries":0,"mode_query_full_scan_entries":0,"mode_query_indexed_candidates":0,"concept_query_full_scan_entries":0,"concept_query_indexed_candidates":0},"language_training_operations":{"dialogue_training_calls":10,"dialogue_tokenizer_encode_calls":30,"redundant_dialogue_encode_calls_avoided":20,"episode_capacity_skips":0,"context_capacity_skips":0,"outcome_update_lookups":100,"linear_outcome_comparisons":20,"indexed_outcome_lookups":80,"outcome_index_builds":1,"outcome_index_entries_built":32,"outcome_index_incremental_inserts":4},"tool_router_training_operations":{"keyword_update_lookups":20,"keyword_capacity_skips":0},"general_training_operations":{"instruction_duplicate_prefilter_lookups":10,"instruction_duplicate_retrievals_avoided":8,"preference_duplicate_lookups":10,"linear_preference_comparisons":0,"indexed_preference_candidates":10,"preference_index_rebuilds":1,"preference_index_entries_built":4,"preference_index_incremental_inserts":2,"active_learning_duplicate_lookups":10,"linear_active_learning_comparisons":0,"indexed_active_learning_candidates":10,"active_learning_index_rebuilds":1,"active_learning_index_entries_built":4,"active_learning_index_incremental_inserts":2}}' >"${WORK}/telemetry.json"
printf 'schema=rlf-general-cuda-training-environment-v1\nprofile=general-v100-32g\npreflight_only=false\ntraining_wall_budget_seconds=900000\nsource_manifest_sha256=%s\nsolstice_binary_sha256=%s\ncuda_local_update_policy=hybrid\ncuda_cached_cosine_policy=precomputed_norms\ncuda_small_cosine_policy=hybrid\nsparse_router_update_policy=incremental\nsparse_rerank_policy=batched\nconcept_update_policy=indexed\nexample_duplicate_policy=indexed\nmode_id_index_policy=persistent\ngrounding_index_policy=persistent\nlanguage_outcome_policy=indexed\nlanguage_dialogue_encoding_policy=fused\ninstruction_duplicate_policy=indexed\ntool_keyword_update_policy=indexed\npreference_duplicate_policy=indexed\nactive_learning_duplicate_policy=indexed\nseparate_vision_analysis=0\ntraining_wall_binding_sha256=%s\ntraining_wall_consumed_seconds=123\n' "${SOURCE_SHA}" "${BINARY_SHA}" "${WALL_BINDING_SHA}" >"${WORK}/environment"
printf 'format_version=6\n' >"${WORK}/inspection"
cat >"${WORK}/resource.json" <<'EOF'
{"schema":"rlf-general-cuda-vram-v1","profile":"general-v100-32g","gpu_uuid":"GPU-TEST-0001","limit_memory_mib":30720,"within_limit":true,"sampler_ok":true,"command_exit_code":0}
EOF
printf '{"schema":"rlf-general-cuda-readiness-v1","ready":true,"test_doubles":false,"require_media_hashes":true,"training_wall_budget_seconds":900000,"profile":"general-v100-32g","gpu_uuid":"GPU-TEST-0001","ledger_sha256":"%s","source_manifest_sha256":"%s","solstice_binary_sha256":"%s"}\n' "${LEDGER_SHA}" "${SOURCE_SHA}" "${BINARY_SHA}" >"${WORK}/readiness.json"
ARGS=(--checkpoint "${WORK}/checkpoint" --ledger "${WORK}/ledger" --source-manifest "${WORK}/source" --data-audit "${WORK}/audit.json" --telemetry "${WORK}/telemetry.json" --resource-summary "${WORK}/resource.json" --vram-trace "${WORK}/vram" --environment "${WORK}/environment" --checkpoint-inspection "${WORK}/inspection" --readiness-report "${WORK}/readiness.json" --output "${WORK}/manifest.tsv")
"${ROOT}/scripts/create_general_cuda_training_artifact_manifest.sh" "${ARGS[@]}"
"${ROOT}/scripts/create_training_artifact_manifest.sh" --verify "${WORK}/manifest.tsv"
sed -i 's/"keyword_capacity_skips":0/"keyword_capacity_skips":1/' "${WORK}/telemetry.json"
set +e
"${ROOT}/scripts/create_general_cuda_training_artifact_manifest.sh" "${ARGS[@]}" >/dev/null 2>&1
STATUS=$?
set -e
[[ "${STATUS}" -ne 0 ]]
sed -i 's/"keyword_capacity_skips":1/"keyword_capacity_skips":0/' "${WORK}/telemetry.json"
BAD_BINARY_SHA="$(printf 'different-binary' | sha256sum | awk '{print $1}')"
sed -i "s/solstice_binary_sha256=${BINARY_SHA}/solstice_binary_sha256=${BAD_BINARY_SHA}/" "${WORK}/environment"
set +e
"${ROOT}/scripts/create_general_cuda_training_artifact_manifest.sh" "${ARGS[@]}" >/dev/null 2>&1
STATUS=$?
set -e
[[ "${STATUS}" -ne 0 ]]
sed -i "s/solstice_binary_sha256=${BAD_BINARY_SHA}/solstice_binary_sha256=${BINARY_SHA}/" "${WORK}/environment"
sed -i 's/"limit_memory_mib":30720/"limit_memory_mib":38912/' "${WORK}/resource.json"
set +e
"${ROOT}/scripts/create_general_cuda_training_artifact_manifest.sh" "${ARGS[@]}" >/dev/null 2>&1
STATUS=$?
set -e
[[ "${STATUS}" -ne 0 ]]

cp "${WORK}/checkpoint" "${WORK}/checkpoint-500m"
CHECKPOINT_500M="$(realpath "${WORK}/checkpoint-500m")"
BINDING_500M="$(printf '%s\n%s\n%s\n' general-v100-32g-500m "${CHECKPOINT_500M}" 5709600 | sha256sum | awk '{print $1}')"
sed -i 's/"training_performed":true/"training_performed":true,"target_training_records":50000000,"cumulative_training_records_after":50000000/' "${WORK}/telemetry.json"
printf 'format_version=6\naudited_training_records=50000000\n' >"${WORK}/inspection"
printf 'schema=rlf-general-cuda-training-environment-v1\nprofile=general-v100-32g-500m\npreflight_only=false\ntraining_wall_budget_seconds=5709600\nsource_manifest_sha256=%s\nsolstice_binary_sha256=%s\ncuda_local_update_policy=hybrid\ncuda_cached_cosine_policy=precomputed_norms\ncuda_small_cosine_policy=hybrid\nsparse_router_update_policy=incremental\nsparse_rerank_policy=batched\nconcept_update_policy=indexed\nexample_duplicate_policy=indexed\nmode_id_index_policy=persistent\ngrounding_index_policy=persistent\nlanguage_outcome_policy=indexed\nlanguage_dialogue_encoding_policy=fused\ninstruction_duplicate_policy=indexed\ntool_keyword_update_policy=indexed\npreference_duplicate_policy=indexed\nactive_learning_duplicate_policy=indexed\nseparate_vision_analysis=0\ntraining_wall_binding_sha256=%s\ntraining_wall_consumed_seconds=456\ncampaign_stage=train-50m\nauthorized_training_records=50000000\ncampaign_binding_sha256=%s\n' "${SOURCE_SHA}" "${BINARY_SHA}" "${BINDING_500M}" "${BINDING_500M}" >"${WORK}/environment-500m"
printf '%s\n' '{"schema":"rlf-general-cuda-vram-v1","profile":"general-v100-32g-500m","gpu_uuid":"GPU-TEST-0001","limit_memory_mib":30720,"within_limit":true,"sampler_ok":true,"command_exit_code":0}' >"${WORK}/resource-500m.json"
printf '{"schema":"rlf-general-cuda-readiness-v1","ready":true,"test_doubles":false,"require_media_hashes":true,"training_wall_budget_seconds":5709600,"training_wall_binding_sha256":"%s","profile":"general-v100-32g-500m","gpu_uuid":"GPU-TEST-0001","ledger_sha256":"%s","source_manifest_sha256":"%s","solstice_binary_sha256":"%s"}\n' "${BINDING_500M}" "${LEDGER_SHA}" "${SOURCE_SHA}" "${BINARY_SHA}" >"${WORK}/readiness-500m.json"
printf 'schema=rlf-wall-budget-v2\nbudget_seconds=5709600\nbinding_sha256=%s\nconsumed_seconds=456\nactive=false\nactive_started_epoch=0\nactive_base_consumed=0\n' "${BINDING_500M}" >"${WORK}/wall-budget-500m.state"
printf 'training_wall_budget_state_sha256=%s\n' "$(sha256sum -- "${WORK}/wall-budget-500m.state" | awk '{print $1}')" >>"${WORK}/environment-500m"
ARGS_500M=(--checkpoint "${CHECKPOINT_500M}" --ledger "${WORK}/ledger" --source-manifest "${WORK}/source" --data-audit "${WORK}/audit.json" --telemetry "${WORK}/telemetry.json" --resource-summary "${WORK}/resource-500m.json" --vram-trace "${WORK}/vram" --environment "${WORK}/environment-500m" --checkpoint-inspection "${WORK}/inspection" --readiness-report "${WORK}/readiness-500m.json" --wall-budget-state "${WORK}/wall-budget-500m.state" --output "${WORK}/manifest-500m.tsv")
"${ROOT}/scripts/create_general_cuda_training_artifact_manifest.sh" "${ARGS_500M[@]}"
"${ROOT}/scripts/create_training_artifact_manifest.sh" --verify "${WORK}/manifest-500m.tsv"
grep -q $'^# wall_budget_state\t' "${WORK}/manifest-500m.tsv"

sed -i 's/"training_wall_budget_seconds":5709600/"training_wall_budget_seconds":5709599/' "${WORK}/readiness-500m.json"
set +e
"${ROOT}/scripts/create_general_cuda_training_artifact_manifest.sh" "${ARGS_500M[@]}" >/dev/null 2>&1
WRONG_BUDGET_STATUS=$?
set -e
[[ "${WRONG_BUDGET_STATUS}" -ne 0 ]]
sed -i 's/"training_wall_budget_seconds":5709599/"training_wall_budget_seconds":5709600/' "${WORK}/readiness-500m.json"

sed -i 's/"profile":"general-v100-32g-500m"/"profile":"general-v100-32g"/' "${WORK}/resource-500m.json"
set +e
"${ROOT}/scripts/create_general_cuda_training_artifact_manifest.sh" "${ARGS_500M[@]}" >/dev/null 2>&1
WRONG_PROFILE_STATUS=$?
set -e
[[ "${WRONG_PROFILE_STATUS}" -ne 0 ]]
sed -i 's/"profile":"general-v100-32g"/"profile":"general-v100-32g-500m"/' "${WORK}/resource-500m.json"

sed -i 's/budget_seconds=5709600/budget_seconds=5709599/' "${WORK}/wall-budget-500m.state"
set +e
"${ROOT}/scripts/create_general_cuda_training_artifact_manifest.sh" "${ARGS_500M[@]}" >/dev/null 2>&1
WRONG_STATE_BUDGET_STATUS=$?
set -e
[[ "${WRONG_STATE_BUDGET_STATUS}" -ne 0 ]]
sed -i 's/budget_seconds=5709599/budget_seconds=5709600/' "${WORK}/wall-budget-500m.state"

cp "${CHECKPOINT_500M}" "${WORK}/wrong-checkpoint-500m"
WRONG_CHECKPOINT_ARGS=("${ARGS_500M[@]}")
WRONG_CHECKPOINT_ARGS[1]="${WORK}/wrong-checkpoint-500m"
set +e
"${ROOT}/scripts/create_general_cuda_training_artifact_manifest.sh" "${WRONG_CHECKPOINT_ARGS[@]}" >/dev/null 2>&1
WRONG_CHECKPOINT_STATUS=$?
set -e
[[ "${WRONG_CHECKPOINT_STATUS}" -ne 0 ]]

printf 'different ledger\n' >"${WORK}/wrong-ledger-500m"
WRONG_LEDGER_ARGS=("${ARGS_500M[@]}")
WRONG_LEDGER_ARGS[3]="${WORK}/wrong-ledger-500m"
set +e
"${ROOT}/scripts/create_general_cuda_training_artifact_manifest.sh" "${WRONG_LEDGER_ARGS[@]}" >/dev/null 2>&1
WRONG_LEDGER_STATUS=$?
set -e
[[ "${WRONG_LEDGER_STATUS}" -ne 0 ]]

printf '%s\n' '{"schema":"rlf-general-cuda-vram-v1","profile":"general-rtx-pro-6000-96g","gpu_uuid":"GPU-TEST-0001","limit_memory_mib":92160,"within_limit":true,"sampler_ok":true,"command_exit_code":0}' >"${WORK}/resource.json"
printf 'schema=rlf-general-cuda-training-environment-v1\nprofile=general-rtx-pro-6000-96g\npreflight_only=false\ntraining_wall_budget_seconds=0\nsource_manifest_sha256=%s\nsolstice_binary_sha256=%s\ncuda_local_update_policy=hybrid\ncuda_cached_cosine_policy=precomputed_norms\ncuda_small_cosine_policy=hybrid\nsparse_router_update_policy=incremental\nsparse_rerank_policy=batched\nconcept_update_policy=indexed\nexample_duplicate_policy=indexed\nmode_id_index_policy=persistent\ngrounding_index_policy=persistent\nlanguage_outcome_policy=indexed\nlanguage_dialogue_encoding_policy=fused\ninstruction_duplicate_policy=indexed\ntool_keyword_update_policy=indexed\npreference_duplicate_policy=indexed\nactive_learning_duplicate_policy=indexed\nseparate_vision_analysis=0\n' "${SOURCE_SHA}" "${BINARY_SHA}" >"${WORK}/environment"
printf '{"schema":"rlf-general-cuda-readiness-v1","ready":true,"test_doubles":false,"require_media_hashes":true,"training_wall_budget_seconds":0,"profile":"general-rtx-pro-6000-96g","gpu_uuid":"GPU-TEST-0001","ledger_sha256":"%s","source_manifest_sha256":"%s","solstice_binary_sha256":"%s"}\n' "${LEDGER_SHA}" "${SOURCE_SHA}" "${BINARY_SHA}" >"${WORK}/readiness.json"
PRO_ARGS=(--checkpoint "${WORK}/checkpoint" --ledger "${WORK}/ledger" --source-manifest "${WORK}/source" --data-audit "${WORK}/audit.json" --telemetry "${WORK}/telemetry.json" --resource-summary "${WORK}/resource.json" --vram-trace "${WORK}/vram" --environment "${WORK}/environment" --checkpoint-inspection "${WORK}/inspection" --readiness-report "${WORK}/readiness.json" --output "${WORK}/pro-manifest.tsv")
"${ROOT}/scripts/create_general_cuda_training_artifact_manifest.sh" "${PRO_ARGS[@]}"
"${ROOT}/scripts/create_training_artifact_manifest.sh" --verify "${WORK}/pro-manifest.tsv"
