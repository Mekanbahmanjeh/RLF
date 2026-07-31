#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFLIGHT_ONLY=false
if [[ "${1:-}" == --preflight-only ]]; then PREFLIGHT_ONLY=true; shift; fi
LEDGER="${1:?Usage: train_general_cuda_audited.sh [--preflight-only] LEDGER [CHECKPOINT] [RESULT_DIR] [READINESS_REPORT]}"
CHECKPOINT="${2:-}"
RESULT_DIR="${3:-}"
READINESS_REPORT="${4:-${ROOT}/results/codex_campaign/general_cuda_readiness/readiness.json}"
[[ -f "${LEDGER}" && -f "${READINESS_REPORT}" ]] || { echo "ledger/readiness report missing" >&2; exit 1; }
grep -Eq '"schema"[[:space:]]*:[[:space:]]*"rlf-general-cuda-readiness-v1"' "${READINESS_REPORT}" || { echo "wrong readiness schema" >&2; exit 1; }
grep -Eq '"ready"[[:space:]]*:[[:space:]]*true' "${READINESS_REPORT}" || { echo "real ready=true report required" >&2; exit 1; }
grep -Eq '"test_doubles"[[:space:]]*:[[:space:]]*false' "${READINESS_REPORT}" || { echo "test-double readiness cannot train" >&2; exit 1; }
json_string() { sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"([^\"]*)\".*/\1/p" "$1" | head -n 1; }
json_number() { sed -nE "s/.*\"$2\"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p" "$1" | head -n 1; }
PROFILE="$(json_string "${READINESS_REPORT}" profile)"; GPU_UUID="$(json_string "${READINESS_REPORT}" gpu_uuid)"; GPU_INDEX="$(json_number "${READINESS_REPORT}" gpu_index)"
READINESS_SOURCE_SHA256="$(json_string "${READINESS_REPORT}" source_manifest_sha256)"
READINESS_BINARY_SHA256="$(json_string "${READINESS_REPORT}" solstice_binary_sha256)"
[[ "${PROFILE}" == general-40g || "${PROFILE}" == general-v100-32g || "${PROFILE}" == general-v100-32g-text || "${PROFILE}" == general-v100-32g-500m || "${PROFILE}" == video-v100-32g || "${PROFILE}" == rtx-pro-6000-96g || "${PROFILE}" == general-rtx-pro-6000-96g || "${PROFILE}" == general-rtx-pro-6000-96g-text || "${PROFILE}" == video-rtx-pro-6000-96g ]] || { echo "unsupported readiness profile" >&2; exit 1; }
[[ -n "${GPU_UUID}" && -n "${GPU_INDEX}" ]] || { echo "readiness device identity missing" >&2; exit 1; }
if [[ "${PROFILE}" == *rtx-pro-6000-96g* ]]; then BIN="${ROOT}/build/ubuntu-rtx-pro-6000-cuda/solstice"; else BIN="${ROOT}/build/ubuntu-general-cuda-compat/solstice"; fi
if [[ -n "${RLF_GENERAL_CUDA_TEST_BIN:-}" ]]; then
  if [[ "${PREFLIGHT_ONLY}" == true && "${RLF_ALLOW_TEST_DOUBLES:-0}" == 1 ]]; then
    BIN="${RLF_GENERAL_CUDA_TEST_BIN}"
  else
    echo "RLF_GENERAL_CUDA_TEST_BIN is allowed only for test-double preflight" >&2
    exit 1
  fi
fi
[[ -x "${BIN}" ]] || { echo "Build the isolated ${PROFILE} CUDA target first." >&2; exit 1; }
[[ "${READINESS_SOURCE_SHA256}" =~ ^[0-9a-f]{64}$ && "${READINESS_BINARY_SHA256}" =~ ^[0-9a-f]{64}$ ]] || { echo "readiness source/binary identity is missing" >&2; exit 1; }
ACTUAL_BINARY_SHA256="$(sha256sum -- "${BIN}" | awk '{print $1}')"
[[ "${ACTUAL_BINARY_SHA256}" == "${READINESS_BINARY_SHA256}" ]] || { echo "Solstice binary changed after readiness" >&2; exit 1; }
if [[ -z "${CHECKPOINT}" ]]; then CHECKPOINT="${ROOT}/models/solstice_${PROFILE}.rlfsp"; fi
if [[ -z "${RESULT_DIR}" ]]; then RESULT_DIR="${ROOT}/results/codex_campaign/${PROFILE}_training"; fi
CHECKPOINT="$(realpath -m "${CHECKPOINT}")"
RESULT_DIR="$(realpath -m "${RESULT_DIR}")"
TRAINING_WALL_BUDGET_SECONDS="$(json_number "${READINESS_REPORT}" training_wall_budget_seconds)"
[[ -n "${TRAINING_WALL_BUDGET_SECONDS}" ]] || { echo "readiness training budget is missing" >&2; exit 1; }
case "${PROFILE}" in
  general-v100-32g|general-v100-32g-text|video-v100-32g)
    [[ "${TRAINING_WALL_BUDGET_SECONDS}" == 900000 ]] || { echo "V100 profiles require a 250-hour budget" >&2; exit 1; } ;;
  general-v100-32g-500m)
    [[ "${TRAINING_WALL_BUDGET_SECONDS}" == 5709600 ]] || { echo "general-v100-32g-500m requires the exact 1,586-hour budget" >&2; exit 1; } ;;
  *)
    [[ "${TRAINING_WALL_BUDGET_SECONDS}" == 0 ]] || { echo "unexpected training budget for ${PROFILE}" >&2; exit 1; } ;;
esac
MAX_AUDIT_RECORDS="$(json_number "${READINESS_REPORT}" maximum_audit_records)"
MAX_TEXT_SHARD_BYTES="$(json_number "${READINESS_REPORT}" maximum_text_shard_bytes)"
MAX_TRAIN_SHARD_BYTES="$(json_number "${READINESS_REPORT}" maximum_train_shard_bytes)"
[[ "${MAX_AUDIT_RECORDS}" =~ ^[1-9][0-9]*$ && \
   "${MAX_TEXT_SHARD_BYTES}" =~ ^[1-9][0-9]*$ && \
   "${MAX_TRAIN_SHARD_BYTES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "readiness audit limits are missing or invalid" >&2; exit 1;
}
for override in RLF_MAX_AUDIT_RECORDS RLF_MAX_TEXT_SHARD_BYTES RLF_MAX_TRAIN_SHARD_BYTES; do
  if [[ -n "${!override:-}" ]]; then
    case "${override}" in
      RLF_MAX_AUDIT_RECORDS) expected="${MAX_AUDIT_RECORDS}" ;;
      RLF_MAX_TEXT_SHARD_BYTES) expected="${MAX_TEXT_SHARD_BYTES}" ;;
      RLF_MAX_TRAIN_SHARD_BYTES) expected="${MAX_TRAIN_SHARD_BYTES}" ;;
    esac
    [[ "${!override}" == "${expected}" ]] || {
      echo "${override} conflicts with the immutable readiness report" >&2; exit 1;
    }
  fi
done
CUDA_LOCAL_UPDATE_POLICY="${RLF_CUDA_LOCAL_UPDATE_POLICY:-hybrid}"
CUDA_CACHED_COSINE_POLICY="${RLF_CUDA_CACHED_COSINE_POLICY:-precomputed_norms}"
CUDA_SMALL_COSINE_POLICY="${RLF_CUDA_SMALL_COSINE_POLICY:-hybrid}"
SPARSE_ROUTER_UPDATE_POLICY="${RLF_SPARSE_ROUTER_UPDATE_POLICY:-incremental}"
SPARSE_RERANK_POLICY="${RLF_SPARSE_RERANK_POLICY:-batched}"
CONCEPT_UPDATE_POLICY="${RLF_CONCEPT_UPDATE_POLICY:-indexed}"
EXAMPLE_DUPLICATE_POLICY="${RLF_EXAMPLE_DUPLICATE_POLICY:-indexed}"
MODE_ID_INDEX_POLICY="${RLF_MODE_ID_INDEX_POLICY:-persistent}"
GROUNDING_INDEX_POLICY="${RLF_GROUNDING_INDEX_POLICY:-persistent}"
LANGUAGE_OUTCOME_POLICY="${RLF_LANGUAGE_OUTCOME_POLICY:-indexed}"
LANGUAGE_DIALOGUE_ENCODING_POLICY="${RLF_LANGUAGE_DIALOGUE_ENCODING_POLICY:-fused}"
INSTRUCTION_DUPLICATE_POLICY="${RLF_INSTRUCTION_DUPLICATE_POLICY:-indexed}"
TOOL_KEYWORD_UPDATE_POLICY="${RLF_TOOL_KEYWORD_UPDATE_POLICY:-indexed}"
PREFERENCE_DUPLICATE_POLICY="${RLF_PREFERENCE_DUPLICATE_POLICY:-indexed}"
ACTIVE_LEARNING_DUPLICATE_POLICY="${RLF_ACTIVE_LEARNING_DUPLICATE_POLICY:-indexed}"
SEPARATE_VISION_ANALYSIS="${RLF_SEPARATE_VISION_ANALYSIS:-0}"
[[ "${CUDA_LOCAL_UPDATE_POLICY}" == hybrid || "${CUDA_LOCAL_UPDATE_POLICY}" == device ]] || { echo "RLF_CUDA_LOCAL_UPDATE_POLICY must be hybrid or device" >&2; exit 1; }
[[ "${CUDA_CACHED_COSINE_POLICY}" == precomputed_norms || "${CUDA_CACHED_COSINE_POLICY}" == inline_norms ]] || { echo "RLF_CUDA_CACHED_COSINE_POLICY must be precomputed_norms or inline_norms" >&2; exit 1; }
[[ "${CUDA_SMALL_COSINE_POLICY}" == hybrid || "${CUDA_SMALL_COSINE_POLICY}" == device ]] || { echo "RLF_CUDA_SMALL_COSINE_POLICY must be hybrid or device" >&2; exit 1; }
[[ "${SPARSE_ROUTER_UPDATE_POLICY}" == incremental || "${SPARSE_ROUTER_UPDATE_POLICY}" == rebuild ]] || { echo "RLF_SPARSE_ROUTER_UPDATE_POLICY must be incremental or rebuild" >&2; exit 1; }
[[ "${SPARSE_RERANK_POLICY}" == batched || "${SPARSE_RERANK_POLICY}" == per_query ]] || { echo "RLF_SPARSE_RERANK_POLICY must be batched or per_query" >&2; exit 1; }
[[ "${CONCEPT_UPDATE_POLICY}" == indexed || "${CONCEPT_UPDATE_POLICY}" == linear ]] || { echo "RLF_CONCEPT_UPDATE_POLICY must be indexed or linear" >&2; exit 1; }
[[ "${EXAMPLE_DUPLICATE_POLICY}" == indexed || "${EXAMPLE_DUPLICATE_POLICY}" == linear ]] || { echo "RLF_EXAMPLE_DUPLICATE_POLICY must be indexed or linear" >&2; exit 1; }
[[ "${MODE_ID_INDEX_POLICY}" == persistent || "${MODE_ID_INDEX_POLICY}" == rebuild ]] || { echo "RLF_MODE_ID_INDEX_POLICY must be persistent or rebuild" >&2; exit 1; }
[[ "${GROUNDING_INDEX_POLICY}" == persistent || "${GROUNDING_INDEX_POLICY}" == rebuild ]] || { echo "RLF_GROUNDING_INDEX_POLICY must be persistent or rebuild" >&2; exit 1; }
[[ "${LANGUAGE_OUTCOME_POLICY}" == indexed || "${LANGUAGE_OUTCOME_POLICY}" == linear ]] || { echo "RLF_LANGUAGE_OUTCOME_POLICY must be indexed or linear" >&2; exit 1; }
[[ "${LANGUAGE_DIALOGUE_ENCODING_POLICY}" == fused || "${LANGUAGE_DIALOGUE_ENCODING_POLICY}" == redundant ]] || { echo "RLF_LANGUAGE_DIALOGUE_ENCODING_POLICY must be fused or redundant" >&2; exit 1; }
[[ "${INSTRUCTION_DUPLICATE_POLICY}" == indexed || "${INSTRUCTION_DUPLICATE_POLICY}" == retrieval ]] || { echo "RLF_INSTRUCTION_DUPLICATE_POLICY must be indexed or retrieval" >&2; exit 1; }
[[ "${TOOL_KEYWORD_UPDATE_POLICY}" == indexed || "${TOOL_KEYWORD_UPDATE_POLICY}" == linear ]] || { echo "RLF_TOOL_KEYWORD_UPDATE_POLICY must be indexed or linear" >&2; exit 1; }
[[ "${PREFERENCE_DUPLICATE_POLICY}" == indexed || "${PREFERENCE_DUPLICATE_POLICY}" == linear ]] || { echo "RLF_PREFERENCE_DUPLICATE_POLICY must be indexed or linear" >&2; exit 1; }
[[ "${ACTIVE_LEARNING_DUPLICATE_POLICY}" == indexed || "${ACTIVE_LEARNING_DUPLICATE_POLICY}" == linear ]] || { echo "RLF_ACTIVE_LEARNING_DUPLICATE_POLICY must be indexed or linear" >&2; exit 1; }
[[ "${SEPARATE_VISION_ANALYSIS}" == 0 || "${SEPARATE_VISION_ANALYSIS}" == 1 ]] || { echo "RLF_SEPARATE_VISION_ANALYSIS must be 0 or 1" >&2; exit 1; }
export RLF_CUDA_LOCAL_UPDATE_POLICY="${CUDA_LOCAL_UPDATE_POLICY}"
export RLF_CUDA_CACHED_COSINE_POLICY="${CUDA_CACHED_COSINE_POLICY}"
export RLF_CUDA_SMALL_COSINE_POLICY="${CUDA_SMALL_COSINE_POLICY}"
export RLF_SPARSE_ROUTER_UPDATE_POLICY="${SPARSE_ROUTER_UPDATE_POLICY}"
export RLF_SPARSE_RERANK_POLICY="${SPARSE_RERANK_POLICY}"
export RLF_CONCEPT_UPDATE_POLICY="${CONCEPT_UPDATE_POLICY}"
export RLF_EXAMPLE_DUPLICATE_POLICY="${EXAMPLE_DUPLICATE_POLICY}"
export RLF_MODE_ID_INDEX_POLICY="${MODE_ID_INDEX_POLICY}"
export RLF_GROUNDING_INDEX_POLICY="${GROUNDING_INDEX_POLICY}"
export RLF_LANGUAGE_OUTCOME_POLICY="${LANGUAGE_OUTCOME_POLICY}"
export RLF_LANGUAGE_DIALOGUE_ENCODING_POLICY="${LANGUAGE_DIALOGUE_ENCODING_POLICY}"
export RLF_INSTRUCTION_DUPLICATE_POLICY="${INSTRUCTION_DUPLICATE_POLICY}"
export RLF_TOOL_KEYWORD_UPDATE_POLICY="${TOOL_KEYWORD_UPDATE_POLICY}"
export RLF_PREFERENCE_DUPLICATE_POLICY="${PREFERENCE_DUPLICATE_POLICY}"
export RLF_ACTIVE_LEARNING_DUPLICATE_POLICY="${ACTIVE_LEARNING_DUPLICATE_POLICY}"
LEDGER_SHA256="$(sha256sum -- "${LEDGER}" | awk '{print $1}')"
for expected in "ledger_sha256:\"${LEDGER_SHA256}\"" "maximum_audit_records:${MAX_AUDIT_RECORDS}" "maximum_text_shard_bytes:${MAX_TEXT_SHARD_BYTES}" "maximum_train_shard_bytes:${MAX_TRAIN_SHARD_BYTES}"; do
  key="${expected%%:*}"; value="${expected#*:}"; grep -Eq "\"${key}\"[[:space:]]*:[[:space:]]*${value}([,[:space:]]|$)" "${READINESS_REPORT}" || { echo "readiness does not authorize ${key}=${value}" >&2; exit 1; }
done
mkdir -p "${RESULT_DIR}" "$(dirname "${CHECKPOINT}")"; export CUDA_MODULE_LOADING=LAZY; export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
"${ROOT}/scripts/create_source_manifest.sh" "${RESULT_DIR}/source_manifest.tsv"
ACTUAL_SOURCE_SHA256="$(sha256sum -- "${RESULT_DIR}/source_manifest.tsv" | awk '{print $1}')"
[[ "${ACTUAL_SOURCE_SHA256}" == "${READINESS_SOURCE_SHA256}" ]] || { echo "source tree changed after readiness" >&2; exit 1; }
WALL_BINDING_SHA256=""
WALL_BUDGET_STATE="${CHECKPOINT}.wall-budget.state"
CAMPAIGN_TARGET_RECORDS=0
if ((TRAINING_WALL_BUDGET_SECONDS > 0)); then
  if [[ "${PROFILE}" == general-v100-32g-500m ]]; then
    WALL_BINDING_SHA256="$(printf '%s\n%s\n%s\n' "${PROFILE}" "${CHECKPOINT}" "${TRAINING_WALL_BUDGET_SECONDS}" | sha256sum | awk '{print $1}')"
    [[ "$(json_string "${READINESS_REPORT}" training_wall_binding_sha256)" == "${WALL_BINDING_SHA256}" ]] || {
      echo "readiness does not authorize this staged profile/checkpoint/budget binding" >&2
      exit 1
    }
    if [[ "${PREFLIGHT_ONLY}" == false ]]; then
      [[ "${RLF_TRAINING_AUTHORIZED:-0}" == 1 ]] || {
        echo "staged V100 training lacks controller authorization" >&2; exit 1;
      }
      [[ "${RLF_CAMPAIGN_PROFILE:-}" == "${PROFILE}" ]] || {
        echo "staged V100 training requires the controller profile binding" >&2; exit 1;
      }
      [[ "${RLF_CAMPAIGN_BINDING_SHA256:-}" == "${WALL_BINDING_SHA256}" ]] || {
        echo "staged V100 training requires the controller campaign binding" >&2; exit 1;
      }
      [[ "${RLF_CAMPAIGN_STATE_DIR:-}" == /* && -f "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" && ! -L "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" ]] || {
        echo "staged V100 training requires a regular controller state" >&2; exit 1;
      }
      grep -Fqx 'schema=rlf-v100-1586h-campaign-v1' "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" || { echo "wrong V100 campaign state schema" >&2; exit 1; }
      grep -Fqx "profile=${PROFILE}" "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" || { echo "V100 campaign state profile mismatch" >&2; exit 1; }
      grep -Fqx "binding_sha256=${WALL_BINDING_SHA256}" "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" || { echo "V100 campaign state binding mismatch" >&2; exit 1; }
      grep -Fqx "total_budget_seconds=${TRAINING_WALL_BUDGET_SECONDS}" "${RLF_CAMPAIGN_STATE_DIR}/campaign.meta" || { echo "V100 campaign state budget mismatch" >&2; exit 1; }
      case "${RLF_CAMPAIGN_STAGE:-}:${RLF_AUTHORIZED_RECORDS:-}" in
        train-50m:50000000|train-200m-primary:200000000|train-500m-promoted:500000000)
          CAMPAIGN_TARGET_RECORDS="${RLF_AUTHORIZED_RECORDS}" ;;
        *) echo "controller stage and exact authorized record target do not match" >&2; exit 1 ;;
      esac
      ((MAX_AUDIT_RECORDS >= CAMPAIGN_TARGET_RECORDS)) || {
        echo "readiness audit ceiling is below the authorized training target" >&2; exit 1;
      }
    fi
  else
    WALL_BINDING_SHA256="$(printf '%s\n%s\n%s\n' "${PROFILE}" "${LEDGER_SHA256}" "${CHECKPOINT}" | sha256sum | awk '{print $1}')"
  fi
  if [[ -f "${CHECKPOINT}" && ! -f "${WALL_BUDGET_STATE}" ]]; then
    echo "existing V100 checkpoint is missing its bound wall-budget state" >&2
    exit 1
  fi
fi
CHECKPOINT_FAILURE_ROOT="${RESULT_DIR}/failed_training_attempts"
if [[ "${PREFLIGHT_ONLY}" == true ]]; then
  "${ROOT}/scripts/run_checkpoint_transaction.sh" --checkpoint "${CHECKPOINT}" \
    --failure-root "${CHECKPOINT_FAILURE_ROOT}" --check-only --
fi
{
  printf 'schema=rlf-general-cuda-training-environment-v1\nprofile=%s\ntimestamp_utc=%s\npreflight_only=%s\ntraining_wall_budget_seconds=%s\nsource_manifest_sha256=%s\nsolstice_binary_sha256=%s\ncuda_local_update_policy=%s\ncuda_cached_cosine_policy=%s\ncuda_small_cosine_policy=%s\nsparse_router_update_policy=%s\nsparse_rerank_policy=%s\nconcept_update_policy=%s\nexample_duplicate_policy=%s\nmode_id_index_policy=%s\ngrounding_index_policy=%s\nlanguage_outcome_policy=%s\nlanguage_dialogue_encoding_policy=%s\ninstruction_duplicate_policy=%s\ntool_keyword_update_policy=%s\npreference_duplicate_policy=%s\nactive_learning_duplicate_policy=%s\nseparate_vision_analysis=%s\n' "${PROFILE}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${PREFLIGHT_ONLY}" "${TRAINING_WALL_BUDGET_SECONDS}" "${ACTUAL_SOURCE_SHA256}" "${ACTUAL_BINARY_SHA256}" "${CUDA_LOCAL_UPDATE_POLICY}" "${CUDA_CACHED_COSINE_POLICY}" "${CUDA_SMALL_COSINE_POLICY}" "${SPARSE_ROUTER_UPDATE_POLICY}" "${SPARSE_RERANK_POLICY}" "${CONCEPT_UPDATE_POLICY}" "${EXAMPLE_DUPLICATE_POLICY}" "${MODE_ID_INDEX_POLICY}" "${GROUNDING_INDEX_POLICY}" "${LANGUAGE_OUTCOME_POLICY}" "${LANGUAGE_DIALOGUE_ENCODING_POLICY}" "${INSTRUCTION_DUPLICATE_POLICY}" "${TOOL_KEYWORD_UPDATE_POLICY}" "${PREFERENCE_DUPLICATE_POLICY}" "${ACTIVE_LEARNING_DUPLICATE_POLICY}" "${SEPARATE_VISION_ANALYSIS}"
  [[ -z "${WALL_BINDING_SHA256}" ]] || printf 'training_wall_binding_sha256=%s\n' "${WALL_BINDING_SHA256}"
  if ((CAMPAIGN_TARGET_RECORDS > 0)); then
    printf 'campaign_stage=%s\nauthorized_training_records=%s\ncampaign_binding_sha256=%s\n' \
      "${RLF_CAMPAIGN_STAGE}" "${CAMPAIGN_TARGET_RECORDS}" "${RLF_CAMPAIGN_BINDING_SHA256}"
  fi
  uname -a; cmake --version | head -n 1; g++ --version | head -n 1; nvcc --version | tail -n 1
  nvidia-smi --id="${GPU_INDEX}" --query-gpu=index,name,uuid,memory.total,compute_cap,driver_version --format=csv,noheader,nounits
} >"${RESULT_DIR}/training_environment.txt"
grep -Eq '"require_media_hashes"[[:space:]]*:[[:space:]]*true' "${READINESS_REPORT}" || { echo "readiness does not require media hashes" >&2; exit 1; }
if [[ "${PREFLIGHT_ONLY}" == true ]]; then
  "${BIN}" audit-data --ledger "${LEDGER}" --output "${RESULT_DIR}/data_audit.json" --require-media-hashes --max-audit-records "${MAX_AUDIT_RECORDS}" --max-text-shard-bytes "${MAX_TEXT_SHARD_BYTES}" --max-train-shard-bytes "${MAX_TRAIN_SHARD_BYTES}"
  EVIDENCE_ELIGIBLE=true
  [[ "${RLF_ALLOW_TEST_DOUBLES:-0}" == 1 ]] && EVIDENCE_ELIGIBLE=false
  TEMPORARY_PREFLIGHT="$(mktemp "${RESULT_DIR}/.training-handoff-preflight.XXXXXX")"
  printf '{\n  "schema": "rlf-general-cuda-training-handoff-v1",\n  "passed": true,\n  "training_performed": false,\n  "evidence_eligible": %s,\n  "profile": "%s",\n  "ledger_sha256": "%s",\n  "source_manifest_sha256": "%s",\n  "solstice_binary_sha256": "%s",\n  "training_wall_budget_seconds": %s\n}\n' \
    "${EVIDENCE_ELIGIBLE}" "${PROFILE}" "${LEDGER_SHA256}" "${ACTUAL_SOURCE_SHA256}" "${ACTUAL_BINARY_SHA256}" "${TRAINING_WALL_BUDGET_SECONDS}" >"${TEMPORARY_PREFLIGHT}"
  mv -f -- "${TEMPORARY_PREFLIGHT}" "${RESULT_DIR}/training_handoff_preflight.json"
  echo "${PROFILE} training handoff preflight passed. No training was performed."
  exit 0
fi
TRAIN_COMMAND=("${BIN}" train-data --ledger "${LEDGER}" --checkpoint "${CHECKPOINT}" --profile "${PROFILE}" --backend cuda --blank --enforce-profile --require-media-hashes --max-audit-records "${MAX_AUDIT_RECORDS}" --max-text-shard-bytes "${MAX_TEXT_SHARD_BYTES}" --max-train-shard-bytes "${MAX_TRAIN_SHARD_BYTES}" --output "${RESULT_DIR}/data_audit.json" --telemetry "${RESULT_DIR}/training_pipeline_telemetry.json" --seed 6003100749088244549)
if ((CAMPAIGN_TARGET_RECORDS > 0)); then
  TRAIN_COMMAND+=(--target-training-records "${CAMPAIGN_TARGET_RECORDS}")
fi
if [[ "${SEPARATE_VISION_ANALYSIS}" == 1 ]]; then
  TRAIN_COMMAND+=(--separate-vision-analysis)
fi
if ((TRAINING_WALL_BUDGET_SECONDS > 0)); then
  TRAIN_COMMAND=("${ROOT}/scripts/run_with_wall_budget.sh" --budget-seconds "${TRAINING_WALL_BUDGET_SECONDS}" --binding "${WALL_BINDING_SHA256}" --state "${WALL_BUDGET_STATE}" -- "${TRAIN_COMMAND[@]}")
fi
"${ROOT}/scripts/run_checkpoint_transaction.sh" --checkpoint "${CHECKPOINT}" \
  --failure-root "${CHECKPOINT_FAILURE_ROOT}" \
  --preserve "${RESULT_DIR}/training_environment.txt" \
  --preserve "${RESULT_DIR}/data_audit.json" \
  --preserve "${RESULT_DIR}/training_pipeline_telemetry.json" \
  --preserve "${RESULT_DIR}/training_vram.csv" \
  --preserve "${RESULT_DIR}/training_resource_summary.json" -- \
  "${ROOT}/scripts/run_general_cuda_with_vram_guard.sh" --profile "${PROFILE}" \
  --gpu-index "${GPU_INDEX}" --expected-uuid "${GPU_UUID}" \
  --trace "${RESULT_DIR}/training_vram.csv" \
  --summary "${RESULT_DIR}/training_resource_summary.json" -- "${TRAIN_COMMAND[@]}"
if ((TRAINING_WALL_BUDGET_SECONDS > 0)); then
  [[ "$(awk -F= '$1 == "schema" { print $2 }' "${WALL_BUDGET_STATE}")" == rlf-wall-budget-v2 ]] || { echo "invalid wall-budget completion schema" >&2; exit 1; }
  [[ "$(awk -F= '$1 == "budget_seconds" { print $2 }' "${WALL_BUDGET_STATE}")" == "${TRAINING_WALL_BUDGET_SECONDS}" ]] || { echo "wall-budget completion total mismatch" >&2; exit 1; }
  [[ "$(awk -F= '$1 == "binding_sha256" { print $2 }' "${WALL_BUDGET_STATE}")" == "${WALL_BINDING_SHA256}" ]] || { echo "wall-budget completion binding mismatch" >&2; exit 1; }
  [[ "$(awk -F= '$1 == "active" { print $2 }' "${WALL_BUDGET_STATE}")" == false ]] || { echo "wall-budget completion state is still active" >&2; exit 1; }
  CONSUMED_SECONDS="$(awk -F= '$1 == "consumed_seconds" { print $2 }' "${WALL_BUDGET_STATE}")"
  [[ "${CONSUMED_SECONDS}" =~ ^[0-9]+$ ]] || { echo "invalid wall-budget completion state" >&2; exit 1; }
  printf 'training_wall_consumed_seconds=%s\n' "${CONSUMED_SECONDS}" >>"${RESULT_DIR}/training_environment.txt"
  printf 'training_wall_budget_state_sha256=%s\n' "$(sha256sum -- "${WALL_BUDGET_STATE}" | awk '{print $1}')" >>"${RESULT_DIR}/training_environment.txt"
fi
"${BIN}" verify-checkpoint --checkpoint "${CHECKPOINT}" --profile "${PROFILE}" --enforce-profile
"${BIN}" inspect-checkpoint --checkpoint "${CHECKPOINT}" --profile "${PROFILE}" --enforce-profile >"${RESULT_DIR}/checkpoint_inspection.txt"
if ((CAMPAIGN_TARGET_RECORDS > 0)); then
  grep -Fqx "audited_training_records=${CAMPAIGN_TARGET_RECORDS}" "${RESULT_DIR}/checkpoint_inspection.txt" || {
    echo "checkpoint does not contain the exact authorized cumulative record count" >&2
    exit 1
  }
  grep -Eq "\"target_training_records\"[[:space:]]*:[[:space:]]*${CAMPAIGN_TARGET_RECORDS}([,[:space:]]|$)" "${RESULT_DIR}/training_pipeline_telemetry.json" || {
    echo "training telemetry does not bind the authorized record target" >&2
    exit 1
  }
  grep -Eq "\"cumulative_training_records_after\"[[:space:]]*:[[:space:]]*${CAMPAIGN_TARGET_RECORDS}([,[:space:]]|$)" "${RESULT_DIR}/training_pipeline_telemetry.json" || {
    echo "training telemetry does not prove the authorized cumulative record count" >&2
    exit 1
  }
fi
sha256sum "${CHECKPOINT}" >"${RESULT_DIR}/checkpoint.sha256"
MANIFEST_COMMAND=("${ROOT}/scripts/create_general_cuda_training_artifact_manifest.sh" --checkpoint "${CHECKPOINT}" --ledger "${LEDGER}" --source-manifest "${RESULT_DIR}/source_manifest.tsv" --data-audit "${RESULT_DIR}/data_audit.json" --telemetry "${RESULT_DIR}/training_pipeline_telemetry.json" --resource-summary "${RESULT_DIR}/training_resource_summary.json" --vram-trace "${RESULT_DIR}/training_vram.csv" --environment "${RESULT_DIR}/training_environment.txt" --checkpoint-inspection "${RESULT_DIR}/checkpoint_inspection.txt" --readiness-report "${READINESS_REPORT}" --output "${RESULT_DIR}/training_artifact_manifest.tsv")
if ((TRAINING_WALL_BUDGET_SECONDS > 0)); then
  MANIFEST_COMMAND+=(--wall-budget-state "${WALL_BUDGET_STATE}")
fi
"${MANIFEST_COMMAND[@]}"
echo "Created and verified ${CHECKPOINT} under ${PROFILE}"
