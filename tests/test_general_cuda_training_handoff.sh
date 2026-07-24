#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${1:?solstice executable required}"
WORK="${ROOT}/results/general_cuda_training_handoff_shell_test"
[[ "$(realpath -m "${WORK}")" == "$(realpath "${ROOT}")/results/general_cuda_training_handoff_shell_test" ]]
rm -rf -- "${WORK}"; trap 'rm -rf -- "${WORK}"' EXIT
mkdir -p "${WORK}/fake-bin" "${WORK}/result"

printf '#!/usr/bin/env bash\necho "Cuda compilation tools, release fixture"\n' >"${WORK}/fake-bin/nvcc"
printf '#!/usr/bin/env bash\necho "0, Tesla V100-SXM2-32GB, GPU-TEST-0001, 32510, 7.0, fixture-driver"\n' >"${WORK}/fake-bin/nvidia-smi"
chmod +x "${WORK}/fake-bin/nvcc" "${WORK}/fake-bin/nvidia-smi"

printf 'alpha beta gamma\n' >"${WORK}/text.txt"
TEXT_SHA="$(sha256sum -- "${WORK}/text.txt" | awk '{print $1}')"
printf 'text-1\ttext\ttrain\ttext\ten\ttest\ttext_lines\ttext.txt\tlocal:test\tCC0-1.0\t2026-07-21\t%s\ttest-v1\tnone\tnone\ttrue\n' \
  "${TEXT_SHA}" >"${WORK}/ledger.tsv"
LEDGER_SHA="$(sha256sum -- "${WORK}/ledger.tsv" | awk '{print $1}')"
"${ROOT}/scripts/create_source_manifest.sh" "${WORK}/source_manifest.tsv" >/dev/null
SOURCE_SHA="$(sha256sum -- "${WORK}/source_manifest.tsv" | awk '{print $1}')"
BINARY_SHA="$(sha256sum -- "${BIN}" | awk '{print $1}')"
printf '{"schema":"rlf-general-cuda-readiness-v1","ready":true,"test_doubles":false,"require_media_hashes":true,"profile":"general-v100-32g","gpu_index":0,"gpu_uuid":"GPU-TEST-0001","ledger_sha256":"%s","source_manifest_sha256":"%s","solstice_binary_sha256":"%s","maximum_audit_records":10000000,"maximum_text_shard_bytes":2147483648,"maximum_train_shard_bytes":4294967296,"training_wall_budget_seconds":900000}\n' \
  "${LEDGER_SHA}" "${SOURCE_SHA}" "${BINARY_SHA}" >"${WORK}/readiness.json"

PATH="${WORK}/fake-bin:${PATH}" RLF_ALLOW_TEST_DOUBLES=1 RLF_GENERAL_CUDA_TEST_BIN="${BIN}" \
  "${ROOT}/scripts/train_general_cuda_audited.sh" --preflight-only \
  "${WORK}/ledger.tsv" "${WORK}/checkpoint.rlfsp" "${WORK}/result" \
  "${WORK}/readiness.json" >/dev/null
grep -q '"passed": true' "${WORK}/result/training_handoff_preflight.json"
grep -q '"training_performed": false' "${WORK}/result/training_handoff_preflight.json"
grep -q '"evidence_eligible": false' "${WORK}/result/training_handoff_preflight.json"
grep -Fqx 'preflight_only=true' "${WORK}/result/training_environment.txt"
[[ ! -e "${WORK}/checkpoint.rlfsp" ]]

sed -i "s/${SOURCE_SHA}/$(printf 'wrong-source' | sha256sum | awk '{print $1}')/" "${WORK}/readiness.json"
set +e
PATH="${WORK}/fake-bin:${PATH}" RLF_ALLOW_TEST_DOUBLES=1 RLF_GENERAL_CUDA_TEST_BIN="${BIN}" \
  "${ROOT}/scripts/train_general_cuda_audited.sh" --preflight-only \
  "${WORK}/ledger.tsv" "${WORK}/checkpoint.rlfsp" "${WORK}/mismatch" \
  "${WORK}/readiness.json" >/dev/null 2>&1
STATUS=$?
set -e
[[ "${STATUS}" -ne 0 && ! -e "${WORK}/checkpoint.rlfsp" ]]

CHECKPOINT_500M="$(realpath -m "${WORK}/checkpoint-500m.rlfsp")"
BINDING_500M="$(printf '%s\n%s\n%s\n' general-v100-32g-500m "${CHECKPOINT_500M}" 5709600 | sha256sum | awk '{print $1}')"
printf '{"schema":"rlf-general-cuda-readiness-v1","ready":true,"test_doubles":false,"require_media_hashes":true,"profile":"general-v100-32g-500m","gpu_index":0,"gpu_uuid":"GPU-TEST-0001","ledger_sha256":"%s","source_manifest_sha256":"%s","solstice_binary_sha256":"%s","maximum_audit_records":10000000,"maximum_text_shard_bytes":2147483648,"maximum_train_shard_bytes":4294967296,"training_wall_budget_seconds":5709600,"training_wall_binding_sha256":"%s"}\n' \
  "${LEDGER_SHA}" "${SOURCE_SHA}" "${BINARY_SHA}" "${BINDING_500M}" >"${WORK}/readiness-500m.json"
PATH="${WORK}/fake-bin:${PATH}" RLF_ALLOW_TEST_DOUBLES=1 RLF_GENERAL_CUDA_TEST_BIN="${BIN}" \
  "${ROOT}/scripts/train_general_cuda_audited.sh" --preflight-only \
  "${WORK}/ledger.tsv" "${CHECKPOINT_500M}" "${WORK}/result-500m" \
  "${WORK}/readiness-500m.json" >/dev/null
grep -q '"profile": "general-v100-32g-500m"' "${WORK}/result-500m/training_handoff_preflight.json"
grep -q '"training_wall_budget_seconds": 5709600' "${WORK}/result-500m/training_handoff_preflight.json"
grep -Fqx 'training_wall_budget_seconds=5709600' "${WORK}/result-500m/training_environment.txt"
grep -Fqx "training_wall_binding_sha256=${BINDING_500M}" "${WORK}/result-500m/training_environment.txt"
[[ ! -e "${CHECKPOINT_500M}" ]]

sed -i 's/"training_wall_budget_seconds":5709600/"training_wall_budget_seconds":5709599/' "${WORK}/readiness-500m.json"
set +e
PATH="${WORK}/fake-bin:${PATH}" RLF_ALLOW_TEST_DOUBLES=1 RLF_GENERAL_CUDA_TEST_BIN="${BIN}" \
  "${ROOT}/scripts/train_general_cuda_audited.sh" --preflight-only \
  "${WORK}/ledger.tsv" "${CHECKPOINT_500M}" "${WORK}/wrong-budget-500m" \
  "${WORK}/readiness-500m.json" >/dev/null 2>&1
WRONG_BUDGET_STATUS=$?
set -e
[[ "${WRONG_BUDGET_STATUS}" -ne 0 && ! -e "${CHECKPOINT_500M}" ]]
sed -i 's/"training_wall_budget_seconds":5709599/"training_wall_budget_seconds":5709600/' "${WORK}/readiness-500m.json"
sed -i 's/general-v100-32g-500m/general-v100-32g-500x/' "${WORK}/readiness-500m.json"
set +e
PATH="${WORK}/fake-bin:${PATH}" RLF_ALLOW_TEST_DOUBLES=1 RLF_GENERAL_CUDA_TEST_BIN="${BIN}" \
  "${ROOT}/scripts/train_general_cuda_audited.sh" --preflight-only \
  "${WORK}/ledger.tsv" "${CHECKPOINT_500M}" "${WORK}/wrong-profile-500m" \
  "${WORK}/readiness-500m.json" >/dev/null 2>&1
WRONG_PROFILE_STATUS=$?
set -e
[[ "${WRONG_PROFILE_STATUS}" -ne 0 && ! -e "${CHECKPOINT_500M}" ]]
