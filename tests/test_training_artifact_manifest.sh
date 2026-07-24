#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE_BIN="${1:-}"
WORK="$(mktemp -d)"
trap 'rm -rf -- "${WORK}"' EXIT

for file in ledger source data_audit vram environment inspection; do
  printf '%s\n' "${file}" >"${WORK}/${file}"
done
cp "${ROOT}/results/codex_campaign/resume_smoke.rlfsp" "${WORK}/checkpoint"
printf '{"ready":false,"test_doubles":true}\n' >"${WORK}/readiness.json"
cat >"${WORK}/resource.json" <<'EOF'
{
  "schema": "rlf-h100-vram-v1",
  "within_limit": true,
  "sampler_ok": true,
  "command_exit_code": 0
}
EOF

ARGS=(
  --checkpoint "${WORK}/checkpoint" --ledger "${WORK}/ledger" \
  --source-manifest "${WORK}/source" --data-audit "${WORK}/data_audit" \
  --resource-summary "${WORK}/resource.json" --vram-trace "${WORK}/vram" \
  --environment "${WORK}/environment" \
  --checkpoint-inspection "${WORK}/inspection" \
  --readiness-report "${WORK}/readiness.json" \
  --output "${WORK}/manifest.tsv"
)
set +e
"${ROOT}/scripts/create_training_artifact_manifest.sh" "${ARGS[@]}" >/dev/null 2>&1
STATUS=$?
set -e
[[ "${STATUS}" -ne 0 ]]

printf '{"ready":true,"test_doubles":false,"profile":"general-h100"}\n' >"${WORK}/readiness.json"
"${ROOT}/scripts/create_training_artifact_manifest.sh" "${ARGS[@]}"
"${ROOT}/scripts/create_training_artifact_manifest.sh" --verify "${WORK}/manifest.tsv"
"${ROOT}/scripts/authorize_serving_checkpoint.sh" \
  "${WORK}/checkpoint" "${WORK}/manifest.tsv"

cp "${WORK}/checkpoint" "${WORK}/wrong-checkpoint"
set +e
"${ROOT}/scripts/authorize_serving_checkpoint.sh" \
  "${WORK}/wrong-checkpoint" "${WORK}/manifest.tsv" >/dev/null 2>&1
STATUS=$?
set -e
[[ "${STATUS}" -ne 0 ]]

if [[ -n "${GATE_BIN}" ]]; then
  [[ -x "${GATE_BIN}" ]]
  SOLSTICE_BIN="$(dirname "${GATE_BIN}")/solstice"
  [[ -x "${SOLSTICE_BIN}" ]]
  "${ROOT}/scripts/authorize_serving_checkpoint.sh" \
    "${WORK}/checkpoint" "${WORK}/manifest.tsv" "${SOLSTICE_BIN}" >/dev/null
  printf 'raw external fixture\n' >"${WORK}/raw.json"
  printf 'contamination fixture\n' >"${WORK}/contamination.json"
  RAW_HASH="$(sha256sum "${WORK}/raw.json" | awk '{print $1}')"
  CONTAMINATION_HASH="$(sha256sum "${WORK}/contamination.json" | awk '{print $1}')"
  CHECKPOINT_HASH="$(sha256sum "${WORK}/checkpoint" | awk '{print $1}')"
  MANIFEST_HASH="$(sha256sum "${WORK}/manifest.tsv" | awk '{print $1}')"
  : >"${WORK}/evidence.tsv"
  while IFS=$'\t' read -r benchmark capability score examples; do
    printf '%s\t%s\t%s\t%s\ttrue\ttrue\ttrue\tofficial-test\tharness\t1\tindependent-test\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${benchmark}" "${capability}" "${score}" "${examples}" \
      "${WORK}/raw.json" "${RAW_HASH}" \
      "${WORK}/contamination.json" "${CONTAMINATION_HASH}" \
      "${CHECKPOINT_HASH}" "${MANIFEST_HASH}" >>"${WORK}/evidence.tsv"
  done <<'EOF'
hle	academic_reasoning	0.568	500
arc_agi_2	abstract_reasoning	0.925	120
gpqa_diamond	scientific_reasoning	0.946	198
swe_bench_verified	agentic_coding	0.939	500
mmmu_pro	multimodal_reasoning	0.830	500
mmmlu	multilingual_knowledge	0.927	1000
tau2_retail	tool_use	0.919	100
mrcr_128k	long_context	0.849	100
EOF
  "${GATE_BIN}" --evidence "${WORK}/evidence.tsv" \
    --training-manifest "${WORK}/manifest.tsv" \
    --output "${WORK}/gate.json"
  grep -q '"broad_frontier_parity_proven": true' "${WORK}/gate.json"

  # The V100/RTX audited trainer emits the ten-artifact v2 schema. It must be
  # accepted by the same unchanged external gate and retain exact model binding.
  printf 'training telemetry fixture\n' >"${WORK}/telemetry"
  printf '# rlf-training-artifact-manifest-v2\n# kind\tsha256\tbytes\tabsolute_path\n' \
    >"${WORK}/manifest-v2.tsv"
  for assignment in \
    "checkpoint:${WORK}/checkpoint" "ledger:${WORK}/ledger" \
    "source_manifest:${WORK}/source" "data_audit:${WORK}/data_audit" \
    "telemetry:${WORK}/telemetry" "resource_summary:${WORK}/resource.json" \
    "vram_trace:${WORK}/vram" "environment:${WORK}/environment" \
    "checkpoint_inspection:${WORK}/inspection" \
    "readiness_report:${WORK}/readiness.json"; do
    kind="${assignment%%:*}"
    path="${assignment#*:}"
    printf '%s\t%s\t%s\t%s\n' "${kind}" \
      "$(sha256sum -- "${path}" | awk '{print $1}')" \
      "$(stat -c '%s' -- "${path}")" "${path}" >>"${WORK}/manifest-v2.tsv"
  done
  (cd "${WORK}" && sha256sum manifest-v2.tsv >manifest-v2.tsv.sha256)
  "${ROOT}/scripts/create_training_artifact_manifest.sh" \
    --verify "${WORK}/manifest-v2.tsv" >/dev/null
  sed 's/^telemetry\t/unexpected\t/' "${WORK}/manifest-v2.tsv" \
    >"${WORK}/wrong-kinds.tsv"
  (cd "${WORK}" && sha256sum wrong-kinds.tsv >wrong-kinds.tsv.sha256)
  set +e
  "${ROOT}/scripts/create_training_artifact_manifest.sh" \
    --verify "${WORK}/wrong-kinds.tsv" >/dev/null 2>"${WORK}/wrong-kinds.err"
  WRONG_KINDS_STATUS=$?
  set -e
  [[ "${WRONG_KINDS_STATUS}" -ne 0 ]]
  grep -q 'unexpected v2 artifact kind' "${WORK}/wrong-kinds.err"
  MANIFEST_V2_HASH="$(sha256sum "${WORK}/manifest-v2.tsv" | awk '{print $1}')"
  sed "s/${MANIFEST_HASH}/${MANIFEST_V2_HASH}/g" "${WORK}/evidence.tsv" \
    >"${WORK}/evidence-v2.tsv"
  "${GATE_BIN}" --evidence "${WORK}/evidence-v2.tsv" \
    --training-manifest "${WORK}/manifest-v2.tsv" \
    --output "${WORK}/gate-v2.json"
  grep -q '"broad_frontier_parity_proven": true' "${WORK}/gate-v2.json"

  # A hash-valid manifest may not use a larger profile merely to relax load
  # limits for a checkpoint serialized under another configuration.
  sed 's/general-h100/general-v100-32g-500m/' "${WORK}/readiness.json" \
    >"${WORK}/wrong-readiness.json"
  sed "s#${WORK}/readiness.json#${WORK}/wrong-readiness.json#" \
    "${WORK}/manifest-v2.tsv" >"${WORK}/wrong-profile-manifest.tsv"
  WRONG_READINESS_HASH="$(sha256sum "${WORK}/wrong-readiness.json" | awk '{print $1}')"
  WRONG_READINESS_BYTES="$(stat -c '%s' "${WORK}/wrong-readiness.json")"
  awk -F '\t' -v OFS='\t' -v hash="${WRONG_READINESS_HASH}" \
    -v bytes="${WRONG_READINESS_BYTES}" \
    '$1 == "readiness_report" {$2=hash; $3=bytes} {print}' \
    "${WORK}/wrong-profile-manifest.tsv" >"${WORK}/wrong-profile-manifest.tmp"
  mv "${WORK}/wrong-profile-manifest.tmp" "${WORK}/wrong-profile-manifest.tsv"
  (cd "${WORK}" && sha256sum wrong-profile-manifest.tsv \
    >wrong-profile-manifest.tsv.sha256)
  set +e
  "${GATE_BIN}" --evidence "${WORK}/evidence-v2.tsv" \
    --training-manifest "${WORK}/wrong-profile-manifest.tsv" \
    --output "${WORK}/wrong-profile-gate.json" \
    >/dev/null 2>"${WORK}/wrong-profile-gate.err"
  WRONG_GATE_STATUS=$?
  "${ROOT}/scripts/authorize_serving_checkpoint.sh" \
    "${WORK}/checkpoint" "${WORK}/wrong-profile-manifest.tsv" \
    "${SOLSTICE_BIN}" >/dev/null 2>"${WORK}/wrong-profile-authorize.err"
  WRONG_AUTHORIZE_STATUS=$?
  set -e
  [[ "${WRONG_GATE_STATUS}" -eq 2 && "${WRONG_AUTHORIZE_STATUS}" -ne 0 ]]
  grep -q 'does not match selected profile' "${WORK}/wrong-profile-gate.err"
  grep -q 'does not match selected profile' "${WORK}/wrong-profile-authorize.err"
fi

printf 'tampered\n' >>"${WORK}/checkpoint"
set +e
"${ROOT}/scripts/create_training_artifact_manifest.sh" --verify "${WORK}/manifest.tsv" >/dev/null 2>&1
STATUS=$?
set -e
[[ "${STATUS}" -ne 0 ]]
