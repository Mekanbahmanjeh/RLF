#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${ROOT}/results/vast_phase1_helper_shell_test"
[[ "$(realpath -m "${WORK}")" == "$(realpath "${ROOT}")/results/vast_phase1_helper_shell_test" ]]
rm -rf -- "${WORK}"
mkdir -p "${WORK}/fake-bin" "${WORK}/artifacts/sub"
trap 'rm -rf -- "${WORK}" 2>/dev/null || true' EXIT

for script in "${ROOT}"/scripts/vast/*.sh; do
  bash -n "${script}"
done

cat >"${WORK}/fake-bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  *"--query-gpu=index,name,uuid,memory.total,memory.free,compute_cap"*)
    printf '0, %s, GPU-VAST-TEST-0001, 97887, 95000, %s\n' \
      "${RLF_FAKE_GPU_NAME:-NVIDIA RTX PRO 6000 Blackwell Workstation Edition}" \
      "${RLF_FAKE_COMPUTE_CAP:-12.0}"
    ;;
  *"--query-gpu=mig.mode.current"*) printf '%s\n' "${RLF_FAKE_MIG_MODE:-N/A}" ;;
  *"-L"*)
    printf 'GPU 0: %s (UUID: GPU-VAST-TEST-0001)\n' \
      "${RLF_FAKE_GPU_NAME:-NVIDIA RTX PRO 6000 Blackwell Workstation Edition}"
    ;;
  *) echo "unsupported fake nvidia-smi call: $*" >&2; exit 2 ;;
esac
EOF
chmod +x "${WORK}/fake-bin/nvidia-smi"

RLF_ALLOW_TEST_DOUBLES=1 RLF_NVIDIA_SMI="${WORK}/fake-bin/nvidia-smi" \
  "${ROOT}/scripts/vast/remote_rtx_pro_6000_preflight.sh" \
  --output "${WORK}/preflight-pass" --work-root "${WORK}" \
  --min-host-ram-gib 0 --min-disk-gib 0
grep -q '"ready": false' "${WORK}/preflight-pass/preflight.json"
grep -q '"synthetic_checks_passed": true' "${WORK}/preflight-pass/preflight.json"
grep -q 'gpu_model[[:space:]]*pass' "${WORK}/preflight-pass/checks.tsv"
grep -q 'mig_mode[[:space:]]*pass' "${WORK}/preflight-pass/checks.tsv"

set +e
RLF_ALLOW_TEST_DOUBLES=1 RLF_NVIDIA_SMI="${WORK}/fake-bin/nvidia-smi" \
RLF_FAKE_GPU_NAME='NVIDIA H100 80GB HBM3' RLF_FAKE_COMPUTE_CAP=9.0 \
  "${ROOT}/scripts/vast/remote_rtx_pro_6000_preflight.sh" \
  --output "${WORK}/preflight-reject" --work-root "${WORK}" \
  --min-host-ram-gib 0 --min-disk-gib 0 >/dev/null 2>&1
STATUS=$?
set -e
[[ "${STATUS}" -eq 1 ]]
grep -q 'gpu_model[[:space:]]*fail' "${WORK}/preflight-reject/checks.tsv"

"${ROOT}/scripts/vast/local_rtx_pro_6000.sh" --print-workflow >"${WORK}/workflow.txt"
grep -q 'compute_cap.*1200' "${WORK}/workflow.txt"
grep -q 'num_gpus=1' "${WORK}/workflow.txt"
grep -q 'cpu_ram.*1120' "${WORK}/workflow.txt"
grep -q 'disk_space.*2400' "${WORK}/workflow.txt"
"${ROOT}/scripts/vast/local_rtx_pro_6000.sh" --render-create 12345 >"${WORK}/create.txt"
grep -q '^# REVIEW THIS COMMAND' "${WORK}/create.txt"
grep -q 'vastai create instance 12345' "${WORK}/create.txt"
grep -q -- '--disk 2400' "${WORK}/create.txt"

printf 'alpha\n' >"${WORK}/artifacts/a.txt"
printf 'beta\n' >"${WORK}/artifacts/sub/b.txt"
"${ROOT}/scripts/vast/artifact_manifest.sh" collect \
  --source "${WORK}/artifacts" --manifest "${WORK}/evidence/artifacts.tsv"
"${ROOT}/scripts/vast/artifact_manifest.sh" verify \
  --source "${WORK}/artifacts" --manifest "${WORK}/evidence/artifacts.tsv"
set +e
printf 'tampered\n' >"${WORK}/artifacts/sub/b.txt"
"${ROOT}/scripts/vast/artifact_manifest.sh" verify \
  --source "${WORK}/artifacts" --manifest "${WORK}/evidence/artifacts.tsv" >/dev/null 2>&1
STATUS=$?
set -e
[[ "${STATUS}" -eq 1 ]]

"${ROOT}/scripts/vast/phase1_118h_controller.sh" --plan >"${WORK}/plan.txt"
grep -Eq '^TOTAL[[:space:]]+118' "${WORK}/plan.txt"
PLANNED_HOURS="$(awk '$1 ~ /^(hardware-preflight|build-validation|ingest-efficiency|controlled-image|50m-candidates|recovery-ablations|frozen-dev-evaluation|artifact-export)$/ { sum += $2 } END { print sum + 0 }' "${WORK}/plan.txt")"
[[ "${PLANNED_HOURS}" -eq 118 ]]
BINDING="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
"${ROOT}/scripts/vast/phase1_118h_controller.sh" --start \
  --state-dir "${WORK}/controller" --binding "${BINDING}" >/dev/null
"${ROOT}/scripts/vast/phase1_118h_controller.sh" --run-stage artifact-export \
  --state-dir "${WORK}/controller" --binding "${BINDING}" -- \
  bash -c 'printf "export tested\n" >/dev/null'
"${ROOT}/scripts/vast/phase1_118h_controller.sh" --status \
  --state-dir "${WORK}/controller" >"${WORK}/controller-status.txt"
grep -q '^stage_artifact-export_consumed_seconds=' "${WORK}/controller-status.txt"
