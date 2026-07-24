#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${ROOT}/results/h200_readiness_shell_test"
[[ "$(realpath -m "${WORK}")" == "$(realpath "${ROOT}")/results/h200_readiness_shell_test" ]]
rm -rf -- "${WORK}"
mkdir -p "${WORK}"
trap 'rm -rf -- "${WORK}"' EXIT
mkdir -p "${WORK}/fake-bin" "${WORK}/build"

for command_name in cmake ninja g++; do
  printf '#!/usr/bin/env bash\necho "%s fake version"\n' "${command_name}" \
    >"${WORK}/fake-bin/${command_name}"
  chmod +x "${WORK}/fake-bin/${command_name}"
done
cat >"${WORK}/fake-bin/nvcc" <<'EOF'
#!/usr/bin/env bash
echo 'Cuda compilation tools, release 12.8, V12.8.0'
EOF
cat >"${WORK}/fake-bin/ctest" <<'EOF'
#!/usr/bin/env bash
echo '100% tests passed, 0 tests failed out of 1'
EOF
chmod +x "${WORK}/fake-bin/nvcc" "${WORK}/fake-bin/ctest"

cat >"${WORK}/build/solstice" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  profile-info)
    printf 'profile=general-h200-141g-30t\ngpu_working_set_gib=132\ncontext_ceiling=2000000000\n'
    ;;
  device-info)
    printf 'backend=cuda-persistent\navailable=true\nprofile_fits_device=true\n'
    ;;
  audit-data)
    while (($# > 0)); do
      if [[ "$1" == --output ]]; then printf '{"passed":true}\n' >"$2"; exit 0; fi
      shift
    done
    exit 2
    ;;
  *) exit 2 ;;
esac
EOF
chmod +x "${WORK}/build/solstice"
cat >"${WORK}/build/CMakeCache.txt" <<'EOF'
RLF_ENABLE_CUDA:BOOL=ON
CMAKE_CUDA_ARCHITECTURES:STRING=90
RLF_WARNINGS_AS_ERRORS:BOOL=ON
CMAKE_BUILD_TYPE:STRING=Release
EOF
printf 'fixture ledger\n' >"${WORK}/ledger.tsv"

PATH="${WORK}/fake-bin:${PATH}" \
RLF_NVIDIA_SMI="${ROOT}/tests/fixtures/fake_nvidia_smi.sh" \
RLF_ALLOW_TEST_DOUBLES=1 RLF_FAKE_GPU_NAME='NVIDIA H200' \
RLF_FAKE_TOTAL_MIB=143771 RLF_FAKE_FREE_MIB=140000 \
bash "${ROOT}/scripts/check_general_h200_readiness.sh" \
  --ledger "${WORK}/ledger.tsv" --output "${WORK}/passed" \
  --build-dir "${WORK}/build" --min-free-mib 0 \
  --min-host-ram-gib 0 --min-disk-gib 0
grep -Fq '"ready": false' "${WORK}/passed/readiness.json"
grep -Fq '"synthetic_checks_passed": true' "${WORK}/passed/readiness.json"
grep -Fq '"profile": "general-h200-141g-30t"' "${WORK}/passed/readiness.json"

set +e
PATH="${WORK}/fake-bin:${PATH}" \
RLF_NVIDIA_SMI="${ROOT}/tests/fixtures/fake_nvidia_smi.sh" \
RLF_ALLOW_TEST_DOUBLES=1 RLF_FAKE_GPU_NAME='NVIDIA H100' \
bash "${ROOT}/scripts/check_general_h200_readiness.sh" \
  --ledger "${WORK}/ledger.tsv" --output "${WORK}/failed" \
  --build-dir "${WORK}/build" --min-free-mib 0 \
  --min-host-ram-gib 0 --min-disk-gib 0 >/dev/null 2>&1
STATUS=$?
set -e
[[ "${STATUS}" -eq 1 ]]
grep -Eq '^gpu_model[[:space:]]+fail' "${WORK}/failed/readiness_checks.tsv"
