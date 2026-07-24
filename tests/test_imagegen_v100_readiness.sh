#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${ROOT}/results/imagegen_v100_readiness_shell_test"
[[ "$(realpath -m "${WORK}")" == "$(realpath "${ROOT}")/results/imagegen_v100_readiness_shell_test" ]]
rm -rf -- "${WORK}"; mkdir -p "${WORK}/fake-bin" "${WORK}/build"
trap 'rm -rf -- "${WORK}"' EXIT
for command_name in cmake; do
  printf '#!/usr/bin/env bash\necho fake\n' >"${WORK}/fake-bin/${command_name}"
  chmod +x "${WORK}/fake-bin/${command_name}"
done
printf '#!/usr/bin/env bash\necho "Cuda compilation tools, release 12.8, V12.8.0"\n' >"${WORK}/fake-bin/nvcc"
chmod +x "${WORK}/fake-bin/nvcc"
printf '#!/usr/bin/env bash\necho "100%% tests passed"\n' >"${WORK}/fake-bin/ctest"
chmod +x "${WORK}/fake-bin/ctest"
cat >"${WORK}/build/solstice" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  imagegen-profile-info) printf 'profile=imagegen-v100-32g\ngpu_working_set_gib=30\nmaximum_modes=48000000\n' ;;
  device-info) printf 'backend=cuda-persistent\navailable=true\nprofile_fits_device=true\n' ;;
  *) exit 2 ;;
esac
EOF
chmod +x "${WORK}/build/solstice"
cat >"${WORK}/build/CMakeCache.txt" <<'EOF'
RLF_ENABLE_CUDA:BOOL=ON
CMAKE_CUDA_ARCHITECTURES:STRING=70;80
RLF_WARNINGS_AS_ERRORS:BOOL=ON
CMAKE_BUILD_TYPE:STRING=Release
EOF
PATH="${WORK}/fake-bin:${PATH}" RLF_ALLOW_TEST_DOUBLES=1 \
RLF_NVIDIA_SMI="${ROOT}/tests/fixtures/fake_nvidia_smi.sh" \
RLF_FAKE_GPU_NAME='Tesla V100-SXM2-32GB' RLF_FAKE_TOTAL_MIB=32510 \
RLF_FAKE_FREE_MIB=31000 RLF_FAKE_COMPUTE_CAP=7.0 \
  "${ROOT}/scripts/check_imagegen_v100_readiness.sh" \
  --output "${WORK}/evidence" --build-dir "${WORK}/build" \
  --min-host-ram-gib 0 --min-disk-gib 0
grep -q '"ready": false' "${WORK}/evidence/readiness.json"
grep -q '"synthetic_checks_passed": true' "${WORK}/evidence/readiness.json"
grep -q '"training_performed": false' "${WORK}/evidence/readiness.json"
grep -q $'cuda_backend\tpass' "${WORK}/evidence/readiness_checks.tsv"

printf '#!/usr/bin/env bash\necho "Cuda compilation tools, release 13.0, V13.0.0"\n' >"${WORK}/fake-bin/nvcc"
set +e
PATH="${WORK}/fake-bin:${PATH}" RLF_ALLOW_TEST_DOUBLES=1 \
  "${ROOT}/scripts/check_imagegen_v100_readiness.sh" \
  --output "${WORK}/cuda13" --build-dir "${WORK}/build" \
  --min-host-ram-gib 0 --min-disk-gib 0 >/dev/null 2>"${WORK}/cuda13.txt"
STATUS=$?
set -e
[[ "${STATUS}" -eq 2 ]]
grep -q 'requires parseable CUDA Toolkit major 12' "${WORK}/cuda13.txt"

printf 'imagegen_v100_readiness_shell_test=pass\nclaim_eligible=false\nphysical_training_performed=false\n'
