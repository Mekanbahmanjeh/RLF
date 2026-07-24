#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${ROOT}/results/general_cuda_readiness_shell_test"
[[ "$(realpath -m "${WORK}")" == "$(realpath "${ROOT}")/results/general_cuda_readiness_shell_test" ]]
rm -rf -- "${WORK}"; mkdir -p "${WORK}/fake-bin" "${WORK}/build"; trap 'rm -rf -- "${WORK}"' EXIT
for command_name in cmake ninja g++; do printf '#!/usr/bin/env bash\necho "%s fake version"\n' "${command_name}" >"${WORK}/fake-bin/${command_name}"; chmod +x "${WORK}/fake-bin/${command_name}"; done
printf '#!/usr/bin/env bash\necho "Cuda compilation tools, release 12.8, V12.8.0"\n' >"${WORK}/fake-bin/nvcc"
chmod +x "${WORK}/fake-bin/nvcc"
cat >"${WORK}/fake-bin/ctest" <<'EOF'
#!/usr/bin/env bash
echo '100% tests passed, 0 tests failed out of 1'
EOF
chmod +x "${WORK}/fake-bin/ctest"
cat >"${WORK}/build/solstice" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  profile-info) while (($#)); do
    if [[ "$1" == --profile ]]; then
      echo "profile=$2"
      if [[ "$2" == general-v100-32g-text || "$2" == general-rtx-pro-6000-96g-text ]]; then
        printf 'vision_training_enabled=false\nvideo_training_enabled=false\n'
      elif [[ "$2" == video-rtx-pro-6000-96g || "$2" == video-v100-32g ]]; then
        printf 'vision_training_enabled=true\nvideo_training_enabled=true\n'
      else
        printf 'vision_training_enabled=true\nvideo_training_enabled=false\n'
      fi
      echo 'gpu_working_set_gib=38'; exit 0
    fi
    shift
  done ;;
  device-info) printf 'backend=cuda-persistent\navailable=true\nprofile_fits_device=true\n' ;;
  audit-data) while (($#)); do [[ "$1" == --output ]] && { printf '{"passed":true}\n' >"$2"; exit 0; }; shift; done; exit 2 ;;
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
printf 'fixture ledger\n' >"${WORK}/ledger.tsv"
PATH="${WORK}/fake-bin:${PATH}" RLF_NVIDIA_SMI="${ROOT}/tests/fixtures/fake_nvidia_smi.sh" RLF_ALLOW_TEST_DOUBLES=1 \
RLF_FAKE_GPU_NAME='NVIDIA A100-SXM4-40GB' RLF_FAKE_TOTAL_MIB=40536 RLF_FAKE_FREE_MIB=40000 RLF_FAKE_COMPUTE_CAP=8.0 \
  "${ROOT}/scripts/check_general_cuda_readiness.sh" --profile general-40g --ledger "${WORK}/ledger.tsv" --output "${WORK}/a100" --build-dir "${WORK}/build" --min-host-ram-gib 0 --min-disk-gib 0
grep -q '"ready": false' "${WORK}/a100/readiness.json"
grep -q '"synthetic_checks_passed": true' "${WORK}/a100/readiness.json"
grep -q '"profile": "general-40g"' "${WORK}/a100/readiness.json"
grep -Eq '"source_manifest_sha256": "[0-9a-f]{64}"' "${WORK}/a100/readiness.json"
grep -Eq '"solstice_binary_sha256": "[0-9a-f]{64}"' "${WORK}/a100/readiness.json"
grep -q $'data_templates/FORMATS.md$' "${WORK}/a100/source_manifest.tsv"
PATH="${WORK}/fake-bin:${PATH}" RLF_NVIDIA_SMI="${ROOT}/tests/fixtures/fake_nvidia_smi.sh" RLF_ALLOW_TEST_DOUBLES=1 \
RLF_FAKE_GPU_NAME='Tesla V100-SXM2-32GB' RLF_FAKE_TOTAL_MIB=32510 RLF_FAKE_FREE_MIB=31000 RLF_FAKE_COMPUTE_CAP=7.0 \
  "${ROOT}/scripts/check_general_cuda_readiness.sh" --profile general-v100-32g --ledger "${WORK}/ledger.tsv" --output "${WORK}/v100" --build-dir "${WORK}/build" --min-host-ram-gib 0 --min-disk-gib 0
grep -q '"synthetic_checks_passed": true' "${WORK}/v100/readiness.json"
grep -q '"training_wall_budget_seconds": 900000' "${WORK}/v100/readiness.json"
PATH="${WORK}/fake-bin:${PATH}" RLF_NVIDIA_SMI="${ROOT}/tests/fixtures/fake_nvidia_smi.sh" RLF_ALLOW_TEST_DOUBLES=1 \
RLF_FAKE_GPU_NAME='Tesla V100-SXM2-32GB' RLF_FAKE_TOTAL_MIB=32510 RLF_FAKE_FREE_MIB=31000 RLF_FAKE_COMPUTE_CAP=7.0 \
  "${ROOT}/scripts/check_general_cuda_readiness.sh" --profile general-v100-32g-text --ledger "${WORK}/ledger.tsv" --output "${WORK}/v100-text" --build-dir "${WORK}/build" --min-host-ram-gib 0 --min-disk-gib 0
grep -q '"profile": "general-v100-32g-text"' "${WORK}/v100-text/readiness.json"
grep -q '"training_wall_budget_seconds": 900000' "${WORK}/v100-text/readiness.json"
PATH="${WORK}/fake-bin:${PATH}" RLF_NVIDIA_SMI="${ROOT}/tests/fixtures/fake_nvidia_smi.sh" RLF_ALLOW_TEST_DOUBLES=1 \
RLF_FAKE_GPU_NAME='Tesla V100-SXM2-32GB' RLF_FAKE_TOTAL_MIB=32510 RLF_FAKE_FREE_MIB=31000 RLF_FAKE_COMPUTE_CAP=7.0 \
  "${ROOT}/scripts/check_general_cuda_readiness.sh" --profile video-v100-32g --ledger "${WORK}/ledger.tsv" --output "${WORK}/v100-video" --build-dir "${WORK}/build" --min-host-ram-gib 0 --min-disk-gib 0
grep -q '"profile": "video-v100-32g"' "${WORK}/v100-video/readiness.json"
grep -q '"training_wall_budget_seconds": 900000' "${WORK}/v100-video/readiness.json"
CHECKPOINT_500M="$(realpath -m "${WORK}/500m-checkpoint.rlfsp")"
LEDGER_SHA="$(sha256sum -- "${WORK}/ledger.tsv" | awk '{print $1}')"
EXPECTED_500M_BINDING="$(printf '%s\n%s\n%s\n' general-v100-32g-500m "${CHECKPOINT_500M}" 5709600 | sha256sum | awk '{print $1}')"
PATH="${WORK}/fake-bin:${PATH}" RLF_NVIDIA_SMI="${ROOT}/tests/fixtures/fake_nvidia_smi.sh" RLF_ALLOW_TEST_DOUBLES=1 \
RLF_FAKE_GPU_NAME='Tesla V100-SXM2-32GB' RLF_FAKE_TOTAL_MIB=32510 RLF_FAKE_FREE_MIB=31000 RLF_FAKE_COMPUTE_CAP=7.0 \
  "${ROOT}/scripts/check_general_cuda_readiness.sh" --profile general-v100-32g-500m \
  --ledger "${WORK}/ledger.tsv" --checkpoint "${CHECKPOINT_500M}" \
  --output "${WORK}/v100-500m" --build-dir "${WORK}/build" --min-host-ram-gib 0 --min-disk-gib 0
grep -q '"profile": "general-v100-32g-500m"' "${WORK}/v100-500m/readiness.json"
grep -q '"training_wall_budget_seconds": 5709600' "${WORK}/v100-500m/readiness.json"
grep -q "\"training_wall_binding_sha256\": \"${EXPECTED_500M_BINDING}\"" "${WORK}/v100-500m/readiness.json"
set +e
PATH="${WORK}/fake-bin:${PATH}" RLF_NVIDIA_SMI="${ROOT}/tests/fixtures/fake_nvidia_smi.sh" RLF_ALLOW_TEST_DOUBLES=1 \
  "${ROOT}/scripts/check_general_cuda_readiness.sh" --profile general-v100-32g-500m \
  --ledger "${WORK}/ledger.tsv" --output "${WORK}/missing-500m-checkpoint" \
  --build-dir "${WORK}/build" --min-host-ram-gib 0 --min-disk-gib 0 >/dev/null 2>&1
MISSING_CHECKPOINT_STATUS=$?
"${ROOT}/scripts/check_general_cuda_readiness.sh" --profile general-v100-32g-500M \
  --ledger "${WORK}/ledger.tsv" --checkpoint "${CHECKPOINT_500M}" \
  --output "${WORK}/wrong-500m-profile" >/dev/null 2>&1
WRONG_PROFILE_STATUS=$?
set -e
[[ "${MISSING_CHECKPOINT_STATUS}" -eq 2 && "${WRONG_PROFILE_STATUS}" -eq 2 ]]
printf '#!/usr/bin/env bash\necho "Cuda compilation tools, release 13.0, V13.0.0"\n' >"${WORK}/fake-bin/nvcc"
set +e
PATH="${WORK}/fake-bin:${PATH}" "${ROOT}/scripts/check_general_cuda_readiness.sh" \
  --profile general-v100-32g --ledger "${WORK}/ledger.tsv" --output "${WORK}/cuda13-rejected" \
  --build-dir "${WORK}/build" --min-host-ram-gib 0 --min-disk-gib 0 >/dev/null 2>"${WORK}/cuda13-error.txt"
CUDA13_STATUS=$?
set -e
[[ "${CUDA13_STATUS}" -eq 2 ]]
grep -q 'require parseable CUDA Toolkit major 12' "${WORK}/cuda13-error.txt"
printf '#!/usr/bin/env bash\necho "unparseable fixture"\n' >"${WORK}/fake-bin/nvcc"
set +e
PATH="${WORK}/fake-bin:${PATH}" "${ROOT}/scripts/check_general_cuda_readiness.sh" \
  --profile general-v100-32g-text --ledger "${WORK}/ledger.tsv" --output "${WORK}/unknown-cuda-rejected" \
  --build-dir "${WORK}/build" --min-host-ram-gib 0 --min-disk-gib 0 >/dev/null 2>"${WORK}/unknown-cuda-error.txt"
UNKNOWN_CUDA_STATUS=$?
set -e
[[ "${UNKNOWN_CUDA_STATUS}" -eq 2 ]]
grep -q 'found unknown' "${WORK}/unknown-cuda-error.txt"
printf '#!/usr/bin/env bash\necho "Cuda compilation tools, release 12.8, V12.8.0"\n' >"${WORK}/fake-bin/nvcc"
set +e
PATH="${WORK}/fake-bin:${PATH}" "${ROOT}/scripts/check_general_cuda_readiness.sh" \
  --profile general-v100-32g-500m --ledger "${WORK}/ledger.tsv" \
  --checkpoint "${CHECKPOINT_500M}" --output "${WORK}/undersized-500m" \
  --build-dir "${WORK}/build" --min-host-ram-gib 1023 --min-disk-gib 2047 \
  >/dev/null 2>"${WORK}/undersized-500m-error.txt"
UNDERSIZED_500M_STATUS=$?
set -e
[[ "${UNDERSIZED_500M_STATUS}" -eq 2 ]]
grep -q 'may not lower the 1024 GiB RAM / 2048 GiB disk floor' "${WORK}/undersized-500m-error.txt"
sed -i 's/CMAKE_CUDA_ARCHITECTURES:STRING=70;80/CMAKE_CUDA_ARCHITECTURES:STRING=120/' "${WORK}/build/CMakeCache.txt"
for pro_profile in rtx-pro-6000-96g general-rtx-pro-6000-96g general-rtx-pro-6000-96g-text video-rtx-pro-6000-96g; do
  output_name="${pro_profile//-/_}"
  PATH="${WORK}/fake-bin:${PATH}" RLF_NVIDIA_SMI="${ROOT}/tests/fixtures/fake_nvidia_smi.sh" RLF_ALLOW_TEST_DOUBLES=1 \
  RLF_FAKE_GPU_NAME='NVIDIA RTX PRO 6000 Blackwell Workstation Edition' RLF_FAKE_TOTAL_MIB=97887 RLF_FAKE_FREE_MIB=95000 RLF_FAKE_COMPUTE_CAP=12.0 \
    "${ROOT}/scripts/check_general_cuda_readiness.sh" --profile "${pro_profile}" --ledger "${WORK}/ledger.tsv" --output "${WORK}/${output_name}" --build-dir "${WORK}/build" --min-host-ram-gib 0 --min-disk-gib 0
  grep -q "\"profile\": \"${pro_profile}\"" "${WORK}/${output_name}/readiness.json"
  grep -q '"training_wall_budget_seconds": 0' "${WORK}/${output_name}/readiness.json"
done
sed -i 's/CMAKE_CUDA_ARCHITECTURES:STRING=120/CMAKE_CUDA_ARCHITECTURES:STRING=70;80/' "${WORK}/build/CMakeCache.txt"
set +e
PATH="${WORK}/fake-bin:${PATH}" RLF_NVIDIA_SMI="${ROOT}/tests/fixtures/fake_nvidia_smi.sh" RLF_ALLOW_TEST_DOUBLES=1 \
RLF_FAKE_GPU_NAME='Tesla V100-SXM2-32GB' RLF_FAKE_TOTAL_MIB=32510 RLF_FAKE_FREE_MIB=31000 RLF_FAKE_COMPUTE_CAP=7.0 \
  "${ROOT}/scripts/check_general_cuda_readiness.sh" --profile general-40g --ledger "${WORK}/ledger.tsv" --output "${WORK}/rejected" --build-dir "${WORK}/build" --min-host-ram-gib 0 --min-disk-gib 0 >/dev/null 2>&1
STATUS=$?
set -e
[[ "${STATUS}" -eq 1 ]]
grep -q 'total_vram[[:space:]]*fail' "${WORK}/rejected/readiness_checks.tsv"
