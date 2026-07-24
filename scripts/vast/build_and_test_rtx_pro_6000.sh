#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: build_and_test_rtx_pro_6000.sh --preflight FILE --output DIR

Builds the existing isolated sm_120 preset, runs the full test suite, checks the
exact CUDA backend, and records source/binary hashes. This script never trains.
FILE must be a real ready=true Vast hardware preflight report.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFLIGHT=""
OUTPUT=""
while (($# > 0)); do
  case "$1" in
    --preflight) PREFLIGHT="${2:?--preflight requires a file}"; shift 2 ;;
    --output) OUTPUT="${2:?--output requires a directory}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "${PREFLIGHT}" && -f "${PREFLIGHT}" && -n "${OUTPUT}" ]] || { usage >&2; exit 2; }
PREFLIGHT="$(realpath "${PREFLIGHT}")"
OUTPUT="$(realpath -m "${OUTPUT}")"
case "${OUTPUT}" in
  "${ROOT}"/results/*) ;;
  *) echo "--output must be a new directory below ${ROOT}/results" >&2; exit 2 ;;
esac
[[ ! -e "${OUTPUT}" ]] || { echo "--output already exists" >&2; exit 2; }
grep -Eq '"schema"[[:space:]]*:[[:space:]]*"rlf-vast-rtx-pro-6000-preflight-v1"' "${PREFLIGHT}" || {
  echo "unsupported preflight schema" >&2; exit 2;
}
grep -Eq '"ready"[[:space:]]*:[[:space:]]*true' "${PREFLIGHT}" || {
  echo "build requires a real ready=true hardware preflight" >&2; exit 2;
}
grep -Eq '"test_doubles"[[:space:]]*:[[:space:]]*false' "${PREFLIGHT}" || {
  echo "test-double preflight cannot authorize a physical build report" >&2; exit 2;
}
for command_name in awk cmake ctest find g++ ninja nproc nvcc sha256sum sort tee xargs; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "required command is missing: ${command_name}" >&2
    exit 2
  }
done

mkdir -p "${OUTPUT}"
cd "${ROOT}"
cmake --preset ubuntu-rtx-pro-6000-cuda 2>&1 | tee "${OUTPUT}/configure.log"
cmake --build --preset ubuntu-rtx-pro-6000-cuda \
  --parallel "${RLF_BUILD_JOBS:-$(nproc)}" 2>&1 | tee "${OUTPUT}/build.log"
ctest --preset ubuntu-rtx-pro-6000-cuda --output-on-failure 2>&1 | tee "${OUTPUT}/ctest.log"

BUILD_DIR="${ROOT}/build/ubuntu-rtx-pro-6000-cuda"
BIN="${BUILD_DIR}/solstice"
CACHE="${BUILD_DIR}/CMakeCache.txt"
[[ -x "${BIN}" && -f "${CACHE}" ]] || { echo "expected sm_120 build outputs are missing" >&2; exit 1; }
grep -Eq '^RLF_ENABLE_CUDA:BOOL=ON$' "${CACHE}"
grep -Eq '^CMAKE_CUDA_ARCHITECTURES:[^=]+=120$' "${CACHE}"
grep -Eq '^RLF_WARNINGS_AS_ERRORS:BOOL=ON$' "${CACHE}"
grep -Eq '^CMAKE_BUILD_TYPE:STRING=Release$' "${CACHE}"
"${BIN}" profile-info --profile rtx-pro-6000-96g >"${OUTPUT}/profile_info.txt"
"${BIN}" device-info --profile rtx-pro-6000-96g --backend cuda >"${OUTPUT}/device_info.txt"
grep -Eq '^backend=cuda-persistent$' "${OUTPUT}/device_info.txt"
grep -Eq '^available=true$' "${OUTPUT}/device_info.txt"
grep -Eq '^profile_fits_device=true$' "${OUTPUT}/device_info.txt"

"${ROOT}/scripts/create_source_manifest.sh" "${OUTPUT}/source_manifest.tsv" \
  >"${OUTPUT}/source_manifest_stdout.txt"
PREFLIGHT_SHA="$(sha256sum -- "${PREFLIGHT}" | awk '{ print $1 }')"
SOURCE_SHA="$(sha256sum -- "${OUTPUT}/source_manifest.tsv" | awk '{ print $1 }')"
BIN_SHA="$(sha256sum -- "${BIN}" | awk '{ print $1 }')"
printf '{\n' >"${OUTPUT}/build_evidence.json"
printf '  "schema": "rlf-vast-rtx-pro-6000-build-v1",\n' >>"${OUTPUT}/build_evidence.json"
printf '  "passed": true,\n' >>"${OUTPUT}/build_evidence.json"
printf '  "training_performed": false,\n' >>"${OUTPUT}/build_evidence.json"
printf '  "cmake_preset": "ubuntu-rtx-pro-6000-cuda",\n' >>"${OUTPUT}/build_evidence.json"
printf '  "cuda_architecture": "sm_120",\n' >>"${OUTPUT}/build_evidence.json"
printf '  "preflight_sha256": "%s",\n' "${PREFLIGHT_SHA}" >>"${OUTPUT}/build_evidence.json"
printf '  "source_manifest_sha256": "%s",\n' "${SOURCE_SHA}" >>"${OUTPUT}/build_evidence.json"
printf '  "solstice_binary_sha256": "%s"\n' "${BIN_SHA}" >>"${OUTPUT}/build_evidence.json"
printf '}\n' >>"${OUTPUT}/build_evidence.json"
find "${OUTPUT}" -type f ! -name complete_build_artifacts.sha256 -print0 | \
  sort -z | xargs -0 sha256sum >"${OUTPUT}/complete_build_artifacts.sha256"
echo "RTX PRO 6000 build and full test suite passed. No training was performed."
