#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:?Usage: create_source_manifest.sh OUTPUT}"
OUTPUT="$(realpath -m "${OUTPUT}")"

case "${OUTPUT}" in
  "${ROOT}"/*) ;;
  *) echo "source manifest output must be below the repository root" >&2; exit 2 ;;
esac

command -v sha256sum >/dev/null 2>&1 || {
  echo "sha256sum was not found" >&2
  exit 2
}

mkdir -p "$(dirname "${OUTPUT}")"
WORK="$(mktemp -d "$(dirname "${OUTPUT}")/.source-manifest.XXXXXX")"
trap 'rm -rf -- "${WORK}"' EXIT
LIST="${WORK}/paths"
MANIFEST="${WORK}/manifest"

cd "${ROOT}"
find . -maxdepth 1 -type f -print0 >"${LIST}"
SOURCE_DIRECTORIES=()
for directory in .github benchmarks cmake configs data_templates docs examples include scripts src tests; do
  [[ -d "${directory}" ]] && SOURCE_DIRECTORIES+=("${directory}")
done
if ((${#SOURCE_DIRECTORIES[@]} > 0)); then
  find "${SOURCE_DIRECTORIES[@]}" \
    \( -path 'benchmarks/efficiency_campaign/generated' -o \
       -path 'benchmarks/efficiency_campaign/evaluation_v1' \) -prune -o \
    -type f ! -name '*.pyc' ! -name '*.tmp' -print0 >>"${LIST}"
fi
LC_ALL=C sort -z -o "${LIST}" "${LIST}"

printf '# rlf-source-manifest-v1\n# sha256\tbytes\tpath\n' >"${MANIFEST}"
while IFS= read -r -d '' path; do
  relative="${path#./}"
  hash="$(sha256sum -- "${relative}" | awk '{print $1}')"
  bytes="$(stat -c '%s' -- "${relative}")"
  printf '%s\t%s\t%s\n' "${hash}" "${bytes}" "${relative}" >>"${MANIFEST}"
done <"${LIST}"

mv -f -- "${MANIFEST}" "${OUTPUT}"
sha256sum -- "${OUTPUT}" >"${OUTPUT}.sha256"
printf 'Source manifest: %s\n' "${OUTPUT}"
