#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

usage() {
  cat <<'EOF'
Usage:
  artifact_manifest.sh collect --source DIR --manifest FILE
  artifact_manifest.sh verify  --source DIR --manifest FILE

Creates or verifies a portable SHA-256/size manifest for every regular file.
Symlinks, special files, tabs, and newlines in paths are rejected. Keep FILE
outside DIR so the evidence does not include itself.
EOF
}

ACTION=""
SOURCE=""
MANIFEST=""
while (($# > 0)); do
  case "$1" in
    collect|verify) [[ -z "${ACTION}" ]] || { usage >&2; exit 2; }; ACTION="$1"; shift ;;
    --source) SOURCE="${2:?--source requires a directory}"; shift 2 ;;
    --manifest) MANIFEST="${2:?--manifest requires a file}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "${ACTION}" && -d "${SOURCE}" && -n "${MANIFEST}" ]] || { usage >&2; exit 2; }
for command_name in find sha256sum stat sort realpath cmp; do
  command -v "${command_name}" >/dev/null 2>&1 || { echo "missing command: ${command_name}" >&2; exit 2; }
done
SOURCE="$(realpath "${SOURCE}")"
MANIFEST="$(realpath -m "${MANIFEST}")"
case "${MANIFEST}" in
  "${SOURCE}"|"${SOURCE}"/*) echo "manifest must be outside the artifact source" >&2; exit 2 ;;
esac

validate_tree() {
  local bad
  bad="$(find "${SOURCE}" -type l -print -quit)"
  [[ -z "${bad}" ]] || { echo "artifact tree contains symlink: ${bad}" >&2; exit 1; }
  bad="$(find "${SOURCE}" ! -type d ! -type f -print -quit)"
  [[ -z "${bad}" ]] || { echo "artifact tree contains special entry: ${bad}" >&2; exit 1; }
}

generate_manifest() {
  local destination="$1"
  local work list temporary path relative hash before after bytes
  work="$(mktemp -d "$(dirname "${destination}")/.artifact-manifest.XXXXXX")"
  list="${work}/paths"
  temporary="${work}/manifest"
  trap 'rm -rf -- "${work}"' RETURN
  find "${SOURCE}" -type f -print0 | sort -z >"${list}"
  printf '# rlf-vast-artifact-manifest-v1\n# sha256\tbytes\tpath\n' >"${temporary}"
  while IFS= read -r -d '' path; do
    relative="${path#"${SOURCE}"/}"
    [[ -n "${relative}" && "${relative}" != *$'\t'* && "${relative}" != *$'\n'* && \
       "${relative}" != *$'\r'* ]] || { echo "unsafe artifact path" >&2; exit 1; }
    before="$(stat -c '%s:%Y' -- "${path}")"
    hash="$(sha256sum -- "${path}" | awk '{ print $1 }')"
    after="$(stat -c '%s:%Y' -- "${path}")"
    [[ "${before}" == "${after}" ]] || { echo "artifact changed while hashing: ${relative}" >&2; exit 1; }
    bytes="${before%%:*}"
    printf '%s\t%s\t%s\n' "${hash}" "${bytes}" "${relative}" >>"${temporary}"
  done <"${list}"
  mv -f -- "${temporary}" "${destination}"
  trap - RETURN
  rm -rf -- "${work}"
}

validate_tree
if [[ "${ACTION}" == collect ]]; then
  [[ ! -e "${MANIFEST}" && ! -e "${MANIFEST}.sha256" ]] || {
    echo "manifest output already exists" >&2; exit 2;
  }
  mkdir -p "$(dirname "${MANIFEST}")"
  generate_manifest "${MANIFEST}"
  MANIFEST_HASH="$(sha256sum -- "${MANIFEST}" | awk '{ print $1 }')"
  printf '%s  %s\n' "${MANIFEST_HASH}" "$(basename "${MANIFEST}")" >"${MANIFEST}.sha256"
  printf 'artifact_manifest=%s\nartifact_manifest_sha256=%s\n' "${MANIFEST}" "${MANIFEST_HASH}"
  exit 0
fi

[[ -f "${MANIFEST}" && -f "${MANIFEST}.sha256" ]] || {
  echo "manifest and manifest SHA-256 sidecar are required" >&2; exit 2;
}
EXPECTED_HASH="$(awk 'NR == 1 { print $1 }' "${MANIFEST}.sha256")"
[[ "${EXPECTED_HASH}" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid manifest SHA-256 sidecar" >&2; exit 1; }
ACTUAL_HASH="$(sha256sum -- "${MANIFEST}" | awk '{ print $1 }')"
[[ "${ACTUAL_HASH}" == "${EXPECTED_HASH}" ]] || { echo "manifest SHA-256 mismatch" >&2; exit 1; }
VERIFY_TEMP="$(mktemp "$(dirname "${MANIFEST}")/.artifact-verify.XXXXXX")"
trap 'rm -f -- "${VERIFY_TEMP}"' EXIT
generate_manifest "${VERIFY_TEMP}"
cmp --silent "${MANIFEST}" "${VERIFY_TEMP}" || {
  diff -u -- "${MANIFEST}" "${VERIFY_TEMP}" >&2 || true
  echo "artifact verification failed" >&2
  exit 1
}
printf 'verified_artifact_manifest=%s\nartifact_manifest_sha256=%s\n' "${MANIFEST}" "${ACTUAL_HASH}"
