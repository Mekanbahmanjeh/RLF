#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  create_training_artifact_manifest.sh --checkpoint FILE --ledger FILE \
    --source-manifest FILE --data-audit FILE --resource-summary FILE \
    --vram-trace FILE --environment FILE --checkpoint-inspection FILE \
    --readiness-report FILE --output FILE
  create_training_artifact_manifest.sh --verify FILE

The manifest is a canonical, hash-linked inventory of the trained checkpoint,
authorizing readiness report, audited data identity, source tree, environment,
and measured VRAM artifacts.
Verification fails if any referenced artifact or the manifest sidecar changed.
EOF
}

VERIFY=""
CHECKPOINT=""
LEDGER=""
SOURCE_MANIFEST=""
DATA_AUDIT=""
RESOURCE_SUMMARY=""
VRAM_TRACE=""
ENVIRONMENT=""
CHECKPOINT_INSPECTION=""
READINESS_REPORT=""
OUTPUT=""

while (($# > 0)); do
  case "$1" in
    --verify) VERIFY="${2:?--verify requires a file}"; shift 2 ;;
    --checkpoint) CHECKPOINT="${2:?--checkpoint requires a file}"; shift 2 ;;
    --ledger) LEDGER="${2:?--ledger requires a file}"; shift 2 ;;
    --source-manifest) SOURCE_MANIFEST="${2:?--source-manifest requires a file}"; shift 2 ;;
    --data-audit) DATA_AUDIT="${2:?--data-audit requires a file}"; shift 2 ;;
    --resource-summary) RESOURCE_SUMMARY="${2:?--resource-summary requires a file}"; shift 2 ;;
    --vram-trace) VRAM_TRACE="${2:?--vram-trace requires a file}"; shift 2 ;;
    --environment) ENVIRONMENT="${2:?--environment requires a file}"; shift 2 ;;
    --checkpoint-inspection) CHECKPOINT_INSPECTION="${2:?--checkpoint-inspection requires a file}"; shift 2 ;;
    --readiness-report) READINESS_REPORT="${2:?--readiness-report requires a file}"; shift 2 ;;
    --output) OUTPUT="${2:?--output requires a file}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v sha256sum >/dev/null 2>&1 || { echo "sha256sum was not found" >&2; exit 2; }

verify_manifest() {
  local manifest="$1"
  [[ -f "${manifest}" ]] || { echo "manifest not found: ${manifest}" >&2; return 2; }
  [[ -f "${manifest}.sha256" ]] || { echo "manifest sidecar not found" >&2; return 2; }
  (cd "$(dirname "${manifest}")" && sha256sum -c "$(basename "${manifest}").sha256") >/dev/null
  local expected_count=0 schema=""
  if grep -Fqx '# rlf-training-artifact-manifest-v1' "${manifest}"; then
    expected_count=9; schema=v1
  elif grep -Fqx '# rlf-training-artifact-manifest-v2' "${manifest}"; then
    expected_count=10; schema=v2
  else
    echo "unsupported training artifact manifest schema" >&2
    return 1
  fi
  local -A required=(
    [checkpoint]=1 [ledger]=1 [source_manifest]=1 [data_audit]=1
    [resource_summary]=1 [vram_trace]=1 [environment]=1
    [checkpoint_inspection]=1 [readiness_report]=1
  )
  [[ "${schema}" != v2 ]] || required[telemetry]=1
  local -A seen=()
  local count=0
  while IFS=$'\t' read -r kind expected_hash expected_bytes path; do
    [[ -n "${kind}" && "${kind}" != \#* ]] || continue
    [[ "${kind}" =~ ^[a-z_]+$ ]] || { echo "invalid artifact kind: ${kind}" >&2; return 1; }
    [[ -n "${required[${kind}]+present}" ]] || { echo "unexpected ${schema} artifact kind: ${kind}" >&2; return 1; }
    [[ -z "${seen[${kind}]+present}" ]] || { echo "duplicate artifact kind: ${kind}" >&2; return 1; }
    seen["${kind}"]=1
    [[ -n "${path}" && -f "${path}" ]] || { echo "missing ${kind} artifact: ${path}" >&2; return 1; }
    local actual_hash actual_bytes
    actual_hash="$(sha256sum -- "${path}" | awk '{print $1}')"
    actual_bytes="$(stat -c '%s' -- "${path}")"
    [[ "${actual_hash}" == "${expected_hash}" ]] || { echo "hash mismatch: ${path}" >&2; return 1; }
    [[ "${actual_bytes}" == "${expected_bytes}" ]] || { echo "size mismatch: ${path}" >&2; return 1; }
    count=$((count + 1))
  done <"${manifest}"
  ((count == expected_count)) || { echo "expected exactly ${expected_count} artifact records, found ${count}" >&2; return 1; }
  local required_kind
  for required_kind in "${!required[@]}"; do
    [[ -n "${seen[${required_kind}]+present}" ]] || { echo "missing ${schema} artifact kind: ${required_kind}" >&2; return 1; }
  done
  printf 'Verified training artifact manifest: %s\n' "${manifest}"
}

if [[ -n "${VERIFY}" ]]; then
  [[ -z "${CHECKPOINT}${LEDGER}${SOURCE_MANIFEST}${DATA_AUDIT}${RESOURCE_SUMMARY}${VRAM_TRACE}${ENVIRONMENT}${CHECKPOINT_INSPECTION}${READINESS_REPORT}${OUTPUT}" ]] || {
    echo "--verify cannot be combined with creation options" >&2
    exit 2
  }
  verify_manifest "$(realpath -m "${VERIFY}")"
  exit $?
fi

for assignment in \
  "checkpoint:${CHECKPOINT}" "ledger:${LEDGER}" \
  "source_manifest:${SOURCE_MANIFEST}" "data_audit:${DATA_AUDIT}" \
  "resource_summary:${RESOURCE_SUMMARY}" "vram_trace:${VRAM_TRACE}" \
  "environment:${ENVIRONMENT}" "checkpoint_inspection:${CHECKPOINT_INSPECTION}" \
  "readiness_report:${READINESS_REPORT}"; do
  kind="${assignment%%:*}"
  path="${assignment#*:}"
  [[ -n "${path}" && -f "${path}" ]] || { echo "${kind} file not found: ${path}" >&2; exit 2; }
done
[[ -n "${OUTPUT}" ]] || { echo "--output is required" >&2; exit 2; }

grep -Eq '"schema"[[:space:]]*:[[:space:]]*"rlf-(h100|h200)-vram-v1"' "${RESOURCE_SUMMARY}" || {
  echo "resource summary has the wrong schema" >&2; exit 1;
}
grep -Eq '"within_limit"[[:space:]]*:[[:space:]]*true' "${RESOURCE_SUMMARY}" || {
  echo "resource summary did not pass the 76 GiB limit" >&2; exit 1;
}
grep -Eq '"sampler_ok"[[:space:]]*:[[:space:]]*true' "${RESOURCE_SUMMARY}" || {
  echo "resource telemetry was not valid" >&2; exit 1;
}
grep -Eq '"command_exit_code"[[:space:]]*:[[:space:]]*0' "${RESOURCE_SUMMARY}" || {
  echo "training command did not exit successfully" >&2; exit 1;
}
grep -Eq '"ready"[[:space:]]*:[[:space:]]*true' "${READINESS_REPORT}" || {
  echo "the target-machine readiness report did not pass" >&2; exit 1;
}
grep -Eq '"test_doubles"[[:space:]]*:[[:space:]]*false' "${READINESS_REPORT}" || {
  echo "test-double readiness cannot authorize an artifact manifest" >&2; exit 1;
}
if grep -Eq '"profile"[[:space:]]*:[[:space:]]*"general-h100"' "${READINESS_REPORT}"; then
  grep -Eq '"schema"[[:space:]]*:[[:space:]]*"rlf-h100-vram-v1"' \
    "${RESOURCE_SUMMARY}" || {
      echo "H100 readiness requires H100 resource evidence" >&2; exit 1;
    }
elif grep -Eq '"profile"[[:space:]]*:[[:space:]]*"general-h200-141g-30t"' \
    "${READINESS_REPORT}"; then
  grep -Eq '"schema"[[:space:]]*:[[:space:]]*"rlf-h200-vram-v1"' \
    "${RESOURCE_SUMMARY}" || {
      echo "H200 readiness requires H200 resource evidence" >&2; exit 1;
    }
else
  echo "readiness report does not bind a supported general profile" >&2
  exit 1
fi

OUTPUT="$(realpath -m "${OUTPUT}")"
mkdir -p "$(dirname "${OUTPUT}")"
WORK="$(mktemp "$(dirname "${OUTPUT}")/.training-manifest.XXXXXX")"
trap 'rm -f -- "${WORK}"' EXIT
printf '# rlf-training-artifact-manifest-v1\n# kind\tsha256\tbytes\tabsolute_path\n' >"${WORK}"
for assignment in \
  "checkpoint:${CHECKPOINT}" "ledger:${LEDGER}" \
  "source_manifest:${SOURCE_MANIFEST}" "data_audit:${DATA_AUDIT}" \
  "resource_summary:${RESOURCE_SUMMARY}" "vram_trace:${VRAM_TRACE}" \
  "environment:${ENVIRONMENT}" "checkpoint_inspection:${CHECKPOINT_INSPECTION}" \
  "readiness_report:${READINESS_REPORT}"; do
  kind="${assignment%%:*}"
  path="$(realpath "${assignment#*:}")"
  hash="$(sha256sum -- "${path}" | awk '{print $1}')"
  bytes="$(stat -c '%s' -- "${path}")"
  printf '%s\t%s\t%s\t%s\n' "${kind}" "${hash}" "${bytes}" "${path}" >>"${WORK}"
done
mv -f -- "${WORK}" "${OUTPUT}"
(cd "$(dirname "${OUTPUT}")" && sha256sum -- "$(basename "${OUTPUT}")" >"$(basename "${OUTPUT}").sha256")
verify_manifest "${OUTPUT}"
