#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: run_checkpoint_transaction.sh --checkpoint FILE --failure-root DIR [--preserve FILE ...] [--check-only] -- COMMAND [ARGS...]" >&2
}

CHECKPOINT=""; FAILURE_ROOT=""; CHECK_ONLY=false; PRESERVE=()
while (($# > 0)); do
  case "$1" in
    --checkpoint) CHECKPOINT="${2:?}"; shift 2 ;;
    --failure-root) FAILURE_ROOT="${2:?}"; shift 2 ;;
    --preserve) PRESERVE+=("${2:?}"); shift 2 ;;
    --check-only) CHECK_ONLY=true; shift ;;
    --) shift; break ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n "${CHECKPOINT}" && -n "${FAILURE_ROOT}" ]] || { usage; exit 2; }
[[ "${CHECK_ONLY}" == true || $# -gt 0 ]] || { usage; exit 2; }
command -v flock >/dev/null 2>&1 || { echo "flock was not found" >&2; exit 2; }
command -v sha256sum >/dev/null 2>&1 || { echo "sha256sum was not found" >&2; exit 2; }

CHECKPOINT="$(realpath -m "${CHECKPOINT}")"
FAILURE_ROOT="$(realpath -m "${FAILURE_ROOT}")"
PARENT="$(dirname "${CHECKPOINT}")"
mkdir -p "${PARENT}" "${FAILURE_ROOT}"
BACKUP="${CHECKPOINT}.attempt-backup"
STATE="${CHECKPOINT}.attempt-state"
TAINT="${CHECKPOINT}.attempt-tainted"
LOCK="${CHECKPOINT}.training.lock"
[[ ! -L "${CHECKPOINT}" && ! -L "${BACKUP}" && ! -L "${STATE}" && \
   ! -L "${TAINT}" && ! -L "${LOCK}" ]] || {
  echo "checkpoint transaction paths may not be symbolic links" >&2; exit 2;
}
exec 9>"${LOCK}"
flock -n 9 || { echo "another process holds the checkpoint training lock" >&2; exit 7; }
[[ ! -e "${TAINT}" ]] || { echo "checkpoint is tainted by an unrecoverable attempt: ${TAINT}" >&2; exit 8; }

state_value() { awk -F= -v key="$2" '$1 == key { print substr($0, length(key) + 2) }' "$1"; }
archive_attempt() {
  local reason="$1" command_status="$2" pre_hash="$3" post_hash="$4"
  local directory="${FAILURE_ROOT}/$(date -u +%Y%m%dT%H%M%SZ)-$$-${reason}"
  local failed=0
  mkdir -p "${directory}" || failed=1
  printf 'schema=rlf-checkpoint-attempt-failure-v1\nreason=%s\ncommand_exit_code=%s\ncheckpoint=%s\npre_checkpoint_sha256=%s\npost_checkpoint_sha256=%s\n' \
    "${reason}" "${command_status}" "${CHECKPOINT}" "${pre_hash}" "${post_hash}" >"${directory}/failure.txt" || failed=1
  if [[ -f "${STATE}" ]]; then cp -- "${STATE}" "${directory}/attempt-state.txt" || failed=1; fi
  local index=0 path
  for path in "${PRESERVE[@]}"; do
    if [[ -f "${path}" ]]; then
      cp -- "${path}" "${directory}/$(printf '%02d' "${index}")-$(basename "${path}")" || failed=1
    fi
    index=$((index + 1))
  done
  return "${failed}"
}
restore_attempt() {
  local preexisted="$1" expected_hash="${2:-}"
  if [[ "${preexisted}" == true ]]; then
    [[ -f "${BACKUP}" ]] || return 1
    if [[ -n "${expected_hash}" && \
          "$(sha256sum -- "${BACKUP}" | awk '{print $1}')" != "${expected_hash}" ]]; then
      return 1
    fi
    mv -f -- "${BACKUP}" "${CHECKPOINT}"
  else
    if [[ -e "${CHECKPOINT}" ]]; then
      [[ -f "${CHECKPOINT}" ]] || return 1
      rm -f -- "${CHECKPOINT}"
    fi
    rm -f -- "${BACKUP}"
  fi
  rm -f -- "${STATE}"
}

if [[ -e "${BACKUP}" || -e "${STATE}" ]]; then
  if [[ "${CHECK_ONLY}" == true ]]; then
    echo "an interrupted checkpoint transaction requires recovery before preflight" >&2
    exit 8
  fi
  PREEXISTED=""
  STALE_PRE_HASH=""
  STALE_PHASE=""
  if [[ -f "${STATE}" ]]; then
    PREEXISTED="$(state_value "${STATE}" checkpoint_preexisted)"
    STALE_PRE_HASH="$(state_value "${STATE}" pre_checkpoint_sha256)"
    STALE_PHASE="$(state_value "${STATE}" phase)"
  fi
  if [[ "${STALE_PHASE}" == committed ]]; then
    STALE_POST_HASH="$(state_value "${STATE}" post_checkpoint_sha256)"
    if [[ "${STALE_POST_HASH}" =~ ^[0-9a-f]{64}$ && -f "${CHECKPOINT}" && \
          "$(sha256sum -- "${CHECKPOINT}" | awk '{print $1}')" == "${STALE_POST_HASH}" ]]; then
      rm -f -- "${BACKUP}" "${STATE}"
    else
      printf 'schema=rlf-checkpoint-attempt-taint-v1\nreason=invalid-committed-state\n' >"${TAINT}"
      echo "committed checkpoint transaction identity is invalid" >&2
      exit 8
    fi
  elif [[ -f "${STATE}" && "${STALE_PHASE}" != active ]]; then
    printf 'schema=rlf-checkpoint-attempt-taint-v1\nreason=invalid-stale-phase\n' >"${TAINT}"
    echo "stale checkpoint transaction phase is invalid" >&2
    exit 8
  fi
  if [[ "${STALE_PHASE}" == committed ]]; then
    :
  else
    if [[ -f "${BACKUP}" ]]; then PREEXISTED=true; fi
    if [[ "${PREEXISTED}" != true && "${PREEXISTED}" != false ]]; then
      printf 'schema=rlf-checkpoint-attempt-taint-v1\nreason=invalid-stale-state\n' >"${TAINT}"
      echo "stale checkpoint transaction state is invalid" >&2
      exit 8
    fi
    POST_HASH=""; [[ ! -f "${CHECKPOINT}" ]] || POST_HASH="$(sha256sum -- "${CHECKPOINT}" | awk '{print $1}')"
    if ! archive_attempt stale-recovery 255 "" "${POST_HASH}"; then
      echo "warning: failed to preserve all stale-attempt evidence" >&2
    fi
    if ! restore_attempt "${PREEXISTED}" "${STALE_PRE_HASH}"; then
      printf 'schema=rlf-checkpoint-attempt-taint-v1\nreason=stale-rollback-failed\n' >"${TAINT}"
      echo "unable to recover stale checkpoint transaction" >&2
      exit 8
    fi
  fi
fi

PROBE_SOURCE="$(mktemp "${PARENT}/.checkpoint-link-source.XXXXXX")"
PROBE_LINK="${PROBE_SOURCE}.link"
printf 'hard-link-probe\n' >"${PROBE_SOURCE}"
if ! ln -- "${PROBE_SOURCE}" "${PROBE_LINK}"; then
  rm -f -- "${PROBE_SOURCE}" "${PROBE_LINK}"
  echo "checkpoint filesystem does not support required hard-link rollback" >&2
  exit 2
fi
rm -f -- "${PROBE_SOURCE}" "${PROBE_LINK}"
if [[ "${CHECK_ONLY}" == true ]]; then
  echo "Checkpoint transaction preflight passed. No checkpoint was changed."
  exit 0
fi

PREEXISTED=false; PRE_HASH=""
if [[ -e "${CHECKPOINT}" ]]; then
  [[ -f "${CHECKPOINT}" ]] || { echo "checkpoint target is not a regular file" >&2; exit 2; }
  PREEXISTED=true
  PRE_HASH="$(sha256sum -- "${CHECKPOINT}" | awk '{print $1}')"
  ln -- "${CHECKPOINT}" "${BACKUP}"
fi
TEMPORARY_STATE="$(mktemp "${PARENT}/.checkpoint-attempt.XXXXXX")"
{
  printf 'schema=rlf-checkpoint-attempt-v1\n'
  printf 'phase=active\n'
  printf 'checkpoint=%s\ncheckpoint_preexisted=%s\npre_checkpoint_sha256=%s\n' \
    "${CHECKPOINT}" "${PREEXISTED}" "${PRE_HASH}"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${TEMPORARY_STATE}"
mv -f -- "${TEMPORARY_STATE}" "${STATE}"

set +e
"$@"
COMMAND_STATUS=$?
set -e
POST_HASH=""; [[ ! -f "${CHECKPOINT}" ]] || POST_HASH="$(sha256sum -- "${CHECKPOINT}" | awk '{print $1}')"
if ((COMMAND_STATUS != 0)); then
  if ! archive_attempt command-failed "${COMMAND_STATUS}" "${PRE_HASH}" "${POST_HASH}"; then
    echo "warning: failed to preserve all failed-attempt evidence" >&2
  fi
  if ! restore_attempt "${PREEXISTED}" "${PRE_HASH}"; then
    printf 'schema=rlf-checkpoint-attempt-taint-v1\nreason=rollback-failed\ncommand_exit_code=%s\n' "${COMMAND_STATUS}" >"${TAINT}"
    echo "checkpoint rollback failed; checkpoint is tainted" >&2
    exit 8
  fi
  echo "checkpoint transaction rolled back after exit ${COMMAND_STATUS}" >&2
  exit "${COMMAND_STATUS}"
fi
if [[ ! -f "${CHECKPOINT}" || ! "${POST_HASH}" =~ ^[0-9a-f]{64}$ ]]; then
  if ! archive_attempt missing-success-checkpoint 8 "${PRE_HASH}" "${POST_HASH}"; then
    echo "warning: failed to preserve missing-checkpoint evidence" >&2
  fi
  if ! restore_attempt "${PREEXISTED}" "${PRE_HASH}"; then
    printf 'schema=rlf-checkpoint-attempt-taint-v1\nreason=missing-success-checkpoint-rollback-failed\n' >"${TAINT}"
  fi
  echo "checkpoint transaction command succeeded without a valid checkpoint" >&2
  exit 8
fi
COMMITTED_STATE="$(mktemp "${PARENT}/.checkpoint-committed.XXXXXX")"
{
  printf 'schema=rlf-checkpoint-attempt-v1\nphase=committed\n'
  printf 'checkpoint=%s\ncheckpoint_preexisted=%s\npre_checkpoint_sha256=%s\npost_checkpoint_sha256=%s\n' \
    "${CHECKPOINT}" "${PREEXISTED}" "${PRE_HASH}" "${POST_HASH}"
} >"${COMMITTED_STATE}"
mv -f -- "${COMMITTED_STATE}" "${STATE}"
rm -f -- "${BACKUP}" "${STATE}"
exit 0
