#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"; trap 'rm -rf -- "${WORK}"' EXIT
CHECKPOINT="${WORK}/model.rlfsp"; FAILURES="${WORK}/failures"

printf 'original\n' >"${CHECKPOINT}"
ORIGINAL_SHA="$(sha256sum -- "${CHECKPOINT}" | awk '{print $1}')"
set +e
"${ROOT}/scripts/run_checkpoint_transaction.sh" --checkpoint "${CHECKPOINT}" \
  --failure-root "${FAILURES}" -- bash -c 'printf "mutated\n" >"$1.tmp"; mv -f -- "$1.tmp" "$1"; exit 4' _ "${CHECKPOINT}"
STATUS=$?
set -e
[[ "${STATUS}" -eq 4 && "$(sha256sum -- "${CHECKPOINT}" | awk '{print $1}')" == "${ORIGINAL_SHA}" ]]
grep -Rqx 'reason=command-failed' "${FAILURES}"

rm -f -- "${CHECKPOINT}"
set +e
"${ROOT}/scripts/run_checkpoint_transaction.sh" --checkpoint "${CHECKPOINT}" \
  --failure-root "${FAILURES}" -- bash -c 'printf "partial\n" >"$1.tmp"; mv -f -- "$1.tmp" "$1"; exit 5' _ "${CHECKPOINT}"
STATUS=$?
set -e
[[ "${STATUS}" -eq 5 && ! -e "${CHECKPOINT}" ]]

"${ROOT}/scripts/run_checkpoint_transaction.sh" --checkpoint "${CHECKPOINT}" \
  --failure-root "${FAILURES}" -- bash -c 'printf "complete\n" >"$1.tmp"; mv -f -- "$1.tmp" "$1"' _ "${CHECKPOINT}"
grep -qx complete "${CHECKPOINT}"
[[ ! -e "${CHECKPOINT}.attempt-backup" && ! -e "${CHECKPOINT}.attempt-state" ]]

printf 'stable\n' >"${CHECKPOINT}"
STABLE_SHA="$(sha256sum -- "${CHECKPOINT}" | awk '{print $1}')"
ln -- "${CHECKPOINT}" "${CHECKPOINT}.attempt-backup"
printf 'schema=rlf-checkpoint-attempt-v1\nphase=active\ncheckpoint=%s\ncheckpoint_preexisted=true\npre_checkpoint_sha256=%s\n' \
  "${CHECKPOINT}" "${STABLE_SHA}" >"${CHECKPOINT}.attempt-state"
printf 'interrupted-mutation\n' >"${CHECKPOINT}.tmp"
mv -f -- "${CHECKPOINT}.tmp" "${CHECKPOINT}"
set +e
"${ROOT}/scripts/run_checkpoint_transaction.sh" --checkpoint "${CHECKPOINT}" \
  --failure-root "${FAILURES}" --check-only --
STATUS=$?
set -e
[[ "${STATUS}" -eq 8 ]]
"${ROOT}/scripts/run_checkpoint_transaction.sh" --checkpoint "${CHECKPOINT}" \
  --failure-root "${FAILURES}" -- true
[[ "$(sha256sum -- "${CHECKPOINT}" | awk '{print $1}')" == "${STABLE_SHA}" ]]
grep -Rqx 'reason=stale-recovery' "${FAILURES}"

printf 'commit-old\n' >"${WORK}/committed.rlfsp"
ln -- "${WORK}/committed.rlfsp" "${WORK}/committed.rlfsp.attempt-backup"
printf 'commit-new\n' >"${WORK}/committed.rlfsp.tmp"
mv -f -- "${WORK}/committed.rlfsp.tmp" "${WORK}/committed.rlfsp"
COMMITTED_SHA="$(sha256sum -- "${WORK}/committed.rlfsp" | awk '{print $1}')"
printf 'schema=rlf-checkpoint-attempt-v1\nphase=committed\ncheckpoint=%s\ncheckpoint_preexisted=true\npre_checkpoint_sha256=unused\npost_checkpoint_sha256=%s\n' \
  "${WORK}/committed.rlfsp" "${COMMITTED_SHA}" >"${WORK}/committed.rlfsp.attempt-state"
"${ROOT}/scripts/run_checkpoint_transaction.sh" --checkpoint "${WORK}/committed.rlfsp" \
  --failure-root "${FAILURES}" -- bash -c 'cp -- "$1" "$1.tmp"; mv -f -- "$1.tmp" "$1"' _ "${WORK}/committed.rlfsp"
[[ "$(sha256sum -- "${WORK}/committed.rlfsp" | awk '{print $1}')" == "${COMMITTED_SHA}" ]]

printf 'tamper-base\n' >"${WORK}/tampered.rlfsp"
set +e
"${ROOT}/scripts/run_checkpoint_transaction.sh" --checkpoint "${WORK}/tampered.rlfsp" \
  --failure-root "${FAILURES}" -- bash -c 'printf "in-place\n" >"$1"; exit 9' _ "${WORK}/tampered.rlfsp" >/dev/null 2>&1
STATUS=$?
set -e
[[ "${STATUS}" -eq 8 && -f "${WORK}/tampered.rlfsp.attempt-tainted" ]]
