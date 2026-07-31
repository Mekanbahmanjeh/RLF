#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RLF_RTX_PRO_6000_PROFILE=video-rtx-pro-6000-96g \
  exec "${ROOT}/scripts/build_ubuntu_rtx_pro_6000.sh" "$@"
