#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RLF_EFFICIENCY_CAMPAIGN_VARIANT=optimized
exec "${ROOT}/scripts/run_efficiency_baseline_rtx_pro_6000.sh" "$@"
