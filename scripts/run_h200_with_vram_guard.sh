#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RLF_REQUIRED_GPU_NAME=H200
export RLF_DEFAULT_VRAM_LIMIT_MIB=135168
export RLF_MAXIMUM_VRAM_LIMIT_MIB=135168
export RLF_VRAM_RESOURCE_SCHEMA=rlf-h200-vram-v1
exec bash "${ROOT}/scripts/run_h100_with_vram_guard.sh" "$@"
