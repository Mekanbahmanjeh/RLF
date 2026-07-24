#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${ROOT}/scripts/setup_ubuntu_3090.sh"

cat <<'MSG'
H100 notes:
- Use a current data-center NVIDIA driver and CUDA Toolkit 12.x or newer.
- The CMake preset targets compute capability 9.0 (sm_90).
- The full general-h100 ceiling requires 1 TiB host RAM and at least 2 TiB free
  fast NVMe for streamed transactional checkpoints. Smaller machines must use
  measured lower curricula; do not treat the capacity estimate as allocation evidence.
- Validate ECC, persistence mode, thermals, and available GPU memory with nvidia-smi.
MSG
