#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${REPO_DIR}/scripts/wsl_tensorflow_env.sh"

cd "${REPO_DIR}"
export CNN_BATCH_SIZE="${CNN_BATCH_SIZE:-32}"
exec python cnn.py "$@"
