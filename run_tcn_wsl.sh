#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${REPO_DIR}/scripts/wsl_tensorflow_env.sh"

cd "${REPO_DIR}"
export TCN_BATCH_SIZE="${TCN_BATCH_SIZE:-16}"
exec python tcn.py "$@"
