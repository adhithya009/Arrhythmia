#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${REPO_DIR}/scripts/wsl_tensorflow_env.sh"

cd "${REPO_DIR}"
export CNN_QAT_BATCH_SIZE="${CNN_QAT_BATCH_SIZE:-32}"
export CNN_QAT_EPOCHS="${CNN_QAT_EPOCHS:-8}"
export CNN_QAT_REP_SAMPLES="${CNN_QAT_REP_SAMPLES:-400}"
exec python cnn_qat.py "$@"
