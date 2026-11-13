#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"

# ============================================================
# AUTOMATIC CONDA ENVIRONMENT
# ============================================================

ENV_NAME="mace_env"

if ! command -v conda >/dev/null 2>&1; then
    module load miniforge
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: Conda is not available."
    exit 1
fi

CONDA_BASE="$(conda info --base)"

source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "ERROR: Conda environment '$ENV_NAME' does not exist."
    exit 1
fi

conda activate "$ENV_NAME"

PYTHON="$(command -v python)"

echo "=============================================="
echo "PRODUCTION WORKFLOW"
echo "=============================================="
echo "Environment : $ENV_NAME"
echo "Python      : $PYTHON"
echo "=============================================="

exec "$PYTHON" "$ROOT/production_submit.py" \
    --input "$ROOT/configs/production.json"
