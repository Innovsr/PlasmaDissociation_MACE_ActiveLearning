#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"

module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate plasma

PYTHON="$(command -v python)"

echo "=============================================="
echo "VALIDATION WORKFLOW"
echo "=============================================="
echo "ROOT   = $ROOT"
echo "PYTHON = $PYTHON"
echo ""

exec "$PYTHON" "$ROOT/01_validation/validation.py" \
    --config "$ROOT/configs/validation.json"
