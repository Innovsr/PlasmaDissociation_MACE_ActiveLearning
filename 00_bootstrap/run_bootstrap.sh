#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"

module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate plasma

PYTHON="$(command -v python)"

echo "=============================================="
echo "BOOTSTRAP MODEL GENERATION"
echo "=============================================="
echo "ROOT   = $ROOT"
echo "PYTHON = $PYTHON"
echo ""

exec "$PYTHON" "$ROOT/bootstrap.py" \
    --config "$ROOT/configs/bootstrap.json"
