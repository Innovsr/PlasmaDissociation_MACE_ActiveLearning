#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"

PYTHON="${PYTHON:-$(command -v python)}"

echo "=============================================="
echo "TRAJECTORY ANALYSIS"
echo "=============================================="
echo
echo "Repository : $ROOT"
echo "Python     : $PYTHON"
echo "Config     : $ROOT/configs/analysis.json"
echo

"$PYTHON" "$ROOT/03_analysis/analysis.py" \
    --config "$ROOT/configs/analysis.json"

echo
echo "=============================================="
echo "ANALYSIS FINISHED"
echo "=============================================="
