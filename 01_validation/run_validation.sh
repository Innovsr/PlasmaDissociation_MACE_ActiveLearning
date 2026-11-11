#!/bin/bash

set -euo pipefail

# ============================================================
# VALIDATION WORKFLOW LAUNCHER
#
# Usage:
#
#     bash run_validation.sh
#
# This script only launches the Python controller.
# The controller automatically submits:
#
#     setup job
#     validation array
#     restart checker
#     restart arrays when required
#     next restart checker
#     finalisation
#     model update
# ============================================================

echo
echo "=============================================="
echo "VALIDATION WORKFLOW"
echo "=============================================="

# ------------------------------------------------------------
# Repository root
# ------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG="$ROOT/configs/validation.json"
CONTROLLER="$ROOT/bomd/run/run_selective.py"

echo
echo "ROOT       = $ROOT"
echo "CONFIG     = $CONFIG"
echo "CONTROLLER = $CONTROLLER"

# ------------------------------------------------------------
# Check files
# ------------------------------------------------------------

if [ ! -f "$CONFIG" ]; then
    echo
    echo "ERROR: validation.json not found:"
    echo "$CONFIG"
    exit 1
fi

if [ ! -f "$CONTROLLER" ]; then
    echo
    echo "ERROR: run_selective.py not found:"
    echo "$CONTROLLER"
    exit 1
fi

# ------------------------------------------------------------
# Load Python environment
# ------------------------------------------------------------

module load miniforge

source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate plasma

PYTHON="$(command -v python)"

echo
echo "PYTHON     = $PYTHON"

# ------------------------------------------------------------
# Run Validation controller
# ------------------------------------------------------------

cd "$ROOT"

echo
echo "Starting Validation controller..."
echo

exec "$PYTHON" "$CONTROLLER" \
    --input "$CONFIG"
