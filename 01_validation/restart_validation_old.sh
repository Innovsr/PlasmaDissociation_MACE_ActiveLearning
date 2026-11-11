#!/bin/bash

#SBATCH --account=chmdsh_project
#SBATCH --job-name=Validation_Check
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -eo pipefail

# ------------------------------------------------------------
# Arguments supplied by run_selective.py
#
# $1 = scratch validation run directory
# $2 = package root
# ------------------------------------------------------------

if [ -z "${1:-}" ]; then
    echo "ERROR: run root was not provided"
    exit 1
fi

if [ -z "${2:-}" ]; then
    echo "ERROR: package root was not provided"
    exit 1
fi

RUN_ROOT="$1"
PACKAGE_ROOT="$2"

# ------------------------------------------------------------
# Python environment
# ------------------------------------------------------------

module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate plasma

PYTHON="$(command -v python)"

# ------------------------------------------------------------
# Validation array ID
# ------------------------------------------------------------

if [ -z "${VALIDATION_ARRAY_JOB_ID:-}" ]; then
    echo "ERROR: VALIDATION_ARRAY_JOB_ID is not set"
    exit 1
fi

echo "Validation array job: $VALIDATION_ARRAY_JOB_ID"
echo "Run root: $RUN_ROOT"
echo "Package root: $PACKAGE_ROOT"
echo "Python: $PYTHON"

# ------------------------------------------------------------
# Run checker
# ------------------------------------------------------------

exec "$PYTHON" "$PACKAGE_ROOT/restart_validation.py" \
    --run-root "$RUN_ROOT" \
    --array-job "$VALIDATION_ARRAY_JOB_ID"
