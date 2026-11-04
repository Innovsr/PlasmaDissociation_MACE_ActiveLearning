#!/bin/bash

#SBATCH --account=chmdsh_project
#SBATCH --job-name=Validation_Controller
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -eo pipefail

# sbatch sets SLURM_SUBMIT_DIR to the directory from which
# the job was submitted. This avoids /var/spool/slurmd.
ROOT="${SLURM_SUBMIT_DIR}"

# If submitted from 01_validation/, move to package root.
if [ ! -f "$ROOT/configs/validation.json" ] && \
   [ -f "$ROOT/../configs/validation.json" ]; then
    ROOT="$(cd "$ROOT/.." && pwd)"
fi

if [ ! -f "$ROOT/configs/validation.json" ]; then
    echo "ERROR: configs/validation.json not found"
    echo "ROOT = $ROOT"
    exit 1
fi

module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate plasma

PYTHON="$(command -v python)"

cd "$ROOT"

echo "Validation controller started"
echo "ROOT = $ROOT"
echo "PYTHON = $PYTHON"

python -c "import sys, numpy; print('Python:', sys.executable); print('NumPy:', numpy.__version__)"

exec "$PYTHON" "$ROOT/bomd/run/run_selective.py" \
    --input "$ROOT/configs/validation.json"
