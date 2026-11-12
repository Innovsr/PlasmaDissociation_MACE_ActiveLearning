#!/bin/bash

#SBATCH --account=chmdsh_project
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=MACE_AL_update

set -euo pipefail


ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

cd "$ROOT"


module load miniforge

source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate mace_env


PYTHON="$(command -v python)"

CONFIG="$ROOT/configs/update_model.json"


# ------------------------------------------------------------
# Read configuration.
# ------------------------------------------------------------

readarray -t CFG < <(
    "$PYTHON" - "$CONFIG" <<'PY'
import json
import sys

config_file = sys.argv[1]

with open(config_file) as f:
    c = json.load(f)

print(c["HPC"]["account"])
print(c["HPC"]["cores"])
print(c["HPC"]["hours"])
print(c["HPC"]["memory"])
print(c["HPC"]["Runfolder"])
print(c["attempt_id"])
PY
)


ACCOUNT="${CFG[0]}"
CORES="${CFG[1]}"
HOURS="${CFG[2]}"
MEM="${CFG[3]}"
RUNFOLDER="${CFG[4]}"
ATTEMPT="${CFG[5]}"


SCRATCH="/mnt/scratch/$USER/$RUNFOLDER/02_model_update/$ATTEMPT"

STATUS="$SCRATCH/STATUS"

NEXT_JOB_FILE="$SCRATCH/next_job_id.txt"


mkdir -p "$SCRATCH"


# ------------------------------------------------------------
# If model update is already finished, do nothing.
# ------------------------------------------------------------

if [ -f "$STATUS" ]; then

    if grep -Eq \
        'CANDIDATE_READY_FOR_USER_APPROVAL|NO_HIGH_ERROR_CONFIGURATIONS|MODEL_OK_USE_EXISTING' \
        "$STATUS"
    then

        echo "Model-update workflow already finished."

        cat "$STATUS"

        exit 0

    fi

fi


# ------------------------------------------------------------
# Submit the next continuation job BEFORE starting this job.
#
# If this job reaches the SLURM walltime, the next job remains
# pending and will start automatically.
#
# If this job finishes normally, model_update.py writes a
# terminal STATUS and the next job exits immediately.
# ------------------------------------------------------------

if [ "${MACE_AL_RESTART_CHAIN:-1}" = "1" ]; then

    NEXT_JOB=$(
        sbatch \
            --parsable \
            --dependency=afterany:${SLURM_JOB_ID} \
            --account="$ACCOUNT" \
            --partition=gpu \
            --cpus-per-task="$CORES" \
            --time="$HOURS" \
            --mem="${MEM}G" \
            --gres=gpu:1 \
            --job-name=MACE_AL_update \
            "$0"
    )

    echo "$NEXT_JOB" > "$NEXT_JOB_FILE"

    echo ""
    echo "============================================================"
    echo "NEXT MODEL-UPDATE JOB"
    echo "============================================================"
    echo "Current job : $SLURM_JOB_ID"
    echo "Next job    : $NEXT_JOB"
    echo "Dependency  : afterany:$SLURM_JOB_ID"
    echo "============================================================"
    echo ""

fi


# ------------------------------------------------------------
# GPU check.
# ------------------------------------------------------------

echo "============================================================"
echo "GPU CHECK"
echo "============================================================"

nvidia-smi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "============================================================"


# ------------------------------------------------------------
# Run model update.
# ------------------------------------------------------------

"$PYTHON" \
    "$ROOT/model_update.py" \
    --config "$CONFIG"


# ------------------------------------------------------------
# Normal successful completion.
#
# model_update.py should have written STATUS.
# The chained next job will see STATUS and exit immediately.
# ------------------------------------------------------------

if [ -f "$STATUS" ]; then

    echo ""
    echo "============================================================"
    echo "MODEL UPDATE STATUS"
    echo "============================================================"

    cat "$STATUS"

    echo "============================================================"

fi
