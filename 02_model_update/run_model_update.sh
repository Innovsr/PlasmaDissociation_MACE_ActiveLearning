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

readarray -t CFG < <("$PYTHON" - "$CONFIG" <<'PY'
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

ACCOUNT="${CFG[0]}"; CORES="${CFG[1]}"; HOURS="${CFG[2]}"; MEM="${CFG[3]}"; RUNFOLDER="${CFG[4]}"; ATTEMPT="${CFG[5]}"
SCRATCH="/mnt/scratch/$USER/$RUNFOLDER/02_model_update/$ATTEMPT"
STATUS="$SCRATCH/STATUS"
if [ -f "$STATUS" ] && grep -Eq 'CANDIDATE_READY_FOR_USER_APPROVAL|NO_HIGH_ERROR_CONFIGURATIONS|MODEL_OK_USE_EXISTING|FAILED' "$STATUS"; then exit 0; fi
mkdir -p "$SCRATCH"
NEXT_JOB=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} --account="$ACCOUNT" --partition=gpu --cpus-per-task="$CORES" --time="$HOURS" --mem="${MEM}G" --gres=gpu:1 --job-name=MACE_AL_update "$0")
echo "$NEXT_JOB" > "$SCRATCH/next_job_id.txt"

echo "===== GPU CHECK ====="
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "===================="

if ! "$PYTHON" "$ROOT/model_update.py" --config "$ROOT/configs/update_model.json"; then
    echo "FAILED" > "$STATUS"
    exit 1
fi
