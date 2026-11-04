#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$(command -v python)}"
readarray -t CFG < <("$PYTHON" - <<'PY'
import json
c=json.load(open("configs/update_model.json")); print(c["HPC"]["account"]); print(c["HPC"]["cores"]); print(c["HPC"]["hours"]); print(c["HPC"]["memory"]); print(c["HPC"]["Runfolder"]); print(c["attempt_id"])
PY
)
ACCOUNT="${CFG[0]}"; CORES="${CFG[1]}"; HOURS="${CFG[2]}"; MEM="${CFG[3]}"; RUNFOLDER="${CFG[4]}"; ATTEMPT="${CFG[5]}"
SCRATCH="/mnt/scratch/$USER/$RUNFOLDER/02_model_update/$ATTEMPT"
STATUS="$SCRATCH/STATUS"
if [ -f "$STATUS" ] && grep -Eq 'CANDIDATE_READY_FOR_USER_APPROVAL|NO_HIGH_ERROR_CONFIGURATIONS|MODEL_OK_USE_EXISTING|FAILED' "$STATUS"; then exit 0; fi
mkdir -p "$SCRATCH"
NEXT_JOB=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} --account="$ACCOUNT" --cpus-per-task="$CORES" --time="${HOURS}:00:00" --mem="${MEM}G" --job-name=MACE_AL_update "$0")
echo "$NEXT_JOB" > "$SCRATCH/next_job_id.txt"
if ! "$PYTHON" "$ROOT/model_update.py" --config "$ROOT/configs/update_model.json"; then
    echo "FAILED" > "$STATUS"
    exit 1
fi
