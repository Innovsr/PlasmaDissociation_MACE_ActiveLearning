#!/bin/bash

# ============================================================
# MODEL UPDATE WORKFLOW
#
# Normal use:
#
#     bash run_model_update.sh
#
# The script submits itself to SLURM.
#
# Worker mode:
#
#     bash run_model_update.sh --worker
#
# The worker runs inside the allocated GPU node.
#
# The project path is determined on the login node and passed
# to the SLURM worker through PROJECT_ROOT.
#
# This keeps the script portable for other users.
# ============================================================

set -euo pipefail


# ============================================================
# LAUNCHER MODE
# ============================================================

if [[ "${1:-}" != "--worker" ]]; then

    # --------------------------------------------------------
    # This part runs BEFORE SLURM stages the script.
    # Therefore BASH_SOURCE points to the real repository.
    # --------------------------------------------------------

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

    CONFIG="$PROJECT_ROOT/configs/update_model.json"


    echo ""
    echo "============================================================"
    echo "MODEL UPDATE LAUNCHER"
    echo "============================================================"

    echo "PROJECT ROOT = $PROJECT_ROOT"
    echo "CONFIG       = $CONFIG"
    echo ""


    if [[ ! -f "$CONFIG" ]]; then

        echo "ERROR: Configuration file not found:"
        echo "$CONFIG"

        exit 1

    fi


    # --------------------------------------------------------
    # Read SLURM settings from update_model.json.
    # --------------------------------------------------------

    readarray -t CFG < <(
        python - "$CONFIG" <<'PY'
import json
import sys

config_file = sys.argv[1]

with open(config_file) as f:
    c = json.load(f)

hpc = c["HPC"]

print(hpc["account"])
print(hpc["cores"])
print(hpc["hours"])
print(hpc["memory"])
print(hpc["Runfolder"])
print(c["attempt_id"])
PY
    )


    ACCOUNT="${CFG[0]}"
    CORES="${CFG[1]}"
    HOURS="${CFG[2]}"
    MEM="${CFG[3]}"
    RUNFOLDER="${CFG[4]}"
    ATTEMPT="${CFG[5]}"


    echo "Account   = $ACCOUNT"
    echo "CPUs      = $CORES"
    echo "Time      = $HOURS"
    echo "Memory    = ${MEM}G"
    echo "Runfolder = $RUNFOLDER"
    echo "Attempt   = $ATTEMPT"

    echo ""


    # --------------------------------------------------------
    # Submit the first GPU worker.
    #
    # PROJECT_ROOT is explicitly passed to the worker.
    # --------------------------------------------------------

    JOB_ID=$(
        sbatch \
            --parsable \
            --export=ALL,PROJECT_ROOT="$PROJECT_ROOT" \
            --account="$ACCOUNT" \
            --partition=gpu \
            --gres=gpu:1 \
            --cpus-per-task="$CORES" \
            --mem="${MEM}G" \
            --time="$HOURS" \
            --job-name=MACE_AL_update \
            --output="$PROJECT_ROOT/model_update_%j.out" \
            --error="$PROJECT_ROOT/model_update_%j.err" \
            "$SCRIPT_DIR/run_model_update.sh" \
            --worker
    )


    echo "============================================================"
    echo "MODEL UPDATE SUBMITTED"
    echo "============================================================"

    echo "Job ID : $JOB_ID"

    echo ""
    echo "Monitor with:"
    echo ""
    echo "    squeue -j $JOB_ID"
    echo ""
    echo "Output:"
    echo ""
    echo "    $PROJECT_ROOT/model_update_${JOB_ID}.out"
    echo ""
    echo "Error:"
    echo ""
    echo "    $PROJECT_ROOT/model_update_${JOB_ID}.err"
    echo ""

    echo "============================================================"

    exit 0

fi


# ============================================================
# WORKER MODE
#
# This part runs inside the SLURM GPU job.
# ============================================================


# ------------------------------------------------------------
# Recover the real project root passed by the launcher.
#
# Do NOT use BASH_SOURCE here to determine the project path,
# because SLURM may execute a staged copy from /var/spool/slurmd.
# ------------------------------------------------------------

ROOT="${PROJECT_ROOT:?PROJECT_ROOT was not supplied by launcher}"

SCRIPT_DIR="$ROOT/02_model_update"

CONFIG="$ROOT/configs/update_model.json"


echo ""
echo "============================================================"
echo "MODEL UPDATE GPU WORKER"
echo "============================================================"

echo "Job ID       : ${SLURM_JOB_ID:-UNKNOWN}"
echo "Node         : $(hostname)"
echo "ROOT         : $ROOT"
echo "CONFIG       : $CONFIG"

echo "============================================================"
echo ""


# ============================================================
# CHECK PROJECT FILES
# ============================================================

if [[ ! -f "$CONFIG" ]]; then

    echo "ERROR: Configuration file not found:"
    echo "$CONFIG"

    exit 1

fi


if [[ ! -f "$ROOT/model_update.py" ]]; then

    echo "ERROR: model_update.py not found:"
    echo "$ROOT/model_update.py"

    exit 1

fi


# ============================================================
# LOAD CONDA
# ============================================================

module load miniforge

source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate mace_env


PYTHON="$(command -v python)"

echo "Python       : $PYTHON"

echo ""


# ============================================================
# VERIFY ENVIRONMENT
# ============================================================

echo "============================================================"
echo "ENVIRONMENT CHECK"
echo "============================================================"

python -c "
import sys
import ase
import mace
import torch

print('Python       :', sys.executable)
print('ASE          :', ase.__version__)
print('MACE         :', getattr(mace, '__version__', 'unknown'))
print('PyTorch      :', torch.__version__)
print('CUDA         :', torch.cuda.is_available())

if torch.cuda.is_available():
    print('GPU          :', torch.cuda.get_device_name(0))
"

echo "============================================================"
echo ""


# ============================================================
# READ CONFIGURATION
# ============================================================

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


# ============================================================
# SCRATCH LOCATION
# ============================================================

SCRATCH="/mnt/scratch/$USER/$RUNFOLDER/02_model_update/$ATTEMPT"

STATUS="$SCRATCH/STATUS"

NEXT_JOB_FILE="$SCRATCH/next_job_id.txt"


mkdir -p "$SCRATCH"


echo "Scratch      : $SCRATCH"
echo ""


# ============================================================
# TERMINAL STATUS CHECK
# ============================================================

if [[ -f "$STATUS" ]]; then

    STATUS_VALUE="$(tr -d '[:space:]' < "$STATUS")"

    case "$STATUS_VALUE" in

        CANDIDATE_READY_FOR_USER_APPROVAL)

            echo "Model update already completed."
            cat "$STATUS"

            exit 0
            ;;

        NO_HIGH_ERROR_CONFIGURATIONS)

            echo "No high-error configurations were found."
            cat "$STATUS"

            exit 0
            ;;

        MODEL_OK_USE_EXISTING)

            echo "Existing model already satisfies the criteria."
            cat "$STATUS"

            exit 0
            ;;

    esac

fi


# ============================================================
# AUTOMATIC RESTART CHAIN
#
# A continuation job is submitted before the current training
# starts.
#
# If this job reaches SLURM walltime:
#
#     current job -> TIMEOUT
#             |
#             v
#     next job starts
#             |
#             v
#     existing MACE checkpoint
#             |
#             v
#     --restart_latest
#
# If the current job completes normally, the unnecessary
# continuation job is cancelled.
# ============================================================

NEXT_JOB_ID=""


if [[ "${MACE_AL_RESTART_CHAIN:-1}" == "1" ]]; then

    NEXT_JOB_ID=$(
        sbatch \
            --parsable \
            --export=ALL,PROJECT_ROOT="$ROOT" \
            --dependency="afterany:${SLURM_JOB_ID}" \
            --account="$ACCOUNT" \
            --partition=gpu \
            --gres=gpu:1 \
            --cpus-per-task="$CORES" \
            --mem="${MEM}G" \
            --time="$HOURS" \
            --job-name=MACE_AL_update \
            --output="$ROOT/model_update_%j.out" \
            --error="$ROOT/model_update_%j.err" \
            "$SCRIPT_DIR/run_model_update.sh" \
            --worker
    )


    echo "$NEXT_JOB_ID" > "$NEXT_JOB_FILE"


    echo "============================================================"
    echo "AUTOMATIC RESTART CHAIN"
    echo "============================================================"

    echo "Current job : $SLURM_JOB_ID"
    echo "Next job    : $NEXT_JOB_ID"
    echo "Dependency  : afterany:$SLURM_JOB_ID"

    echo ""
    echo "If this job times out, the next job will start."
    echo "The next job will continue from the MACE checkpoint."

    echo "============================================================"
    echo ""

fi


# ============================================================
# GPU CHECK
# ============================================================

echo "============================================================"
echo "GPU CHECK"
echo "============================================================"

nvidia-smi

echo ""
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"

echo "============================================================"
echo ""


# ============================================================
# RUN MODEL UPDATE
# ============================================================

MODEL_UPDATE_EXIT=0


set +e

"$PYTHON" \
    "$ROOT/model_update.py" \
    --config "$CONFIG"

MODEL_UPDATE_EXIT=$?

set -e


# ============================================================
# NORMAL COMPLETION
# ============================================================

if [[ "$MODEL_UPDATE_EXIT" -eq 0 ]]; then

    echo ""
    echo "============================================================"
    echo "MODEL UPDATE SCRIPT FINISHED"
    echo "============================================================"


    FINAL_STATUS=""

    if [[ -f "$STATUS" ]]; then

        FINAL_STATUS="$(
            tr -d '[:space:]' < "$STATUS"
        )"

        echo ""
        echo "STATUS:"
        cat "$STATUS"

    fi


    # --------------------------------------------------------
    # Cancel the pre-submitted continuation job if the current
    # model update reached a terminal state.
    # --------------------------------------------------------

    if [[ -n "$NEXT_JOB_ID" ]]; then

        case "$FINAL_STATUS" in

            CANDIDATE_READY_FOR_USER_APPROVAL|\
            NO_HIGH_ERROR_CONFIGURATIONS|\
            MODEL_OK_USE_EXISTING)

                echo ""
                echo "Cancelling unnecessary continuation job:"
                echo "$NEXT_JOB_ID"

                scancel "$NEXT_JOB_ID" || true

                ;;

            *)

                echo ""
                echo "WARNING:"
                echo "model_update.py returned successfully but"
                echo "no terminal STATUS was found."

                echo "The continuation job will remain pending."

                ;;

        esac

    fi


    echo "============================================================"

    exit 0

fi


# ============================================================
# MODEL UPDATE FAILED
# ============================================================

echo ""
echo "============================================================"
echo "MODEL UPDATE FAILED"
echo "============================================================"

echo "Exit code: $MODEL_UPDATE_EXIT"


# ------------------------------------------------------------
# This was a normal application failure rather than a SLURM
# walltime timeout, so cancel the continuation job.
# ------------------------------------------------------------

if [[ -n "$NEXT_JOB_ID" ]]; then

    echo ""
    echo "Cancelling continuation job because"
    echo "model_update.py failed."

    scancel "$NEXT_JOB_ID" || true

fi


echo "============================================================"

exit "$MODEL_UPDATE_EXIT"
