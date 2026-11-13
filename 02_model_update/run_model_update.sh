#!/bin/bash

# ============================================================
# MODEL UPDATE WORKFLOW
#
# Normal use:
#
#     bash run_model_update.sh
#
# Worker mode:
#
#     bash run_model_update.sh --worker
#
# ============================================================
#
# AUTOMATIC RESTART DESIGN
# ------------------------
#
# A continuation job is submitted before training starts.
#
# The continuation uses afterany, BUT it does NOT blindly
# continue.
#
# When the continuation starts it checks the parent job state.
#
# ONLY:
#
#     TIMEOUT
#
# allows the continuation to run MACE with --restart_latest.
#
# COMPLETED / CANCELLED / FAILED / etc.
# causes the continuation to exit immediately.
#
# This prevents an infinite restart chain.
#
# ============================================================

set -euo pipefail


# ============================================================
# LAUNCHER MODE
# ============================================================

if [[ "${1:-}" != "--worker" ]]; then

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


    # ========================================================
    # READ HPC SETTINGS
    # ========================================================

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


    # ========================================================
    # SLURM LOG DIRECTORY
    # ========================================================

    SLURM_LOG_DIR="/mnt/scratch/$USER/$RUNFOLDER/02_model_update/$ATTEMPT/slurm_logs"

    mkdir -p "$SLURM_LOG_DIR"


    echo "Account     = $ACCOUNT"
    echo "CPUs        = $CORES"
    echo "Time        = $HOURS"
    echo "Memory      = ${MEM}G"
    echo "Runfolder   = $RUNFOLDER"
    echo "Attempt     = $ATTEMPT"
    echo "SLURM logs  = $SLURM_LOG_DIR"

    echo "============================================================"
    echo ""


    # ========================================================
    # SUBMIT FIRST GPU WORKER
    # ========================================================

    JOB_ID="$(
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
            --output="$SLURM_LOG_DIR/model_update_%j.out" \
            --error="$SLURM_LOG_DIR/model_update_%j.err" \
            "$SCRIPT_DIR/run_model_update.sh" \
            --worker
    )"


    echo "============================================================"
    echo "MODEL UPDATE SUBMITTED"
    echo "============================================================"

    echo "Job ID : $JOB_ID"
    echo ""

    echo "Monitor with:"
    echo "    squeue -j $JOB_ID"
    echo ""

    echo "SLURM output:"
    echo "    $SLURM_LOG_DIR/model_update_${JOB_ID}.out"
    echo ""

    echo "SLURM error:"
    echo "    $SLURM_LOG_DIR/model_update_${JOB_ID}.err"
    echo ""

    echo "============================================================"

    exit 0

fi


# ============================================================
# WORKER MODE
# ============================================================

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
# CHECK REQUIRED FILES
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
# ENVIRONMENT CHECK
# ============================================================

echo "============================================================"
echo "ENVIRONMENT CHECK"
echo "============================================================"

"$PYTHON" -c 'import sys, ase, mace, torch; print("Python       :", sys.executable); print("ASE          :", ase.__version__); print("MACE         :", getattr(mace, "__version__", "unknown")); print("PyTorch      :", torch.__version__); print("CUDA         :", torch.cuda.is_available()); print("GPU          :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")'

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
# SCRATCH DIRECTORIES
# ============================================================

SCRATCH="/mnt/scratch/$USER/$RUNFOLDER/02_model_update/$ATTEMPT"

STATUS="$SCRATCH/STATUS"

NEXT_JOB_FILE="$SCRATCH/next_job_id.txt"

SLURM_LOG_DIR="$SCRATCH/slurm_logs"

mkdir -p "$SCRATCH"
mkdir -p "$SLURM_LOG_DIR"


echo "Scratch      : $SCRATCH"
echo "SLURM logs   : $SLURM_LOG_DIR"
echo ""


# ============================================================
# PARENT JOB INFORMATION
# ============================================================

PARENT_JOB_ID="${PARENT_JOB_ID:-}"


# ============================================================
# CONTINUATION WORKER
# ============================================================

if [[ -n "$PARENT_JOB_ID" ]]; then

    echo "============================================================"
    echo "CONTINUATION WORKER"
    echo "============================================================"

    echo "Current job : $SLURM_JOB_ID"
    echo "Parent job  : $PARENT_JOB_ID"
    echo ""


    # ========================================================
    # FIND PARENT SLURM STATE
    # ========================================================

    PARENT_STATE="$(
        sacct -X \
            -j "$PARENT_JOB_ID" \
            --format=State \
            --noheader \
            --parsable2 \
            2>/dev/null |
        head -n 1 |
        tr -d '[:space:]'
    )"


    echo "Parent state: ${PARENT_STATE:-UNKNOWN}"
    echo ""


    # ========================================================
    # ONLY TIMEOUT MAY CONTINUE
    # ========================================================

    case "$PARENT_STATE" in

        TIMEOUT|TIMEOUT*)

            echo "Parent job timed out."
            echo "Automatic MACE restart is allowed."

            ;;


        CANCELLED|CANCELLED*)

            echo "Parent job was CANCELLED."
            echo "NO restart will be submitted."
            echo "Continuation worker terminating."

            exit 0

            ;;


        FAILED|FAILED*)

            echo "Parent job FAILED."
            echo "NO restart will be submitted."
            echo "Continuation worker terminating."

            exit 0

            ;;


        OUT_OF_MEMORY|OUT_OF_MEMORY*)

            echo "Parent job ran out of memory."
            echo "NO automatic restart will be submitted."
            echo "Continuation worker terminating."

            exit 0

            ;;


        NODE_FAIL|NODE_FAIL*)

            echo "Parent job failed because of NODE_FAIL."
            echo "NO automatic restart will be submitted."
            echo "Continuation worker terminating."

            exit 0

            ;;


        PREEMPTED|PREEMPTED*)

            echo "Parent job was PREEMPTED."
            echo "NO automatic restart will be submitted."
            echo "Continuation worker terminating."

            exit 0

            ;;


        COMPLETED|COMPLETED*)

            echo "Parent job COMPLETED normally."
            echo "NO restart is required."
            echo "Continuation worker terminating."

            exit 0

            ;;


        DEADLINE|DEADLINE*)

            echo "Parent job reached a deadline."
            echo "NO automatic restart will be submitted."
            echo "Continuation worker terminating."

            exit 0

            ;;


        *)

            echo "WARNING: Parent state could not be safely identified."
            echo "Parent state = ${PARENT_STATE:-UNKNOWN}"
            echo ""
            echo "NO automatic restart will be submitted."
            echo "Continuation worker terminating."

            exit 0

            ;;

    esac


    echo "============================================================"
    echo ""

fi


# ============================================================
# TERMINAL STATUS CHECK
# ============================================================

if [[ -f "$STATUS" ]]; then

    STATUS_VALUE="$(tr -d '[:space:]' < "$STATUS")"

    echo "Current model-update STATUS:"
    echo "$STATUS_VALUE"
    echo ""

    case "$STATUS_VALUE" in

        CANDIDATE_READY_FOR_USER_APPROVAL|READY_FOR_MANUAL_REVIEW)

            echo "Model update already completed."
            echo "No further continuation is required."

            cat "$STATUS"

            exit 0

            ;;


        NO_HIGH_ERROR_CONFIGURATIONS)

            echo "No high-error configurations were found."
            echo "No further continuation is required."

            cat "$STATUS"

            exit 0

            ;;


        MODEL_OK_USE_EXISTING)

            echo "Existing model already satisfies the criteria."
            echo "No further continuation is required."

            cat "$STATUS"

            exit 0

            ;;

    esac

fi


# ============================================================
# AUTOMATIC TIMEOUT CONTINUATION
# ============================================================
#
# One child is prepared before training.
#
# The child will start because of afterany, but it will only
# continue if the parent state is TIMEOUT.
#
# ============================================================

NEXT_JOB_ID=""


if [[ "${MACE_AL_RESTART_CHAIN:-1}" == "1" ]]; then

    echo "============================================================"
    echo "PREPARING AUTOMATIC TIMEOUT CONTINUATION"
    echo "============================================================"

    echo "Current job : $SLURM_JOB_ID"

    if [[ -n "$PARENT_JOB_ID" ]]; then

        echo "Parent job  : $PARENT_JOB_ID"
        echo "Parent state: TIMEOUT"
        echo "This is a genuine continuation."

    else

        echo "This is the first model-update worker."
        echo "Preparing first timeout continuation."

    fi

    echo ""


    # ========================================================
    # SUBMIT ONE CHILD
    # ========================================================

    NEXT_JOB_ID="$(
        sbatch \
            --parsable \
            --export=ALL,PROJECT_ROOT="$ROOT",PARENT_JOB_ID="$SLURM_JOB_ID" \
            --dependency="afterany:${SLURM_JOB_ID}" \
            --account="$ACCOUNT" \
            --partition=gpu \
            --gres=gpu:1 \
            --cpus-per-task="$CORES" \
            --mem="${MEM}G" \
            --time="$HOURS" \
            --job-name=MACE_AL_update \
            --output="$SLURM_LOG_DIR/model_update_%j.out" \
            --error="$SLURM_LOG_DIR/model_update_%j.err" \
            "$SCRIPT_DIR/run_model_update.sh" \
            --worker
    )"


    echo "$NEXT_JOB_ID" > "$NEXT_JOB_FILE"


    echo "Current job : $SLURM_JOB_ID"
    echo "Next job    : $NEXT_JOB_ID"
    echo "Dependency  : afterany:$SLURM_JOB_ID"
    echo ""

    echo "The next job will ONLY continue if this job"
    echo "actually reaches SLURM TIMEOUT."

    echo ""

    echo "If this job completes normally, the next job"
    echo "will be cancelled."

    echo ""

    echo "If this job is cancelled or fails, the next job"
    echo "will start briefly, detect the parent state,"
    echo "and exit without creating another job."

    echo ""

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

"$PYTHON" "$ROOT/model_update.py" --config "$CONFIG"

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


    # ========================================================
    # CANCEL PRE-SUBMITTED CONTINUATION
    # ========================================================

    if [[ -n "$NEXT_JOB_ID" ]]; then

        case "$FINAL_STATUS" in

            CANDIDATE_READY_FOR_USER_APPROVAL|READY_FOR_MANUAL_REVIEW|NO_HIGH_ERROR_CONFIGURATIONS|MODEL_OK_USE_EXISTING)

                echo ""
                echo "============================================================"
                echo "CANCELLING UNNECESSARY CONTINUATION"
                echo "============================================================"

                echo "Continuation job: $NEXT_JOB_ID"

                scancel "$NEXT_JOB_ID" || true

                echo "Continuation cancelled."

                echo "============================================================"

                ;;


            *)

                echo ""
                echo "WARNING:"
                echo "model_update.py returned successfully but"
                echo "no recognised terminal STATUS was found."

                echo ""

                if [[ -f "$STATUS" ]]; then
                    echo "STATUS:"
                    cat "$STATUS"
                else
                    echo "NO STATUS FILE"
                fi

                echo ""

                echo "The continuation job will remain available"
                echo "for a genuine SLURM TIMEOUT."

                ;;

        esac

    fi


    echo ""
    echo "============================================================"
    echo "MODEL UPDATE WORKER FINISHED SUCCESSFULLY"
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


# ============================================================
# CANCEL PRE-SUBMITTED CONTINUATION
# ============================================================

if [[ -n "$NEXT_JOB_ID" ]]; then

    echo ""
    echo "Cancelling continuation job because"
    echo "model_update.py failed."

    echo "Continuation job: $NEXT_JOB_ID"

    scancel "$NEXT_JOB_ID" || true

fi


echo ""
echo "============================================================"

exit "$MODEL_UPDATE_EXIT"
