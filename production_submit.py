#!/usr/bin/env python3

from __future__ import annotations

import argparse
import getpass
import json
import shutil
import subprocess
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

BOMD = ROOT / "bomd"

RESTART = ROOT / "restart_manager.py"


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):
    path = Path(path)

    with path.open() as f:
        return json.load(f)


# ============================================================
# FIND MODEL
# ============================================================

def find_model(model_id):

    model_dir = ROOT / "models" / model_id

    model_file = model_dir / "model.model"

    if not model_file.exists():

        raise SystemExit(
            "\n"
            "ERROR: MACE model was not found.\n"
            "\n"
            f"Model ID : {model_id}\n"
            f"Expected : {model_file}\n"
        )

    return model_dir


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # COMMAND-LINE ARGUMENTS
    # ========================================================

    parser = argparse.ArgumentParser(
        description="Submit production BOMD/MACE workflow."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to production JSON configuration."
    )

    args = parser.parse_args()

    config_file = Path(args.input).resolve()


    # ========================================================
    # LOAD CONFIGURATION
    # ========================================================

    if not config_file.exists():

        raise SystemExit(
            f"ERROR: Production configuration not found:\n"
            f"{config_file}"
        )

    config = load_json(config_file)


    # ========================================================
    # MODEL
    # ========================================================

    model_id = config["model"]

    model_dir = find_model(model_id)

    model_file = model_dir / "model.model"


    # ========================================================
    # HPC
    # ========================================================

    hpc = config["HPC"]

    user = getpass.getuser()

    parent = (
        Path("/mnt/scratch")
        / user
        / hpc["Runfolder"]
    )

    stage = parent / "03_production"

    logs = stage / "logs"

    setup_dir = stage / "setup"

    repetitions_dir = stage / "repetitions"


    # ========================================================
    # CREATE DIRECTORIES
    # ========================================================

    stage.mkdir(
        parents=True,
        exist_ok=True
    )

    logs.mkdir(
        parents=True,
        exist_ok=True
    )

    setup_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    repetitions_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # COPY PRODUCTION CONFIGURATION
    #
    # setup.py will explicitly receive:
    #
    #     ../inputs.json
    #
    # ========================================================

    shutil.copy2(
        config_file,
        stage / "inputs.json"
    )


    # ========================================================
    # INITIAL GEOMETRY
    # ========================================================

    molecule = config["Molecule_data"]["Molecule"]

    initial_geometry = (
        BOMD
        / "Initial_Geometry"
        / f"{molecule}.txt"
    )

    if not initial_geometry.exists():

        raise SystemExit(
            "\n"
            "ERROR: Initial geometry not found:\n"
            f"{initial_geometry}\n"
        )


    shutil.copy2(
        initial_geometry,
        setup_dir / "initial_guess.txt"
    )


    # ========================================================
    # PYTHON FILES
    # ========================================================

    setup_py = BOMD / "code" / "setup.py"

    main_py = BOMD / "code" / "main.py"


    if not setup_py.exists():

        raise SystemExit(
            f"ERROR: setup.py not found:\n{setup_py}"
        )

    if not main_py.exists():

        raise SystemExit(
            f"ERROR: main.py not found:\n{main_py}"
        )

    if not RESTART.exists():

        raise SystemExit(
            f"ERROR: restart_manager.py not found:\n{RESTART}"
        )


    # ========================================================
    # PRODUCTION PARAMETERS
    # ========================================================

    repeats = int(
        hpc["repeats"]
    )

    target_steps = int(
        config["Propagation"]["Tot_timesteps"]
    )

    timestep = float(
        config["Propagation"]["Timestep"]
    )


    # ========================================================
    # MACE SETTINGS
    # ========================================================

    mace_config = config.get(
        "MACE",
        {}
    )

    mace_device = mace_config.get(
        "device",
        "cpu"
    )

    mace_dtype = mace_config.get(
        "default_dtype",
        "float64"
    )


    # ========================================================
    # SETUP SCRIPT
    #
    # SETUP ALWAYS USES:
    #
    #     plasma
    #
    # because pyqchem is installed there.
    # ========================================================

    setup_script = stage / "setup_production.sh"

    setup_script.write_text(
        f'''#!/bin/bash

#SBATCH --account={hpc["account"]}
#SBATCH --job-name=Setup_{hpc["Runfolder"]}
#SBATCH --time={hpc["hours"]}:00:00
#SBATCH --cpus-per-task={hpc["cores"]}
#SBATCH --mem={hpc["memory"]}G
#SBATCH --output={logs}/setup_%j.out
#SBATCH --error={logs}/setup_%j.err

set -euo pipefail


# ============================================================
# MINIFORGE
# ============================================================

module load miniforge

source "$(conda info --base)/etc/profile.d/conda.sh"


# ============================================================
# PLASMA ENVIRONMENT
# ============================================================

conda activate plasma

PYTHON="$(command -v python)"


echo ""
echo "============================================================"
echo "BOMD SETUP JOB"
echo "============================================================"

echo "Job ID       : ${{SLURM_JOB_ID:-UNKNOWN}}"
echo "Node         : $(hostname)"
echo "Environment  : $CONDA_DEFAULT_ENV"
echo "Python       : $PYTHON"

echo "============================================================"
echo ""


# ============================================================
# CHECK PYQCHEM
# ============================================================

"$PYTHON" - <<'PY'
import pyqchem

from pyqchem import QchemInput, Structure
from pyqchem import get_output_from_qchem
from pyqchem.parsers.parser_optimization import basic_optimization
from pyqchem.parsers.parser_frequencies import basic_frequencies
from pyqchem.parsers.basic import basic_parser_qchem
from pyqchem import errors

print("pyqchem import: SUCCESS")
print("pyqchem location:", pyqchem.__file__)
PY


# ============================================================
# Q-CHEM
# ============================================================

module unload qchem 2>/dev/null || true
module load qchem/6.0.2

export LM_LICENSE_FILE=27000@uol-lnx-lic01.leeds.ac.uk

export QC="$QCHEM_HOME"
export QCAUX="$QC/qcaux"
export QCPROG="$QC/exe/qcprog.exe"
export QCPROG_S="$QC/exe/qcprog.exe_s"

export PATH="$PATH:$QC/exe:$QC/bin"


# ============================================================
# Q-CHEM SCRATCH
# ============================================================

export QCSCRATCH="{setup_dir}/tmp"

mkdir -p "$QCSCRATCH"


# ============================================================
# SETUP DIRECTORY
# ============================================================

cd "{setup_dir}"


echo ""
echo "============================================================"
echo "RUNNING BOMD SETUP"
echo "============================================================"

echo "Input : ../inputs.json"

echo "============================================================"
echo ""


# ============================================================
# SETUP
# ============================================================

exec "$PYTHON" "{setup_py}" --input "../inputs.json"
'''
    )

    setup_script.chmod(0o755)


    # ========================================================
    # PRODUCTION ARRAY SCRIPT
    #
    # IMPORTANT RESTART DESIGN:
    #
    # The array pre-submits ONE child.
    #
    # The child starts because of afterany.
    #
    # The child checks the parent state.
    #
    # ONLY TIMEOUT allows continuation.
    #
    # FAILED/CANCELLED/COMPLETED/etc. simply terminate.
    #
    # Therefore:
    #
    #     FAILED → STOP
    #     CANCELLED → STOP
    #     COMPLETED → STOP
    #     TIMEOUT → CONTINUE
    #
    # This prevents the infinite resubmission problem.
    # ========================================================

    production_script = stage / "production_array.sh"

    production_script.write_text(
        f'''#!/bin/bash

#SBATCH --account={hpc["account"]}
#SBATCH --job-name=MACE_{hpc["Runfolder"]}
#SBATCH --time={hpc["hours"]}:00:00
#SBATCH --cpus-per-task={hpc["cores"]}
#SBATCH --mem={hpc["memory"]}G
#SBATCH --array=1-{repeats}
#SBATCH --output={logs}/mace_%A_%a.out
#SBATCH --error={logs}/mace_%A_%a.err

set -euo pipefail


# ============================================================
# MINIFORGE
# ============================================================

module load miniforge

source "$(conda info --base)/etc/profile.d/conda.sh"


# ============================================================
# MACE ENVIRONMENT
# ============================================================

conda activate mace_env

PYTHON="$(command -v python)"


# ============================================================
# BASIC INFORMATION
# ============================================================

echo ""
echo "============================================================"
echo "MACE PRODUCTION WORKER"
echo "============================================================"

echo "Job ID        : ${{SLURM_JOB_ID:-UNKNOWN}}"
echo "Array Job ID  : ${{SLURM_ARRAY_JOB_ID:-UNKNOWN}}"
echo "Array Task ID : ${{SLURM_ARRAY_TASK_ID:-UNKNOWN}}"

echo "Environment   : $CONDA_DEFAULT_ENV"
echo "Python        : $PYTHON"

echo "============================================================"
echo ""


# ============================================================
# MACE / PYTORCH CHECK
# ============================================================

"$PYTHON" - <<'PY'
import torch
import mace

print("MACE import     : SUCCESS")
print("MACE version    :", getattr(mace, "__version__", "unknown"))
print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU             :", torch.cuda.get_device_name(0))
PY


# ============================================================
# MODEL
# ============================================================

export MACE_MODEL_PATH="{model_file}"

export MACE_DEVICE="{mace_device}"

export MACE_DTYPE="{mace_dtype}"


echo ""
echo "============================================================"
echo "MODEL"
echo "============================================================"

echo "Model ID    : {model_id}"
echo "Model file  : $MACE_MODEL_PATH"
echo "MACE device : $MACE_DEVICE"
echo "MACE dtype  : $MACE_DTYPE"

echo "============================================================"
echo ""


if [[ ! -f "$MACE_MODEL_PATH" ]]; then

    echo "ERROR: MACE model does not exist:"
    echo "$MACE_MODEL_PATH"

    exit 1

fi


# ============================================================
# CONTINUATION INFORMATION
#
# A normal first worker has:
#
#     PARENT_JOB_ID=""
#
# A continuation worker has:
#
#     PARENT_JOB_ID=<previous array job>
#
# The continuation worker MUST verify the actual SLURM state
# of the previous array before doing anything.
# ============================================================

PARENT_JOB_ID="${{PARENT_JOB_ID:-}}"


if [[ -n "$PARENT_JOB_ID" ]]; then

    echo ""
    echo "============================================================"
    echo "CONTINUATION WORKER"
    echo "============================================================"

    echo "Current job : $SLURM_ARRAY_JOB_ID"
    echo "Parent job  : $PARENT_JOB_ID"

    echo ""


    # ========================================================
    # GET PARENT ARRAY STATE
    # ========================================================

    PARENT_STATE="$(
        sacct \
            -X \
            -j "$PARENT_JOB_ID" \
            --format=State \
            --noheader \
            --parsable2 \
            2>/dev/null \
        | head -n 1 \
        | tr -d '[:space:]'
    )"


    echo "Parent state: ${{PARENT_STATE:-UNKNOWN}}"

    echo ""


    # ========================================================
    # ONLY TIMEOUT MAY CONTINUE
    # ========================================================

    case "$PARENT_STATE" in

        TIMEOUT|TIMEOUT*)

            echo "Parent array reached SLURM TIMEOUT."

            echo "Automatic continuation is allowed."

            ;;


        COMPLETED|COMPLETED*)

            echo "Parent array COMPLETED normally."

            echo "No restart required."

            echo "Continuation worker terminating."

            exit 0

            ;;


        FAILED|FAILED*)

            echo "Parent array FAILED."

            echo "No automatic restart will be submitted."

            echo "Continuation worker terminating."

            exit 0

            ;;


        CANCELLED|CANCELLED*)

            echo "Parent array was CANCELLED."

            echo "No automatic restart will be submitted."

            echo "Continuation worker terminating."

            exit 0

            ;;


        OUT_OF_MEMORY|OUT_OF_MEMORY*)

            echo "Parent array ran out of memory."

            echo "No automatic restart will be submitted."

            echo "Continuation worker terminating."

            exit 0

            ;;


        NODE_FAIL|NODE_FAIL*)

            echo "Parent array failed because of NODE_FAIL."

            echo "No automatic restart will be submitted."

            echo "Continuation worker terminating."

            exit 0

            ;;


        PREEMPTED|PREEMPTED*)

            echo "Parent array was PREEMPTED."

            echo "No automatic restart will be submitted."

            echo "Continuation worker terminating."

            exit 0

            ;;


        DEADLINE|DEADLINE*)

            echo "Parent array reached a deadline."

            echo "No automatic restart will be submitted."

            echo "Continuation worker terminating."

            exit 0

            ;;


        *)

            echo "WARNING: Parent state could not be safely identified."

            echo "Parent state = ${{PARENT_STATE:-UNKNOWN}}"

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
# SUBMIT ONE CONTINUATION
#
# ONLY ARRAY TASK 1 does this.
#
# This child is submitted before the current array begins.
#
# The child will NOT automatically continue unless this parent
# actually reaches TIMEOUT.
#
# A failed Python job therefore cannot create an endless chain.
# ============================================================

NEXT_JOB_ID=""


if [[ "$SLURM_ARRAY_TASK_ID" -eq 1 ]]; then

    echo ""
    echo "============================================================"
    echo "PREPARING TIMEOUT CONTINUATION"
    echo "============================================================"

    echo "Current array : $SLURM_ARRAY_JOB_ID"

    if [[ -n "$PARENT_JOB_ID" ]]; then

        echo "Parent array  : $PARENT_JOB_ID"
        echo "Parent state  : TIMEOUT"
        echo "This is a genuine continuation."

    else

        echo "This is the first production array."

    fi

    echo ""


    NEXT_JOB_ID="$(
        sbatch \
            --parsable \
            --export=ALL,PROJECT_ROOT="{ROOT}",PARENT_JOB_ID="$SLURM_ARRAY_JOB_ID" \
            --dependency="afterany:${{SLURM_ARRAY_JOB_ID}}" \
            --account="{hpc["account"]}" \
            --partition=nodes \
            --cpus-per-task="{hpc["cores"]}" \
            --mem="{hpc["memory"]}G" \
            --time="{hpc["hours"]}" \
            --array=1-{repeats} \
            --job-name=MACE_{hpc["Runfolder"]} \
            --output="{logs}/mace_%A_%a.out" \
            --error="{logs}/mace_%A_%a.err" \
            "$0"
    )"


    echo "$NEXT_JOB_ID" > "{stage}/next_job_id.txt"


    echo "Next array   : $NEXT_JOB_ID"

    echo "Dependency   : afterany:$SLURM_ARRAY_JOB_ID"

    echo ""

    echo "IMPORTANT: the next array will only continue if"

    echo "this array reaches SLURM TIMEOUT."

    echo "If this array FAILS, the child will exit."

    echo "If this array is CANCELLED, the child will exit."

    echo "If this array COMPLETES, the child will exit."

    echo ""

    echo "============================================================"
    echo ""

fi


# ============================================================
# REPETITION DIRECTORY
# ============================================================

REP_DIR="{repetitions_dir}/rep-$SLURM_ARRAY_TASK_ID"


if [[ ! -d "$REP_DIR" ]]; then

    echo ""
    echo "ERROR: repetition directory does not exist:"
    echo "$REP_DIR"

    exit 1

fi


# ============================================================
# RUN MACE PRODUCTION
# ============================================================

cd "$REP_DIR"


echo ""
echo "============================================================"
echo "STARTING MACE TRAJECTORY"
echo "============================================================"

echo "Repetition : $SLURM_ARRAY_TASK_ID"

echo "Directory  : $(pwd)"

echo "============================================================"
echo ""


exec "$PYTHON" "{main_py}"
'''
    )

    production_script.chmod(0o755)


    # ========================================================
    # SUBMIT SETUP
    # ========================================================

    setup_job = (
        subprocess
        .check_output(
            [
                "sbatch",
                "--parsable",
                str(setup_script)
            ]
        )
        .decode()
        .strip()
        .split(";")[0]
    )


    # ========================================================
    # SUBMIT PRODUCTION ARRAY
    #
    # Production starts ONLY if setup COMPLETES successfully.
    # ========================================================

    production_job = (
        subprocess
        .check_output(
            [
                "sbatch",
                "--parsable",
                f"--dependency=afterok:{setup_job}",
                str(production_script)
            ]
        )
        .decode()
        .strip()
        .split(";")[0]
    )


    # ========================================================
    # FINAL INFORMATION
    # ========================================================

    print()

    print("=" * 70)

    print("PRODUCTION WORKFLOW SUBMITTED")

    print("=" * 70)

    print()

    print(f"Model ID          : {model_id}")

    print(f"Model file        : {model_file}")

    print(f"Setup job         : {setup_job}")

    print(f"Production job    : {production_job}")

    print(f"Repeats           : {repeats}")

    print(
        f"Temperature       : "
        f"{config['Molecule_data']['Temp']} K"
    )

    print(
        f"Multiplicity      : "
        f"{config['Molecule_data']['Multiplicity']}"
    )

    print(f"Timestep          : {timestep}")

    print(f"Total timesteps   : {target_steps}")

    print(f"MACE device       : {mace_device}")

    print(f"MACE dtype        : {mace_dtype}")

    print(f"Production dir    : {stage}")

    print(f"Logs              : {logs}")

    print()

    print("=" * 70)

    print("WORKFLOW ORDER")

    print("=" * 70)

    print()

    print(f"1. Setup job {setup_job}")

    print("   ↓")

    print("2. Environment: plasma")

    print("   ↓")

    print("3. Q-Chem geometry optimisation")

    print("   ↓")

    print("4. Random momentum generation")

    print("   ↓")

    print("5. rep-1 ... rep-N created")

    print("   ↓")

    print(f"6. Production array {production_job}")

    print("   ↓")

    print("7. Environment: mace_env")

    print("   ↓")

    print("8. MACE production trajectories")

    print()

    print(
        f"Production dependency: afterok:{setup_job}"
    )

    print()

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
