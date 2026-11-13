#!/usr/bin/env python3

from __future__ import annotations

import argparse
import getpass
import json
import os
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

def load(path):
    return json.loads(Path(path).read_text())


# ============================================================
# FIND MODEL
#
# Production uses the model ID from production.json:
#
#     "model": "CF3-CHF-CHF2_ea_001"
#
# and automatically looks for:
#
#     models/CF3-CHF-CHF2_ea_001/model.model
#
# No registry or approval status is required.
# ============================================================

def find_model(model_id):

    model_dir = ROOT / "models" / model_id

    model_file = model_dir / "model.model"

    if not model_file.exists():

        raise SystemExit(
            "\n"
            "ERROR: MACE model was not found.\n"
            "\n"
            f"Model ID   : {model_id}\n"
            f"Expected   : {model_file}\n"
            "\n"
            "Please check that the model directory exists and\n"
            "contains model.model.\n"
        )

    return model_dir


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # COMMAND-LINE ARGUMENTS
    # ========================================================

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
        help="Path to production JSON configuration"
    )

    args = ap.parse_args()


    # ========================================================
    # LOAD CONFIGURATION
    # ========================================================

    config_file = Path(args.input).resolve()

    if not config_file.exists():

        raise SystemExit(
            f"ERROR: Production configuration not found:\n"
            f"{config_file}"
        )

    c = load(config_file)


    # ========================================================
    # MODEL
    # ========================================================

    model_id = c["model"]

    model_dir = find_model(model_id)

    model_file = model_dir / "model.model"


    # ========================================================
    # HPC SETTINGS
    # ========================================================

    h = c["HPC"]

    user = getpass.getuser()

    parent = (
        Path("/mnt/scratch")
        / user
        / h["Runfolder"]
    )

    stage = parent / "03_production"


    # ========================================================
    # CREATE PRODUCTION DIRECTORIES
    # ========================================================

    stage.mkdir(
        parents=True,
        exist_ok=True
    )

    (stage / "setup").mkdir(
        exist_ok=True
    )

    (stage / "setup" / "tmp").mkdir(
        parents=True,
        exist_ok=True
    )

    (stage / "repetitions").mkdir(
        exist_ok=True
    )

    (stage / "logs").mkdir(
        exist_ok=True
    )


    # ========================================================
    # COPY INPUT CONFIGURATION
    # ========================================================

    shutil.copy2(
        config_file,
        stage / "inputs.json"
    )


    # ========================================================
    # INITIAL GEOMETRY
    # ========================================================

    molecule = c["Molecule_data"]["Molecule"]

    geom = (
        BOMD
        / "Initial_Geometry"
        / f"{molecule}.txt"
    )

    if not geom.exists():

        raise SystemExit(
            "\n"
            "ERROR: Initial geometry not found.\n"
            f"{geom}\n"
        )


    # ========================================================
    # COPY INITIAL GEOMETRY INTO SETUP DIRECTORY
    # ========================================================

    shutil.copy2(
        geom,
        stage / "setup" / "initial_guess.txt"
    )


    # ========================================================
    # PYTHON
    #
    # The Python executable used to submit this workflow.
    #
    # run_production.sh activates mace_env automatically,
    # therefore this should normally be:
    #
    # /users/<user>/.conda/envs/mace_env/bin/python
    # ========================================================

    py = os.sys.executable


    # ========================================================
    # PYTHON SCRIPTS
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
    # RUN NAME
    # ========================================================

    runname = h["Runfolder"]


    # ========================================================
    # SETUP SCRIPT
    #
    # IMPORTANT:
    #
    # This job runs FIRST.
    #
    # setup.py:
    #
    #   1. performs the initial calculation
    #   2. obtains the required initial conditions
    #   3. generates random momenta
    #   4. creates rep-1, rep-2, ...
    #
    # Only after this job COMPLETES successfully will
    # the production array be released.
    # ========================================================

    setup_script = (
        stage
        / "setup_production.sh"
    )


    setup_script.write_text(
        f'''#!/bin/bash

#SBATCH --account={h["account"]}
#SBATCH --job-name=Setup_{runname}
#SBATCH --time={h["hours"]}:00:00
#SBATCH --cpus-per-task={h["cores"]}
#SBATCH --mem={h["memory"]}G
#SBATCH --output={stage}/logs/setup_%j.out
#SBATCH --error={stage}/logs/setup_%j.err

set -euo pipefail


# ============================================================
# Q-CHEM ENVIRONMENT
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

export QCSCRATCH="{stage}/setup/tmp"


# ============================================================
# SETUP DIRECTORY
# ============================================================

cd "{stage}/setup"


# ============================================================
# RUN SETUP
# ============================================================

exec "{py}" "{setup_py}" --input "../inputs.json"
'''
    )


    setup_script.chmod(0o755)


    # ========================================================
    # PRODUCTION PARAMETERS
    # ========================================================

    target = int(
        c["Propagation"]["Tot_timesteps"]
    )

    dt = float(
        c["Propagation"]["Timestep"]
    )

    n = int(
        h["repeats"]
    )


    # ========================================================
    # MACE SETTINGS
    # ========================================================

    mace_config = c.get(
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
    # PRODUCTION ARRAY SCRIPT
    # ========================================================

    arr = (
        stage
        / "production_array.sh"
    )


    arr.write_text(
        f'''#!/bin/bash

#SBATCH --account={h["account"]}
#SBATCH --job-name=MACE_{runname}
#SBATCH --time={h["hours"]}:00:00
#SBATCH --cpus-per-task={h["cores"]}
#SBATCH --mem={h["memory"]}G
#SBATCH --array=1-{n}
#SBATCH --output={stage}/logs/mace_%A_%a.out
#SBATCH --error={stage}/logs/mace_%A_%a.err

set -euo pipefail


# ============================================================
# MACE MODEL
# ============================================================

export MACE_MODEL_PATH="{model_file}"

export MACE_DEVICE="{mace_device}"

export MACE_DTYPE="{mace_dtype}"


# ============================================================
# DISPLAY MODEL INFORMATION
# ============================================================

echo "============================================================"
echo "MACE PRODUCTION WORKER"
echo "============================================================"

echo "Job ID          : ${{SLURM_JOB_ID:-UNKNOWN}}"
echo "Array Job ID    : ${{SLURM_ARRAY_JOB_ID:-UNKNOWN}}"
echo "Array Task ID   : ${{SLURM_ARRAY_TASK_ID:-UNKNOWN}}"

echo "Model ID        : {model_id}"

echo "Model file      : $MACE_MODEL_PATH"

echo "MACE device     : $MACE_DEVICE"

echo "MACE dtype      : $MACE_DTYPE"

echo "Python          : {py}"

echo "============================================================"


# ============================================================
# CHECK MODEL
# ============================================================

if [[ ! -f "$MACE_MODEL_PATH" ]]; then

    echo "ERROR: MACE model file does not exist:"
    echo "$MACE_MODEL_PATH"

    exit 1

fi


# ============================================================
# ARRAY TASK 1
#
# The restart manager is checked only by task 1.
#
# If all trajectories are already complete, task 1 exits.
#
# Otherwise it prepares a continuation job after the array.
# ============================================================

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then

    if "{py}" "{RESTART}" \
        --run-root "{stage}" \
        --target-steps {target} \
        --timestep {dt} \
        --all-complete
    then

        echo "All production trajectories are complete."

        exit 0

    fi


    # --------------------------------------------------------
    # DO NOT blindly create an unlimited restart chain.
    #
    # This is the existing production restart mechanism.
    # --------------------------------------------------------

    sbatch \
        --parsable \
        --dependency=afterany:${{SLURM_ARRAY_JOB_ID}} \
        "$0" \
        > "{stage}/logs/next_job_id.txt"

fi


# ============================================================
# SELECT TRAJECTORY DIRECTORY
# ============================================================

REP_DIR="{stage}/repetitions/rep-$SLURM_ARRAY_TASK_ID"


if [[ ! -d "$REP_DIR" ]]; then

    echo "ERROR: Repetition directory does not exist:"
    echo "$REP_DIR"

    echo ""
    echo "This normally means the setup job did not create"
    echo "the required random-momentum initial condition."

    exit 1

fi


# ============================================================
# ENTER TRAJECTORY DIRECTORY
# ============================================================

cd "$REP_DIR"


# ============================================================
# RUN PRODUCTION
# ============================================================

exec "{py}" "{main_py}"
'''
    )


    arr.chmod(0o755)


    # ========================================================
    # CHECK EXISTING REPETITIONS
    #
    # If all repetitions already exist, setup does not need
    # to run again.
    # ========================================================

    reps = list(
        (stage / "repetitions").glob("rep-*")
    )


    if reps and len(reps) >= n:

        setup_job = None

        print(
            "Existing repetition directories detected."
        )

        print(
            f"Found {len(reps)} repetition directories."
        )

        print(
            f"Required {n} repetition directories."
        )

        print(
            "Setup job will NOT be submitted."
        )


    else:

        # ====================================================
        # SUBMIT SETUP FIRST
        # ====================================================

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


        print(
            f"Setup job submitted: {setup_job}"
        )


    # ========================================================
    # SUBMIT PRODUCTION ARRAY
    #
    # IMPORTANT:
    #
    # If setup_job exists:
    #
    #     production -> afterok -> setup
    #
    # Therefore production CANNOT start before setup has
    # completed successfully.
    # ========================================================

    if setup_job:

        jid = (
            subprocess
            .check_output(
                [
                    "sbatch",
                    "--parsable",
                    f"--dependency=afterok:{setup_job}",
                    str(arr)
                ]
            )
            .decode()
            .strip()
            .split(";")[0]
        )

    else:

        jid = (
            subprocess
            .check_output(
                [
                    "sbatch",
                    "--parsable",
                    str(arr)
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

    print(
        f"Model ID          : {model_id}"
    )

    print(
        f"Model file        : {model_file}"
    )

    print(
        f"Setup job         : {setup_job or 'not required'}"
    )

    print(
        f"Production job    : {jid}"
    )

    print(
        f"Repeats           : {n}"
    )

    print(
        f"Temperature       : "
        f"{c['Molecule_data']['Temp']} K"
    )

    print(
        f"Multiplicity      : "
        f"{c['Molecule_data']['Multiplicity']}"
    )

    print(
        f"Timestep          : {dt}"
    )

    print(
        f"Total timesteps   : {target}"
    )

    print(
        f"MACE device       : {mace_device}"
    )

    print(
        f"MACE dtype        : {mace_dtype}"
    )

    print(
        f"Production dir    : {stage}"
    )

    print(
        f"Logs              : {stage / 'logs'}"
    )

    print("=" * 70)

    if setup_job:

        print()
        print(
            "WORKFLOW ORDER:"
        )

        print(
            f"  1. Setup job {setup_job}"
        )

        print(
            "     ↓"
        )

        print(
            "  2. Geometry / initial-condition calculation"
        )

        print(
            "     ↓"
        )

        print(
            "  3. Random momentum generation"
        )

        print(
            "     ↓"
        )

        print(
            "  4. rep-1 ... rep-N created"
        )

        print(
            "     ↓"
        )

        print(
            f"  5. Production array {jid}"
        )

        print(
            "     ↓"
        )

        print(
            "  6. MACE production trajectories"
        )

        print()

        print(
            f"Production dependency: afterok:{setup_job}"
        )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
