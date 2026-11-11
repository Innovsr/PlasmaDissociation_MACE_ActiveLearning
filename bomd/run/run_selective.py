#!/usr/bin/env python3

"""
Stage-1 controller: select validation trajectories and submit Q-Chem BOMD.

The user supplies:

    configs/validation.json

All stage output lives under:

    /mnt/scratch/$USER/<Runfolder>/01_validation

Workflow:

    run_validation.sh
          |
          v
    run_selective.py
          |
          +--> setup job
          |
          +--> validation array
          |
          +--> restart checker
                    |
                    +--> restart array if TIMEOUT + incomplete
                    |
                    +--> next restart checker
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import shutil
import subprocess
import getpass
import textwrap

from pathlib import Path


# ============================================================
# Package locations
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

BOMD_CODE = ROOT / "bomd" / "code"

SETUP = BOMD_CODE / "setup.py"
MAIN = BOMD_CODE / "main.py"

GEOM_DIR = ROOT / "bomd" / "Initial_Geometry"

# Validation geometry source
GEOM_FLG = "Initial"


# ============================================================
# Utility functions
# ============================================================

def load(path):
    """
    Load a JSON input file.
    """

    return json.loads(
        Path(path).read_text()
    )


def make_runtime_inputs(inputs):
    """
    Create the input file used by the actual BOMD trajectories.

    The complete validation/controller configuration is stored in:

        validation.json

    The actual BOMD trajectories use:

        inputs.json

    Trajectory-selection information is deliberately NOT copied
    into inputs.json.
    """

    runtime = {
        "model": inputs.get("model"),

        "HPC": {
            "account": inputs["HPC"]["account"],
            "Runfolder": inputs["HPC"]["Runfolder"],
            "cores": inputs["HPC"]["cores"],
            "hours": inputs["HPC"]["hours"],
            "memory": inputs["HPC"]["memory"]
        },

        "Molecule_data": {
            "Molecule": inputs["Molecule_data"]["Molecule"],
            "Multiplicity": inputs["Molecule_data"]["Multiplicity"],
            "Temp": inputs["Molecule_data"]["Temp"]
        },

        "Elec_structure": dict(
            inputs["Elec_structure"]
        ),

        "Propagation": dict(
            inputs["Propagation"]
        )
    }

    return runtime


def scratch_parent(runfolder):
    """
    Return the user's scratch directory for this run.
    """

    return (
        Path("/mnt/scratch")
        / getpass.getuser()
        / runfolder
    )


# ============================================================
# Main controller
# ============================================================

def main():

    # --------------------------------------------------------
    # Read command-line argument
    # --------------------------------------------------------

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
        help="Path to configs/validation.json"
    )

    args = ap.parse_args()

    # --------------------------------------------------------
    # Read validation configuration
    # --------------------------------------------------------

    inputs = load(args.input)

    h = inputs["HPC"]

    # --------------------------------------------------------
    # Basic validation of inputs
    # --------------------------------------------------------

    if h["cores"] < 1:
        raise SystemExit(
            "HPC cores must be >= 1"
        )

    if h["repeats"] < 1:
        raise SystemExit(
            "HPC repeats must be >= 1"
        )

    # Stage 1 must use QChem
    if inputs["Elec_structure"].get("method") != "QChem":
        raise SystemExit(
            "Validation dynamics must use QChem"
        )

    # --------------------------------------------------------
    # Read trajectory selection
    # --------------------------------------------------------

    sel = inputs.get(
        "Trajectory_selection",
        {}
    )

    if int(sel.get("enabled", 0)) != 1:
        raise SystemExit(
            "Stage 1 requires "
            "Trajectory_selection.enabled = 1"
        )

    selected_count = sum(
        int(x.get("select", 0))
        for x in sel.get("categories", [])
    )

    if selected_count < 1:
        raise SystemExit(
            "No validation trajectories requested"
        )

    print(
        f"Selected validation trajectories: "
        f"{selected_count}"
    )

    # ========================================================
    # Create scratch directory
    # ========================================================

    parent = scratch_parent(
        h["Runfolder"]
    )

    stage = parent / "01_validation"

    stage.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create required directories
    # --------------------------------------------------------

    (stage / "logs").mkdir(
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

    (stage / "validation").mkdir(
        exist_ok=True
    )

    # ========================================================
    # Save validation.json
    # ========================================================

    # This is the complete controller configuration.
    #
    # It includes Trajectory_selection and is kept for
    # validation/setup purposes.

    (stage / "validation.json").write_text(
        json.dumps(
            inputs,
            indent=4
        )
    )

    # ========================================================
    # Create runtime inputs.json
    # ========================================================

    # This is deliberately stripped of trajectory-selection
    # settings.

    runtime_inputs = make_runtime_inputs(
        inputs
    )

    (stage / "inputs.json").write_text(
        json.dumps(
            runtime_inputs,
            indent=4
        )
    )

    # ========================================================
    # Copy initial geometry
    # ========================================================

    geom = (
        GEOM_DIR
        / (
            inputs["Molecule_data"]["Molecule"]
            + ".txt"
        )
    )

    if not geom.exists():
        raise SystemExit(
            f"Initial geometry not found: {geom}"
        )

    shutil.copy2(
        geom,
        stage / "setup" / "initial_guess.txt"
    )

    # ========================================================
    # Copy restart checker into scratch
    # ========================================================

    restart_template = (
        ROOT
        / "01_validation"
        / "restart_validation.sh"
    )

    if not restart_template.exists():
        raise SystemExit(
            "restart_validation.sh was not found at:\n"
            f"{restart_template}"
        )

    restart_checker = (
        stage
        / "restart_validation.sh"
    )

    shutil.copy2(
        restart_template,
        restart_checker
    )

    restart_checker.chmod(
        0o755
    )

    # ========================================================
    # Create SETUP SLURM script
    # ========================================================

    setup_script = (
        stage
        / "setup_validation.sh"
    )

    setup_script.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash

            #SBATCH --account={h["account"]}
            #SBATCH --job-name=Setup_{h["Runfolder"]}
            #SBATCH --time=02:00:00
            #SBATCH --cpus-per-task={h["cores"]}
            #SBATCH --mem={h["memory"]}G
            #SBATCH --output={stage}/logs/setup_%j.out
            #SBATCH --error={stage}/logs/setup_%j.err

            set -eo pipefail

            module unload qchem 2>/dev/null || true
            module load qchem/6.0.2

            export LM_LICENSE_FILE=27000@uol-lnx-lic01.leeds.ac.uk

            export QC="$QCHEM_HOME"
            export QCAUX="$QC/qcaux"
            export QCPROG="$QC/exe/qcprog.exe"
            export QCPROG_S="$QC/exe/qcprog.exe_s"

            export PATH="$PATH:$QC/exe:$QC/bin"

            export QCSCRATCH="{stage}/setup/tmp"

            cd "{stage}/setup"

            exec "{sys.executable}" "{SETUP}"
            """
        )
    )

    setup_script.chmod(
        0o755
    )

    # ========================================================
    # Create VALIDATION ARRAY SLURM script
    # ========================================================

    array_script = (
        stage
        / "validation_array.sh"
    )

    array_script.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash

            #SBATCH --account={h["account"]}
            #SBATCH --job-name=Validation_{h["Runfolder"]}
            #SBATCH --time={hpc["bomd_time"]}
            #SBATCH --cpus-per-task={h["cores"]}
            #SBATCH --mem={h["memory"]}G
            #SBATCH --array=1-{selected_count}
            #SBATCH --output={stage}/logs/validation_%A_%a.out
            #SBATCH --error={stage}/logs/validation_%A_%a.err

            set -eo pipefail

            module unload qchem 2>/dev/null || true
            module load qchem/6.0.2

            export LM_LICENSE_FILE=27000@uol-lnx-lic01.leeds.ac.uk

            export QC="$QCHEM_HOME"
            export QCAUX="$QC/qcaux"
            export QCPROG="$QC/exe/qcprog.exe"
            export QCPROG_S="$QC/exe/qcprog.exe_s"

            export PATH="$PATH:$QC/exe:$QC/bin"

            export QCSCRATCH="{stage}/repetitions/rep-$SLURM_ARRAY_TASK_ID/tmp"

            cd "{stage}/repetitions/rep-$SLURM_ARRAY_TASK_ID"

            exec "{sys.executable}" "{MAIN}"
            """
        )
    )

    array_script.chmod(
        0o755
    )

    # ========================================================
    # Decide whether SETUP is required
    # ========================================================

    reps = list(
        (stage / "repetitions").glob("rep-*")
    )

    if (
        reps
        and len(reps) >= selected_count
    ):

        setup_job = None

        print(
            "Existing repetition directories found."
        )

        print(
            "Skipping setup submission."
        )

    else:

        setup_job = (
            subprocess
            .check_output(
                [
                    "sbatch",
                    "--parsable",
                    str(setup_script)
                ],
                text=True
            )
            .strip()
            .split(";")[0]
        )

        print(
            f"Validation setup job: "
            f"{setup_job}"
        )

    # ========================================================
    # Submit VALIDATION ARRAY
    # ========================================================

    if setup_job:

        # Validation starts ONLY after setup succeeds.

        jid = (
            subprocess
            .check_output(
                [
                    "sbatch",
                    "--parsable",
                    "--dependency=afterok:" + setup_job,
                    str(array_script)
                ],
                text=True
            )
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
                    str(array_script)
                ],
                text=True
            )
            .strip()
            .split(";")[0]
        )

    print(
        f"Validation array job: {jid}"
    )

    # ========================================================
    # Submit FIRST restart checker
    # ========================================================

    # The checker waits until the ENTIRE validation array
    # reaches a terminal state.
    #
    # afterany is intentional:
    #
    #   COMPLETED -> checker runs
    #   TIMEOUT   -> checker runs
    #   FAILED    -> checker runs
    #
    # restart_validation.py itself decides whether an
    # automatic restart is permitted.
    #
    # At present, only TIMEOUT triggers restart.

    checker_job = (
        subprocess
        .check_output(
            [
                "sbatch",
                "--parsable",
                "--dependency=afterany:" + jid,
                "--export=ALL,VALIDATION_ARRAY_JOB_ID=" + jid,
                str(restart_checker),
                str(stage),
                str(ROOT)
            ],
            text=True
        )
        .strip()
        .split(";")[0]
    )

    print(
        f"Validation restart checker job: "
        f"{checker_job}"
    )

    # ========================================================
    # Final information
    # ========================================================

    print(
        f"Validation directory: {stage}"
    )

    print("")

    print(
        "Validation workflow submitted:"
    )

    if setup_job:

        print(
            f"    Setup job:        {setup_job}"
        )

    print(
        f"    Validation array: {jid}"
    )

    print(
        f"    Restart checker:  {checker_job}"
    )

    print("")

    print(
        "The restart checker will run automatically "
        "after the validation array finishes."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
