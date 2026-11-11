#!/usr/bin/env python3

"""
Stage-1 Validation workflow launcher/controller.

Usage:

    bash run_validation.sh

This script:

    1. Reads configs/validation.json
    2. Creates the 01_validation scratch directory
    3. Creates the runtime inputs.json
    4. Copies the initial geometry
    5. Submits the setup job
    6. The setup job performs the trajectory selection
    7. Submits the selected validation BOMD array
    8. Submits validation_controller.py as the automatic checker

The restart mechanism is handled by:

    validation_controller.py
            |
            v
    restart_manager.py

No restart_validation.sh is required.
"""

from __future__ import annotations

import argparse
import getpass
import json
import shutil
import subprocess
import sys
import textwrap

from pathlib import Path


# ============================================================
# REPOSITORY LOCATIONS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BOMD_CODE = ROOT / "bomd" / "code"

SETUP = BOMD_CODE / "setup.py"
MAIN = BOMD_CODE / "main.py"

GEOM_DIR = ROOT / "bomd" / "Initial_Geometry"

CONTROLLER = (
    ROOT
    / "01_validation"
    / "validation_controller.py"
)


# ============================================================
# UTILITIES
# ============================================================

def load_config(path: Path) -> dict:

    with path.open() as f:
        return json.load(f)


def scratch_parent(runfolder: str) -> Path:

    return (
        Path("/mnt/scratch")
        / getpass.getuser()
        / runfolder
    )

def slurm_time(hours) -> str:
    """
    Accept either:

        1
        "1"
        "01:00:00"
        "00:15:00"
        "2-00:00:00"

    and return a valid Slurm time specification.
    """

    if isinstance(hours, int):
        if hours < 1:
            raise ValueError("HPC hours must be >= 1")
        return f"{hours:02d}:00:00"

    hours = str(hours).strip()

    if ":" in hours or "-" in hours:
        return hours

    value = int(hours)

    if value < 1:
        raise ValueError("HPC hours must be >= 1")

    return f"{value:02d}:00:00"


def make_runtime_inputs(inputs: dict) -> dict:
    """
    Create the inputs.json used by the actual BOMD trajectories.

    Trajectory-selection settings remain in validation.json.
    They are not needed by main.py during BOMD propagation.
    """

    return {
        "model": inputs.get("model"),

        "HPC": {
            "account": inputs["HPC"]["account"],
            "Runfolder": inputs["HPC"]["Runfolder"],
            "cores": inputs["HPC"]["cores"],
            "hours": inputs["HPC"]["hours"],
            "memory": inputs["HPC"]["memory"],
            "repeats": inputs["HPC"]["repeats"]
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


def submit(command):

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )

    return (
        result.stdout
        .strip()
        .split(";")[0]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to configs/validation.json"
    )

    args = parser.parse_args()

    config_path = Path(
        args.config
    ).resolve()

    inputs = load_config(
        config_path
    )

    hpc = inputs["HPC"]

    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------

    if int(hpc["cores"]) < 1:
        raise SystemExit(
            "HPC cores must be >= 1"
        )

    if int(hpc["repeats"]) < 1:
        raise SystemExit(
            "HPC repeats must be >= 1"
        )

    if inputs["Elec_structure"].get("method") != "QChem":
        raise SystemExit(
            "Validation dynamics must use QChem"
        )

    selection = inputs.get(
        "Trajectory_selection",
        {}
    )

    if int(selection.get("enabled", 0)) != 1:
        raise SystemExit(
            "Validation requires "
            "Trajectory_selection.enabled = 1"
        )

    selected_count = sum(
        int(category.get("select", 0))
        for category in selection.get(
            "categories",
            []
        )
    )

    if selected_count < 1:
        raise SystemExit(
            "No validation trajectories requested"
        )

    # Keep repeats consistent with the number of selected
    # trajectories.

    if int(hpc["repeats"]) != selected_count:

        print(
            "WARNING: HPC.repeats does not match "
            "the number of selected trajectories."
        )

        print(
            f"    HPC.repeats      = {hpc['repeats']}"
        )

        print(
            f"    selected_count   = {selected_count}"
        )

        print(
            "Using selected_count for the validation array."
        )

    print(
        f"Selected validation trajectories: "
        f"{selected_count}"
    )

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    parent = scratch_parent(
        hpc["Runfolder"]
    )

    stage = (
        parent
        / "01_validation"
    )

    stage.mkdir(
        parents=True,
        exist_ok=True
    )

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

    # --------------------------------------------------------
    # Save complete validation configuration
    # --------------------------------------------------------

    (stage / "validation.json").write_text(
        json.dumps(
            inputs,
            indent=4
        )
    )

    # --------------------------------------------------------
    # Create runtime inputs.json
    # --------------------------------------------------------

    runtime_inputs = make_runtime_inputs(
        inputs
    )

    (stage / "inputs.json").write_text(
        json.dumps(
            runtime_inputs,
            indent=4
        )
    )

    # --------------------------------------------------------
    # Copy initial geometry
    # --------------------------------------------------------

    molecule = inputs[
        "Molecule_data"
    ]["Molecule"]

    geometry = (
        GEOM_DIR
        / f"{molecule}.txt"
    )

    if not geometry.exists():

        raise SystemExit(
            f"Initial geometry not found:\n{geometry}"
        )

    shutil.copy2(
        geometry,
        stage
        / "setup"
        / "initial_guess.txt"
    )

    # --------------------------------------------------------
    # Slurm times
    # --------------------------------------------------------

    setup_time = "02:00:00"

    bomd_time = slurm_time(
        hpc["hours"]
    )

    # --------------------------------------------------------
    # Create setup job
    #
    # The setup job runs bomd/code/setup.py.
    #
    # setup.py performs:
    #
    #   geometry optimisation
    #   normal-mode generation
    #   momentum generation
    #   C-C/C-F/C-H selection
    #   creation of rep-1 ... rep-30
    # --------------------------------------------------------

    setup_script = (
        stage
        / "setup_validation.sh"
    )

    setup_script.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash

            #SBATCH --account={hpc["account"]}
            #SBATCH --job-name=ValidationSetup
            #SBATCH --time={setup_time}
            #SBATCH --cpus-per-task={hpc["cores"]}
            #SBATCH --mem={hpc["memory"]}G
            #SBATCH --output={stage}/logs/setup_%j.out
            #SBATCH --error={stage}/logs/setup_%j.err

            set -euo pipefail

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

    # --------------------------------------------------------
    # Create validation BOMD array
    # --------------------------------------------------------

    array_script = (
        stage
        / "validation_array.sh"
    )

    array_script.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash

            #SBATCH --account={hpc["account"]}
            #SBATCH --job-name=ValidationBOMD
            #SBATCH --time={bomd_time}
            #SBATCH --cpus-per-task={hpc["cores"]}
            #SBATCH --mem={hpc["memory"]}G
            #SBATCH --array=1-{selected_count}
            #SBATCH --output={stage}/logs/validation_%A_%a.out
            #SBATCH --error={stage}/logs/validation_%A_%a.err

            set -euo pipefail

            module unload qchem 2>/dev/null || true
            module load qchem/6.0.2

            export LM_LICENSE_FILE=27000@uol-lnx-lic01.leeds.ac.uk

            export QC="$QCHEM_HOME"
            export QCAUX="$QC/qcaux"
            export QCPROG="$QC/exe/qcprog.exe"
            export QCPROG_S="$QC/exe/qcprog.exe_s"

            export PATH="$PATH:$QC/exe:$QC/bin"

            export QCSCRATCH="{stage}/repetitions/rep-$SLURM_ARRAY_TASK_ID/tmp"

            mkdir -p "$QCSCRATCH"

            cd "{stage}/repetitions/rep-$SLURM_ARRAY_TASK_ID"

            exec "{sys.executable}" "{MAIN}"
            """
        )
    )

    array_script.chmod(
        0o755
    )

    # --------------------------------------------------------
    # Create first validation controller submission
    # --------------------------------------------------------

    # The controller will start only after the validation
    # array reaches a terminal state.

    print()
    print("=" * 60)
    print("VALIDATION WORKFLOW")
    print("=" * 60)

    print(
        "Run directory:",
        stage
    )

    print(
        "Setup time:",
        setup_time
    )

    print(
        "BOMD time:",
        bomd_time
    )

    print(
        "Trajectories:",
        selected_count
    )

    print("=" * 60)

    setup_job = submit(
        [
            "sbatch",
            "--parsable",
            str(setup_script)
        ]
    )

    print(
        "Setup job:",
        setup_job
    )

    array_job = submit(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterok:{setup_job}",
            str(array_script)
        ]
    )

    print(
        "Validation array:",
        array_job
    )

    controller_script = stage / "validation_controller.sh"

    controller_script.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
    
            #SBATCH --account={hpc["account"]}
            #SBATCH --job-name=ValidationCheck
            #SBATCH --time=00:15:00
            #SBATCH --cpus-per-task=1
            #SBATCH --mem=1G
            #SBATCH --output={stage}/logs/controller_%j.out
            #SBATCH --error={stage}/logs/controller_%j.err
    
            set -euo pipefail
    
            exec "{sys.executable}" "{CONTROLLER}" \
                --run-root "{stage}" \
                --array-job "{array_job}"
            """
        )
    )
    
    controller_script.chmod(0o755)

    controller_job = submit(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{array_job}",
            str(controller_script)
        ]
    )

    #controller_job = submit(
    #    [
    #        "sbatch",
    #        "--parsable",
    #        f"--dependency=afterany:{array_job}",
    #        "--cpus-per-task=1",
    #        "--mem=1G",
    #        "--time=00:15:00",
    #        "--job-name=ValidationCheck",
    #        f"--output={stage}/logs/controller_%j.out",
    #        f"--error={stage}/logs/controller_%j.err",
    #        sys.executable,
    #        str(CONTROLLER),
    #        "--run-root",
    #        str(stage),
    #        "--array-job",
    #        str(array_job)
    #    ]
    #)

    print(
        "Validation controller:",
        controller_job
    )

    print()
    print(
        "Validation workflow submitted successfully."
    )

    print(
        "Monitor with:"
    )

    print(
        "    squeue -u $USER"
    )


if __name__ == "__main__":

    main()
