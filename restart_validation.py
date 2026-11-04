#!/usr/bin/env python3

"""
Automatic validation/restart controller for Stage 1.

Workflow:

    validation array
          |
          v
    restart_validation.py
          |
          +---- all trajectories complete
          |             |
          |             v
          |       finalize validation
          |             |
          |             v
          |       submit MACE training
          |
          +---- incomplete trajectories
                        |
                        v
                  restart array
                        |
                        v
                restart_validation.py
                        |
                       ...

The controller is data-driven.

A trajectory is considered complete only when the latest common
timestep in:

    xyz.all
    momenta.all
    forces.all
    time.all

has reached:

    Propagation.Tot_timesteps

No .bomd_complete file is used.

Only TIMEOUT trajectories are automatically restarted.

FAILED / CANCELLED trajectories are not automatically restarted.
"""


from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ============================================================
# SLURM ARRAY STATE
# ============================================================

def get_array_states(array_job_id):
    """
    Return SLURM state for each array task.

    Example:

        9029631_1    COMPLETED
        9029631_2    TIMEOUT
        9029631_3    COMPLETED
    """

    result = subprocess.run(
        [
            "sacct",
            "-n",
            "-P",
            "-j",
            str(array_job_id),
            "--format=JobID,State",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    states = {}

    for line in result.stdout.splitlines():

        if not line.strip():
            continue

        parts = line.strip().split("|")

        if len(parts) != 2:
            continue

        jobid = parts[0].strip()
        state = parts[1].strip().split()[0]

        # Ignore batch/extern subjobs
        if "." in jobid:
            continue

        states[jobid] = state

    return states


# ============================================================
# TIMESTEP READING
# ============================================================

def check_timestep_file(filename):
    """
    Return all timestep numbers found in a trajectory file.
    """

    timesteps = set()

    if not filename.exists():
        return timesteps

    with filename.open(
        "r",
        errors="replace"
    ) as f:

        for line in f:

            if "Timestep:" not in line:
                continue

            parts = line.split()

            try:
                timestep = int(float(parts[1]))
                timesteps.add(timestep)

            except (
                ValueError,
                IndexError
            ):
                continue

    return timesteps


# ============================================================
# COMMON TIMESTEP
# ============================================================

def get_common_timestep(repdir):
    """
    Find the latest timestep present in ALL required files.

        xyz.all
        momenta.all
        forces.all
        time.all

    This is the only timestep used to determine completion.
    """

    output = repdir / "output"

    xyz = check_timestep_file(
        output / "xyz.all"
    )

    momenta = check_timestep_file(
        output / "momenta.all"
    )

    forces = check_timestep_file(
        output / "forces.all"
    )

    times = check_timestep_file(
        output / "time.all"
    )

    if not xyz or not momenta or not forces or not times:
        return None

    common = (
        xyz
        & momenta
        & forces
        & times
    )

    if not common:
        return None

    return max(common)


# ============================================================
# CHECK ALL TRAJECTORIES
# ============================================================

def inspect_all_trajectories(
    run_root,
    target_steps,
):
    """
    Inspect every repetition directory.

    Returns:

        complete_tasks
        incomplete_tasks
        missing_tasks
    """

    repetitions = (
        run_root
        / "repetitions"
    )

    complete = []
    incomplete = []
    missing = []

    for repdir in sorted(
        repetitions.glob("rep-*"),
        key=lambda p: int(
            p.name.split("-")[1]
        )
    ):

        try:
            task_id = int(
                repdir.name.split("-")[1]
            )
        except (
            ValueError,
            IndexError
        ):
            continue

        common = get_common_timestep(
            repdir
        )

        if common is None:

            missing.append(task_id)

            print(
                f"rep-{task_id}: "
                "NO COMMON TIMESTEP"
            )

        elif common >= target_steps:

            complete.append(task_id)

            print(
                f"rep-{task_id}: "
                f"COMPLETE "
                f"(common timestep={common}, "
                f"target={target_steps})"
            )

        else:

            incomplete.append(task_id)

            print(
                f"rep-{task_id}: "
                f"INCOMPLETE "
                f"(common timestep={common}, "
                f"target={target_steps})"
            )

    return (
        complete,
        incomplete,
        missing,
    )


# ============================================================
# FINALIZE VALIDATION
# ============================================================

def finalize_validation(
    run_root
):
    """
    Convert the completed BOMD trajectories into validation.extxyz.
    """

    package_root = (
        Path(__file__)
        .resolve()
        .parent
    )

    finalize_script = (
        package_root
        / "finalize_validation.py"
    )

    output_file = (
        run_root
        / "validation"
        / "validation.extxyz"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 72)
    print("FINALIZING VALIDATION DATA")
    print("=" * 72)

    print(
        "Output:",
        output_file
    )

    subprocess.run(
        [
            sys.executable,
            str(finalize_script),
            "--run-root",
            str(run_root),
            "--output",
            str(output_file),
        ],
        check=True,
    )

    print(
        "Validation finalization completed."
    )

    return output_file


# ============================================================
# SUBMIT MACE TRAINING
# ============================================================

def submit_training(
    run_root,
    package_root,
):
    """
    Submit the MACE model-update controller after ALL
    validation trajectories are complete.

    Uses the existing:

        02_model_update/run_model_update.sh

    """

    training_script = (
        package_root
        / "02_model_update"
        / "run_model_update.sh"
    )

    if not training_script.exists():

        raise FileNotFoundError(
            "MACE training controller not found:\n"
            f"{training_script}"
        )

    print()
    print("=" * 72)
    print("ALL VALIDATION TRAJECTORIES COMPLETE")
    print("=" * 72)

    print(
        "Submitting MACE training controller:"
    )

    print(
        training_script
    )

    # Submit training as a separate Slurm job.
    #
    # The validator has already performed the completion check,
    # so training is submitted only from this successful path.

    result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            str(training_script),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    training_job_id = (
        result.stdout
        .strip()
        .split(";")[0]
    )

    print(
        "MACE training job submitted:",
        training_job_id
    )

    # Save the job ID for easy tracking.
    training_id_file = (
        run_root
        / "training_job_id.txt"
    )

    training_id_file.write_text(
        training_job_id + "\n"
    )

    return training_job_id


# ============================================================
# CREATE RESTART SCRIPT
# ============================================================

def create_restart_script(
    run_root,
    inputs,
    restart_tasks,
):
    """
    Create a Slurm array containing ONLY incomplete
    trajectories.

    The trajectory itself determines the restart point.
    """

    array_spec = ",".join(
        str(x)
        for x in restart_tasks
    )

    restart_script = (
        run_root
        / "validation_restart.sh"
    )

    package_root = (
        Path(__file__)
        .resolve()
        .parent
    )

    main_py = (
        package_root
        / "bomd"
        / "code"
        / "main.py"
    )

    restart_script.write_text(
        f"""#!/bin/bash

#SBATCH --account={inputs["HPC"]["account"]}
#SBATCH --job-name=ValidationRestart
#SBATCH --time={inputs["HPC"]["hours"]}:00:00
#SBATCH --cpus-per-task={inputs["HPC"]["cores"]}
#SBATCH --mem={inputs["HPC"]["memory"]}G
#SBATCH --array={array_spec}
#SBATCH --output={run_root}/logs/restart_%A_%a.out
#SBATCH --error={run_root}/logs/restart_%A_%a.err

set -eo pipefail

module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate plasma

PYTHON="$(command -v python)"

module unload qchem 2>/dev/null || true
module load qchem/6.0.2

export LM_LICENSE_FILE=27000@uol-lnx-lic01.leeds.ac.uk

export QC="$QCHEM_HOME"
export QCAUX="$QC/qcaux"
export QCPROG="$QC/exe/qcprog.exe"
export QCPROG_S="$QC/exe/qcprog.exe_s"

export PATH="$PATH:$QC/exe:$QC/bin"

export QCSCRATCH="{run_root}/repetitions/rep-$SLURM_ARRAY_TASK_ID/tmp"

# Tell init.py that this is a restart.
export BOMD_RESTART=1

cd "{run_root}/repetitions/rep-$SLURM_ARRAY_TASK_ID"

echo "=========================================="
echo "VALIDATION RESTART"
echo "=========================================="
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Run root: {run_root}"
echo "Restart directory: $(pwd)"
echo "Python: $PYTHON"
echo "=========================================="

exec "$PYTHON" "{main_py}"
"""
    )

    restart_script.chmod(
        0o755
    )

    return restart_script


# ============================================================
# SUBMIT NEXT VALIDATION CHECKER
# ============================================================

def submit_next_checker(
    restart_job_id,
    run_root,
    package_root,
):
    """
    Submit another validation checker after the newly submitted
    restart array.

    This is what makes the restart system continuous.

        restart array
              |
              v
        validation checker
              |
              v
        restart array
              |
              v
             ...

    """

    checker_script = (
        package_root
        / "01_validation"
        / "restart_validation.sh"
    )

    if not checker_script.exists():

        raise FileNotFoundError(
            "Validation checker script not found:\n"
            f"{checker_script}"
        )

    command = [
        "sbatch",
        "--parsable",

        "--dependency=afterany:"
        + str(restart_job_id),

        "--export=ALL,"
        "VALIDATION_ARRAY_JOB_ID="
        + str(restart_job_id),

        str(checker_script),

        str(run_root),

        str(package_root),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    checker_job_id = (
        result.stdout
        .strip()
        .split(";")[0]
    )

    print(
        "Next validation checker submitted:",
        checker_job_id
    )

    print(
        "Checker dependency:",
        restart_job_id
    )

    return checker_job_id


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-root",
        required=True,
    )

    parser.add_argument(
        "--array-job",
        required=True,
    )

    args = parser.parse_args()

    run_root = Path(
        args.run_root
    ).resolve()

    package_root = (
        Path(__file__)
        .resolve()
        .parent
    )

    # --------------------------------------------------------
    # Load runtime input
    # --------------------------------------------------------

    inputs_file = (
        run_root
        / "inputs.json"
    )

    if not inputs_file.exists():

        raise FileNotFoundError(
            f"Missing inputs.json: {inputs_file}"
        )

    with inputs_file.open() as f:

        inputs = json.load(f)
    
    nsteps = int(
    inputs["Propagation"]["Tot_timesteps"]
    )
    
    dt = float(
        inputs["Propagation"]["Timestep"]
    )
    
    target_steps = nsteps * dt

    #target_steps = (
    #int(inputs["Propagation"]["Tot_timesteps"])
    #* float(inputs["Propagation"]["Timestep"])
    #)   

    # --------------------------------------------------------
    # Print controller information
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PLASMA VALIDATION / RESTART CONTROLLER")
    print("=" * 72)

    print(
        "Run root:",
        run_root
    )

    print(
        "Package root:",
        package_root
    )

    print(
        "Previous array job:",
        args.array_job
    )

    print(
        "Target timestep:",
        target_steps
    )

    print("=" * 72)

    # --------------------------------------------------------
    # Read SLURM states
    # --------------------------------------------------------

    states = get_array_states(
        args.array_job
    )

    print()
    print(
        "Previous array states:"
    )

    for jobid, state in states.items():

        print(
            f"    {jobid}: {state}"
        )

    # --------------------------------------------------------
    # Determine whether any task failed
    # --------------------------------------------------------

    failed_tasks = []

    cancelled_tasks = []

    timeout_tasks = []

    for jobid, state in states.items():

        if "_" not in jobid:
            continue

        try:

            task_id = int(
                jobid.rsplit("_", 1)[1]
            )

        except ValueError:

            continue

        if state == "TIMEOUT":

            timeout_tasks.append(
                task_id
            )

        elif state in {
            "FAILED",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
            "PREEMPTED",
        }:

            failed_tasks.append(
                (task_id, state)
            )

        elif state == "CANCELLED":

            cancelled_tasks.append(
                task_id
            )

    timeout_tasks.sort()

    print()
    print(
        "TIMEOUT tasks:",
        timeout_tasks
    )

    print(
        "FAILED tasks:",
        failed_tasks
    )

    print(
        "CANCELLED tasks:",
        cancelled_tasks
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT immediately finalize simply because there is
    # no TIMEOUT.
    #
    # We ALWAYS inspect the actual trajectory files.
    # --------------------------------------------------------

    (
        complete_tasks,
        incomplete_tasks,
        missing_tasks,
    ) = inspect_all_trajectories(
        run_root,
        target_steps,
    )

    print()
    print("=" * 72)
    print("TRAJECTORY SUMMARY")
    print("=" * 72)

    print(
        "Complete:",
        complete_tasks
    )

    print(
        "Incomplete:",
        incomplete_tasks
    )

    print(
        "Missing/no common timestep:",
        missing_tasks
    )

    # ========================================================
    # CASE 1
    #
    # ALL TRAJECTORIES COMPLETE
    #
    # This is the ONLY condition that allows training.
    # ========================================================

    if (
        len(complete_tasks)
        == int(inputs["HPC"]["repeats"])
    ):

        print()
        print(
            "ALL TRAJECTORIES HAVE REACHED "
            "THE REQUESTED FINAL TIMESTEP."
        )

        # ----------------------------------------------------
        # Finalize validation dataset
        # ----------------------------------------------------

        validation_file = finalize_validation(
            run_root
        )

        # ----------------------------------------------------
        # Prevent accidental duplicate training submission
        # ----------------------------------------------------

        training_id_file = (
            run_root
            / "training_job_id.txt"
        )

        if training_id_file.exists():

            existing_id = (
                training_id_file
                .read_text()
                .strip()
            )

            print()
            print(
                "Training has already been submitted."
            )

            print(
                "Existing training job:",
                existing_id
            )

            return 0

        # ----------------------------------------------------
        # Submit training
        # ----------------------------------------------------

        submit_training(
            run_root,
            package_root,
        )

        print()
        print(
            "VALIDATION WORKFLOW COMPLETE."
        )

        print(
            "MACE TRAINING HAS BEEN INITIATED."
        )

        return 0

    # ========================================================
    # CASE 2
    #
    # Some trajectories are incomplete.
    #
    # Only TIMEOUT trajectories are automatically restarted.
    # ========================================================

    restart_tasks = []

    for task_id in timeout_tasks:

        repdir = (
            run_root
            / "repetitions"
            / f"rep-{task_id}"
        )

        common = get_common_timestep(
            repdir
        )

        print()
        print(
            f"Checking TIMEOUT rep-{task_id}: "
            f"common timestep = {common}"
        )

        if common is None:

            print(
                f"rep-{task_id}: "
                "No safe common timestep. "
                "Will NOT restart automatically."
            )

            continue

        if common >= target_steps:

            print(
                f"rep-{task_id}: "
                "already complete."
            )

            continue

        restart_tasks.append(
            task_id
        )

    restart_tasks.sort()

    # ========================================================
    # CASE 3
    #
    # Incomplete trajectories exist but none are safely
    # restartable TIMEOUT trajectories.
    #
    # Do NOT train.
    # ========================================================

    if not restart_tasks:

        print()
        print("=" * 72)
        print("NO SAFE AUTOMATIC RESTART AVAILABLE")
        print("=" * 72)

        if failed_tasks:

            print(
                "Some trajectories failed:"
            )

            for task_id, state in failed_tasks:

                print(
                    f"    rep-{task_id}: {state}"
                )

        if cancelled_tasks:

            print(
                "Some trajectories were cancelled:"
            )

            for task_id in cancelled_tasks:

                print(
                    f"    rep-{task_id}: CANCELLED"
                )

        if missing_tasks:

            print(
                "Some trajectories have no valid "
                "common timestep:"
            )

            for task_id in missing_tasks:

                print(
                    f"    rep-{task_id}"
                )

        print()
        print(
            "Training will NOT be submitted."
        )

        print(
            "The workflow requires ALL trajectories "
            "to reach the target timestep."
        )

        return 2

    # ========================================================
    # CASE 4
    #
    # Restart incomplete TIMEOUT trajectories.
    # ========================================================

    print()
    print("=" * 72)
    print("AUTOMATIC RESTART")
    print("=" * 72)

    print(
        "Restarting trajectories:",
        restart_tasks
    )

    restart_script = create_restart_script(
        run_root,
        inputs,
        restart_tasks,
    )

    print(
        "Restart script:",
        restart_script
    )

    # --------------------------------------------------------
    # Submit restart array
    # --------------------------------------------------------

    result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            str(restart_script),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    restart_job = (
        result.stdout
        .strip()
        .split(";")[0]
    )

    print(
        "Restart array submitted:",
        restart_job
    )

    # --------------------------------------------------------
    # CRITICAL:
    #
    # Submit another checker after the restart array.
    # --------------------------------------------------------

    next_checker = submit_next_checker(
        restart_job,
        run_root,
        package_root,
    )

    print()
    print("=" * 72)
    print("RESTART CHAIN CONTINUED")
    print("=" * 72)

    print(
        "Restart array:",
        restart_job
    )

    print(
        "Next checker:",
        next_checker
    )

    print()
    print(
        "The next checker will repeat:"
    )

    print(
        "    check"
    )

    print(
        "    -> restart if necessary"
    )

    print(
        "    -> check again"
    )

    print(
        "    -> continue until all trajectories "
        "are complete"
    )

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        sys.exit(
            main()
        )

    except Exception as exc:

        print(
            "ERROR:",
            exc,
            file=sys.stderr
        )

        sys.exit(1)

