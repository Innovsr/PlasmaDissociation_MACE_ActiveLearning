#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path


# ============================================================
# SLURM ARRAY STATE
# ============================================================

def get_array_states(array_job_id):
    """
    Return the SLURM state of each array task.

    Example:
        9029631_1  COMPLETED
        9029631_2  TIMEOUT
        ...
    """

    result = subprocess.run(
        [
            "sacct",
            "-n",
            "-P",
            "-j",
            array_job_id,
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

        jobid = parts[0]
        state = parts[1].split()[0]

        states[jobid] = state

    return states


# ============================================================
# TIMESTEP READING
# ============================================================

def check_timestep_file(filename):
    """
    Read timestep numbers from one trajectory output file.

    Only lines containing:
        Timestep:

    are considered.
    """

    timesteps = set()

    if not filename.exists():
        return timesteps

    with filename.open() as f:

        for line in f:

            if "Timestep:" not in line:
                continue

            parts = line.split()

            try:
                timestep = int(parts[1])
                timesteps.add(timestep)

            except (ValueError, IndexError):
                continue

    return timesteps


# ============================================================
# COMMON TIMESTEP
# ============================================================

def get_common_timestep(repdir):
    """
    Find the latest timestep which exists in ALL four files:

        xyz.all
        momenta.all
        forces.all
        time.all

    A timestep is considered valid for restart only if it
    occurs in all four files.
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

    common = xyz & momenta & forces & times

    if not common:
        return None

    return max(common)


# ============================================================
# FINALIZATION
# ============================================================

def finalize_validation(run_root):
    """
    Combine completed validation trajectories into validation.extxyz.
    """

    package_root = Path(__file__).resolve().parent

    finalize_script = (
        package_root / "finalize_validation.py"
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

    print("Finalizing validation trajectories...")
    print(f"Output: {output_file}")

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

    print("Validation finalization completed.")


# ============================================================
# RESTART SCRIPT
# ============================================================

def create_restart_script(
    run_root,
    inputs,
    restart_tasks,
):
    """
    Create a SLURM array containing only the trajectories
    that timed out and are still incomplete.
    """

    array_spec = ",".join(
        str(x) for x in restart_tasks
    )

    restart_script = (
        run_root / "validation_restart.sh"
    )

    package_root = Path(__file__).resolve().parent

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

exec "$PYTHON" "{main_py}"
"""
    )

    restart_script.chmod(0o755)

    return restart_script


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

    run_root = Path(args.run_root)

    # --------------------------------------------------------
    # Read runtime input
    # --------------------------------------------------------

    with open(run_root / "inputs.json") as f:
        inputs = json.load(f)

    target_steps = int(
        inputs["Propagation"]["Tot_timesteps"]
    )

    # --------------------------------------------------------
    # Read SLURM states
    # --------------------------------------------------------

    states = get_array_states(
        args.array_job
    )

    timeout_tasks = []

    for jobid, state in states.items():

        if "_" not in jobid:
            continue

        # Ignore .batch/.extern records.
        if "." in jobid:
            continue

        if state != "TIMEOUT":
            continue

        try:
            task_id = int(
                jobid.rsplit("_", 1)[1]
            )
        except ValueError:
            continue

        timeout_tasks.append(task_id)

    timeout_tasks.sort()

    print(
        "TIMEOUT trajectories:",
        timeout_tasks
    )

    # --------------------------------------------------------
    # NO TIMEOUT
    #
    # This means every trajectory finished without timeout.
    # DO NOT restart anything.
    #
    # Instead finalize the validation data.
    # --------------------------------------------------------

    if not timeout_tasks:

        print(
            "No TIMEOUT detected."
        )

        print(
            "No automatic restart required."
        )

        finalize_validation(run_root)

        return 0

    # --------------------------------------------------------
    # Examine only TIMEOUT trajectories
    # --------------------------------------------------------

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

        print(
            f"rep-{task_id}: "
            f"latest common timestep = {common}"
        )

        # No common timestep means we cannot safely restart.
        if common is None:

            print(
                f"rep-{task_id}: "
                "NO common timestep found."
            )

            print(
                "This trajectory will NOT be restarted "
                "from timestep 0."
            )

            continue

        # The target is already reached.
        if common >= target_steps:

            print(
                f"rep-{task_id}: "
                "target timestep already reached."
            )

            continue

        # This TIMEOUT trajectory is incomplete.
        restart_tasks.append(task_id)

    # --------------------------------------------------------
    # Nothing needs restarting
    # --------------------------------------------------------

    if not restart_tasks:

        print(
            "No incomplete TIMEOUT trajectories "
            "require restarting."
        )

        finalize_validation(run_root)

        return 0

    # --------------------------------------------------------
    # Submit restart array
    # --------------------------------------------------------

    print(
        "Trajectories to restart:",
        restart_tasks
    )

    restart_script = create_restart_script(
        run_root,
        inputs,
        restart_tasks,
    )

    restart_job = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            str(restart_script),
        ],
        text=True,
    ).strip().split(";")[0]

    print(
        "Restart array submitted:",
        restart_job
    )

    return 0


if __name__ == "__main__":
    main()
