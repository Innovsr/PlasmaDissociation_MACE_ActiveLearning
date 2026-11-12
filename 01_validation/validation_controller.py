#!/usr/bin/env python3

"""
Validation controller.

This is the Validation equivalent of bootstrap_controller.py.

Workflow:

    validation array
          |
          v
    validation_controller.py
          |
          v
    restart_manager.py
          |
          +---- ALL COMPLETE
          |          |
          |          v
          |     finalize validation
          |          |
          |          v
          |     model update
          |
          +---- INCOMPLETE + TIMEOUT
                     |
                     v
                restart array
                     |
                     v
             validation_controller.py
                     |
                     v
              repeat until complete

Only incomplete TIMEOUT trajectories are automatically
restarted.

FAILED / CANCELLED / NODE_FAIL / OUT_OF_MEMORY /
PREEMPTED trajectories are NOT automatically restarted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
import re


# ============================================================
# PATHS
# ============================================================

PACKAGE_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

ROOT = (
    PACKAGE_ROOT
    .parent
)

RESTART_MANAGER_PATH = (
    ROOT
    / "restart_manager.py"
)
FINALIZE_SCRIPT = (
    PACKAGE_ROOT.parent
    / "finalize_validation.py"
)

MODEL_UPDATE_SCRIPT = (
    ROOT
    / "02_model_update"
    / "run_model_update.sh"
)


# ============================================================
# LOAD COMMON RESTART MANAGER
# ============================================================

if not RESTART_MANAGER_PATH.exists():

    raise FileNotFoundError(
        "restart_manager.py not found at:\n"
        f"{RESTART_MANAGER_PATH}"
    )

import importlib.util

spec = importlib.util.spec_from_file_location(
    "restart_manager",
    RESTART_MANAGER_PATH
)

if spec is None or spec.loader is None:

    raise ImportError(
        "Could not load restart_manager.py"
    )

restart_manager = (
    importlib.util.module_from_spec(spec)
)

spec.loader.exec_module(
    restart_manager
)


# ============================================================
# SLURM
# ============================================================

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


def get_array_states(
    array_job_id,
    repetitions
):
    """
    Query each array task explicitly.

    This is more reliable on Aire than relying only on the
    parent array JobID.
    """

    states = {}

    for rep in repetitions:

        command = [
            "sacct",
            "-j",
            f"{array_job_id}_{rep}",
            "-X",
            "-n",
            "-P",
            "--format=State"
        ]

        try:

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )

            output = (
                result.stdout
                .strip()
            )

            if output:

                state = (
                    output
                    .splitlines()[0]
                    .strip()
                    .split()[0]
                )

            else:

                state = "UNKNOWN"

        except subprocess.CalledProcessError:

            state = "UNKNOWN"

        states[rep] = state

    return states


# ============================================================
# TRAJECTORY STATUS
# ============================================================

def inspect_trajectories(
    run_root,
    target
):
    """
    Use restart_manager.py to determine the latest common
    timestep of every repetition.
    """

    repetitions_dir = (
        run_root
        / "repetitions"
    )

    results = {}

    for repdir in sorted(
        repetitions_dir.glob("rep-*"),
        key=lambda p: int(
            p.name.split("-")[1]
        )
    ):

        try:

            rep = int(
                repdir.name.split("-")[1]
            )

        except (
            ValueError,
            IndexError
        ):

            continue

        results[rep] = (
            restart_manager.status(
                repdir,
                target
            )
        )

    return results


# ============================================================
# NORMALISE OLD TIMESTEP HEADERS
# ============================================================

def normalise_timestep_headers(
    output_dir
):
    """
    Convert:

        Timestep: 730.0

    to:

        Timestep: 730

    for compatibility with the existing BOMD restart
    parser in init.py.

    This does NOT alter the numerical timestep value.
    """

    pattern = re.compile(
        r"^(\s*Timestep:\s*)(-?\d+)\.0+(\s*)$",
        re.MULTILINE
    )

    for filename in (
        "xyz.all",
        "momenta.all",
        "forces.all",
        "time.all"
    ):

        path = (
            output_dir
            / filename
        )

        if not path.exists():
            continue

        text = path.read_text(
            errors="replace"
        )

        new_text = pattern.sub(
            r"\1\2\3",
            text
        )

        if new_text != text:

            path.write_text(
                new_text
            )


# ============================================================
# CREATE RESTART ARRAY
# ============================================================

def create_restart_array(
    run_root,
    inputs,
    restart_tasks
):

    hpc = inputs["HPC"]

    array_spec = ",".join(
        str(x)
        for x in restart_tasks
    )

    restart_script = (
        run_root
        / "validation_restart.sh"
    )

    bomd_main = (
        ROOT
        / "bomd"
        / "code"
        / "main.py"
    )

    bomd_time = str(hpc["hours"]).strip()

    python_exe = sys.executable

    restart_script.write_text(
        f"""#!/bin/bash

#SBATCH --account={hpc["account"]}
#SBATCH --job-name=ValidationRestart
#SBATCH --time={bomd_time}
#SBATCH --cpus-per-task={hpc["cores"]}
#SBATCH --mem={hpc["memory"]}G
#SBATCH --array={array_spec}
#SBATCH --output={run_root}/logs/restart_%A_%a.out
#SBATCH --error={run_root}/logs/restart_%A_%a.err

set -euo pipefail

module unload qchem 2>/dev/null || true
module load qchem/6.0.2

export LM_LICENSE_FILE=27000@uol-lnx-lic01.leeds.ac.uk

export QC="$QCHEM_HOME"
export QCAUX="$QC/qcaux"
export QCPROG="$QC/exe/qcprog.exe"
export QCPROG_S="$QC/exe/qcprog.exe_s"

export PATH="$PATH:$QC/exe:$QC/bin"

export QCSCRATCH="{run_root}/repetitions/rep-$SLURM_ARRAY_TASK_ID/tmp"

mkdir -p "$QCSCRATCH"

cd "{run_root}/repetitions/rep-$SLURM_ARRAY_TASK_ID"

echo "=============================================="
echo "VALIDATION BOMD RESTART"
echo "=============================================="

echo "Repository : {ROOT}"
echo "Repetition : $SLURM_ARRAY_TASK_ID"
echo "Working dir: $(pwd)"
echo "Python     : {python_exe}"

echo "=============================================="
echo ""

echo "Existing trajectory:"
grep "Timestep:" output/xyz.all | tail -3 || true

echo ""

echo "Normalising Timestep headers..."

python - <<'PY'
from pathlib import Path
import re

output = Path("output")

pattern = re.compile(
    r"^(\\s*Timestep:\\s*)(-?\\d+)\\.0+(\\s*)$",
    re.MULTILINE
)

for name in (
    "xyz.all",
    "momenta.all",
    "forces.all",
    "time.all",
):

    path = output / name

    if not path.exists():
        continue

    text = path.read_text(
        errors="replace"
    )

    new_text = pattern.sub(
        r"\\1\\2\\3",
        text
    )

    if new_text != text:
        path.write_text(new_text)

print("Timestep headers normalised.")
PY

echo ""

echo "Trajectory after normalisation:"
grep "Timestep:" output/xyz.all | tail -3 || true

echo ""

echo "No BOMD_RESTART variable is being used."
echo "Existing trajectory files determine the restart point."

echo ""

exec "{python_exe}" "{bomd_main}"
"""
    )

    restart_script.chmod(
        0o755
    )

    return restart_script


# ============================================================
# FINALISE VALIDATION
# ============================================================

def finalize_validation(
    run_root
):

    if not FINALIZE_SCRIPT.exists():

        raise FileNotFoundError(
            "finalize_validation.py not found:\n"
            f"{FINALIZE_SCRIPT}"
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
    print("=" * 60)
    print("FINALISING VALIDATION DATA")
    print("=" * 60)

    print(
        "Output:",
        output_file
    )

    subprocess.run(
        [
            sys.executable,
            str(FINALIZE_SCRIPT),
            "--run-root",
            str(run_root),
            "--output",
            str(output_file)
        ],
        check=True
    )

    print(
        "Validation finalisation completed."
    )

    return output_file


# ============================================================
# MODEL UPDATE
# ============================================================

def submit_model_update(
    run_root
):

    if not MODEL_UPDATE_SCRIPT.exists():

        raise FileNotFoundError(
            "Model update script not found:\n"
            f"{MODEL_UPDATE_SCRIPT}"
        )

    job_id = submit(
        [
            "sbatch",
            "--parsable",
            str(MODEL_UPDATE_SCRIPT)
        ]
    )

    (
        run_root
        / "training_job_id.txt"
    ).write_text(
        job_id + "\n"
    )

    print(
        "MACE model-update job:",
        job_id
    )

    return job_id


# ============================================================
# SUBMIT NEXT CONTROLLER
# ============================================================

#def submit_next_controller(
#    run_root,
#    restart_job
#):
#
#    controller_job = submit(
#        [
#            "sbatch",
#            "--parsable",
#
#            f"--dependency=afterany:{restart_job}",
#
#            "--cpus-per-task=1",
#            "--mem=1G",
#            "--time=00:15:00",
#
#            "--job-name=ValidationCheck",
#
#            f"--output={run_root}/logs/controller_%j.out",
#            f"--error={run_root}/logs/controller_%j.err",
#
#            sys.executable,
#            str(
#                PACKAGE_ROOT
#                / "validation_controller.py"
#            ),
#
#            "--run-root",
#            str(run_root),
#
#            "--array-job",
#            str(restart_job)
#        ]
#    )
#
#    return controller_job

def submit_next_controller(
    run_root,
    restart_job
):
    """
    Submit the next validation controller through a proper
    Slurm batch script.

    The controller waits until the restart array has finished,
    then checks all trajectories again.
    """

    run_root = Path(run_root).resolve()

    controller_script = (
        run_root
        / "next_validation_controller.sh"
    )

    controller_python = (
        PACKAGE_ROOT
        / "validation_controller.py"
    )

    controller_script.write_text(
        f"""#!/bin/bash

#SBATCH --account={json.load(open(run_root / "inputs.json"))["HPC"]["account"]}
#SBATCH --job-name=ValidationCheck
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output={run_root}/logs/controller_%j.out
#SBATCH --error={run_root}/logs/controller_%j.err

set -euo pipefail

exec "{sys.executable}" "{controller_python}" \\
    --run-root "{run_root}" \\
    --array-job "{restart_job}"
"""
    )

    controller_script.chmod(0o755)

    controller_job = submit(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{restart_job}",
            str(controller_script)
        ]
    )

    print(
        "Next controller:",
        controller_job
    )

    return controller_job

# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-root",
        required=True
    )

    parser.add_argument(
        "--array-job",
        required=True
    )

    args = parser.parse_args()

    run_root = (
        Path(args.run_root)
        .resolve()
    )

    inputs_file = (
        run_root
        / "inputs.json"
    )

    if not inputs_file.exists():

        raise FileNotFoundError(
            f"inputs.json not found:\n"
            f"{inputs_file}"
        )

    with inputs_file.open() as f:
        inputs = json.load(f)

    target = (
        float(
            inputs["Propagation"]["Timestep"]
        )
        *
        int(
            inputs["Propagation"]["Tot_timesteps"]
        )
    )

    expected_repetitions = int(
        inputs["HPC"]["repeats"]
    )

    print()
    print("=" * 70)
    print("VALIDATION TRAJECTORY CONTROLLER")
    print("=" * 70)

    print(
        "Run root:",
        run_root
    )

    print(
        "Previous array job:",
        args.array_job
    )

    print(
        "Target timestep:",
        target
    )

    print(
        "Expected trajectories:",
        expected_repetitions
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Inspect trajectory files
    # --------------------------------------------------------

    trajectory_status = inspect_trajectories(
        run_root,
        target
    )

    repetitions = sorted(
        trajectory_status
    )

    # --------------------------------------------------------
    # Slurm state
    # --------------------------------------------------------

    states = get_array_states(
        args.array_job,
        repetitions
    )

    print()
    print("TRAJECTORY STATUS")
    print("-" * 70)

    complete = []
    incomplete = []

    for rep in repetitions:

        info = trajectory_status[rep]

        state = states.get(
            rep,
            "UNKNOWN"
        )

        print(
            f"rep-{rep}: "
            f"Slurm={state:<12} "
            f"current={info['current_timestep']} "
            f"target={target} "
            f"complete={info['complete']}"
        )

        if info["complete"]:

            complete.append(rep)

        else:

            incomplete.append(rep)

    print("-" * 70)

    print(
        "Complete:",
        complete
    )

    print(
        "Incomplete:",
        incomplete
    )

    # --------------------------------------------------------
    # Check whether every expected trajectory exists.
    # --------------------------------------------------------

    expected = set(
        range(
            1,
            expected_repetitions + 1
        )
    )

    actual = set(
        repetitions
    )

    missing_reps = sorted(
        expected - actual
    )

    if missing_reps:

        print()
        print(
            "Missing repetition directories:",
            missing_reps
        )

    # ========================================================
    # CASE 1: ALL COMPLETE
    # ========================================================

    if (
        len(complete) == expected_repetitions
        and not incomplete
        and not missing_reps
    ):

        print()
        print("=" * 70)
        print("ALL VALIDATION TRAJECTORIES COMPLETE")
        print("=" * 70)

        # Prevent duplicate model-update submission.

        training_id_file = (
            run_root
            / "training_job_id.txt"
        )

        if training_id_file.exists():

            print(
                "Model update has already been submitted:"
            )

            print(
                training_id_file
                .read_text()
                .strip()
            )

            return 0

        # Finalise validation dataset.

        finalize_validation(
            run_root
        )

        # Submit model update.

        submit_model_update(
            run_root
        )

        print()
        print(
            "VALIDATION WORKFLOW COMPLETE."
        )

        return 0

    # ========================================================
    # CASE 2: FIND TIMEOUT TRAJECTORIES THAT CAN RESTART
    # ========================================================

    restart_tasks = []

    for rep in incomplete:

        state = states.get(
            rep,
            "UNKNOWN"
        )

        if state != "TIMEOUT":

            print(
                f"rep-{rep}: incomplete but Slurm state "
                f"is {state}; not automatically restarting."
            )

            continue

        repdir = (
            run_root
            / "repetitions"
            / f"rep-{rep}"
        )

        common = (
            restart_manager.latest_common(
                repdir
            )
        )

        print(
            f"rep-{rep}: TIMEOUT, "
            f"latest common timestep={common}"
        )

        if common is None:

            print(
                f"rep-{rep}: no safe common timestep; "
                "not restarting."
            )

            continue

        if common >= target:

            print(
                f"rep-{rep}: already reached target; "
                "no restart needed."
            )

            continue

        restart_tasks.append(
            rep
        )

    restart_tasks.sort()

    # ========================================================
    # CASE 3: NOTHING SAFE TO RESTART
    # ========================================================

    if not restart_tasks:

        print()
        print("=" * 70)
        print("NO SAFE AUTOMATIC RESTART AVAILABLE")
        print("=" * 70)

        print(
            "Incomplete trajectories:",
            incomplete
        )

        print(
            "No restart job will be submitted."
        )

        print(
            "Model update will NOT be submitted."
        )

        return 2

    # ========================================================
    # CASE 4: SUBMIT RESTART
    # ========================================================

    print()
    print("=" * 70)
    print("AUTOMATIC VALIDATION RESTART")
    print("=" * 70)

    print(
        "Restarting:",
        restart_tasks
    )

    restart_script = create_restart_array(
        run_root,
        inputs,
        restart_tasks
    )

    restart_job = submit(
        [
            "sbatch",
            "--parsable",
            str(restart_script)
        ]
    )

    print(
        "Restart array:",
        restart_job
    )

    # --------------------------------------------------------
    # Submit another controller after restart.
    # --------------------------------------------------------

    next_controller = (
        submit_next_controller(
            run_root,
            restart_job
        )
    )

    print(
        "Next controller:",
        next_controller
    )

    print()
    print(
        "Restart chain continues automatically."
    )

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )
