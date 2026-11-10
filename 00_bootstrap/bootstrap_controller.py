#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path) as f:
        return json.load(f)


def submit(cmd):
    """
    Submit a Slurm command and return the job ID.
    """
    result = subprocess.check_output(
        cmd,
        text=True
    ).strip()

    return result.split(";")[0]


def get_sacct_states(job_id, repetitions):
    """
    Query each Slurm array task explicitly.

    On Aire, sacct does not reliably expose the array task number
    in JobIDRaw. However, querying <job_id>_<rep> directly works.

    Example:

        sacct -j 9182236_5

    returns the historical state for array task 5.

    Returns:

        {
            1: "COMPLETED",
            2: "COMPLETED",
            5: "TIMEOUT",
            ...
        }
    """

    states = {}

    for rep in repetitions:

        cmd = [
            "sacct",
            "-j",
            f"{job_id}_{rep}",
            "-X",
            "-n",
            "-P",
            "--format=State",
        ]

        try:

            output = subprocess.check_output(
                cmd,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()

            if output:
                state = output.splitlines()[0].strip()
                states[rep] = state
            else:
                states[rep] = "UNKNOWN"

        except subprocess.CalledProcessError:

            states[rep] = "UNKNOWN"

    return states


import importlib.util

def get_trajectory_status(stage, target):

    restart_manager_path = ROOT / "restart_manager.py"

    if not restart_manager_path.exists():
        raise FileNotFoundError(
            f"restart_manager.py not found at {restart_manager_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "restart_manager",
        restart_manager_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load restart_manager.py from "
            f"{restart_manager_path}"
        )

    restart_manager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(restart_manager)

    repetitions = stage / "repetitions"
    results = {}

    for repdir in sorted(
        repetitions.glob("rep-*"),
        key=lambda p: int(p.name.split("-")[1])
    ):
        rep = int(repdir.name.split("-")[1])

        results[rep] = restart_manager.status(
            repdir,
            target
        )

    return results

#def get_trajectory_status(stage, target):
#
#    """
#    Use restart_manager.py to determine the latest common
#    timestep of every repetition.
#    """
#
#    restart_manager_path = ROOT / "restart_manager.py"
#
#    if not restart_manager_path.exists():
#        raise FileNotFoundError(
#            f"restart_manager.py not found at {restart_manager_path}"
#        )
#
#    sys.path.insert(0, str(ROOT))
#
#    from restart_manager import status


#def get_trajectory_status(stage, target):
#    """
#    Use restart_manager.py to determine the latest common
#    timestep of every repetition.
#    """
#
#    sys.path.insert(
#        0,
#        str(ROOT)
#    )
#
#    from restart_manager import status

    repetitions = stage / "repetitions"

    results = {}

    for repdir in sorted(
        repetitions.glob("rep-*"),
        key=lambda p: int(p.name.split("-")[1])
    ):

        rep = int(
            repdir.name.split("-")[1]
        )

        results[rep] = status(
            repdir,
            target
        )

    return results


def write_restart_array(
    cfg,
    stage,
    repetitions
):
    """
    Create a Slurm array containing ONLY the incomplete
    TIMEOUT trajectories.

    No BOMD_RESTART=1 is used.

    The normal BOMD code is allowed to detect the existing
    trajectory files and continue from the latest common
    timestep.
    """

    hpc = cfg["HPC"]

    array_spec = ",".join(
        str(x)
        for x in repetitions
    )

    script = stage / "restart_bomd.sh"

    python_exe = sys.executable

    script.write_text(
        f"""#!/bin/bash
#SBATCH --account={hpc["account"]}
#SBATCH --job-name=BootstrapRestart
#SBATCH --time={hpc["bomd_time"]}
#SBATCH --cpus-per-task={hpc["cores"]}
#SBATCH --mem={hpc["memory"]}G
#SBATCH --array={array_spec}
#SBATCH --output={stage}/logs/restart_%A_%a.out
#SBATCH --error={stage}/logs/restart_%A_%a.err

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

echo "=============================================="
echo "BOOTSTRAP BOMD RESTART"
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

echo "IMPORTANT:"
echo "No BOMD_RESTART variable is being used."
echo "The existing trajectory files are used for restart."
echo ""

exec "{python_exe}" "{ROOT}/bomd/code/main.py"
"""
    )

    script.chmod(0o755)

    return script


def write_training_job(
    cfg,
    stage
):
    """
    Create the GPU-only MACE training job.
    """

    hpc = cfg["HPC"]

    script = stage / "bootstrap_train_gpu.sh"

    python_exe = sys.executable

    script.write_text(
        f"""#!/bin/bash
#SBATCH --account={hpc["account"]}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --job-name=BootstrapMACE
#SBATCH --output={stage}/logs/mace_%j.out
#SBATCH --error={stage}/logs/mace_%j.err

set -euo pipefail

module load miniforge

source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate mace_env

echo "=============================================="
echo "BOOTSTRAP MACE GPU TRAINING"
echo "=============================================="

echo "Host:"
hostname

echo ""

echo "GPU:"
nvidia-smi

echo ""

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo ""

echo "Python:"
which python

python - <<'PY'

import torch

print("PyTorch:", torch.__version__)

print(
    "CUDA available:",
    torch.cuda.is_available()
)

if not torch.cuda.is_available():

    raise SystemExit(
        "ERROR: CUDA is not available. "
        "MACE training will NOT run on CPU."
    )

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

PY

cd "{ROOT}"

exec python bootstrap_train.py \\
    --config "{ROOT}/configs/bootstrap.json"
"""
    )

    script.chmod(0o755)

    return script


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        required=True
    )

    parser.add_argument(
        "--run-root",
        required=True
    )

    parser.add_argument(
        "--config",
        required=True
    )

    parser.add_argument(
        "--job-id",
        required=True,
        help="Previous BOMD array job ID"
    )

    args = parser.parse_args()

    stage = Path(
        args.run_root
    ).resolve()

    cfg = load_config(
        args.config
    )

    target = (
        float(cfg["Propagation"]["Timestep"])
        *
        int(cfg["Propagation"]["Tot_timesteps"])
    )

    print("=" * 60)
    print("BOOTSTRAP TRAJECTORY CONTROLLER")
    print("=" * 60)

    print("Stage:", args.stage)
    print("Run root:", stage)
    print("Previous BOMD job:", args.job_id)
    print("Target timestep:", target)

    print("=" * 60)

    # --------------------------------------------------------
    # Check trajectory files first.
    # --------------------------------------------------------

    trajectory = get_trajectory_status(
        stage,
        target
    )

    repetitions = sorted(
        trajectory
    )

    # --------------------------------------------------------
    # Get Slurm state for each array task.
    # --------------------------------------------------------

    states = get_sacct_states(
        args.job_id,
        repetitions
    )

    complete = []
    incomplete = []

    for rep in repetitions:

        info = trajectory[rep]

        slurm_state = states.get(
            rep,
            "UNKNOWN"
        )

        print(
            f"rep-{rep}: "
            f"Slurm={slurm_state:<10} "
            f"current={info['current_timestep']} "
            f"target={target} "
            f"complete={info['complete']}"
        )

        if info["complete"]:

            complete.append(rep)

        else:

            incomplete.append(rep)

    print("-" * 60)

    print("Complete:", complete)
    print("Incomplete:", incomplete)

    # --------------------------------------------------------
    # Everything complete.
    # --------------------------------------------------------

    if not incomplete:

        print("")
        print(
            "ALL BOOTSTRAP TRAJECTORIES ARE COMPLETE."
        )

        print(
            "Submitting MACE GPU training."
        )

        training_script = write_training_job(
            cfg,
            stage
        )

        training_job = submit(
            [
                "sbatch",
                "--parsable",
                str(training_script)
            ]
        )

        print(
            "MACE training job:",
            training_job
        )

        return 0

    # --------------------------------------------------------
    # Identify ONLY incomplete TIMEOUT jobs.
    # --------------------------------------------------------

    timeout_reps = [
        rep
        for rep in incomplete
        if states.get(rep) == "TIMEOUT"
    ]

    non_timeout = [
        rep
        for rep in incomplete
        if states.get(rep) != "TIMEOUT"
    ]

    # --------------------------------------------------------
    # Do NOT automatically restart non-TIMEOUT jobs.
    # --------------------------------------------------------

    if non_timeout:
    print("")
    print(
        "These incomplete trajectories will NOT be restarted "
        "because their Slurm state is not TIMEOUT:"
    )

    for rep in non_timeout:
        print(
            f"  rep-{rep}: "
            f"Slurm={states.get(rep, 'UNKNOWN')}"
        )

    print("")
    print(
        "They will be excluded from the automatic restart."
    )

    #if non_timeout:

    #    print("")
    #    print(
    #        "ERROR: These trajectories are incomplete "
    #        "but did not finish with TIMEOUT:"
    #    )

    #    for rep in non_timeout:

    #        print(
    #            f"  rep-{rep}: "
    #            f"Slurm={states.get(rep, 'UNKNOWN')}"
    #        )

    #    print("")

    #    print(
    #        "No automatic restart will be submitted "
    #        "for these trajectories."
    #    )

    #    return 2

    # --------------------------------------------------------
    # Restart ONLY incomplete TIMEOUT trajectories.
    # --------------------------------------------------------

    print("")

    print(
        "TIMEOUT trajectories requiring restart:"
    )

    print(
        timeout_reps
    )

    restart_script = write_restart_array(
        cfg,
        stage,
        timeout_reps
    )

    restart_job = submit(
        [
            "sbatch",
            "--parsable",
            str(restart_script)
        ]
    )

    print("")

    print(
        "Restart array submitted:",
        restart_job
    )

    # --------------------------------------------------------
    # Submit another controller after restart.
    # --------------------------------------------------------

    next_controller = submit(
        [
            "sbatch",
            "--parsable",

            f"--dependency=afterany:{restart_job}",

            "--cpus-per-task=1",
            "--mem=1G",
            "--time=00:15:00",

            "--job-name=BootstrapCheck",

            f"--output={stage}/logs/controller_%j.out",
            f"--error={stage}/logs/controller_%j.err",

            str(Path(__file__).resolve()),

            "--stage",
            str(int(args.stage) + 1),

            "--run-root",
            str(stage),

            "--config",
            str(Path(args.config).resolve()),

            "--job-id",
            restart_job
        ]
    )

    print(
        "Next controller:",
        next_controller
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
