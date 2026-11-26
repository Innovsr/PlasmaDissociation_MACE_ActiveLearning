#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_json(path):
    return json.loads(Path(path).read_text())


def scratch_root(runfolder):
    return (
        Path("/mnt/scratch")
        / os.environ.get("USER", os.environ.get("LOGNAME", "unknown"))
        / runfolder
        / "00_bootstrap"
    )


def create_runtime_files(cfg, stage):

    stage.mkdir(parents=True, exist_ok=True)

    (stage / "setup").mkdir(exist_ok=True)
    (stage / "repetitions").mkdir(exist_ok=True)
    (stage / "logs").mkdir(exist_ok=True)
    (stage / "datasets").mkdir(exist_ok=True)
    (stage / "model_work").mkdir(exist_ok=True)

    runtime = {
        "model": None,

        "HPC": {
            "account": cfg["HPC"]["account"],
            "Runfolder": cfg["HPC"]["Runfolder"],
            "cores": cfg["HPC"]["cores"],
            "memory": cfg["HPC"]["memory"]
        },

        "Molecule_data": cfg["Molecule_data"],

        "Elec_structure": cfg["Elec_structure"],

        "Propagation": cfg["Propagation"]
    }

    # setup.py expects validation.json
    setup_cfg = dict(runtime)

    setup_cfg["HPC"]["repeats"] = cfg["HPC"]["repeats"]

    # IMPORTANT:
    # Bootstrap deliberately disables trajectory selection.
    setup_cfg["Trajectory_selection"] = {
        "enabled": 0
    }

    (stage / "validation.json").write_text(
    json.dumps(setup_cfg, indent=4)
    )
    #(stage / "setup" / "validation.json").write_text(
    #    json.dumps(setup_cfg, indent=4)
    #)

    # main.py/global_vars.py reads ../../inputs.json
    (stage / "inputs.json").write_text(
        json.dumps(runtime, indent=4)
    )


def copy_geometry(cfg, stage):

    molecule = cfg["Molecule_data"]["Molecule"]

    source = (
        ROOT
        / "bomd"
        / "Initial_Geometry"
        / f"{molecule}.txt"
    )

    if not source.exists():
        raise SystemExit(
            f"Initial geometry not found:\n{source}"
        )

    shutil.copy2(
        source,
        stage / "setup" / "initial_guess.txt"
    )


def submit_setup(cfg, stage):

    setup_script = stage / "setup_bootstrap.sh"

    h = cfg["HPC"]

    setup_script.write_text(
        f"""#!/bin/bash
#SBATCH --account={h["account"]}
#SBATCH --job-name=BootstrapSetup
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task={h["cores"]}
#SBATCH --mem={h["memory"]}G
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

mkdir -p "$QCSCRATCH"

cd "{stage}/setup"

exec "{sys.executable}" "{ROOT}/bomd/code/setup.py"
"""
    )

    setup_script.chmod(0o755)

    job = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            str(setup_script)
        ],
        text=True
    ).strip().split(";")[0]

    return job


def submit_bomd_array(cfg, stage, setup_job):

    h = cfg["HPC"]

    repeats = cfg["HPC"]["repeats"]

    array_script = stage / "bootstrap_array.sh"

    array_script.write_text(
        f"""#!/bin/bash
#SBATCH --account={h["account"]}
#SBATCH --job-name=BootstrapBOMD
#SBATCH --time={h["bomd_time"]}
#SBATCH --cpus-per-task={h["cores"]}
#SBATCH --mem={h["memory"]}G
#SBATCH --array=1-{repeats}
#SBATCH --output={stage}/logs/bomd_%A_%a.out
#SBATCH --error={stage}/logs/bomd_%A_%a.err

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

exec "{sys.executable}" "{ROOT}/bomd/code/main.py"
"""
    )

    array_script.chmod(0o755)

    job = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterok:{setup_job}",
            str(array_script)
        ],
        text=True
    ).strip().split(";")[0]

    return job


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True
    )

    args = parser.parse_args()

    cfg = load_json(args.config)

    stage = scratch_root(
        cfg["HPC"]["Runfolder"]
    )

    print("==============================================")
    print("BOOTSTRAP WORKFLOW")
    print("==============================================")
    print(f"Run directory : {stage}")
    print(
        f"Molecule     : "
        f"{cfg['Molecule_data']['Molecule']}"
    )
    print(
        f"Temperature  : "
        f"{cfg['Molecule_data']['Temp']} K"
    )
    print(
        f"Trajectories : "
        f"{cfg['HPC']['repeats']}"
    )
    print("")

    create_runtime_files(
        cfg,
        stage
    )

    copy_geometry(
        cfg,
        stage
    )

    setup_job = submit_setup(
        cfg,
        stage
    )

    print(
        f"Bootstrap setup job: {setup_job}"
    )

    bomd_job = submit_bomd_array(
        cfg,
        stage,
        setup_job
    )
    
    # ------------------------------------------------------------
    # Submit automatic trajectory controller.
    #
    # The controller waits for the BOMD array, checks every
    # trajectory, restarts incomplete trajectories, checks again,
    # and launches GPU MACE training only when ALL trajectories
    # are genuinely complete.
    # ------------------------------------------------------------

    controller_script = (
        ROOT / "00_bootstrap" / "bootstrap_controller.py"
    )

    #controller_job = subprocess.check_output(
    #    [
    #        "sbatch",
    #        "--parsable",
    #        "--account",
    #        cfg["HPC"]["account"],
    #        "--dependency",
    #        f"afterany:{bomd_job}",
    #        "--cpus-per-task=1",
    #        "--mem=1G",
    #        "--time=00:15:00",
    #        "--job-name=BootstrapCheck",
    #        f"--output={stage}/logs/controller_%j.out",
    #        f"--error={stage}/logs/controller_%j.err",
    #        str(controller_script),
    #        "--stage",
    #        "0",
    #        "--run-root",
    #        str(stage),
    #        "--config",
    #        str(Path(args.config).resolve()),
    #        "--job-id",
    #        str(bomd_job)
    #    ],
    #    text=True
    #).strip().split(";")[0]

    controller_job = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--account",
            cfg["HPC"]["account"],
            "--dependency",
            f"afterany:{bomd_job}",
            "--cpus-per-task=1",
            "--mem=1G",
            "--time=00:15:00",
            "--job-name=BootstrapCheck",
            f"--output={stage}/logs/controller_%j.out",
            f"--error={stage}/logs/controller_%j.err",
    
            # Use the real repository controller script,
            # not the Slurm temporary copy
            str(ROOT / "00_bootstrap" / "bootstrap_controller.py"),
    
            "--stage",
            "0",
    
            "--run-root",
            str(stage),
    
            "--config",
            str(Path(args.config).resolve()),
    
            "--job-id",
            str(bomd_job),
    
            # Explicitly tell the controller where the repository is
            "--repo-root",
            str(ROOT),
        ],
        text=True
    ).strip()

    print("")
    print("BOMD trajectories submitted.")
    print("")
    print(
        f"Automatic bootstrap controller: {controller_job}"
    )
    print("")
    print(
        "The controller will:"
    )
    print(
        "  1. Check all trajectories."
    )
    print(
        "  2. Restart ONLY incomplete trajectories whose Slurm state is TIMEOUT."
    )
    print(
        "  3. Check again after every restart."
    )
    print(
        "  4. Launch MACE training on GPU only when ALL "
        "trajectories are complete."
    )

    #print(
    #    f"Bootstrap BOMD array: {bomd_job}"
    #)

    #print("")
    #print(
    #    "BOMD trajectories submitted."
    #)
    #print(
    #    "After they finish, run:"
    #)
    #print(
    #    f"  python bootstrap_train.py "
    #    f"--config {args.config}"
    #)


if __name__ == "__main__":
    main()
