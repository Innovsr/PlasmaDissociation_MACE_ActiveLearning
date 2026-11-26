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


# ============================================================
# MACHINE / SCRATCH
# ============================================================

def scratch_root(cfg):
    """
    Return the machine-specific scratch directory.

    This uses the machine configuration through get_scratch_root().
    """

    from scripts.machine import get_scratch_root

    return (
        get_scratch_root(cfg)
        / cfg["HPC"]["Runfolder"]
        / "00_bootstrap"
    )


# ============================================================
# Q-CHEM CONFIGURATION
# ============================================================

def get_qchem_root(cfg):
    """
    Return the Q-Chem installation to use.

    Priority:

    1. HPC.qchem_root from config, if provided.
    2. Machine-specific Calder installation.
    3. QCHEM_HOME environment variable.

    This avoids relying on a Q-Chem module that may not exist
    on the current machine.
    """

    hpc = cfg.get("HPC", {})

    # --------------------------------------------------------
    # 1. Explicit path from configuration
    # --------------------------------------------------------

    configured = hpc.get("qchem_root")

    if configured:
        path = Path(
            os.path.expandvars(
                os.path.expanduser(configured)
            )
        )

        if path.exists():
            return path

        raise RuntimeError(
            "Configured Q-Chem installation does not exist:\n"
            f"  {path}"
        )

    # --------------------------------------------------------
    # 2. Calder local installation
    # --------------------------------------------------------

    user = os.environ.get(
        "USER",
        os.environ.get("LOGNAME", "unknown")
    )

    calder_qchem = Path(
        f"/scratch-calder/users/{user}/software/qchem-6.0.2"
    )

    if calder_qchem.exists():
        return calder_qchem

    # --------------------------------------------------------
    # 3. Existing QCHEM_HOME
    # --------------------------------------------------------

    qchem_home = os.environ.get("QCHEM_HOME")

    if qchem_home:
        path = Path(qchem_home)

        if path.exists():
            return path

    raise RuntimeError(
        "Could not locate Q-Chem installation.\n\n"
        "Checked:\n"
        f"  {calder_qchem}\n"
        "  $QCHEM_HOME\n\n"
        "Set HPC.qchem_root in the configuration if necessary."
    )


def qchem_environment(cfg, stage, scratch_subdir):
    """
    Generate the shell commands required to initialise Q-Chem.

    This deliberately does NOT use:

        module load qchem/6.0.2

    because Calder does not provide that module.

    Instead it directly uses the Q-Chem installation.
    """

    qchem_root = get_qchem_root(cfg)

    qchem_scratch = (
        Path(stage)
        / scratch_subdir
    )

    return f"""
# ------------------------------------------------------------
# Q-CHEM
# ------------------------------------------------------------

export QCHEM_HOME="{qchem_root}"

export QC="$QCHEM_HOME"
export QCAUX="$QC/qcaux"
export QCPROG="$QC/exe/qcprog.exe"
export QCPROG_S="$QC/exe/qcprog.exe_s"

export PATH="$QC/bin:$QC/exe:$PATH"

export QCSCRATCH="{qchem_scratch}"

mkdir -p "$QCSCRATCH"

# Q-Chem licence
export LM_LICENSE_FILE=27000@uol-lnx-lic01.leeds.ac.uk

# Basic validation
if [ ! -x "$QCPROG" ]; then
    echo "ERROR: Q-Chem executable not found:"
    echo "  $QCPROG"
    exit 1
fi

if [ ! -d "$QCAUX" ]; then
    echo "ERROR: Q-Chem qcaux directory not found:"
    echo "  $QCAUX"
    exit 1
fi

echo "Q-Chem installation:"
echo "  $QC"

echo "Q-Chem executable:"
echo "  $QCPROG"

echo "Q-Chem scratch:"
echo "  $QCSCRATCH"
"""


# ============================================================
# RUNTIME FILES
# ============================================================

def create_runtime_files(cfg, stage):

    stage.mkdir(
        parents=True,
        exist_ok=True
    )

    (stage / "setup").mkdir(
        exist_ok=True
    )

    (stage / "repetitions").mkdir(
        exist_ok=True
    )

    (stage / "logs").mkdir(
        exist_ok=True
    )

    (stage / "datasets").mkdir(
        exist_ok=True
    )

    (stage / "model_work").mkdir(
        exist_ok=True
    )

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

    # --------------------------------------------------------
    # setup.py expects validation.json
    # --------------------------------------------------------

    setup_cfg = dict(runtime)

    setup_cfg["HPC"]["repeats"] = (
        cfg["HPC"]["repeats"]
    )

    # Bootstrap deliberately disables trajectory selection.
    setup_cfg["Trajectory_selection"] = {
        "enabled": 0
    }

    (stage / "validation.json").write_text(
        json.dumps(
            setup_cfg,
            indent=4
        )
    )

    # --------------------------------------------------------
    # main.py/global_vars.py reads ../../inputs.json
    # --------------------------------------------------------

    (stage / "inputs.json").write_text(
        json.dumps(
            runtime,
            indent=4
        )
    )


# ============================================================
# INITIAL GEOMETRY
# ============================================================

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
            "Initial geometry not found:\n"
            f"{source}"
        )

    shutil.copy2(
        source,
        stage / "setup" / "initial_guess.txt"
    )


# ============================================================
# SETUP JOB
# ============================================================

def submit_setup(cfg, stage):

    setup_script = (
        stage / "setup_bootstrap.sh"
    )

    h = cfg["HPC"]

    qchem_root = get_qchem_root(cfg)

    qchem_scratch = (
        stage / "setup" / "tmp"
    )

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


# ============================================================
# Q-CHEM
# ============================================================

export QCHEM_HOME="{qchem_root}"

export QC="$QCHEM_HOME"
export QCAUX="$QC/qcaux"
export QCPROG="$QC/exe/qcprog.exe"
export QCPROG_S="$QC/exe/qcprog.exe_s"

export PATH="$QC/bin:$QC/exe:$PATH"

export LM_LICENSE_FILE=27000@uol-lnx-lic01.leeds.ac.uk

export QCSCRATCH="{qchem_scratch}"

mkdir -p "$QCSCRATCH"


echo "=============================================="
echo "BOOTSTRAP SETUP"
echo "=============================================="

echo "Hostname:"
hostname

echo "Q-Chem:"
echo "$QC"

echo "QCPROG:"
echo "$QCPROG"

echo "QCSCRATCH:"
echo "$QCSCRATCH"

echo "Python:"
{sys.executable} --version

echo "=============================================="


# ------------------------------------------------------------
# Validate Q-Chem
# ------------------------------------------------------------

if [ ! -x "$QCPROG" ]; then
    echo "ERROR: Q-Chem executable not found:"
    echo "$QCPROG"
    exit 1
fi

if [ ! -d "$QCAUX" ]; then
    echo "ERROR: Q-Chem qcaux directory not found:"
    echo "$QCAUX"
    exit 1
fi


# ============================================================
# RUN SETUP
# ============================================================

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


# ============================================================
# BOMD ARRAY
# ============================================================

def submit_bomd_array(
    cfg,
    stage,
    setup_job
):

    h = cfg["HPC"]

    repeats = cfg["HPC"]["repeats"]

    array_script = (
        stage / "bootstrap_array.sh"
    )

    qchem_root = get_qchem_root(cfg)

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


# ============================================================
# Q-CHEM
# ============================================================

export QCHEM_HOME="{qchem_root}"

export QC="$QCHEM_HOME"
export QCAUX="$QC/qcaux"
export QCPROG="$QC/exe/qcprog.exe"
export QCPROG_S="$QC/exe/qcprog.exe_s"

export PATH="$QC/bin:$QC/exe:$PATH"

export LM_LICENSE_FILE=27000@uol-lnx-lic01.leeds.ac.uk

export QCSCRATCH="{stage}/repetitions/rep-$SLURM_ARRAY_TASK_ID/tmp"

mkdir -p "$QCSCRATCH"


echo "=============================================="
echo "BOOTSTRAP BOMD"
echo "=============================================="

echo "Array task:"
echo "$SLURM_ARRAY_TASK_ID"

echo "Hostname:"
hostname

echo "Q-Chem:"
echo "$QC"

echo "QCPROG:"
echo "$QCPROG"

echo "QCSCRATCH:"
echo "$QCSCRATCH"

echo "=============================================="


# ------------------------------------------------------------
# Validate Q-Chem
# ------------------------------------------------------------

if [ ! -x "$QCPROG" ]; then
    echo "ERROR: Q-Chem executable not found:"
    echo "$QCPROG"
    exit 1
fi

if [ ! -d "$QCAUX" ]; then
    echo "ERROR: Q-Chem qcaux directory not found:"
    echo "$QCAUX"
    exit 1
fi


# ============================================================
# RUN BOMD
# ============================================================

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


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True
    )

    args = parser.parse_args()

    cfg = load_json(
        args.config
    )

    # --------------------------------------------------------
    # Determine scratch directory
    # --------------------------------------------------------

    stage = scratch_root(cfg)

    print(
        "=============================================="
    )

    print(
        "BOOTSTRAP WORKFLOW"
    )

    print(
        "=============================================="
    )

    print(
        f"Run directory : {stage}"
    )

    print(
        f"Molecule      : "
        f"{cfg['Molecule_data']['Molecule']}"
    )

    print(
        f"Temperature   : "
        f"{cfg['Molecule_data']['Temp']} K"
    )

    print(
        f"Trajectories  : "
        f"{cfg['HPC']['repeats']}"
    )

    # --------------------------------------------------------
    # Q-Chem check before submitting jobs
    # --------------------------------------------------------

    qchem_root = get_qchem_root(
        cfg
    )

    print(
        f"Q-Chem        : {qchem_root}"
    )

    print(
        "=============================================="
    )

    # --------------------------------------------------------
    # Create runtime files
    # --------------------------------------------------------

    create_runtime_files(
        cfg,
        stage
    )

    # --------------------------------------------------------
    # Copy initial geometry
    # --------------------------------------------------------

    copy_geometry(
        cfg,
        stage
    )

    # --------------------------------------------------------
    # Submit setup
    # --------------------------------------------------------

    setup_job = submit_setup(
        cfg,
        stage
    )

    print(
        f"Bootstrap setup job: {setup_job}"
    )

    # --------------------------------------------------------
    # Submit BOMD array
    # --------------------------------------------------------

    bomd_job = submit_bomd_array(
        cfg,
        stage,
        setup_job
    )

    # --------------------------------------------------------
    # Automatic bootstrap controller
    # --------------------------------------------------------

    controller_script = (
        ROOT
        / "00_bootstrap"
        / "bootstrap_controller.py"
    )

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

            # Use the real repository controller
            str(controller_script),

            "--stage",
            "0",

            "--run-root",
            str(stage),

            "--config",
            str(
                Path(
                    args.config
                ).resolve()
            ),

            "--job-id",
            str(bomd_job),

            # Explicit repository root
            "--repo-root",
            str(ROOT)
        ],
        text=True
    ).strip()

    print("")
    print(
        "BOMD trajectories submitted."
    )

    print("")
    print(
        f"Automatic bootstrap controller: "
        f"{controller_job}"
    )

    print("")

    print(
        "The controller will:"
    )

    print(
        "  1. Check all trajectories."
    )

    print(
        "  2. Restart ONLY incomplete trajectories "
        "whose Slurm state is TIMEOUT."
    )

    print(
        "  3. Check again after every restart."
    )

    print(
        "  4. Launch MACE training on GPU only "
        "when ALL trajectories are complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
