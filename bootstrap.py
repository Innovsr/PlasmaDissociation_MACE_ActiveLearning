#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.machine import get_machine, get_scratch_root


ROOT = Path(__file__).resolve().parent


def load_json(path):
    return json.loads(Path(path).read_text())


# ============================================================
# MACHINE CONFIGURATION
# ============================================================

def get_machine_config(cfg):
    """
    Return:

        machine name
        machine-specific configuration
    """

    return get_machine(cfg)


def scratch_root(cfg):
    """
    Return the machine-specific bootstrap scratch directory.
    """

    return (
        get_scratch_root(cfg)
        / cfg["HPC"]["Runfolder"]
        / "00_bootstrap"
    )


# ============================================================
# SLURM CONFIGURATION
# ============================================================

def get_bomd_slurm_settings(cfg):
    """
    Read the BOMD partition/QOS from configs/machines.json.
    """

    machine, machine_cfg = get_machine_config(cfg)

    bomd_cfg = machine_cfg.get("bomd", {})

    partition = bomd_cfg.get("partition")

    if not partition:
        raise RuntimeError(
            f"No BOMD partition configured for machine '{machine}'."
        )

    qos = bomd_cfg.get("qos")

    return machine, partition, qos


def slurm_directives(cfg):
    """
    Generate machine-specific Slurm directives.
    """

    machine, partition, qos = get_bomd_slurm_settings(cfg)

    lines = [
        f"#SBATCH --partition={partition}"
    ]

    if qos:
        lines.append(
            f"#SBATCH --qos={qos}"
        )

    return "\n".join(lines)


# ============================================================
# Q-CHEM CONFIGURATION
# ============================================================

def get_qchem_config(cfg):
    """
    Read the Q-Chem configuration from configs/machines.json.

    Aire:
        qchem_module

    Calder:
        qchem
    """

    machine, machine_cfg = get_machine_config(cfg)

    bomd_cfg = machine_cfg.get("bomd", {})

    if machine == "aire":

        qchem_module = bomd_cfg.get(
            "qchem_module"
        )

        if not qchem_module:
            raise RuntimeError(
                "No qchem_module configured for Aire."
            )

        return {
            "machine": machine,
            "mode": "module",
            "module": qchem_module
        }

    if machine == "calder":

        qchem_root = bomd_cfg.get(
            "qchem"
        )

        if not qchem_root:
            raise RuntimeError(
                "No Q-Chem installation configured "
                "for Calder."
            )

        qchem_root = qchem_root.replace(
            "$USER",
            os.environ.get(
                "USER",
                os.environ.get(
                    "LOGNAME",
                    "unknown"
                )
            )
        )

        qchem_root = os.path.expandvars(
            os.path.expanduser(qchem_root)
        )

        qchem_path = Path(qchem_root)

        if not qchem_path.exists():
            raise RuntimeError(
                "Configured Calder Q-Chem installation "
                "does not exist:\n"
                f"  {qchem_path}"
            )

        return {
            "machine": machine,
            "mode": "direct",
            "root": qchem_path
        }

    raise RuntimeError(
        f"Unsupported machine '{machine}'."
    )


def qchem_environment(
    cfg,
    qchem_scratch
):
    """
    Generate the Q-Chem environment for a Slurm job.

    Aire:
        load Q-Chem module.

    Calder:
        use the local Q-Chem installation directly.
    """

    qchem = get_qchem_config(cfg)

    if qchem["mode"] == "module":

        return f"""
# ============================================================
# Q-CHEM
# ============================================================

module unload qchem 2>/dev/null || true
module load {qchem["module"]}

export LM_LICENSE_FILE=27000@uol-lnx-lic01.leeds.ac.uk

export QC="$QCHEM_HOME"
export QCAUX="$QC/qcaux"
export QCPROG="$QC/exe/qcprog.exe"
export QCPROG_S="$QC/exe/qcprog.exe_s"

export PATH="$QC/bin:$QC/exe:$PATH"

export QCSCRATCH="{qchem_scratch}"

mkdir -p "$QCSCRATCH"

echo "Q-Chem module : {qchem["module"]}"
echo "Q-Chem root   : $QC"
echo "Q-Chem exe    : $QCPROG"
echo "Q-Chem scratch: $QCSCRATCH"

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
"""

    if qchem["mode"] == "direct":

        root = qchem["root"]

        return f"""
# ============================================================
# Q-CHEM
# ============================================================

export QCHEM_HOME="{root}"

export QC="$QCHEM_HOME"
export QCAUX="$QC/qcaux"
export QCPROG="$QC/exe/qcprog.exe"
export QCPROG_S="$QC/exe/qcprog.exe_s"

export PATH="$QC/bin:$QC/exe:$PATH"

export LM_LICENSE_FILE=27000@uol-lnx-lic01.leeds.ac.uk

export QCSCRATCH="{qchem_scratch}"

mkdir -p "$QCSCRATCH"

echo "Q-Chem root   : $QC"
echo "Q-Chem exe    : $QCPROG"
echo "Q-Chem scratch: $QCSCRATCH"

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
"""

    raise RuntimeError(
        "Unknown Q-Chem configuration mode."
    )


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

    setup_cfg = dict(runtime)

    setup_cfg["HPC"]["repeats"] = (
        cfg["HPC"]["repeats"]
    )

    setup_cfg["Trajectory_selection"] = {
        "enabled": 0
    }

    (stage / "validation.json").write_text(
        json.dumps(
            setup_cfg,
            indent=4
        )
    )

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

    partition_directives = (
        slurm_directives(cfg)
    )

    qchem_scratch = (
        stage / "setup" / "tmp"
    )

    qchem_env = qchem_environment(
        cfg,
        qchem_scratch
    )

    setup_script.write_text(
        f"""#!/bin/bash

#SBATCH --account={h["account"]}
{partition_directives}
#SBATCH --job-name=BootstrapSetup
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task={h["cores"]}
#SBATCH --mem={h["memory"]}G
#SBATCH --output={stage}/logs/setup_%j.out
#SBATCH --error={stage}/logs/setup_%j.err

set -euo pipefail

{qchem_env}

echo "=============================================="
echo "BOOTSTRAP SETUP"
echo "=============================================="

echo "Hostname:"
hostname

echo "Machine:"
echo "{get_machine_config(cfg)[0]}"

echo "Python:"
{sys.executable} --version

echo "=============================================="

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

    partition_directives = (
        slurm_directives(cfg)
    )

    qchem_scratch = (
        f"{stage}/repetitions/"
        f"rep-$SLURM_ARRAY_TASK_ID/tmp"
    )

    qchem_env = qchem_environment(
        cfg,
        qchem_scratch
    )

    array_script.write_text(
        f"""#!/bin/bash

#SBATCH --account={h["account"]}
{partition_directives}
#SBATCH --job-name=BootstrapBOMD
#SBATCH --time={h["bomd_time"]}
#SBATCH --cpus-per-task={h["cores"]}
#SBATCH --mem={h["memory"]}G
#SBATCH --array=1-{repeats}
#SBATCH --output={stage}/logs/bomd_%A_%a.out
#SBATCH --error={stage}/logs/bomd_%A_%a.err

set -euo pipefail

{qchem_env}

echo "=============================================="
echo "BOOTSTRAP BOMD"
echo "=============================================="

echo "Array task:"
echo "$SLURM_ARRAY_TASK_ID"

echo "Hostname:"
hostname

echo "Machine:"
{get_machine_config(cfg)[0]}

echo "Python:"
{sys.executable} --version

echo "=============================================="

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

    machine, machine_cfg = (
        get_machine_config(cfg)
    )

    stage = scratch_root(cfg)

    print("==============================================")
    print("BOOTSTRAP WORKFLOW")
    print("==============================================")

    print(
        f"Machine       : {machine}"
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

    print(
        f"BOMD partition: "
        f"{machine_cfg['bomd']['partition']}"
    )

    if machine_cfg["bomd"].get("qos"):
        print(
            f"BOMD QOS      : "
            f"{machine_cfg['bomd']['qos']}"
        )

    qchem_cfg = get_qchem_config(cfg)

    if qchem_cfg["mode"] == "direct":
        print(
            f"Q-Chem       : "
            f"{qchem_cfg['root']}"
        )
    else:
        print(
            f"Q-Chem module: "
            f"{qchem_cfg['module']}"
        )

    print("==============================================")

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
