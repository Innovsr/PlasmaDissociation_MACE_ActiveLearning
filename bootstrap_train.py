#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(
        Path(path).read_text()
    )


def scratch_stage(cfg):

    return (
        Path("/mnt/scratch")
        / os.environ.get(
            "USER",
            os.environ.get("LOGNAME", "unknown")
        )
        / cfg["HPC"]["Runfolder"]
        / "00_bootstrap"
    )


def next_model_version(models_dir, molecule):

    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    safe = molecule.replace(
        "/",
        "_"
    )

    existing = []

    for d in models_dir.glob(
        f"{safe}_*"
    ):

        if not d.is_dir():
            continue

        try:
            number = int(
                d.name.rsplit("_", 1)[1]
            )

            existing.append(number)

        except ValueError:
            pass

    number = (
        max(existing) + 1
        if existing
        else 1
    )

    return f"{safe}_{number:03d}"


def archive(source, destination):

    with zipfile.ZipFile(
        destination,
        "w",
        zipfile.ZIP_DEFLATED
    ) as z:

        z.write(
            source,
            Path(source).name
        )


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--config",
        required=True
    )

    args = p.parse_args()

    cfg = load(
        args.config
    )

    stage = scratch_stage(
        cfg
    )

    repetitions = (
        stage / "repetitions"
    )

    if not repetitions.exists():
        raise SystemExit(
            f"Missing trajectories: {repetitions}"
        )

    # ------------------------------------------------------------
    # Convert BOMD trajectories to EXTXYZ.
    # ------------------------------------------------------------

    full_dataset = (
        stage
        / "datasets"
        / "bootstrap_full.extxyz"
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "bomd_to_extxyz.py"),
            "--run-root",
            str(stage),
            "--output",
            str(full_dataset)
        ],
        check=True
    )

    # ------------------------------------------------------------
    # Split by COMPLETE trajectory.
    # ------------------------------------------------------------

    dataset_dir = (
        stage / "datasets"
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "split_trajectory_dataset.py"),

            "--input",
            str(full_dataset),

            "--output-dir",
            str(dataset_dir),

            "--train-percent",
            str(
                cfg["dataset_split"]["training_percent"]
            ),

            "--valid-percent",
            str(
                cfg["dataset_split"]["validation_percent"]
            ),

            "--test-percent",
            str(
                cfg["dataset_split"]["test_percent"]
            ),

            "--seed",
            str(
                cfg["dataset_split"]["seed"]
            )
        ],
        check=True
    )

    train = dataset_dir / "train.extxyz"
    valid = dataset_dir / "valid.extxyz"
    test = dataset_dir / "test.extxyz"

    # ------------------------------------------------------------
    # Train from scratch.
    # ------------------------------------------------------------

    work = (
        stage / "model_work"
    )

    work.mkdir(
        parents=True,
        exist_ok=True
    )

    cfg_path = (
        stage / "bootstrap_training_config.json"
    )

    cfg_path.write_text(
        json.dumps(
            cfg,
            indent=4
        )
    )

    molecule = (
        cfg["Molecule_data"]["Molecule"]
    )

    model_name = (
        f"bootstrap_{molecule}"
    )

    mace = cfg["mace"]

    cmd = [
        mace["command"],

        f"--name={model_name}",

        f"--work_dir={work}",

        f"--train_file={train}",

        f"--valid_file={valid}",

        f"--test_file={test}",

        f"--energy_key={mace['energy_key']}",

        f"--forces_key={mace['forces_key']}",

        "--model=MACE",

        "--loss=weighted",

        "--E0s=average",

        f"--r_max={mace['r_max']}",

        f"--radial_type={mace['radial_type']}",

        f"--num_radial_basis={mace['num_radial_basis']}",

        f"--num_cutoff_basis={mace['num_cutoff_basis']}",

        f"--max_ell={mace['max_ell']}",

        f"--correlation={mace['correlation']}",

        f"--num_interactions={mace['num_interactions']}",

        f"--hidden_irreps={mace['hidden_irreps']}",

        f"--batch_size={mace['batch_size']}",

        f"--valid_batch_size={mace['valid_batch_size']}",

        f"--energy_weight={mace['energy_weight']}",

        f"--forces_weight={mace['forces_weight']}",

        f"--lr={mace['lr']}",

        f"--max_num_epochs={mace['max_num_epochs']}",

        f"--patience={mace['patience']}",

        f"--eval_interval={mace['eval_interval']}",

        f"--device={mace['device']}",

        f"--default_dtype={mace['default_dtype']}",

        f"--seed={mace['seed']}"
    ]

    print(
        "Starting MACE training FROM SCRATCH."
    )

    subprocess.run(
        cmd,
        check=True
    )

    # ------------------------------------------------------------
    # Locate resulting model.
    # ------------------------------------------------------------

    candidates = list(
        work.rglob("*.model")
    )

    if not candidates:
        raise SystemExit(
            "MACE finished but no .model file was found."
        )

    model = max(
        candidates,
        key=lambda x: x.stat().st_mtime
    )

    # ------------------------------------------------------------
    # Create unique model directory.
    # ------------------------------------------------------------

    model_id = next_model_version(
        ROOT / "models",
        molecule
    )

    output = (
        ROOT / "models" / model_id
    )

    output.mkdir(
        parents=True,
        exist_ok=False
    )

    shutil.copy2(
        model,
        output / "model.model"
    )

    # ------------------------------------------------------------
    # Preserve datasets.
    # ------------------------------------------------------------

    archive(
        train,
        output / "training_data.zip"
    )

    archive(
        valid,
        output / "validation_data.zip"
    )

    archive(
        test,
        output / "test_data.zip"
    )

    # ------------------------------------------------------------
    # Preserve checkpoints.
    # ------------------------------------------------------------

    checkpoints = work / "checkpoints"

    if checkpoints.exists():

        shutil.copytree(
            checkpoints,
            output / "checkpoints"
        )

    # ------------------------------------------------------------
    # Metadata.
    # ------------------------------------------------------------

    metadata = {
        "model_id": model_id,

        "parent_model": None,

        "training_mode": "from_scratch",

        "molecules": [
            molecule
        ],

        "temperature_K": cfg["Molecule_data"]["Temp"],

        "multiplicity": cfg["Molecule_data"]["Multiplicity"],

        "electronic_structure": cfg["Elec_structure"],

        "propagation": cfg["Propagation"],

        "number_of_trajectories":
            cfg["HPC"]["repeats"],

        "dataset_split":
            cfg["dataset_split"],

        "dataset_files": {
            "training": "training_data.zip",
            "validation": "validation_data.zip",
            "test": "test_data.zip"
        },

        "mace": cfg["mace"],

        "model_file": "model.model",

        "status": "bootstrap_candidate"
    }

    (
        output / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=4
        )
    )

    # Preserve exact bootstrap configuration.
    shutil.copy2(
        cfg_path,
        output / "bootstrap_config.json"
    )

    print("")
    print("==============================================")
    print("BOOTSTRAP MODEL CREATED")
    print("==============================================")
    print(f"Model ID : {model_id}")
    print(f"Directory: {output}")
    print("")
    print(
        "The model has NOT been marked as production-approved."
    )
    print(
        "Review metadata.json and independent test results first."
    )


if __name__ == "__main__":
    main()
