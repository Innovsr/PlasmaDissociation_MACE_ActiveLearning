#!/usr/bin/env python3

"""
Active-learning model update workflow.

Workflow
--------
1. Take the completed validation.extxyz from Stage 1.
2. Split trajectories in blocks of 10:
       7 trajectories -> training pool
       2 trajectories -> validation
       1 trajectory   -> test
3. Evaluate the source model on the training pool.
4. Select high-error configurations ONLY from the training pool.
5. Combine old source-model training data with selected new data.
6. Train a candidate MACE model using:
       training.extxyz
       validation_set.extxyz
7. Evaluate the candidate on the independent validation set.
8. Evaluate the candidate on the independent test set.
9. Compare source and candidate model performance.
10. Store the candidate for user approval.

Restart behaviour
-----------------
The model-update MACE training uses the preserved checkpoint
from the source model as its starting point.

First model-update run:
    source model checkpoint
            |
            v
    02_model_update/<attempt>/mace_work/checkpoints/
            |
            v
       --restart
            |
            v
      MACE --restart_latest

If the model-update SLURM job times out:
    existing candidate checkpoint
            |
            v
    next chained SLURM job
            |
            v
       --restart
            |
            v
      MACE --restart_latest

Bootstrap workflow is not modified by this script.
"""

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

from ase.io import read, write


ROOT = Path(__file__).resolve().parent


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def load_config(path):
    return json.loads(
        Path(path).read_text()
    )


def latest_model(root, requested):
    model_dir = (
        root
        / "models"
        / requested
    )

    if not model_dir.exists():
        raise SystemExit(
            f"Approved model not found: {requested}"
        )

    model_file = (
        model_dir
        / "model.model"
    )

    if not model_file.exists():
        raise SystemExit(
            f"Missing model file: {model_file}"
        )

    return model_dir


def extract_training(zip_path, output_dir):
    """
    Extract the largest XYZ/EXTXYZ dataset from training_data.zip.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with zipfile.ZipFile(zip_path) as archive:

        members = [
            name
            for name in archive.namelist()
            if name.endswith(".xyz")
            or name.endswith(".extxyz")
        ]

        if not members:
            raise SystemExit(
                "Training archive contains no EXTXYZ/XYZ dataset."
            )

        member = max(
            members,
            key=lambda name:
                archive.getinfo(name).file_size
        )

        archive.extract(
            member,
            output_dir
        )

        return (
            output_dir
            / member
        )


def registry_row(root, model_id):
    registry_file = (
        root
        / "registry"
        / "models.csv"
    )

    if not registry_file.exists():
        return {}

    with registry_file.open() as f:

        for row in csv.DictReader(f):

            if row.get("model_id") == model_id:
                return row

    return {}


def run_command(command):
    """
    Print and execute a subprocess command.
    """

    print()
    print("Running:")
    print(
        " ".join(
            str(x)
            for x in command
        )
    )
    print()

    subprocess.run(
        command,
        check=True
    )


# ----------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=str(
            ROOT
            / "configs"
            / "update_model.json"
        ),
    )

    parser.add_argument(
        "--validation",
        default="",
        help="Path to completed Stage-1 validation.extxyz",
    )

    parser.add_argument(
        "--source-model",
        default=None,
        help="Override source model from config",
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    root = ROOT

    # ------------------------------------------------------------------
    # Source model
    # ------------------------------------------------------------------

    source = (
        args.source_model
        or config["source_model"]
    )

    model_dir = latest_model(
        root,
        source,
    )

    metadata = registry_row(
        root,
        source,
    )

    source_epoch = int(
        metadata.get("best_epoch")
        or config.get(
            "old_epoch",
            0
        )
    )

    print("=" * 70)
    print("ACTIVE-LEARNING MODEL UPDATE")
    print("=" * 70)

    print(
        f"Source model : {source}"
    )

    print(
        f"Source epoch : {source_epoch}"
    )

    # ------------------------------------------------------------------
    # Locate Stage-1 validation dataset
    # ------------------------------------------------------------------

    if args.validation:

        validation_path = Path(
            args.validation
        )

    else:

        validation_path = (
            Path("/mnt/scratch")
            / os.environ.get(
                "USER",
                os.environ.get(
                    "LOGNAME",
                    "unknown"
                )
            )
            / config["HPC"]["Runfolder"]
            / "01_validation"
            / "validation"
            / "validation.extxyz"
        )

    if not validation_path.exists():

        raise SystemExit(
            f"Validation dataset not found:\n"
            f"{validation_path}\n\n"
            f"Complete Stage 1 first."
        )

    print(
        f"Validation dataset: {validation_path}"
    )

    # ------------------------------------------------------------------
    # Attempt directory
    # ------------------------------------------------------------------

    scratch_root = (
        Path("/mnt/scratch")
        / os.environ.get(
            "USER",
            os.environ.get(
                "LOGNAME",
                "unknown"
            )
        )
        / config["HPC"]["Runfolder"]
    )

    attempt_dir = (
        scratch_root
        / "02_model_update"
        / config.get(
            "attempt_id",
            "attempt_001"
        )
    )

    attempt_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    status_file = (
        attempt_dir
        / "STATUS"
    )

    if status_file.exists():

        status = (
            status_file
            .read_text()
            .strip()
        )

        if status in {
            "CANDIDATE_READY_FOR_USER_APPROVAL",
            "NO_HIGH_ERROR_CONFIGURATIONS",
            "MODEL_OK_USE_EXISTING",
        }:

            print(
                f"Attempt already terminal: {status}"
            )

            return 0

    # ------------------------------------------------------------------
    # Save source information
    # ------------------------------------------------------------------

    (
        attempt_dir
        / "source_model.txt"
    ).write_text(
        f"{source}\n"
        f"source_epoch={source_epoch}\n"
    )

    full_validation_copy = (
        attempt_dir
        / "validation.extxyz"
    )

    if not full_validation_copy.exists():

        shutil.copy2(
            validation_path,
            full_validation_copy,
        )

    # ------------------------------------------------------------------
    # STEP 1
    # Split trajectories:
    #
    # Every 10:
    #     7 training
    #     2 validation
    #     1 test
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 1: SPLIT VALIDATION TRAJECTORIES")
    print("=" * 70)

    split_dir = (
        attempt_dir
        / "split"
    )

    split_script = (
        root
        / "split_validation_trajectories.py"
    )

    if not split_script.exists():

        raise SystemExit(
            f"Missing split script:\n"
            f"{split_script}"
        )

    run_command(
        [
            sys.executable,
            str(split_script),

            "--input",
            str(full_validation_copy),

            "--output-dir",
            str(split_dir),
        ]
    )

    training_pool = (
        split_dir
        / "training_pool.extxyz"
    )

    validation_set = (
        split_dir
        / "validation_set.extxyz"
    )

    test_set = (
        split_dir
        / "test_set.extxyz"
    )

    # ------------------------------------------------------------------
    # Verify split files
    # ------------------------------------------------------------------

    for path in [
        training_pool,
        validation_set,
        test_set,
    ]:

        if not path.exists():

            raise SystemExit(
                f"Expected split file was not created:\n"
                f"{path}"
            )

    # ------------------------------------------------------------------
    # STEP 2
    # Evaluate source model on TRAINING POOL.
    #
    # This is deliberately separate from the independent validation
    # and test sets.
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 2: SOURCE MODEL ON TRAINING POOL")
    print("=" * 70)

    training_pool_results = (
        attempt_dir
        / "training_pool_results.csv"
    )

    run_command(
        [
            sys.executable,
            str(
                root
                / "validate_model.py"
            ),

            "--model",
            str(
                model_dir
                / "model.model"
            ),

            "--validation",
            str(training_pool),

            "--output",
            str(training_pool_results),

            "--device",
            config["mace"]["device"],

            "--default-dtype",
            config["mace"]["default_dtype"],

            "--summary",
            str(
                attempt_dir
                / "training_pool_summary.json"
            ),
        ]
    )

    # ------------------------------------------------------------------
    # STEP 3
    # Evaluate source model on INDEPENDENT VALIDATION SET.
    #
    # This set is NOT used to select training configurations.
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 3: SOURCE MODEL ON VALIDATION SET")
    print("=" * 70)

    source_validation_results = (
        attempt_dir
        / "source_validation.csv"
    )

    run_command(
        [
            sys.executable,
            str(
                root
                / "validate_model.py"
            ),

            "--model",
            str(
                model_dir
                / "model.model"
            ),

            "--validation",
            str(validation_set),

            "--output",
            str(source_validation_results),

            "--device",
            config["mace"]["device"],

            "--default-dtype",
            config["mace"]["default_dtype"],

            "--summary",
            str(
                attempt_dir
                / "source_summary.json"
            ),
        ]
    )

    source_summary = json.loads(
        (
            attempt_dir
            / "source_summary.json"
        ).read_text()
    )

    source_force = float(
        source_summary[
            "force_rmse_mev_A"
        ]
    )

    source_energy = float(
        source_summary[
            "energy_rmse_mev_atom"
        ]
    )

    print()
    print("SOURCE MODEL VALIDATION")

    print(
        f"Force RMSE  : "
        f"{source_force:.6f} meV/A"
    )

    print(
        f"Energy RMSE : "
        f"{source_energy:.6f} meV/atom"
    )

    # ------------------------------------------------------------------
    # If source model already satisfies the validation targets,
    # there is no reason to update it.
    # ------------------------------------------------------------------

    if (
        source_force
        <= config["validation"]
        ["force_rmse_target_mev_A"]
        and
        source_energy
        <= config["validation"]
        ["energy_rmse_target_mev_atom"]
    ):

        (
            attempt_dir
            / "STATUS"
        ).write_text(
            "MODEL_OK_USE_EXISTING\n"
        )

        print()
        print(
            "Source model meets both validation targets."
        )

        print(
            "No model update required."
        )

        return 0

    # ------------------------------------------------------------------
    # STEP 4
    # Select high-error configurations ONLY from training pool.
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "STEP 4: SELECT HIGH-ERROR "
        "TRAINING CONFIGURATIONS"
    )
    print("=" * 70)

    high_error_csv = (
        attempt_dir
        / "high_error.csv"
    )

    run_command(
        [
            sys.executable,
            str(
                root
                / "select_high_error.py"
            ),

            "--results",
            str(training_pool_results),

            "--output",
            str(high_error_csv),

            "--max-configurations",
            str(
                config["validation"]
                ["max_new_configurations"]
            ),

            "--force-threshold",
            str(
                config["validation"]
                ["high_error_force_threshold_mev_A"]
            ),
        ]
    )

    # ------------------------------------------------------------------
    # Extract selected configurations from TRAINING POOL.
    # ------------------------------------------------------------------

    training_pool_frames = read(
        str(training_pool),
        index=":"
    )

    with high_error_csv.open() as f:

        high_error_rows = list(
            csv.DictReader(f)
        )

    indices = [
        int(row["configuration"])
        for row in high_error_rows
    ]

    new_frames = [
        training_pool_frames[i]
        for i in indices
        if 0 <= i < len(training_pool_frames)
    ]

    new_problematic = (
        attempt_dir
        / "new_problematic.extxyz"
    )

    write(
        str(new_problematic),
        new_frames,
        format="extxyz",
    )

    print(
        f"Selected high-error configurations: "
        f"{len(new_frames)}"
    )

    # ------------------------------------------------------------------
    # No useful new data
    # ------------------------------------------------------------------

    if not new_frames:

        (
            attempt_dir
            / "STATUS"
        ).write_text(
            "NO_HIGH_ERROR_CONFIGURATIONS\n"
        )

        print(
            "No high-error configurations found."
        )

        return 0

    # ------------------------------------------------------------------
    # STEP 5
    # Extract old training data from source model.
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 5: PREPARE OLD TRAINING DATA")
    print("=" * 70)

    old_training_zip = (
        model_dir
        / "training_data.zip"
    )

    if not old_training_zip.exists():

        raise SystemExit(
            f"Approved model {source} has no "
            f"training_data.zip"
        )

    old_training_dir = (
        attempt_dir
        / "old_training"
    )

    old_training_extxyz = extract_training(
        old_training_zip,
        old_training_dir,
    )

    # ------------------------------------------------------------------
    # STEP 6
    # Build updated training set.
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 6: BUILD UPDATED TRAINING DATASET")
    print("=" * 70)

    total_configurations = (
        config["training_update"]
        .get(
            "total_configurations"
        )
    )

    if total_configurations is None:

        total_configurations = len(
            read(
                str(
                    old_training_extxyz
                ),
                index=":"
            )
        )

    training_path = (
        attempt_dir
        / "training.extxyz"
    )

    run_command(
        [
            sys.executable,
            str(
                root
                / "build_training_set.py"
            ),

            "--old",
            str(old_training_extxyz),

            "--new",
            str(new_problematic),

            "--output",
            str(training_path),

            "--old-percent",
            str(
                config["training_update"]
                ["old_data_percent"]
            ),

            "--new-percent",
            str(
                config["training_update"]
                ["new_data_percent"]
            ),

            "--total",
            str(total_configurations),

            "--seed",
            str(
                config["mace"]
                ["seed"]
            ),
        ]
    )

    # ------------------------------------------------------------------
    # STEP 7
    # Prepare MACE work directory.
    #
    # IMPORTANT:
    #
    # The candidate model starts from the existing source-model
    # checkpoint, NOT from a newly initialised model.
    #
    # Source:
    #
    # models/<source>/checkpoints/*.pt
    #
    # copied to:
    #
    # 02_model_update/<attempt>/mace_work/checkpoints/
    #
    # train_mace.py then receives --restart and adds
    # --restart_latest to the MACE command.
    #
    # If this script is called again after a timeout, the candidate
    # checkpoints already exist and are NOT overwritten by the
    # source checkpoints.
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 7: PREPARE MACE TRAINING")
    print("=" * 70)

    work = (
        attempt_dir
        / "mace_work"
    )

    work.mkdir(
        parents=True,
        exist_ok=True
    )

    source_checkpoints = (
        model_dir
        / "checkpoints"
    )

    candidate_checkpoints = (
        work
        / "checkpoints"
    )

    # ------------------------------------------------------------------
    # Source checkpoints must exist.
    # ------------------------------------------------------------------

    if not source_checkpoints.exists():

        raise SystemExit(
            f"Source model {source} has no preserved "
            f"MACE checkpoints.\n\n"
            f"Expected:\n"
            f"{source_checkpoints}\n\n"
            f"The active-learning workflow requires "
            f"the previous optimised epoch."
        )

    source_pt_files = list(
        source_checkpoints.glob("*.pt")
    )

    if not source_pt_files:

        raise SystemExit(
            f"No .pt checkpoint files found in:\n"
            f"{source_checkpoints}"
        )

    # ------------------------------------------------------------------
    # First model-update run:
    #
    # copy source checkpoint directory.
    # ------------------------------------------------------------------

    if not candidate_checkpoints.exists():

        print(
            "No candidate checkpoint directory exists."
        )

        print(
            "Copying source-model checkpoints:"
        )

        print(
            f"  {source_checkpoints}"
        )

        print(
            "to:"
        )

        print(
            f"  {candidate_checkpoints}"
        )

        shutil.copytree(
            source_checkpoints,
            candidate_checkpoints
        )

        print()
        print(
            "Source checkpoint copied successfully."
        )

    # ------------------------------------------------------------------
    # Restart/continuation:
    #
    # candidate checkpoint directory already exists.
    #
    # DO NOT overwrite it with the source checkpoint again.
    # This is essential after a SLURM timeout.
    # ------------------------------------------------------------------

    else:

        existing_candidate_checkpoints = list(
            candidate_checkpoints.glob("*.pt")
        )

        if not existing_candidate_checkpoints:

            raise SystemExit(
                "Candidate checkpoint directory exists "
                "but contains no .pt checkpoint:\n"
                f"{candidate_checkpoints}"
            )

        print(
            "Candidate checkpoint directory already exists."
        )

        print(
            "Existing candidate checkpoints will be preserved."
        )

        print(
            f"Checkpoint files: "
            f"{len(existing_candidate_checkpoints)}"
        )

    # ------------------------------------------------------------------
    # Display checkpoint information.
    # ------------------------------------------------------------------

    candidate_checkpoint_files = sorted(
        candidate_checkpoints.glob("*.pt"),
        key=lambda path: path.stat().st_mtime
    )

    print()
    print(
        "MACE CHECKPOINT STATE"
    )
    print(
        "-" * 70
    )

    for checkpoint in candidate_checkpoint_files:

        print(
            checkpoint.name
        )

    latest_checkpoint = (
        candidate_checkpoint_files[-1]
    )

    print()
    print(
        f"Latest checkpoint: "
        f"{latest_checkpoint.name}"
    )

    print(
        f"Checkpoint directory: "
        f"{candidate_checkpoints}"
    )

    print()

    # ------------------------------------------------------------------
    # STEP 8
    # Train candidate.
    #
    # The candidate always starts/resumes from the checkpoint in
    # mace_work/checkpoints.
    #
    # train_mace.py receives --restart.
    #
    # train_mace.py then adds:
    #
    #     --restart_latest
    #
    # to the mace_run_train command.
    #
    # Therefore:
    #
    # first run:
    #
    # source checkpoint -> candidate training
    #
    # timeout:
    #
    # candidate checkpoint -> next candidate training
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 8: TRAIN CANDIDATE MACE MODEL")
    print("=" * 70)

    candidate = (
        f"{source}"
        f"_candidate_"
        f"{attempt_dir.name}"
    )

    print(
        f"Candidate name : {candidate}"
    )

    print(
        f"Training data  : {training_path}"
    )

    print(
        f"Validation data: {validation_set}"
    )

    print(
        f"Work directory : {work}"
    )

    print(
        f"Restart mode   : YES"
    )

    print(
        f"Checkpoint     : {latest_checkpoint.name}"
    )

    run_command(
        [
            sys.executable,
            str(
                root
                / "train_mace.py"
            ),

            "--train",
            str(training_path),

            "--valid",
            str(validation_set),

            "--name",
            candidate,

            "--work-dir",
            str(work),

            "--device",
            config["mace"]["device"],

            "--restart",
        ]
    )

    # ------------------------------------------------------------------
    # STEP 9
    # Locate compiled candidate model.
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 9: LOCATE CANDIDATE MODEL")
    print("=" * 70)

    candidates = list(
        work.rglob("*.model")
    )

    if not candidates:

        raise SystemExit(
            "No candidate .model file found after MACE training."
        )

    candidate_model_source = max(
        candidates,
        key=lambda path:
            path.stat().st_mtime
    )

    candidate_model = (
        attempt_dir
        / "candidate.model"
    )

    shutil.copy2(
        candidate_model_source,
        candidate_model,
    )

    print(
        f"Candidate model: "
        f"{candidate_model}"
    )

    # ------------------------------------------------------------------
    # STEP 10
    # Evaluate candidate on independent validation set.
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 10: CANDIDATE VALIDATION")
    print("=" * 70)

    candidate_validation_results = (
        attempt_dir
        / "candidate_validation.csv"
    )

    run_command(
        [
            sys.executable,
            str(
                root
                / "validate_model.py"
            ),

            "--model",
            str(candidate_model),

            "--validation",
            str(validation_set),

            "--output",
            str(candidate_validation_results),

            "--device",
            config["mace"]["device"],

            "--default-dtype",
            config["mace"]["default_dtype"],

            "--summary",
            str(
                attempt_dir
                / "candidate_summary.json"
            ),
        ]
    )

    # ------------------------------------------------------------------
    # STEP 11
    # FINAL TEST
    #
    # This is the first time the candidate model sees the test set.
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 11: FINAL INDEPENDENT TEST")
    print("=" * 70)

    candidate_test_results = (
        attempt_dir
        / "candidate_test.csv"
    )

    run_command(
        [
            sys.executable,
            str(
                root
                / "validate_model.py"
            ),

            "--model",
            str(candidate_model),

            "--validation",
            str(test_set),

            "--output",
            str(candidate_test_results),

            "--device",
            config["mace"]["device"],

            "--default-dtype",
            config["mace"]["default_dtype"],

            "--summary",
            str(
                attempt_dir
                / "candidate_test_summary.json"
            ),
        ]
    )

    # ------------------------------------------------------------------
    # STEP 12
    # Compare source and candidate on VALIDATION.
    # ------------------------------------------------------------------

    candidate_summary = json.loads(
        (
            attempt_dir
            / "candidate_summary.json"
        ).read_text()
    )

    candidate_test_summary = json.loads(
        (
            attempt_dir
            / "candidate_test_summary.json"
        ).read_text()
    )

    candidate_force = float(
        candidate_summary[
            "force_rmse_mev_A"
        ]
    )

    candidate_energy = float(
        candidate_summary[
            "energy_rmse_mev_atom"
        ]
    )

    test_force = float(
        candidate_test_summary[
            "force_rmse_mev_A"
        ]
    )

    test_energy = float(
        candidate_test_summary[
            "energy_rmse_mev_atom"
        ]
    )

    comparison = {

        "source_model":
            source,

        "candidate":
            candidate,

        # Source model on independent validation set
        "source_validation_force_rmse_mev_A":
            source_force,

        "source_validation_energy_rmse_mev_atom":
            source_energy,

        # Candidate on independent validation set
        "candidate_validation_force_rmse_mev_A":
            candidate_force,

        "candidate_validation_energy_rmse_mev_atom":
            candidate_energy,

        # Candidate on completely untouched test set
        "candidate_test_force_rmse_mev_A":
            test_force,

        "candidate_test_energy_rmse_mev_atom":
            test_energy,

        # Validation improvement
        "validation_force_improvement_percent":
            (
                (
                    source_force
                    - candidate_force
                )
                / source_force
                * 100
                if source_force
                else 0.0
            ),

        "validation_energy_improvement_percent":
            (
                (
                    source_energy
                    - candidate_energy
                )
                / source_energy
                * 100
                if source_energy
                else 0.0
            ),

        # Targets
        "validation_target_force_met":
            (
                candidate_force
                <= config["validation"]
                ["force_rmse_target_mev_A"]
            ),

        "validation_target_energy_met":
            (
                candidate_energy
                <= config["validation"]
                ["energy_rmse_target_mev_atom"]
            ),

        "test_target_force_met":
            (
                test_force
                <= config["validation"]
                ["force_rmse_target_mev_A"]
            ),

        "test_target_energy_met":
            (
                test_energy
                <= config["validation"]
                ["energy_rmse_target_mev_atom"]
            ),
    }

    # ------------------------------------------------------------------
    # Save comparison
    # ------------------------------------------------------------------

    (
        attempt_dir
        / "comparison.json"
    ).write_text(
        json.dumps(
            comparison,
            indent=4
        )
        + "\n"
    )

    (
        attempt_dir
        / "STATUS"
    ).write_text(
        "CANDIDATE_READY_FOR_USER_APPROVAL\n"
    )

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL UPDATE COMPLETE")
    print("=" * 70)

    print(
        json.dumps(
            comparison,
            indent=2
        )
    )

    print()

    print(
        f"Candidate model: "
        f"{candidate_model}"
    )

    print(
        f"Attempt directory: "
        f"{attempt_dir}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
