#!/usr/bin/env python3

"""
Active-learning model update workflow.

Workflow
--------

1. Take the completed Stage-1 validation.extxyz.

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

10. Create:

       attempt_001/
       |
       +-- model/
       |
       +-- results/

The model directory contains the complete candidate model package.

The results directory contains all evaluation information.

No model is automatically promoted into models/.

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

The existing candidate checkpoint is NOT overwritten by the
source checkpoint on restart.

Bootstrap workflow is not modified by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
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
    """
    Load JSON configuration.
    """

    return json.loads(
        Path(path).read_text()
    )


def latest_model(root, requested):
    """
    Locate an approved source model.
    """

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
    """
    Return the registry row for a model if the registry exists.
    """

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


def find_source_checkpoint(model_dir):
    """
    Find the highest-numbered MACE .pt checkpoint.

    Example:

        bootstrap_x_epoch-49.pt
        bootstrap_x_epoch-50.pt

    returns epoch 50.
    """

    checkpoint_dir = (
        model_dir
        / "checkpoints"
    )

    if not checkpoint_dir.exists():

        raise SystemExit(
            f"Source model has no checkpoint directory:\n"
            f"{checkpoint_dir}"
        )

    checkpoint_files = sorted(
        checkpoint_dir.glob("*.pt")
    )

    if not checkpoint_files:

        raise SystemExit(
            f"No MACE checkpoint (*.pt) found in:\n"
            f"{checkpoint_dir}"
        )

    def checkpoint_epoch(path):

        match = re.search(
            r"epoch[-_]?(\d+)",
            path.name,
            flags=re.IGNORECASE,
        )

        return (
            int(match.group(1))
            if match
            else -1
        )

    numbered = [
        (
            checkpoint_epoch(path),
            path
        )
        for path in checkpoint_files
    ]

    numbered.sort(
        key=lambda item: (
            item[0],
            item[1].stat().st_mtime
        )
    )

    epoch, checkpoint = numbered[-1]

    if epoch < 0:

        raise SystemExit(
            "Could not determine epoch from checkpoint filename."
        )

    return (
        checkpoint,
        epoch
    )


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
# Package candidate model and results
# ----------------------------------------------------------------------

def package_candidate(
    attempt_dir,
    candidate_model,
    work,
    source_model_dir,
    source_model,
    source_checkpoint,
    source_epoch,
    training_path,
    validation_set,
    test_set,
    comparison,
    config_path,
):
    """
    Create the final candidate package.

    Structure:

        attempt_001/
        |
        +-- model/
        |   |
        |   +-- update_model.json
        |   +-- metadata.json
        |   +-- model.model
        |   +-- training_data.zip
        |   +-- validation_data.zip
        |   +-- test_data.zip
        |   |
        |   +-- checkpoints/
        |   |       +-- *.pt
        |   |
        |   +-- logs/
        |
        +-- results/
            |
            +-- comparison.json
            +-- candidate_summary.json
            +-- candidate_test_summary.json
            +-- candidate_validation.csv
            +-- candidate_test.csv
            +-- source_summary.json
            +-- source_validation.csv
            +-- training_pool_summary.json
            +-- training_pool_results.csv
            +-- high_error.csv

    Nothing is automatically copied into the official models/
    directory.

    The model directory is therefore the candidate package that
    can later be manually promoted.
    """

    # ==============================================================
    # DIRECTORIES
    # ==============================================================

    model_package = (
        attempt_dir
        / "model"
    )

    results_dir = (
        attempt_dir
        / "results"
    )

    logs_dir = (
        model_package
        / "logs"
    )

    checkpoints_dir = (
        model_package
        / "checkpoints"
    )

    # --------------------------------------------------------------
    # Remove an old package if it is being regenerated.
    # --------------------------------------------------------------

    if model_package.exists():

        shutil.rmtree(
            model_package
        )

    if results_dir.exists():

        shutil.rmtree(
            results_dir
        )

    model_package.mkdir(
        parents=True,
        exist_ok=False
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    logs_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==============================================================
    # 1. CANDIDATE MODEL
    # ==============================================================

    if not Path(candidate_model).exists():

        raise SystemExit(
            "Candidate model does not exist:\n"
            f"{candidate_model}"
        )

    shutil.copy2(
        candidate_model,
        model_package
        / "model.model"
    )

    # ==============================================================
    # 2. CANDIDATE CHECKPOINTS
    # ==============================================================

    source_checkpoint_dir = (
        work
        / "checkpoints"
    )

    if not source_checkpoint_dir.exists():

        raise SystemExit(
            "Candidate checkpoint directory does not exist:\n"
            f"{source_checkpoint_dir}"
        )

    checkpoint_files = list(
        source_checkpoint_dir.glob("*.pt")
    )

    if not checkpoint_files:

        raise SystemExit(
            "No candidate .pt checkpoints found:\n"
            f"{source_checkpoint_dir}"
        )

    shutil.copytree(
        source_checkpoint_dir,
        checkpoints_dir
    )

    # ==============================================================
    # 3. DATA ZIP HELPER
    # ==============================================================

    def create_dataset_zip(
        input_file,
        output_file,
        archive_name,
    ):

        input_file = Path(
            input_file
        )

        if not input_file.exists():

            raise SystemExit(
                "Cannot package dataset because it does not exist:\n"
                f"{input_file}"
            )

        with zipfile.ZipFile(
            output_file,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:

            archive.write(
                input_file,
                arcname=archive_name
            )

    # ==============================================================
    # 4. TRAINING DATA
    # ==============================================================

    create_dataset_zip(
        training_path,
        model_package
        / "training_data.zip",
        "training.extxyz",
    )

    # ==============================================================
    # 5. VALIDATION DATA
    # ==============================================================

    create_dataset_zip(
        validation_set,
        model_package
        / "validation_data.zip",
        "validation.extxyz",
    )

    # ==============================================================
    # 6. TEST DATA
    # ==============================================================

    create_dataset_zip(
        test_set,
        model_package
        / "test_data.zip",
        "test.extxyz",
    )

    # ==============================================================
    # 7. UPDATE_MODEL.JSON
    #
    # Copy the EXACT configuration used for this run.
    # ==============================================================

    config_source = Path(
        config_path
    )

    if not config_source.exists():

        raise SystemExit(
            "Could not find update_model.json used for this run:\n"
            f"{config_source}"
        )

    shutil.copy2(
        config_source,
        model_package
        / "update_model.json"
    )

    # ==============================================================
    # 8. METADATA.JSON
    # ==============================================================

    promotion_ok = (
        comparison[
            "validation_target_force_met"
        ]
        and
        comparison[
            "validation_target_energy_met"
        ]
        and
        comparison[
            "test_target_force_met"
        ]
        and
        comparison[
            "test_target_energy_met"
        ]
    )

    metadata = {

        "model_id":
            "MANUAL_RENAME_REQUIRED",

        "source_model":
            source_model,

        "source_checkpoint":
            source_checkpoint.name,

        "source_epoch":
            source_epoch,

        "candidate":
            comparison["candidate"],

        "workflow":
            "active_learning",

        "status":
            "READY_FOR_MANUAL_REVIEW",

        "promotion_criteria_met":
            promotion_ok,

        "files": {

            "model":
                "model.model",

            "training_data":
                "training_data.zip",

            "validation_data":
                "validation_data.zip",

            "test_data":
                "test_data.zip",

            "checkpoints":
                "checkpoints/",

            "logs":
                "logs/",

            "configuration":
                "update_model.json",

            "results":
                "../results/",
        },

        "performance": {

            "source_validation_force_rmse_mev_A":
                comparison[
                    "source_validation_force_rmse_mev_A"
                ],

            "source_validation_energy_rmse_mev_atom":
                comparison[
                    "source_validation_energy_rmse_mev_atom"
                ],

            "candidate_validation_force_rmse_mev_A":
                comparison[
                    "candidate_validation_force_rmse_mev_A"
                ],

            "candidate_validation_energy_rmse_mev_atom":
                comparison[
                    "candidate_validation_energy_rmse_mev_atom"
                ],

            "candidate_test_force_rmse_mev_A":
                comparison[
                    "candidate_test_force_rmse_mev_A"
                ],

            "candidate_test_energy_rmse_mev_atom":
                comparison[
                    "candidate_test_energy_rmse_mev_atom"
                ],
        },
    }

    (
        model_package
        / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=4
        )
        + "\n"
    )

    # ==============================================================
    # 9. RESULTS
    # ==============================================================

    result_files = [

        "comparison.json",

        "candidate_summary.json",

        "candidate_test_summary.json",

        "candidate_validation.csv",

        "candidate_test.csv",

        "source_summary.json",

        "source_validation.csv",

        "training_pool_summary.json",

        "training_pool_results.csv",

        "high_error.csv",
    ]

    for filename in result_files:

        source_file = (
            attempt_dir
            / filename
        )

        if source_file.exists():

            shutil.copy2(
                source_file,
                results_dir
                / filename
            )

    # ==============================================================
    # 10. LOGS
    #
    # Copy useful logs from mace_work and attempt directory.
    # ==============================================================

    if work.exists():

        for log_file in work.rglob("*"):

            if not log_file.is_file():

                continue

            if log_file.suffix.lower() not in {
                ".log",
                ".out",
                ".err",
            }:

                continue

            relative_path = (
                log_file.relative_to(work)
            )

            destination = (
                logs_dir
                / "mace_work"
                / relative_path
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.copy2(
                log_file,
                destination
            )

    # --------------------------------------------------------------
    # Copy attempt-level logs.
    # --------------------------------------------------------------

    for log_file in attempt_dir.iterdir():

        if not log_file.is_file():

            continue

        if log_file.suffix.lower() not in {
            ".log",
            ".out",
            ".err",
        }:

            continue

        if log_file.name in result_files:

            continue

        shutil.copy2(
            log_file,
            logs_dir
            / log_file.name
        )

    # ==============================================================
    # 11. SAVE PACKAGE INFORMATION
    # ==============================================================

    package_info = {

        "source_model":
            source_model,

        "source_checkpoint":
            source_checkpoint.name,

        "source_epoch":
            source_epoch,

        "candidate":
            comparison["candidate"],

        "status":
            "READY_FOR_MANUAL_REVIEW",

        "promotion_criteria_met":
            promotion_ok,

        "model_directory":
            str(model_package),

        "results_directory":
            str(results_dir),

        "model_file":
            "model.model",

        "configuration":
            "update_model.json",

        "training_data":
            "training_data.zip",

        "validation_data":
            "validation_data.zip",

        "test_data":
            "test_data.zip",

        "checkpoints":
            "checkpoints/",

        "logs":
            "logs/",
    }

    (
        model_package
        / "package_info.json"
    ).write_text(
        json.dumps(
            package_info,
            indent=4
        )
        + "\n"
    )

    # ==============================================================
    # 12. PRINT FINAL PACKAGE INFORMATION
    # ==============================================================

    print()
    print("=" * 70)
    print("MODEL PACKAGE CREATED")
    print("=" * 70)

    print(
        f"Model package : {model_package}"
    )

    print(
        f"Results       : {results_dir}"
    )

    print(
        f"Promotion criteria met: {promotion_ok}"
    )

    print()
    print("MODEL PACKAGE CONTENTS")
    print("-" * 70)

    print(
        "update_model.json"
    )

    print(
        "metadata.json"
    )

    print(
        "package_info.json"
    )

    print(
        "model.model"
    )

    print(
        "training_data.zip"
    )

    print(
        "validation_data.zip"
    )

    print(
        "test_data.zip"
    )

    print(
        "checkpoints/"
    )

    print(
        "logs/"
    )

    print()
    print("RESULT FILES")
    print("-" * 70)

    for filename in result_files:

        if (
            results_dir
            / filename
        ).exists():

            print(
                filename
            )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Nothing has been copied to the official models/"
        " directory."
    )

    print(
        "Manual promotion is required."
    )

    return model_package


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

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Find source checkpoint
    #
    # This determines the actual highest epoch checkpoint from the
    # approved source model.
    # ------------------------------------------------------------------

    source_checkpoint, source_epoch = (
        find_source_checkpoint(
            model_dir
        )
    )

    print("=" * 70)
    print("ACTIVE-LEARNING MODEL UPDATE")
    print("=" * 70)

    print(
        f"Source model      : {source}"
    )

    print(
        f"Source checkpoint : "
        f"{source_checkpoint.name}"
    )

    print(
        f"Source epoch      : "
        f"{source_epoch}"
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
        f"Validation dataset: "
        f"{validation_path}"
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

    # ------------------------------------------------------------------
    # Only stop automatically for a genuinely completed attempt.
    #
    # Do NOT stop merely because model/ or results/ exists.
    # ------------------------------------------------------------------

    if status_file.exists():

        status = (
            status_file
            .read_text()
            .strip()
        )

        if status in {
            "CANDIDATE_READY_FOR_USER_APPROVAL",
            "READY_FOR_MANUAL_REVIEW",
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
        f"source_checkpoint={source_checkpoint.name}\n"
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
    #
    # Split validation trajectories:
    #
    # Every 10:
    #
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

            str(
                split_script
            ),

            "--input",

            str(
                full_validation_copy
            ),

            "--output-dir",

            str(
                split_dir
            ),
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
    #
    # Evaluate source model on TRAINING POOL.
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

            str(
                training_pool
            ),

            "--output",

            str(
                training_pool_results
            ),

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
    #
    # Evaluate source model on independent validation set.
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

            str(
                validation_set
            ),

            "--output",

            str(
                source_validation_results
            ),

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
    # If source model already satisfies validation targets,
    # no model update is required.
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
    #
    # Select high-error configurations ONLY from training pool.
    #
    # IMPORTANT:
    # There is NO maximum number of configurations here.
    # Every configuration whose force error exceeds the threshold
    # is retained as candidate new data.
    # ------------------------------------------------------------------
    
    print()
    print("=" * 70)
    print("STEP 4: SELECT HIGH-ERROR TRAINING CONFIGURATIONS")
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
    
            str(
                training_pool_results
            ),
    
            "--output",
    
            str(
                high_error_csv
            ),
    
            "--force-threshold",
    
            str(
                config["validation"]
                ["high_error_force_threshold_mev_A"]
            ),
        ]
    )

#    # ------------------------------------------------------------------
#    # STEP 4
#    #
#    # Select high-error configurations ONLY from training pool.
#    # ------------------------------------------------------------------
#
#    print()
#    print("=" * 70)
#    print(
#        "STEP 4: SELECT HIGH-ERROR "
#        "TRAINING CONFIGURATIONS"
#    )
#    print("=" * 70)
#
#    high_error_csv = (
#        attempt_dir
#        / "high_error.csv"
#    )
#
#    run_command(
#        [
#            sys.executable,
#
#            str(
#                root
#                / "select_high_error.py"
#            ),
#
#            "--results",
#
#            str(
#                training_pool_results
#            ),
#
#            "--output",
#
#            str(
#                high_error_csv
#            ),
#
#            "--max-configurations",
#
#            str(
#                config["validation"]
#                ["max_new_configurations"]
#            ),
#
#            "--force-threshold",
#
#            str(
#                config["validation"]
#                ["high_error_force_threshold_mev_A"]
#            ),
#        ]
#    )

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
        int(
            row["configuration"]
        )
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
    #
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
    #
    # Build updated training set.
    #
    # IMPORTANT:
    #
    # There is NO total_configurations parameter.
    #
    # old_data_percent controls the percentage of the OLD training
    # dataset that is selected.
    #
    # new_data_percent controls the percentage of the NEW
    # high-error dataset that is selected.
    #
    # Examples:
    #
    # old_data_percent = 80
    # new_data_percent = 100
    #
    # means:
    #
    #     80% of old training data
    #     100% of available high-error new data
    #
    # The final number of configurations is therefore determined
    # automatically by the sizes of the two datasets.
    # ------------------------------------------------------------------
    
    print()
    print("=" * 70)
    print("STEP 6: BUILD UPDATED TRAINING DATASET")
    print("=" * 70)
    
    training_path = (
        attempt_dir
        / "training.extxyz"
    )
    
    old_percent = float(
        config["training_update"]
        ["old_data_percent"]
    )
    
    new_percent = float(
        config["training_update"]
        ["new_data_percent"]
    )
    
    # --------------------------------------------------------------
    # Validate percentages
    # --------------------------------------------------------------
    
    if not 0.0 <= old_percent <= 100.0:
        raise SystemExit(
            "ERROR: old_data_percent must be between 0 and 100."
        )
    
    if not 0.0 <= new_percent <= 100.0:
        raise SystemExit(
            "ERROR: new_data_percent must be between 0 and 100."
        )
    
    print()
    print(
        f"Old training data percentage : "
        f"{old_percent:.2f}%"
    )
    
    print(
        f"New training data percentage : "
        f"{new_percent:.2f}%"
    )
    
    print()
    
    run_command(
        [
            sys.executable,
    
            str(
                root
                / "build_training_set.py"
            ),
    
            "--old",
    
            str(
                old_training_extxyz
            ),
    
            "--new",
    
            str(
                new_problematic
            ),
    
            "--output",
    
            str(
                training_path
            ),
    
            "--old-percent",
    
            str(
                old_percent
            ),
    
            "--new-percent",
    
            str(
                new_percent
            ),
    
            "--seed",
    
            str(
                config["mace"]
                ["seed"]
            ),
        ]
    )
    
    print()
    
    print(
        f"Final training dataset: "
        f"{training_path}"
    )

#    # ------------------------------------------------------------------
#    # STEP 6
#    #
#    # Build updated training set.
#    # ------------------------------------------------------------------
#
#    print()
#    print("=" * 70)
#    print("STEP 6: BUILD UPDATED TRAINING DATASET")
#    print("=" * 70)
#
#    total_configurations = (
#        config["training_update"]
#        .get(
#            "total_configurations"
#        )
#    )
#
#    if total_configurations is None:
#
#        total_configurations = len(
#            read(
#                str(
#                    old_training_extxyz
#                ),
#                index=":"
#            )
#        )
#
#    training_path = (
#        attempt_dir
#        / "training.extxyz"
#    )
#
#    run_command(
#        [
#            sys.executable,
#
#            str(
#                root
#                / "build_training_set.py"
#            ),
#
#            "--old",
#
#            str(
#                old_training_extxyz
#            ),
#
#            "--new",
#
#            str(
#                new_problematic
#            ),
#
#            "--output",
#
#            str(
#                training_path
#            ),
#
#            "--old-percent",
#
#            str(
#                config["training_update"]
#                ["old_data_percent"]
#            ),
#
#            "--new-percent",
#
#            str(
#                config["training_update"]
#                ["new_data_percent"]
#            ),
#
#            "--total",
#
#            str(
#                total_configurations
#            ),
#
#            "--seed",
#
#            str(
#                config["mace"]
#                ["seed"]
#            ),
#        ]
#    )

    # ------------------------------------------------------------------
    # STEP 7
    #
    # Prepare MACE work directory.
    #
    # IMPORTANT:
    #
    # The candidate model starts from the existing source-model
    # checkpoint, NOT from a newly initialised model.
    #
    # Source:
    #
    #     models/<source>/checkpoints/*.pt
    #
    # copied to:
    #
    #     02_model_update/<attempt>/mace_work/checkpoints/
    #
    # train_mace.py receives --restart.
    #
    # train_mace.py then adds --restart_latest to MACE.
    #
    # If this script is restarted after timeout, the existing
    # candidate checkpoints are preserved.
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
    # Copy source checkpoint directory.
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
    # Candidate checkpoint directory already exists.
    #
    # DO NOT overwrite it with source checkpoint again.
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
        key=lambda path:
            path.stat().st_mtime
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
    #
    # Train candidate.
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
        f"Candidate name : "
        f"{candidate}"
    )

    print(
        f"Training data  : "
        f"{training_path}"
    )

    print(
        f"Validation data: "
        f"{validation_set}"
    )

    print(
        f"Work directory : "
        f"{work}"
    )

    print(
        "Restart mode   : YES"
    )

    print(
        f"Checkpoint     : "
        f"{latest_checkpoint.name}"
    )

    run_command(
        [
            sys.executable,

            str(
                root
                / "train_mace.py"
            ),

            "--train",

            str(
                training_path
            ),

            "--valid",

            str(
                validation_set
            ),

            "--name",

            candidate,

            "--work-dir",

            str(
                work
            ),

            "--device",

            config["mace"]["device"],

            "--restart",
        ]
    )

    # ------------------------------------------------------------------
    # STEP 9
    #
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
    #
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

            str(
                candidate_model
            ),

            "--validation",

            str(
                validation_set
            ),

            "--output",

            str(
                candidate_validation_results
            ),

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
    #
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

            str(
                candidate_model
            ),

            "--validation",

            str(
                test_set
            ),

            "--output",

            str(
                candidate_test_results
            ),

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
    #
    # Compare source and candidate.
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

        "source_checkpoint":
            source_checkpoint.name,

        "source_epoch":
            source_epoch,

        "candidate":
            candidate,

        # ----------------------------------------------------------
        # Source model on independent validation set
        # ----------------------------------------------------------

        "source_validation_force_rmse_mev_A":
            source_force,

        "source_validation_energy_rmse_mev_atom":
            source_energy,

        # ----------------------------------------------------------
        # Candidate on independent validation set
        # ----------------------------------------------------------

        "candidate_validation_force_rmse_mev_A":
            candidate_force,

        "candidate_validation_energy_rmse_mev_atom":
            candidate_energy,

        # ----------------------------------------------------------
        # Candidate on untouched test set
        # ----------------------------------------------------------

        "candidate_test_force_rmse_mev_A":
            test_force,

        "candidate_test_energy_rmse_mev_atom":
            test_energy,

        # ----------------------------------------------------------
        # Validation improvement
        # ----------------------------------------------------------

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

        # ----------------------------------------------------------
        # Validation targets
        # ----------------------------------------------------------

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

        # ----------------------------------------------------------
        # Test targets
        # ----------------------------------------------------------

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
    # Save comparison at attempt level first.
    #
    # package_candidate() will copy it into results/.
    # ------------------------------------------------------------------

    comparison_path = (
        attempt_dir
        / "comparison.json"
    )

    comparison_path.write_text(
        json.dumps(
            comparison,
            indent=4
        )
        + "\n"
    )

    # ------------------------------------------------------------------
    # STEP 13
    #
    # Create final model and results packages.
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 13: PACKAGE MODEL AND RESULTS")
    print("=" * 70)

    model_package = package_candidate(
        attempt_dir=attempt_dir,

        candidate_model=candidate_model,

        work=work,

        source_model_dir=model_dir,

        source_model=source,

        source_checkpoint=source_checkpoint,

        source_epoch=source_epoch,

        training_path=training_path,

        validation_set=validation_set,

        test_set=test_set,

        comparison=comparison,

        config_path=args.config,
    )

    # ------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------

    (
        attempt_dir
        / "STATUS"
    ).write_text(
        "READY_FOR_MANUAL_REVIEW\n"
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
        "============================================================"
    )

    print(
        "CANDIDATE MODEL PACKAGE"
    )

    print(
        "============================================================"
    )

    print(
        f"{model_package}"
    )

    print()

    print(
        "RESULTS"
    )

    print(
        f"{attempt_dir / 'results'}"
    )

    print()

    print(
        "REVIEW COMPARISON"
    )

    print(
        f"{attempt_dir / 'results' / 'comparison.json'}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The candidate has NOT been automatically promoted."
    )

    print(
        "Manual review is required."
    )

    print()

    print(
        "If approved manually, copy the model directory to"
    )

    print(
        "the next approved model directory, for example:"
    )

    print()

    print(
        f"models/CF3-CHF-CHF2_ea_002/"
    )

    print()

    print(
        "Do NOT overwrite the existing approved model."
    )

    print()

    return 0


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":

    raise SystemExit(
        main()
    )
