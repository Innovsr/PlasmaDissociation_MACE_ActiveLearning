#!/usr/bin/env python3

"""
Split validation.extxyz by complete trajectory.

For every block of 10 trajectories:

    7 trajectories -> training pool
    2 trajectories -> validation
    1 trajectory   -> test

No trajectory is split between datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ase.io import read, write


def trajectory_number(rep_name):
    try:
        return int(
            str(rep_name).split("-")[-1]
        )
    except Exception:
        return 10**9


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise SystemExit(
            f"Input file not found: {input_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("VALIDATION TRAJECTORY SPLIT")
    print("=" * 70)

    frames = read(
        str(input_path),
        index=":"
    )

    if not frames:
        raise SystemExit(
            "No configurations found."
        )

    # Identify complete trajectories.
    rep_names = sorted(
        {
            str(frame.info["rep"])
            for frame in frames
            if "rep" in frame.info
        },
        key=trajectory_number
    )

    if not rep_names:
        raise SystemExit(
            "No 'rep' information found in validation.extxyz."
        )

    print(
        f"Trajectories found: {len(rep_names)}"
    )

    print(
        f"Configurations found: {len(frames)}"
    )

    if len(rep_names) % 10 != 0:

        raise SystemExit(
            f"Found {len(rep_names)} trajectories. "
            "The number must be a multiple of 10."
        )

    training_reps = []
    validation_reps = []
    test_reps = []

    # --------------------------------------------------------------
    # 7 / 2 / 1 split for EVERY block of 10.
    # --------------------------------------------------------------

    for start in range(
        0,
        len(rep_names),
        10
    ):

        block = rep_names[
            start:start + 10
        ]

        training_reps.extend(
            block[0:7]
        )

        validation_reps.extend(
            block[7:9]
        )

        test_reps.extend(
            block[9:10]
        )

    print()
    print("TRAINING TRAJECTORIES:")
    print(training_reps)

    print()
    print("VALIDATION TRAJECTORIES:")
    print(validation_reps)

    print()
    print("TEST TRAJECTORIES:")
    print(test_reps)

    # --------------------------------------------------------------
    # Save trajectory lists.
    # --------------------------------------------------------------

    (
        output_dir
        / "training_trajectories.txt"
    ).write_text(
        "\n".join(training_reps)
        + "\n"
    )

    (
        output_dir
        / "validation_trajectories.txt"
    ).write_text(
        "\n".join(validation_reps)
        + "\n"
    )

    (
        output_dir
        / "test_trajectories.txt"
    ).write_text(
        "\n".join(test_reps)
        + "\n"
    )

    # --------------------------------------------------------------
    # Select complete trajectories.
    # --------------------------------------------------------------

    training_set = set(
        training_reps
    )

    validation_set = set(
        validation_reps
    )

    test_set = set(
        test_reps
    )

    training_frames = [
        frame
        for frame in frames
        if str(frame.info.get("rep"))
        in training_set
    ]

    validation_frames = [
        frame
        for frame in frames
        if str(frame.info.get("rep"))
        in validation_set
    ]

    test_frames = [
        frame
        for frame in frames
        if str(frame.info.get("rep"))
        in test_set
    ]

    if not training_frames:
        raise SystemExit(
            "Training pool is empty."
        )

    if not validation_frames:
        raise SystemExit(
            "Validation set is empty."
        )

    if not test_frames:
        raise SystemExit(
            "Test set is empty."
        )

    # --------------------------------------------------------------
    # Write datasets.
    # --------------------------------------------------------------

    training_path = (
        output_dir
        / "training_pool.extxyz"
    )

    validation_path = (
        output_dir
        / "validation_set.extxyz"
    )

    test_path = (
        output_dir
        / "test_set.extxyz"
    )

    write(
        str(training_path),
        training_frames,
        format="extxyz"
    )

    write(
        str(validation_path),
        validation_frames,
        format="extxyz"
    )

    write(
        str(test_path),
        test_frames,
        format="extxyz"
    )

    # --------------------------------------------------------------
    # Final report.
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("SPLIT COMPLETE")
    print("=" * 70)

    print(
        f"Training:   {len(training_reps)} trajectories, "
        f"{len(training_frames)} configurations"
    )

    print(
        f"Validation: {len(validation_reps)} trajectories, "
        f"{len(validation_frames)} configurations"
    )

    print(
        f"Test:       {len(test_reps)} trajectories, "
        f"{len(test_frames)} configurations"
    )

    print()
    print(
        f"Training:   {training_path}"
    )

    print(
        f"Validation: {validation_path}"
    )

    print(
        f"Test:       {test_path}"
    )


if __name__ == "__main__":
    main()
