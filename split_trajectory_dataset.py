#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from ase.io import read, write


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        required=True
    )

    p.add_argument(
        "--output-dir",
        required=True
    )

    p.add_argument(
        "--train-percent",
        type=float,
        default=80
    )

    p.add_argument(
        "--valid-percent",
        type=float,
        default=10
    )

    p.add_argument(
        "--test-percent",
        type=float,
        default=10
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = p.parse_args()

    if abs(
        args.train_percent
        + args.valid_percent
        + args.test_percent
        - 100
    ) > 1e-8:

        raise SystemExit(
            "Train + validation + test percentages must equal 100."
        )

    frames = read(
        args.input,
        index=":"
    )

    if not frames:
        raise SystemExit(
            "No configurations found."
        )

    # ------------------------------------------------------------
    # Group configurations by complete BOMD trajectory.
    # ------------------------------------------------------------

    trajectories = defaultdict(list)

    for frame in frames:

        rep = frame.info.get("rep")

        if rep is None:
            raise SystemExit(
                "EXTXYZ frame has no 'rep' field. "
                "Cannot perform trajectory-level splitting."
            )

        trajectories[rep].append(frame)

    reps = sorted(trajectories)

    print(
        f"Configurations : {len(frames)}"
    )

    print(
        f"Trajectories   : {len(reps)}"
    )

    # ------------------------------------------------------------
    # Randomise TRAJECTORIES, not configurations.
    # ------------------------------------------------------------

    random.seed(args.seed)

    random.shuffle(reps)

    n = len(reps)

    n_train = round(
        n * args.train_percent / 100
    )

    n_valid = round(
        n * args.valid_percent / 100
    )

    n_test = n - n_train - n_valid

    train_reps = reps[:n_train]

    valid_reps = reps[
        n_train:n_train + n_valid
    ]

    test_reps = reps[
        n_train + n_valid:
    ]

    output = Path(args.output_dir)

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    def collect(selected):

        result = []

        for rep in selected:
            result.extend(
                trajectories[rep]
            )

        return result

    train = collect(train_reps)
    valid = collect(valid_reps)
    test = collect(test_reps)

    write(
        output / "train.extxyz",
        train,
        format="extxyz"
    )

    write(
        output / "valid.extxyz",
        valid,
        format="extxyz"
    )

    write(
        output / "test.extxyz",
        test,
        format="extxyz"
    )

    (output / "split.txt").write_text(
        "BOOTSTRAP DATASET SPLIT\n"
        "=======================\n\n"
        f"Total trajectories: {n}\n"
        f"Training trajectories: {len(train_reps)}\n"
        f"Validation trajectories: {len(valid_reps)}\n"
        f"Test trajectories: {len(test_reps)}\n\n"
        f"Training configurations: {len(train)}\n"
        f"Validation configurations: {len(valid)}\n"
        f"Test configurations: {len(test)}\n\n"
        "Training trajectories:\n"
        + "\n".join(train_reps)
        + "\n\nValidation trajectories:\n"
        + "\n".join(valid_reps)
        + "\n\nTest trajectories:\n"
        + "\n".join(test_reps)
        + "\n"
    )

    print(
        f"Training   : {len(train_reps)} trajectories, "
        f"{len(train)} configurations"
    )

    print(
        f"Validation : {len(valid_reps)} trajectories, "
        f"{len(valid)} configurations"
    )

    print(
        f"Test       : {len(test_reps)} trajectories, "
        f"{len(test)} configurations"
    )


if __name__ == "__main__":
    main()
