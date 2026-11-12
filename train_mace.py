#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_config():
    with open(ROOT / "configs" / "update_model.json") as f:
        return json.load(f)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        required=True
    )

    parser.add_argument(
        "--valid",
        required=True
    )

    parser.add_argument(
        "--name",
        required=True,
        help="MACE experiment name. Must remain unchanged during restart."
    )

    parser.add_argument(
        "--work-dir",
        required=True
    )

    parser.add_argument(
        "--device",
        default="cpu"
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="Resume from the latest checkpoint in work-dir."
    )

    args = parser.parse_args()

    config = load_config()
    mace = config["mace"]

    work = Path(args.work_dir)

    work.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoints = work / "checkpoints"

    # ------------------------------------------------------------
    # Check restart state
    # ------------------------------------------------------------

    restart = args.restart

    if restart:

        if not checkpoints.exists():

            raise SystemExit(
                "Restart requested, but checkpoint directory does "
                f"not exist:\n{checkpoints}"
            )

        checkpoint_files = list(
            checkpoints.glob("*.pt")
        )

        if not checkpoint_files:

            raise SystemExit(
                "Restart requested, but no .pt checkpoint was found "
                f"in:\n{checkpoints}"
            )

        print("=" * 70)
        print("MACE RESTART")
        print("=" * 70)
        print(f"Work directory : {work}")
        print(f"Checkpoint dir : {checkpoints}")
        print("Restart mode   : --restart_latest")
        print("=" * 70)

    else:

        print("=" * 70)
        print("MACE NEW TRAINING")
        print("=" * 70)
        print(f"Work directory : {work}")
        print("Restart mode   : NO")
        print("=" * 70)

    # ------------------------------------------------------------
    # Build MACE command
    # ------------------------------------------------------------

    command = [

        mace["command"],

        f"--name={args.name}",

        f"--work_dir={work}",

        f"--train_file={args.train}",

        f"--valid_file={args.valid}",

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

        f"--device={args.device}",

        f"--default_dtype={mace['default_dtype']}",

        f"--seed={mace['seed']}",
    ]

    # ------------------------------------------------------------
    # Only add restart_latest when a checkpoint really exists.
    # ------------------------------------------------------------

    if restart:

        command.append(
            "--restart_latest"
        )

    print()
    print("MACE COMMAND")
    print("-" * 70)
    print(
        " ".join(str(x) for x in command)
    )
    print("-" * 70)
    print()

    subprocess.run(
        command,
        check=True
    )


if __name__ == "__main__":
    main()
