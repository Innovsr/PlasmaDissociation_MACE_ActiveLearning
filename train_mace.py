#!/usr/bin/env python3

import argparse
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent


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
    "--test",
    default=""
)

parser.add_argument(
    "--name",
    required=True
)

parser.add_argument(
    "--work-dir",
    required=True
)

parser.add_argument(
    "--restart-from",
    default=""
)

parser.add_argument(
    "--device",
    default="cpu"
)

args = parser.parse_args()


config = json.load(
    open(
        ROOT / "configs" / "update_model.json"
    )
)

mace = config["mace"]

work = Path(
    args.work_dir
)

work.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Restore checkpoints when requested.
# ------------------------------------------------------------

if args.restart_from:

    source = (
        Path(args.restart_from)
        / "checkpoints"
    )

    destination = (
        work / "checkpoints"
    )

    if (
        source.exists()
        and source.resolve()
        != destination.resolve()
    ):

        if destination.exists():
            shutil.rmtree(destination)

        shutil.copytree(
            source,
            destination
        )


# ------------------------------------------------------------
# MACE command.
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
# IMPORTANT:
#
# The test set is NOT passed to MACE during training.
#
# It will be evaluated separately after training by
# model_update.py.
# ------------------------------------------------------------

if args.restart_from:

    command.append(
        "--restart_latest"
    )


print("=" * 70)
print("MACE TRAINING")
print("=" * 70)

print(
    "Training dataset:",
    args.train
)

print(
    "Validation dataset:",
    args.valid
)

print(
    "Test dataset:",
    "NOT USED DURING TRAINING"
)

print("=" * 70)

print()
print(
    "Running:",
    " ".join(str(x) for x in command)
)

print()

subprocess.run(
    command,
    check=True
)
