#!/usr/bin/env python3

import argparse
import random
from ase.io import read, write


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Build a MACE training dataset by selecting a percentage "
        "of configurations from the old and new datasets."
    )
)

parser.add_argument(
    "--old",
    required=True,
    help="Old training dataset (.extxyz)"
)

parser.add_argument(
    "--new",
    required=True,
    help="New/problematic dataset (.extxyz)"
)

parser.add_argument(
    "--output",
    required=True,
    help="Output training dataset (.extxyz)"
)

parser.add_argument(
    "--old-percent",
    type=float,
    required=True,
    help="Percentage of old configurations to include"
)

parser.add_argument(
    "--new-percent",
    type=float,
    required=True,
    help="Percentage of new configurations to include"
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed"
)


args = parser.parse_args()


# ============================================================
# CHECK PERCENTAGES
# ============================================================

if not 0 <= args.old_percent <= 100:
    raise SystemExit(
        "ERROR: --old-percent must be between 0 and 100."
    )

if not 0 <= args.new_percent <= 100:
    raise SystemExit(
        "ERROR: --new-percent must be between 0 and 100."
    )


# ============================================================
# READ DATASETS
# ============================================================

print()
print("=" * 70)
print("READING TRAINING DATA")
print("=" * 70)

print(f"Old dataset : {args.old}")
print(f"New dataset : {args.new}")

old = read(args.old, index=":")
new = read(args.new, index=":")

old_total = len(old)
new_total = len(new)

print()
print(f"Old configurations available : {old_total}")
print(f"New configurations available : {new_total}")


# ============================================================
# CALCULATE NUMBER TO SELECT
# ============================================================

old_number = round(
    old_total * args.old_percent / 100.0
)

new_number = round(
    new_total * args.new_percent / 100.0
)


# ============================================================
# REPORT SELECTION
# ============================================================

print()
print("=" * 70)
print("DATA SELECTION")
print("=" * 70)

print(
    f"Old data percentage requested : "
    f"{args.old_percent:.2f}%"
)

print(
    f"New data percentage requested : "
    f"{args.new_percent:.2f}%"
)

print()

print(
    f"Old configurations selected  : "
    f"{old_number} / {old_total}"
)

print(
    f"New configurations selected  : "
    f"{new_number} / {new_total}"
)


# ============================================================
# RANDOM SELECTION
# ============================================================

random.seed(args.seed)

selected_old = random.sample(
    old,
    old_number
)

selected_new = random.sample(
    new,
    new_number
)


# ============================================================
# COMBINE DATASETS
# ============================================================

training_data = (
    selected_old +
    selected_new
)

random.shuffle(training_data)


# ============================================================
# WRITE OUTPUT
# ============================================================

write(
    args.output,
    training_data,
    format="extxyz"
)


# ============================================================
# FINAL REPORT
# ============================================================

actual_total = len(training_data)

print()
print("=" * 70)
print("TRAINING DATASET CREATED")
print("=" * 70)

print(
    f"Output file                 : {args.output}"
)

print(
    f"Old configurations available: {old_total}"
)

print(
    f"Old configurations selected : {old_number}"
)

print(
    f"Old percentage used          : "
    f"{args.old_percent:.2f}%"
)

print()

print(
    f"New configurations available: {new_total}"
)

print(
    f"New configurations selected : {new_number}"
)

print(
    f"New percentage used          : "
    f"{args.new_percent:.2f}%"
)

print()

print(
    f"TOTAL configurations created : "
    f"{actual_total}"
)

if actual_total > 0:

    print(
        f"Old fraction of final set   : "
        f"{old_number / actual_total * 100:.2f}%"
    )

    print(
        f"New fraction of final set   : "
        f"{new_number / actual_total * 100:.2f}%"
    )

print("=" * 70)
