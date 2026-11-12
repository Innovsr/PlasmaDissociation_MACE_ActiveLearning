#!/usr/bin/env python3

import argparse
import random

from ase.io import read, write


p = argparse.ArgumentParser()

p.add_argument(
    "--old",
    required=True
)

p.add_argument(
    "--new",
    required=True
)

p.add_argument(
    "--output",
    required=True
)

p.add_argument(
    "--old-percent",
    type=float,
    required=True
)

p.add_argument(
    "--new-percent",
    type=float,
    required=True
)

p.add_argument(
    "--total",
    type=int,
    required=True
)

p.add_argument(
    "--seed",
    type=int,
    default=42
)


a = p.parse_args()


# ============================================================
# CHECK PERCENTAGES
# ============================================================

if abs(
    a.old_percent
    + a.new_percent
    - 100
) > 1e-9:

    raise SystemExit(
        "old+new must equal 100"
    )


# ============================================================
# READ INPUT DATA
# ============================================================

old = read(
    a.old,
    index=":"
)

new = read(
    a.new,
    index=":"
)


# ============================================================
# REQUESTED NUMBERS
# ============================================================

requested_old = round(
    a.total
    * a.old_percent
    / 100
)

requested_new = (
    a.total
    - requested_old
)


# ============================================================
# ACTUAL NUMBERS
#
# If fewer configurations are available than requested,
# use ALL available configurations instead of failing.
# ============================================================

actual_old = min(
    requested_old,
    len(old)
)

actual_new = min(
    requested_new,
    len(new)
)


# ============================================================
# WARN IF DATA IS LIMITED
# ============================================================

data_limited = False


if len(old) < requested_old:

    data_limited = True

    print()
    print("=" * 70)
    print("WARNING: NOT ENOUGH OLD TRAINING DATA")
    print("=" * 70)

    print(
        f"Requested old configurations : {requested_old}"
    )

    print(
        f"Available old configurations : {len(old)}"
    )

    print(
        "Using ALL available old configurations."
    )

    print("=" * 70)


if len(new) < requested_new:

    data_limited = True

    print()
    print("=" * 70)
    print("WARNING: NOT ENOUGH NEW CONFIGURATIONS")
    print("=" * 70)

    print(
        f"Requested new configurations : {requested_new}"
    )

    print(
        f"Available new configurations : {len(new)}"
    )

    print(
        "Using ALL available new configurations."
    )

    print("=" * 70)


# ============================================================
# RANDOM SELECTION
# ============================================================

random.seed(
    a.seed
)


selected_old = random.sample(
    old,
    actual_old
)

selected_new = random.sample(
    new,
    actual_new
)


# ============================================================
# COMBINE
# ============================================================

out = (
    selected_old
    + selected_new
)


random.shuffle(
    out
)


# ============================================================
# WRITE OUTPUT
# ============================================================

write(
    a.output,
    out,
    format="extxyz"
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 70)
print("TRAINING DATASET CREATED")
print("=" * 70)

print(
    f"Output file              : {a.output}"
)

print(
    f"Requested total          : {a.total}"
)

print(
    f"Actual total             : {len(out)}"
)

print(
    f"Requested old            : {requested_old}"
)

print(
    f"Actual old               : {actual_old}"
)

print(
    f"Requested new            : {requested_new}"
)

print(
    f"Actual new               : {actual_new}"
)


if len(out) > 0:

    print(
        f"Actual old percentage   : "
        f"{actual_old / len(out) * 100:.2f}%"
    )

    print(
        f"Actual new percentage   : "
        f"{actual_new / len(out) * 100:.2f}%"
    )


if data_limited:

    print()
    print(
        "NOTE: The requested total/ratio could not be "
        "reached because one or both input datasets "
        "contained fewer configurations than requested."
    )

    print(
        "All available configurations from the limited "
        "dataset were included."
    )


print("=" * 70)
