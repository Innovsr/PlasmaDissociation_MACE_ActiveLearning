#!/usr/bin/env python3

import argparse
from pathlib import Path

from ase.io import read


parser = argparse.ArgumentParser(
    description="Count configurations in an XYZ/EXTXYZ file."
)

parser.add_argument(
    "file",
    help="XYZ/EXTXYZ file to count"
)

args = parser.parse_args()

path = Path(args.file)


if not path.is_file():

    raise SystemExit(
        f"ERROR: File not found:\n{path}"
    )


try:

    configurations = read(
        str(path),
        index=":"
    )

except Exception as exc:

    raise SystemExit(
        f"ERROR: Could not read file:\n{exc}"
    )


print()
print("=" * 70)
print("CONFIGURATION COUNT")
print("=" * 70)

print(
    f"File           : {path}"
)

print(
    f"Configurations : {len(configurations)}"
)

print("=" * 70)
