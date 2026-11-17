#!/usr/bin/env python3

"""
Independent analysis program for:

    PlasmaDissociation_MACE_ActiveLearning

This program analyses:

    00_bootstrap
    01_validation
    03_production

independently.

It does NOT modify the simulation data.

It can:

    1. Analyse selected datasets.
    2. Count trajectories.
    3. Check completed/incomplete trajectories.
    4. Detect first bond-breaking events from geometry.
    5. Use dissociation flags when available.
    6. Analyse first-break timing.
    7. Analyse final timesteps.
    8. Analyse SCF energy when available.
    9. Produce CSV summaries.
   10. Produce PNG graphs.
   11. Compare two datasets qualitatively.

Usage:

    python analysis.py --config analysis_config.json
"""

from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, median, stdev
import argparse
import csv
import json
import math
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# COVALENT RADII
# ============================================================

COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "F": 0.57,
    "O": 0.66,
    "N": 0.71,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}


DEFAULT_BOND_FACTOR = 1.35


# ============================================================
# GENERAL UTILITIES
# ============================================================

def safe_name(value):

    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value)
    )


def load_json(path):

    with open(path, "r") as f:
        return json.load(f)


def first_number(text):

    match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        str(text)
    )

    if match:
        return float(match.group())

    return None


def pair_label(a, b):

    return "-".join(sorted([a, b]))


def write_csv(path, rows):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not rows:
        return

    keys = []

    for row in rows:

        for key in row:

            if key not in keys:
                keys.append(key)

    with open(path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=keys,
            extrasaction="ignore"
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# XYZ PARSER
# ============================================================

def parse_xyz_all(path):

    """
    Reads the output format produced by output.py:

        Timestep: 0

        1 C x y z
        2 F x y z
        ...

        ----------------------------------------
    """

    frames = []

    current = None

    if not path.exists():
        return frames

    with open(path, "r", errors="replace") as f:

        for raw in f:

            line = raw.strip()

            if line.startswith("Timestep:"):

                if (
                    current is not None
                    and current["symbols"]
                ):

                    frames.append(current)

                current = {

                    "timestep":
                        first_number(
                            line.split(":", 1)[1]
                        ),

                    "symbols": [],

                    "coords": []
                }

                continue

            if current is None:
                continue

            if not line:
                continue

            if line.startswith("-"):
                continue

            match = re.match(

                r"^\s*\d+\s+"
                r"([A-Za-z][A-Za-z]?)\s+"
                r"([-+0-9.eE]+)\s+"
                r"([-+0-9.eE]+)\s+"
                r"([-+0-9.eE]+)",

                line
            )

            if match:

                symbol = match.group(1)

                x = float(match.group(2))
                y = float(match.group(3))
                z = float(match.group(4))

                current["symbols"].append(symbol)

                current["coords"].append(
                    [x, y, z]
                )

    if (
        current is not None
        and current["symbols"]
    ):

        frames.append(current)

    for frame in frames:

        frame["coords"] = np.asarray(
            frame["coords"],
            dtype=float
        )

    return frames


# ============================================================
# FORCES PARSER
# ============================================================

def parse_forces_all(path):

    """
    Reads:

        Timestep:
        SCF Energy:
        Dissociation Flags:
    """

    records = {}

    current = None

    if not path.exists():
        return records

    with open(path, "r", errors="replace") as f:

        for raw in f:

            line = raw.strip()

            if line.startswith("Timestep:"):

                timestep = first_number(
                    line.split(":", 1)[1]
                )

                current = {

                    "timestep": timestep,

                    "scf_energy": None,

                    "flags": None
                }

                if timestep is not None:

                    records[timestep] = current

                continue

            if current is None:
                continue

            if line.startswith("SCF Energy:"):

                value = line.split(
                    ":",
                    1
                )[1]

                numbers = re.findall(

                    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
                    r"(?:[eE][-+]?\d+)?",

                    value
                )

                if numbers:

                    current["scf_energy"] = [
                        float(x)
                        for x in numbers
                    ]

            elif line.startswith(
                "Dissociation Flags:"
            ):

                value = line.split(
                    ":",
                    1
                )[1]

                flags = re.findall(
                    r"['\"]([^'\"]+)['\"]",
                    value
                )

                if not flags:

                    flags = (
                        value
                        .strip("[]")
                        .replace(",", " ")
                        .split()
                    )

                current["flags"] = flags

    return records


# ============================================================
# TIME PARSER
# ============================================================

def parse_time_all(path):

    records = {}

    current_timestep = None

    if not path.exists():
        return records

    with open(path, "r", errors="replace") as f:

        for raw in f:

            line = raw.strip()

            if line.startswith("Timestep:"):

                current_timestep = first_number(
                    line.split(":", 1)[1]
                )

            elif line.startswith(
                "Time of this step:"
            ):

                if current_timestep is not None:

                    value = first_number(
                        line.split(":", 1)[1]
                    )

                    if value is not None:

                        records[current_timestep] = value

    return records


# ============================================================
# BOND CALCULATION
# ============================================================

def initial_bonds(
    symbols,
    coordinates,
    factor
):

    bonds = []

    for i in range(len(symbols)):

        ri = COVALENT_RADII.get(
            symbols[i],
            0.75
        )

        for j in range(i + 1, len(symbols)):

            rj = COVALENT_RADII.get(
                symbols[j],
                0.75
            )

            cutoff = factor * (ri + rj)

            distance = np.linalg.norm(
                coordinates[i]
                - coordinates[j]
            )

            if distance <= cutoff:

                bonds.append(
                    (i, j)
                )

    return bonds


def bond_distance(
    coordinates,
    i,
    j
):

    return float(
        np.linalg.norm(
            coordinates[i]
            - coordinates[j]
        )
    )


# ============================================================
# GEOMETRIC BOND BREAKING
# ============================================================

def geometry_break_events(
    frames,
    factor
):

    if not frames:
        return []

    symbols = frames[0]["symbols"]

    bonds = initial_bonds(
        symbols,
        frames[0]["coords"],
        factor
    )

    events = []

    already_broken = set()

    for frame in frames[1:]:

        timestep = frame["timestep"]

        for i, j in bonds:

            if (i, j) in already_broken:
                continue

            cutoff = factor * (

                COVALENT_RADII.get(
                    symbols[i],
                    0.75
                )

                +

                COVALENT_RADII.get(
                    symbols[j],
                    0.75
                )
            )

            distance = bond_distance(
                frame["coords"],
                i,
                j
            )

            if distance > cutoff:

                already_broken.add(
                    (i, j)
                )

                events.append({

                    "timestep":
                        timestep,

                    "atom_i":
                        i + 1,

                    "atom_j":
                        j + 1,

                    "bond":
                        pair_label(
                            symbols[i],
                            symbols[j]
                        ),

                    "distance":
                        distance,

                    "source":
                        "geometry"
                })

    return events


# ============================================================
# DISSOCIATION FLAG EVENTS
# ============================================================

def flag_break_events(
    frames,
    force_records
):

    events = []

    previous_flags = None

    for frame in frames:

        timestep = frame["timestep"]

        record = force_records.get(
            timestep
        )

        if not record:
            continue

        flags = record.get(
            "flags"
        )

        if not flags:
            continue

        if previous_flags is not None:

            for index, (
                old,
                new
            ) in enumerate(
                zip(
                    previous_flags,
                    flags
                )
            ):

                if (
                    old == "NO"
                    and new != "NO"
                ):

                    events.append({

                        "timestep":
                            timestep,

                        "atom_i":
                            index + 1,

                        "atom_j":
                            "",

                        "bond":
                            "FLAGGED_DISSOCIATION",

                        "distance":
                            "",

                        "source":
                            "dissociation_flags"
                    })

        previous_flags = list(flags)

    return events


# ============================================================
# FIRST BREAK
# ============================================================

def determine_first_break(
    frames,
    force_records,
    factor
):

    geometry_events = geometry_break_events(
        frames,
        factor
    )

    flag_events = flag_break_events(
        frames,
        force_records
    )

    if geometry_events:

        first = min(
            geometry_events,
            key=lambda x: x["timestep"]
        )

        return (
            first,
            geometry_events,
            flag_events
        )

    if flag_events:

        first = min(
            flag_events,
            key=lambda x: x["timestep"]
        )

        return (
            first,
            geometry_events,
            flag_events
        )

    return (
        None,
        geometry_events,
        flag_events
    )


# ============================================================
# TRAJECTORY DISCOVERY
# ============================================================

def discover_trajectories(root):

    trajectories = []

    for xyz in root.rglob(
        "output/xyz.all"
    ):

        trajectories.append(
            xyz.parent.parent
        )

    return sorted(
        set(trajectories)
    )


# ============================================================
# METADATA
# ============================================================

def get_metadata(
    root,
    user_metadata
):

    metadata = dict(
        user_metadata
    )

    possible_files = [

        root / "metadata.json",

        root / "bootstrap_config.json",

        root / "validation.json",

        root / "inputs.json",

        root.parent / "metadata.json",

        root.parent / "bootstrap_config.json",

        root.parent / "validation.json",

        root.parent / "inputs.json"
    ]

    for path in possible_files:

        if not path.exists():
            continue

        try:

            data = load_json(path)

        except Exception:

            continue

        for key, value in data.items():

            if key not in metadata:

                metadata[key] = value

    molecule = (

        metadata.get("molecule")

        or metadata.get("Molecule")

        or metadata.get(
            "Molecule_data",
            {}
        ).get("Molecule")

        or (
            metadata.get(
                "molecules",
                [None]
            )[0]
        )
    )

    temperature = (

        metadata.get("temperature_K")

        or metadata.get("temperature")

        or metadata.get("Temp")

        or metadata.get(
            "Molecule_data",
            {}
        ).get("Temp")
    )

    multiplicity = (

        metadata.get("multiplicity")

        or metadata.get("Multiplicity")

        or metadata.get(
            "Molecule_data",
            {}
        ).get("Multiplicity")
    )

    method = (

        metadata.get("method")

        or metadata.get(
            "electronic_structure",
            {}
        ).get("method")

        or metadata.get(
            "Elec_structure",
            {}
        ).get("method")
    )

    return {

        "molecule":
            molecule,

        "temperature_K":
            temperature,

        "multiplicity":
            multiplicity,

        "method":
            method
    }


# ============================================================
# ANALYSE ONE TRAJECTORY
# ============================================================

def analyse_trajectory(
    trajectory,
    target_timestep,
    bond_break_factor
):

    output = (
        trajectory
        / "output"
    )

    xyz_file = (
        output
        / "xyz.all"
    )

    forces_file = (
        output
        / "forces.all"
    )

    time_file = (
        output
        / "time.all"
    )

    frames = parse_xyz_all(
        xyz_file
    )

    force_records = parse_forces_all(
        forces_file
    )

    time_records = parse_time_all(
        time_file
    )

    result = {

        "trajectory":
            trajectory.name,

        "path":
            str(trajectory),

        "n_frames":
            len(frames),

        "start_timestep":
            None,

        "final_timestep":
            None,

        "n_atoms":
            None,

        "status":
            "EMPTY",

        "first_break_bond":
            None,

        "first_break_timestep":
            None,

        "first_break_distance":
            None,

        "first_break_source":
            None,

        "n_geometry_breaks":
            0,

        "n_flag_breaks":
            0,

        "scf_energy_initial":
            None,

        "scf_energy_final":
            None,

        "scf_energy_change":
            None
    }

    if not frames:

        return (
            result,
            frames,
            force_records,
            time_records
        )

    result["start_timestep"] = (
        frames[0]["timestep"]
    )

    result["final_timestep"] = (
        frames[-1]["timestep"]
    )

    result["n_atoms"] = len(
        frames[0]["symbols"]
    )

    if (
        target_timestep is not None
        and frames[-1]["timestep"]
        >= float(target_timestep)
    ):

        result["status"] = "COMPLETED"

    else:

        result["status"] = "INCOMPLETE"

    (
        first,
        geometry_events,
        flag_events
    ) = determine_first_break(

        frames,

        force_records,

        bond_break_factor
    )

    result["n_geometry_breaks"] = len(
        geometry_events
    )

    result["n_flag_breaks"] = len(
        flag_events
    )

    if first:

        result["first_break_bond"] = (
            first["bond"]
        )

        result["first_break_timestep"] = (
            first["timestep"]
        )

        result["first_break_distance"] = (
            first["distance"]
        )

        result["first_break_source"] = (
            first["source"]
        )

    # --------------------------------------------------------
    # ENERGY
    # --------------------------------------------------------

    energies = []

    for frame in frames:

        timestep = frame["timestep"]

        record = force_records.get(
            timestep
        )

        if (
            record
            and record.get("scf_energy")
        ):

            energy = sum(
                record["scf_energy"]
            )

            energies.append(
                (
                    timestep,
                    energy
                )
            )

    if energies:

        result[
            "scf_energy_initial"
        ] = energies[0][1]

        result[
            "scf_energy_final"
        ] = energies[-1][1]

        result[
            "scf_energy_change"
        ] = (
            energies[-1][1]
            -
            energies[0][1]
        )

    return (
        result,
        frames,
        force_records,
        time_records
    )


# ============================================================
# PLOT: FIRST BREAK DISTRIBUTION
# ============================================================

def plot_first_break_distribution(
    summary,
    output
):

    counts = summary[
        "first_break_counts"
    ]

    if not counts:
        return

    labels = list(
        counts.keys()
    )

    values = [
        counts[x]
        for x in labels
    ]

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        labels,
        values
    )

    plt.xlabel(
        "First detected event"
    )

    plt.ylabel(
        "Number of trajectories"
    )

    plt.title(
        "First dissociation / bond-breaking event"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        output
        / "first_break_distribution.png",
        dpi=180
    )

    plt.close()


# ============================================================
# PLOT: FINAL TIMESTEPS
# ============================================================

def plot_final_timesteps(
    trajectory_results,
    output
):

    values = [

        x["final_timestep"]

        for x in trajectory_results

        if x["final_timestep"]
        is not None
    ]

    if not values:
        return

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        values,
        bins=min(
            20,
            max(
                5,
                len(values)
            )
        )
    )

    plt.xlabel(
        "Final timestep"
    )

    plt.ylabel(
        "Number of trajectories"
    )

    plt.title(
        "Final timestep distribution"
    )

    plt.tight_layout()

    plt.savefig(
        output
        / "final_timestep_distribution.png",
        dpi=180
    )

    plt.close()


# ============================================================
# PLOT: FIRST BREAK TIMING
# ============================================================

def plot_first_break_timing(
    trajectory_results,
    output
):

    groups = defaultdict(list)

    for result in trajectory_results:

        bond = result[
            "first_break_bond"
        ]

        timestep = result[
            "first_break_timestep"
        ]

        if (
            bond
            and timestep is not None
        ):

            groups[
                bond
            ].append(
                float(timestep)
            )

    if not groups:
        return

    labels = list(
        groups.keys()
    )

    data = [
        groups[x]
        for x in labels
    ]

    plt.figure(
        figsize=(10, 5)
    )

    plt.boxplot(
        data,
        labels=labels
    )

    plt.xlabel(
        "First detected event"
    )

    plt.ylabel(
        "First-break timestep"
    )

    plt.title(
        "First-break timing"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        output
        / "first_break_timing.png",
        dpi=180
    )

    plt.close()


# ============================================================
# PLOT: SURVIVAL
# ============================================================

def plot_survival(
    trajectory_results,
    output
):

    final_times = [

        x["final_timestep"]

        for x in trajectory_results

        if x["final_timestep"]
        is not None
    ]

    if not final_times:
        return

    maximum = max(
        final_times
    )

    if maximum <= 0:
        return

    grid = np.linspace(
        0,
        maximum,
        200
    )

    survival = []

    for t in grid:

        survivors = 0

        for result in trajectory_results:

            break_time = result[
                "first_break_timestep"
            ]

            if (
                break_time is None
                or break_time > t
            ):

                survivors += 1

        survival.append(

            survivors
            /
            len(trajectory_results)
        )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        grid,
        survival
    )

    plt.xlabel(
        "Timestep"
    )

    plt.ylabel(
        "Fraction without detected break"
    )

    plt.ylim(
        0,
        1.05
    )

    plt.title(
        "Qualitative survival curve"
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output
        / "survival_probability.png",
        dpi=180
    )

    plt.close()


# ============================================================
# PLOT: BOND DISTANCES
# ============================================================

def plot_example_bonds(
    frames,
    output,
    factor
):

    if len(frames) < 2:
        return

    symbols = frames[0][
        "symbols"
    ]

    bonds = initial_bonds(
        symbols,
        frames[0]["coords"],
        factor
    )

    if not bonds:
        return

    selected = bonds[
        :min(
            8,
            len(bonds)
        )
    ]

    plt.figure(
        figsize=(10, 6)
    )

    for i, j in selected:

        x = [
            frame["timestep"]
            for frame in frames
        ]

        y = [

            bond_distance(
                frame["coords"],
                i,
                j
            )

            for frame in frames
        ]

        plt.plot(
            x,
            y,
            label=(
                f"{i+1}-{j+1} "
                f"{symbols[i]}-{symbols[j]}"
            )
        )

    plt.xlabel(
        "Timestep"
    )

    plt.ylabel(
        "Bond distance (Å)"
    )

    plt.title(
        "Example bond-distance evolution"
    )

    plt.legend(
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        output
        / "example_bond_distances.png",
        dpi=180
    )

    plt.close()


# ============================================================
# PLOT: ENERGY
# ============================================================

def plot_example_energy(
    frames,
    force_records,
    output
):

    x = []
    y = []

    for frame in frames:

        timestep = frame[
            "timestep"
        ]

        record = force_records.get(
            timestep
        )

        if (
            record
            and record.get(
                "scf_energy"
            )
        ):

            x.append(
                timestep
            )

            y.append(
                sum(
                    record[
                        "scf_energy"
                    ]
                )
            )

    if len(y) < 2:
        return

    y = np.asarray(y)

    y = y - y[0]

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        x,
        y
    )

    plt.xlabel(
        "Timestep"
    )

    plt.ylabel(
        "Change in SCF energy"
    )

    plt.title(
        "Example energy evolution"
    )

    plt.tight_layout()

    plt.savefig(
        output
        / "example_energy_evolution.png",
        dpi=180
    )

    plt.close()


# ============================================================
# DATASET ANALYSIS
# ============================================================

def analyse_dataset(
    name,
    specification,
    master_output
):

    root = Path(
        specification["path"]
    ).expanduser().resolve()

    if not root.exists():

        raise FileNotFoundError(
            f"Dataset '{name}' does not exist:\n"
            f"{root}"
        )

    output = (
        master_output
        /
        safe_name(name)
    )

    plots = (
        output
        /
        "plots"
    )

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    plots.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = get_metadata(
        root,
        specification.get(
            "metadata",
            {}
        )
    )

    target_timestep = specification.get(
        "target_timestep"
    )

    factor = float(
        specification.get(
            "bond_break_factor",
            DEFAULT_BOND_FACTOR
        )
    )

    trajectories = discover_trajectories(
        root
    )

    print(
        f"\nDataset: {name}"
    )

    print(
        f"Root: {root}"
    )

    print(
        f"Trajectories found: "
        f"{len(trajectories)}"
    )

    results = []

    example_frames = None
    example_forces = None

    for trajectory in trajectories:

        (
            result,
            frames,
            force_records,
            time_records
        ) = analyse_trajectory(

            trajectory,

            target_timestep,

            factor
        )

        results.append(
            result
        )

        if (
            example_frames is None
            and frames
        ):

            example_frames = frames
            example_forces = force_records

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    write_csv(

        output
        /
        "trajectory_summary.csv",

        results
    )

    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    first_break_counts = Counter(

        result["first_break_bond"]

        for result in results

        if result[
            "first_break_bond"
        ]
    )

    completed = sum(

        result["status"]
        == "COMPLETED"

        for result in results
    )

    incomplete = sum(

        result["status"]
        == "INCOMPLETE"

        for result in results
    )

    empty = sum(

        result["status"]
        == "EMPTY"

        for result in results
    )

    # --------------------------------------------------------
    # BREAK TIMING
    # --------------------------------------------------------

    break_times = defaultdict(list)

    for result in results:

        bond = result[
            "first_break_bond"
        ]

        timestep = result[
            "first_break_timestep"
        ]

        if (
            bond
            and timestep is not None
        ):

            break_times[
                bond
            ].append(
                float(timestep)
            )

    timing_summary = {}

    for bond, values in break_times.items():

        timing_summary[bond] = {

            "n":
                len(values),

            "mean":
                mean(values),

            "median":
                median(values),

            "stdev":
                (
                    stdev(values)
                    if len(values) >= 2
                    else None
                )
        }

    summary = {

        "name":
            name,

        "root":
            str(root),

        "molecule":
            metadata["molecule"],

        "temperature_K":
            metadata["temperature_K"],

        "multiplicity":
            metadata["multiplicity"],

        "method":
            metadata["method"],

        "n_trajectories":
            len(results),

        "completed":
            completed,

        "incomplete":
            incomplete,

        "empty":
            empty,

        "first_break_counts":
            dict(
                first_break_counts
            ),

        "first_break_timing":
            timing_summary
    }

    with open(
        output / "summary.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # TEXT REPORT
    # --------------------------------------------------------

    report = []

    report.append(
        "=" * 70
    )

    report.append(
        f"DATASET ANALYSIS: {name}"
    )

    report.append(
        "=" * 70
    )

    report.append("")

    report.append(
        f"Root:          {root}"
    )

    report.append(
        f"Molecule:      "
        f"{metadata['molecule']}"
    )

    report.append(
        f"Temperature:   "
        f"{metadata['temperature_K']} K"
    )

    report.append(
        f"Multiplicity:  "
        f"{metadata['multiplicity']}"
    )

    report.append(
        f"Method:        "
        f"{metadata['method']}"
    )

    report.append("")

    report.append(
        f"Trajectories:  "
        f"{len(results)}"
    )

    report.append(
        f"Completed:     "
        f"{completed}"
    )

    report.append(
        f"Incomplete:    "
        f"{incomplete}"
    )

    report.append(
        f"Empty:         "
        f"{empty}"
    )

    report.append("")

    report.append(
        "FIRST BREAK EVENTS"
    )

    report.append(
        "-" * 70
    )

    if first_break_counts:

        total = len(results)

        for bond, count in sorted(
            first_break_counts.items(),
            key=lambda x: -x[1]
        ):

            percentage = (

                100.0
                * count
                / total
                if total
                else 0
            )

            report.append(

                f"{bond:30s}"
                f"{count:6d}"
                f" ({percentage:6.2f}%)"
            )

    else:

        report.append(
            "No first-break event detected."
        )

    report.append("")

    report.append(
        "FIRST BREAK TIMING"
    )

    report.append(
        "-" * 70
    )

    for bond, stats in timing_summary.items():

        report.append(

            f"{bond:30s} "
            f"n={stats['n']} "
            f"mean={stats['mean']:.2f} "
            f"median={stats['median']:.2f}"
        )

    with open(
        output / "summary.txt",
        "w"
    ) as f:

        f.write(
            "\n".join(report)
        )

    # --------------------------------------------------------
    # GRAPHS
    # --------------------------------------------------------

    plot_first_break_distribution(
        summary,
        plots
    )

    plot_final_timesteps(
        results,
        plots
    )

    plot_first_break_timing(
        results,
        plots
    )

    plot_survival(
        results,
        plots
    )

    if example_frames:

        plot_example_bonds(
            example_frames,
            plots,
            factor
        )

        plot_example_energy(
            example_frames,
            example_forces,
            plots
        )

    return summary


# ============================================================
# QUALITATIVE COMPARISON
# ============================================================

def qualitative_compare(
    dataset_a,
    dataset_b,
    output
):

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    warnings = []

    # --------------------------------------------------------
    # METADATA CHECK
    # --------------------------------------------------------

    molecule_a = dataset_a[
        "molecule"
    ]

    molecule_b = dataset_b[
        "molecule"
    ]

    if (
        molecule_a
        and molecule_b
        and molecule_a != molecule_b
    ):

        warnings.append(

            f"Molecule mismatch: "
            f"{molecule_a} vs "
            f"{molecule_b}"
        )

    temperature_a = dataset_a[
        "temperature_K"
    ]

    temperature_b = dataset_b[
        "temperature_K"
    ]

    if (
        temperature_a is not None
        and temperature_b is not None
        and str(temperature_a)
        != str(temperature_b)
    ):

        warnings.append(

            f"Temperature mismatch: "
            f"{temperature_a} K vs "
            f"{temperature_b} K"
        )

    multiplicity_a = dataset_a[
        "multiplicity"
    ]

    multiplicity_b = dataset_b[
        "multiplicity"
    ]

    if (
        multiplicity_a is not None
        and multiplicity_b is not None
        and str(multiplicity_a)
        != str(multiplicity_b)
    ):

        warnings.append(

            f"Multiplicity mismatch: "
            f"{multiplicity_a} vs "
            f"{multiplicity_b}"
        )

    # --------------------------------------------------------
    # BREAK DISTRIBUTIONS
    # --------------------------------------------------------

    counts_a = dataset_a[
        "first_break_counts"
    ]

    counts_b = dataset_b[
        "first_break_counts"
    ]

    total_a = dataset_a[
        "n_trajectories"
    ]

    total_b = dataset_b[
        "n_trajectories"
    ]

    all_events = sorted(

        set(counts_a)
        |
        set(counts_b)
    )

    comparison_rows = []

    for event in all_events:

        count_a = counts_a.get(
            event,
            0
        )

        count_b = counts_b.get(
            event,
            0
        )

        fraction_a = (

            count_a
            /
            total_a
            if total_a
            else 0
        )

        fraction_b = (

            count_b
            /
            total_b
            if total_b
            else 0
        )

        comparison_rows.append({

            "event":
                event,

            "dataset_A_count":
                count_a,

            "dataset_A_fraction":
                fraction_a,

            "dataset_B_count":
                count_b,

            "dataset_B_fraction":
                fraction_b
        })

    write_csv(

        output
        /
        "qualitative_comparison.csv",

        comparison_rows
    )

    # --------------------------------------------------------
    # DOMINANT EVENTS
    # --------------------------------------------------------

    dominant_a = (

        max(
            counts_a,
            key=counts_a.get
        )
        if counts_a
        else None
    )

    dominant_b = (

        max(
            counts_b,
            key=counts_b.get
        )
        if counts_b
        else None
    )

    if (
        dominant_a
        and dominant_b
        and dominant_a == dominant_b
    ):

        agreement = "CONSISTENT"

        pathway_statement = (

            "Both datasets show the same "
            "dominant first detected event: "
            f"{dominant_a}."
        )

    elif (
        dominant_a
        and dominant_b
    ):

        agreement = "DIFFERENT"

        pathway_statement = (

            "The dominant first detected "
            "event differs between datasets: "
            f"{dataset_a['name']} = "
            f"{dominant_a}; "
            f"{dataset_b['name']} = "
            f"{dominant_b}."
        )

    else:

        agreement = "INSUFFICIENT_DATA"

        pathway_statement = (

            "There is insufficient first-break "
            "information for a pathway comparison."
        )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    if all_events:

        x = np.arange(
            len(all_events)
        )

        width = 0.38

        fractions_a = [

            counts_a.get(
                event,
                0
            )
            /
            total_a
            if total_a
            else 0

            for event in all_events
        ]

        fractions_b = [

            counts_b.get(
                event,
                0
            )
            /
            total_b
            if total_b
            else 0

            for event in all_events
        ]

        plt.figure(
            figsize=(11, 6)
        )

        plt.bar(
            x - width / 2,
            fractions_a,
            width,
            label=dataset_a["name"]
        )

        plt.bar(
            x + width / 2,
            fractions_b,
            width,
            label=dataset_b["name"]
        )

        plt.xticks(
            x,
            all_events,
            rotation=30,
            ha="right"
        )

        plt.xlabel(
            "First detected event"
        )

        plt.ylabel(
            "Fraction of trajectories"
        )

        plt.title(
            "Qualitative first-break comparison"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            output
            /
            "first_break_comparison.png",
            dpi=180
        )

        plt.close()

    # --------------------------------------------------------
    # TEXT REPORT
    # --------------------------------------------------------

    report = []

    report.append(
        "=" * 70
    )

    report.append(
        "QUALITATIVE DATASET COMPARISON"
    )

    report.append(
        "=" * 70
    )

    report.append("")

    report.append(
        f"Dataset A: "
        f"{dataset_a['name']}"
    )

    report.append(
        f"Dataset B: "
        f"{dataset_b['name']}"
    )

    report.append("")

    report.append(
        "DATASET A"
    )

    report.append(
        f"  Molecule: "
        f"{dataset_a['molecule']}"
    )

    report.append(
        f"  Temperature: "
        f"{dataset_a['temperature_K']} K"
    )

    report.append(
        f"  Multiplicity: "
        f"{dataset_a['multiplicity']}"
    )

    report.append(
        f"  Method: "
        f"{dataset_a['method']}"
    )

    report.append("")

    report.append(
        "DATASET B"
    )

    report.append(
        f"  Molecule: "
        f"{dataset_b['molecule']}"
    )

    report.append(
        f"  Temperature: "
        f"{dataset_b['temperature_K']} K"
    )

    report.append(
        f"  Multiplicity: "
        f"{dataset_b['multiplicity']}"
    )

    report.append(
        f"  Method: "
        f"{dataset_b['method']}"
    )

    report.append("")

    report.append(
        "COMPARABILITY"
    )

    report.append(
        "-" * 70
    )

    if warnings:

        for warning in warnings:

            report.append(
                f"WARNING: {warning}"
            )

        report.append("")

        report.append(
            "The comparison is therefore "
            "qualitative rather than quantitative."
        )

    else:

        report.append(
            "No basic metadata mismatch was detected."
        )

    report.append("")

    report.append(
        "FIRST DISSOCIATION / BREAKING"
    )

    report.append(
        "-" * 70
    )

    report.append(
        pathway_statement
    )

    report.append("")

    report.append(
        f"Overall qualitative agreement: "
        f"{agreement}"
    )

    with open(
        output
        /
        "comparison_report.txt",
        "w"
    ) as f:

        f.write(
            "\n".join(report)
        )

    return {

        "dataset_A":
            dataset_a["name"],

        "dataset_B":
            dataset_b["name"],

        "agreement":
            agreement,

        "dominant_A":
            dominant_a,

        "dominant_B":
            dominant_b,

        "warnings":
            warnings
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Independent analysis of "
            "bootstrap, validation and "
            "production trajectories."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to analysis_config.json"
    )

    args = parser.parse_args()

    config_path = Path(
        args.config
    ).expanduser().resolve()

    config = load_json(
        config_path
    )

    output_dir = Path(
        config.get(
            "output_dir",
            "./results"
        )
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    datasets = config.get(
        "datasets",
        {}
    )

    enabled = []

    for name, specification in datasets.items():

        if specification.get(
            "enabled",
            False
        ):

            enabled.append(
                (
                    name,
                    specification
                )
            )

    if not enabled:

        raise SystemExit(
            "No datasets are enabled."
        )

    summaries = {}

    # --------------------------------------------------------
    # ANALYSE DATASETS
    # --------------------------------------------------------

    for name, specification in enabled:

        summary = analyse_dataset(

            name,

            specification,

            output_dir
        )

        summaries[name] = summary

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    comparison = config.get(
        "comparison",
        {}
    )

    if (
        comparison.get(
            "enabled",
            False
        )
        and len(enabled) >= 2
    ):

        first_name = enabled[0][0]
        second_name = enabled[1][0]

        comparison_dir = (

            output_dir
            /
            "comparison"
            /
            (
                safe_name(first_name)
                +
                "_vs_"
                +
                safe_name(second_name)
            )
        )

        result = qualitative_compare(

            summaries[first_name],

            summaries[second_name],

            comparison_dir
        )

        with open(
            output_dir
            /
            "comparison"
            /
            "comparison_summary.json",
            "w"
        ) as f:

            json.dump(
                result,
                f,
                indent=4
            )

    # --------------------------------------------------------
    # MASTER SUMMARY
    # --------------------------------------------------------

    with open(
        output_dir
        /
        "master_summary.json",
        "w"
    ) as f:

        json.dump(
            summaries,
            f,
            indent=4
        )

    print("")
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("")
    print(
        f"Results directory:"
    )
    print(
        output_dir
    )
    print("")


if __name__ == "__main__":

    main()
