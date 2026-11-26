#!/usr/bin/env python3

from pathlib import Path
import re
import csv
import math


# ============================================================
# FIXED VALIDATION RUN
# ============================================================

VALIDATION_ROOT = Path(
    "/mnt/scratch/lgjs798/"
    "HFC236ea_model_2000/01_validation"
)

SELECTED_MOMENTA = (
    VALIDATION_ROOT
    / "selection"
    / "selected_momenta.txt"
)

REPETITIONS_DIR = (
    VALIDATION_ROOT
    / "repetitions"
)

# ============================================================
# DISSOCIATION CRITERION
# ============================================================

# IMPORTANT:
# The plasma dissociation criterion is 5 Bohr.
#
# A bond is considered broken when:
#
#       bond distance > 5.0 Bohr
#
DISSOCIATION_THRESHOLD = 5.0


# ============================================================
# INITIAL BONDS FOR HFC236
# ============================================================

# HFC236 = C3H2F6
#
# Atom numbering used by your validation trajectories:
#
# F1-C2
# C2-F3
# C2-F4
# C2-C5
# C5-F6
# C5-C7
# C7-F8
# C7-F9
# C5-H10
# C7-H11
#
# These are the bonds whose distances are monitored.

INITIAL_BONDS = [
    (1, 2),    # F1-C2
    (2, 3),    # C2-F3
    (2, 4),    # C2-F4
    (2, 5),    # C2-C5
    (5, 6),    # C5-F6
    (5, 7),    # C5-C7
    (7, 8),    # C7-F8
    (7, 9),    # C7-F9
    (5, 10),   # C5-H10
    (7, 11),   # C7-H11
]


# ============================================================
# BASIC BOND FUNCTIONS
# ============================================================

def canonical_bond(a, b):
    a = int(a)
    b = int(b)

    if a < b:
        return (a, b)

    return (b, a)


def bond_key(bond):
    return canonical_bond(
        bond[0],
        bond[1]
    )


# ============================================================
# ATOM SYMBOLS
# ============================================================

DEFAULT_SYMBOLS = {
    1: "F",
    2: "C",
    3: "F",
    4: "F",
    5: "C",
    6: "F",
    7: "C",
    8: "F",
    9: "F",
    10: "H",
    11: "H",
}


def read_geometry_symbols(geometry_file):

    """
    Read atom symbols from the repetition's Geometry file.

    The atom order is used as the atom number if the Geometry
    file does not explicitly contain atom numbers.
    """

    symbols = {}

    if not geometry_file.exists():
        return DEFAULT_SYMBOLS.copy()

    with open(
        geometry_file,
        "r",
        errors="replace"
    ) as f:

        lines = f.readlines()

    atom_index = 0

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if not parts:
            continue

        # Skip obvious headers
        if parts[0].lower() in (
            "geometry",
            "coordinates",
            "atoms",
            "momentum",
        ):
            continue

        # ----------------------------------------------------
        # Case 1:
        #
        # 1 F x y z
        #
        # ----------------------------------------------------

        if len(parts) >= 5:

            try:
                number = int(parts[0])

                float(parts[2])
                float(parts[3])
                float(parts[4])

                symbols[number] = parts[1]

                continue

            except (ValueError, TypeError):
                pass

        # ----------------------------------------------------
        # Case 2:
        #
        # F x y z
        #
        # ----------------------------------------------------

        if len(parts) >= 4:

            try:

                float(parts[1])
                float(parts[2])
                float(parts[3])

                atom_index += 1

                symbols[atom_index] = parts[0]

            except (ValueError, TypeError):

                continue

    # If parsing was incomplete, use known HFC236 mapping
    for number, symbol in DEFAULT_SYMBOLS.items():

        if number not in symbols:

            symbols[number] = symbol

    return symbols


def formatted_bond(bond, symbols):

    if bond is None:
        return ""

    a, b = bond

    sa = symbols.get(
        a,
        f"A{a}"
    )

    sb = symbols.get(
        b,
        f"A{b}"
    )

    return (
        f"{sa}{a}-{sb}{b}"
    )


def plain_bond(bond):

    if bond is None:
        return ""

    return (
        f"{bond[0]}-{bond[1]}"
    )


# ============================================================
# SELECTED MOMENTA PARSER
# ============================================================

def parse_selected_momenta():

    """
    Extract the selected bond for each repetition.

    The parser is deliberately flexible because selected_momenta.txt
    can contain different explanatory text.

    It looks for:

        rep-1
        ...
        Selected bond: 2-5

    and also accepts variations such as:

        Selected Bond = 2-5
        selected_bond: 2-5
        Bond: 2-5
        bond = 2-5
    """

    if not SELECTED_MOMENTA.exists():

        raise FileNotFoundError(
            f"Cannot find:\n"
            f"{SELECTED_MOMENTA}"
        )

    selected = {}

    current_rep = None

    rep_pattern = re.compile(
        r"\brep(?:etition)?[-_\s]*(\d+)\b",
        re.IGNORECASE
    )

    # Most specific patterns first
    bond_patterns = [

        re.compile(
            r"selected\s*[_\-]?\s*bond"
            r"\s*(?:is|=|:)?\s*"
            r"(\d+)\s*[-–]\s*(\d+)",
            re.IGNORECASE
        ),

        re.compile(
            r"selected"
            r"\s*(?:bond)?"
            r"\s*(?:is|=|:)\s*"
            r"(\d+)\s*[-–]\s*(\d+)",
            re.IGNORECASE
        ),

        re.compile(
            r"\bbond\b"
            r"\s*(?:is|=|:)\s*"
            r"(\d+)\s*[-–]\s*(\d+)",
            re.IGNORECASE
        ),
    ]

    with open(
        SELECTED_MOMENTA,
        "r",
        errors="replace"
    ) as f:

        lines = f.readlines()

    for line in lines:

        stripped = line.strip()

        if not stripped:
            continue

        # ----------------------------------------------------
        # Find repetition number
        # ----------------------------------------------------

        rep_match = rep_pattern.search(
            stripped
        )

        if rep_match:

            current_rep = int(
                rep_match.group(1)
            )

        # ----------------------------------------------------
        # Find selected bond
        # ----------------------------------------------------

        if current_rep is None:
            continue

        for pattern in bond_patterns:

            match = pattern.search(
                stripped
            )

            if match:

                a = int(
                    match.group(1)
                )

                b = int(
                    match.group(2)
                )

                selected[current_rep] = (
                    canonical_bond(a, b)
                )

                break

    return selected


# ============================================================
# XYZ TRAJECTORY READER
# ============================================================

def parse_xyz_frames(xyz_file):

    """
    Read xyz.all.

    The function supports both common formats:

        11
        Timestep 100
        C ...
        F ...

    and indexed formats:

        1 F x y z
        2 C x y z

    The timestep is obtained from the comment/header whenever
    possible. If no explicit timestep can be found, the frame
    number is used.
    """

    if not xyz_file.exists():

        return []

    with open(
        xyz_file,
        "r",
        errors="replace"
    ) as f:

        lines = f.readlines()

    frames = []

    i = 0
    frame_number = 0

    while i < len(lines):

        line = lines[i].strip()

        # ----------------------------------------------------
        # Look for standard XYZ atom-count line
        # ----------------------------------------------------

        try:

            natoms = int(line)

        except ValueError:

            i += 1
            continue

        if natoms <= 0:

            i += 1
            continue

        # Need comment line + atoms
        if i + 1 + natoms >= len(lines):

            break

        comment = lines[i + 1].strip()

        frame_number += 1

        # ----------------------------------------------------
        # Extract timestep from comment
        # ----------------------------------------------------

        timestep = None

        timestep_patterns = [

            r"\bTimestep\s*[:=]?\s*"
            r"([-+]?\d+(?:\.\d+)?)",

            r"\btime\s*[:=]?\s*"
            r"([-+]?\d+(?:\.\d+)?)",

            r"\bstep\s*[:=]?\s*"
            r"([-+]?\d+(?:\.\d+)?)",
        ]

        for pattern in timestep_patterns:

            match = re.search(
                pattern,
                comment,
                re.IGNORECASE
            )

            if match:

                timestep = float(
                    match.group(1)
                )

                break

        if timestep is None:

            timestep = float(
                frame_number
            )

        coordinates = {}

        # ----------------------------------------------------
        # Read atoms
        # ----------------------------------------------------

        for j in range(
            natoms
        ):

            atom_line = lines[
                i + 2 + j
            ].strip()

            parts = atom_line.split()

            if len(parts) < 4:
                continue

            # ------------------------------------------------
            # Indexed:
            #
            # 1 F x y z
            # ------------------------------------------------

            if len(parts) >= 5:

                try:

                    atom_number = int(
                        parts[0]
                    )

                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])

                    coordinates[
                        atom_number
                    ] = (
                        x,
                        y,
                        z
                    )

                    continue

                except ValueError:

                    pass

            # ------------------------------------------------
            # Standard XYZ:
            #
            # F x y z
            #
            # Atom number = order in frame
            # ------------------------------------------------

            try:

                atom_number = j + 1

                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])

                coordinates[
                    atom_number
                ] = (
                    x,
                    y,
                    z
                )

            except ValueError:

                continue

        if coordinates:

            frames.append(
                (
                    timestep,
                    coordinates
                )
            )

        i += (
            natoms + 2
        )

    # --------------------------------------------------------
    # If standard XYZ parsing found nothing, try a more
    # general timestep-based parser.
    # --------------------------------------------------------

    if frames:

        return frames

    return parse_nonstandard_xyz(
        lines
    )


# ============================================================
# NON-STANDARD XYZ FALLBACK
# ============================================================

def parse_nonstandard_xyz(lines):

    """
    Fallback reader for xyz.all files that don't use ordinary
    XYZ atom-count blocks.

    Looks for lines containing 'Timestep' and then reads atom
    coordinate lines following them.
    """

    frames = []

    timestep_pattern = re.compile(
        r"\bTimestep\s*[:=]?\s*"
        r"([-+]?\d+(?:\.\d+)?)",
        re.IGNORECASE
    )

    current_timestep = None
    coordinates = {}
    atom_counter = 0

    for line in lines:

        stripped = line.strip()

        if not stripped:
            continue

        timestep_match = (
            timestep_pattern.search(
                stripped
            )
        )

        if timestep_match:

            if coordinates:

                frames.append(
                    (
                        current_timestep,
                        coordinates
                    )
                )

            current_timestep = float(
                timestep_match.group(1)
            )

            coordinates = {}
            atom_counter = 0

            continue

        parts = stripped.split()

        # ----------------------------------------------------
        # Indexed atom line
        # ----------------------------------------------------

        if len(parts) >= 5:

            try:

                atom_number = int(
                    parts[0]
                )

                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])

                coordinates[
                    atom_number
                ] = (
                    x,
                    y,
                    z
                )

                continue

            except ValueError:

                pass

        # ----------------------------------------------------
        # Non-indexed atom line
        # ----------------------------------------------------

        if len(parts) >= 4:

            try:

                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])

                atom_counter += 1

                coordinates[
                    atom_counter
                ] = (
                    x,
                    y,
                    z
                )

            except ValueError:

                pass

    if coordinates:

        frames.append(
            (
                current_timestep,
                coordinates
            )
        )

    return frames


# ============================================================
# DISTANCE
# ============================================================

def distance(p1, p2):

    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]

    return math.sqrt(
        dx * dx
        + dy * dy
        + dz * dz
    )


# ============================================================
# FIND ACTUAL DISSOCIATION EVENTS
# ============================================================

def find_dissociation_events(
    xyz_file
):

    """
    Scan the trajectory chronologically.

    For every initial bond:

        distance > 5.0 Bohr

    means the bond has dissociated.

    A bond is recorded only once, at its FIRST crossing.

    Therefore:

        event 1 = actual first break
        event 2 = actual second break
        event 3 = actual third break
        ...

    Returns a list of dictionaries.
    """

    frames = parse_xyz_frames(
        xyz_file
    )

    if not frames:

        return []

    broken = set()
    events = []

    for timestep, coordinates in frames:

        for bond in INITIAL_BONDS:

            bond = bond_key(bond)

            if bond in broken:
                continue

            a, b = bond

            if (
                a not in coordinates
                or b not in coordinates
            ):
                continue

            d = distance(
                coordinates[a],
                coordinates[b]
            )

            # =================================================
            # DISSOCIATION CRITERION
            # =================================================

            if d > DISSOCIATION_THRESHOLD:

                broken.add(
                    bond
                )

                events.append(
                    {
                        "bond": bond,
                        "timestep": timestep,
                        "distance": d,
                    }
                )

    return events


# ============================================================
# NATURAL POSITION
# ============================================================

def position_name(position):

    if position == 1:
        return "1st"

    if position == 2:
        return "2nd"

    if position == 3:
        return "3rd"

    return f"{position}th"


# ============================================================
# ANALYSE ONE REPETITION
# ============================================================

def analyse_rep(
    rep_number,
    selected_bond
):

    rep_name = (
        f"rep-{rep_number}"
    )

    rep_dir = (
        REPETITIONS_DIR
        / rep_name
    )

    geometry_file = (
        rep_dir
        / "Geometry"
    )

    xyz_file = (
        rep_dir
        / "output"
        / "xyz.all"
    )

    symbols = read_geometry_symbols(
        geometry_file
    )

    result = {

        "rep":
            rep_name,

        "selected_bond":
            formatted_bond(
                selected_bond,
                symbols
            ),

        "actual_1st_break":
            "",

        "1st_break_timestep":
            "",

        "actual_2nd_break":
            "",

        "2nd_break_timestep":
            "",

        "actual_3rd_break":
            "",

        "3rd_break_timestep":
            "",

        "actual_4th_break":
            "",

        "4th_break_timestep":
            "",

        "all_breaks":
            "",

        "selected_bond_actual_timestep":
            "",

        "selected_bond_distance":
            "",

        "matched_position":
            "NO",

        "selected_bond_status":
            "",

        "trajectory_status":
            "",

    }

    # ========================================================
    # Check trajectory
    # ========================================================

    if not xyz_file.exists():

        result["trajectory_status"] = (
            "NO_XYZ"
        )

        result["selected_bond_status"] = (
            "NOT_ANALYSED"
        )

        return result

    # ========================================================
    # Find breaks
    # ========================================================

    events = find_dissociation_events(
        xyz_file
    )

    if not events:

        result["trajectory_status"] = (
            "NO_BREAKS"
        )

        result["selected_bond_status"] = (
            "NOT_OBSERVED"
        )

        return result

    result["trajectory_status"] = (
        "BREAKS_FOUND"
    )

    # ========================================================
    # Store first 4 breaks explicitly
    # ========================================================

    if len(events) >= 1:

        event = events[0]

        result["actual_1st_break"] = (
            formatted_bond(
                event["bond"],
                symbols
            )
        )

        result["1st_break_timestep"] = (
            str(event["timestep"])
        )

    if len(events) >= 2:

        event = events[1]

        result["actual_2nd_break"] = (
            formatted_bond(
                event["bond"],
                symbols
            )
        )

        result["2nd_break_timestep"] = (
            str(event["timestep"])
        )

    if len(events) >= 3:

        event = events[2]

        result["actual_3rd_break"] = (
            formatted_bond(
                event["bond"],
                symbols
            )
        )

        result["3rd_break_timestep"] = (
            str(event["timestep"])
        )

    if len(events) >= 4:

        event = events[3]

        result["actual_4th_break"] = (
            formatted_bond(
                event["bond"],
                symbols
            )
        )

        result["4th_break_timestep"] = (
            str(event["timestep"])
        )

    # ========================================================
    # All breaks
    # ========================================================

    result["all_breaks"] = " | ".join(
        formatted_bond(
            event["bond"],
            symbols
        )
        for event in events
    )

    # ========================================================
    # No selected bond
    # ========================================================

    if selected_bond is None:

        result["selected_bond_status"] = (
            "NO_SELECTED_BOND"
        )

        return result

    selected_bond = bond_key(
        selected_bond
    )

    # ========================================================
    # Find selected bond in actual break sequence
    # ========================================================

    selected_event = None
    selected_position = None

    for index, event in enumerate(
        events
    ):

        if bond_key(
            event["bond"]
        ) == selected_bond:

            selected_event = event

            selected_position = (
                index + 1
            )

            break

    # ========================================================
    # Selected bond never broke
    # ========================================================

    if selected_event is None:

        result["selected_bond_status"] = (
            "NOT_OBSERVED"
        )

        result["matched_position"] = (
            "NO"
        )

        return result

    # ========================================================
    # Selected bond did break
    # ========================================================

    result["selected_bond_status"] = (
        "OBSERVED"
    )

    result["selected_bond_actual_timestep"] = (
        str(
            selected_event["timestep"]
        )
    )

    result["selected_bond_distance"] = (
        f"{selected_event['distance']:.6f}"
    )

    result["matched_position"] = (
        position_name(
            selected_position
        )
    )

    return result


# ============================================================
# WRITE CSV
# ============================================================

def write_csv(
    results,
    output_file
):

    fields = [

        "rep",

        "selected_bond",

        "actual_1st_break",
        "1st_break_timestep",

        "actual_2nd_break",
        "2nd_break_timestep",

        "actual_3rd_break",
        "3rd_break_timestep",

        "actual_4th_break",
        "4th_break_timestep",

        "all_breaks",

        "selected_bond_actual_timestep",

        "selected_bond_distance",

        "matched_position",

        "selected_bond_status",

        "trajectory_status",
    ]

    with open(
        output_file,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for result in results:

            writer.writerow(
                result
            )


# ============================================================
# WRITE HUMAN-READABLE TXT
# ============================================================

def write_txt(
    results,
    output_file
):

    with open(
        output_file,
        "w"
    ) as f:

        f.write(
            "HFC236 VALIDATION TRAJECTORY ANALYSIS\n"
        )

        f.write(
            "=" * 100 + "\n\n"
        )

        f.write(
            f"Validation directory:\n"
            f"{VALIDATION_ROOT}\n\n"
        )

        f.write(
            "Selected bond source:\n"
            f"{SELECTED_MOMENTA}\n\n"
        )

        f.write(
            "Dissociation criterion:\n"
            f"distance > {DISSOCIATION_THRESHOLD} Bohr\n\n"
        )

        f.write(
            "Initial monitored bonds:\n"
        )

        for bond in INITIAL_BONDS:

            f.write(
                f"    {plain_bond(bond)}\n"
            )

        f.write(
            "\n"
        )

        f.write(
            "=" * 100 + "\n\n"
        )

        for result in results:

            f.write(
                f"{result['rep']}\n"
            )

            f.write(
                f"Selected bond: "
                f"{result['selected_bond']}\n"
            )

            f.write(
                f"1st break: "
                f"{result['actual_1st_break']} "
                f"@ "
                f"{result['1st_break_timestep']}\n"
            )

            f.write(
                f"2nd break: "
                f"{result['actual_2nd_break']} "
                f"@ "
                f"{result['2nd_break_timestep']}\n"
            )

            f.write(
                f"3rd break: "
                f"{result['actual_3rd_break']} "
                f"@ "
                f"{result['3rd_break_timestep']}\n"
            )

            f.write(
                f"4th break: "
                f"{result['actual_4th_break']} "
                f"@ "
                f"{result['4th_break_timestep']}\n"
            )

            f.write(
                f"All breaks: "
                f"{result['all_breaks']}\n"
            )

            f.write(
                f"Selected bond actual timestep: "
                f"{result['selected_bond_actual_timestep']}\n"
            )

            f.write(
                f"Selected bond distance: "
                f"{result['selected_bond_distance']} Bohr\n"
            )

            f.write(
                f"Matched position: "
                f"{result['matched_position']}\n"
            )

            f.write(
                f"Selected bond status: "
                f"{result['selected_bond_status']}\n"
            )

            f.write(
                f"Trajectory status: "
                f"{result['trajectory_status']}\n"
            )

            f.write(
                "-" * 100 + "\n"
            )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(results):

    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    total = len(results)

    observed = [
        r for r in results
        if r["selected_bond_status"]
        == "OBSERVED"
    ]

    not_observed = [
        r for r in results
        if r["selected_bond_status"]
        == "NOT_OBSERVED"
    ]

    first = [
        r for r in observed
        if r["matched_position"]
        == "1st"
    ]

    second = [
        r for r in observed
        if r["matched_position"]
        == "2nd"
    ]

    later = [
        r for r in observed
        if r["matched_position"]
        not in ("1st", "2nd")
    ]

    no_xyz = [
        r for r in results
        if r["trajectory_status"]
        == "NO_XYZ"
    ]

    print(
        f"Total repetitions              : {total}"
    )

    print(
        f"Selected bond observed         : "
        f"{len(observed)}"
    )

    print(
        f"Selected bond NOT observed     : "
        f"{len(not_observed)}"
    )

    print(
        f"Selected bond was 1st break    : "
        f"{len(first)}"
    )

    print(
        f"Selected bond was 2nd break    : "
        f"{len(second)}"
    )

    print(
        f"Selected bond was later break  : "
        f"{len(later)}"
    )

    print(
        f"Missing xyz.all                : "
        f"{len(no_xyz)}"
    )

    if total > 0:

        print()

        print(
            f"1st-break prediction rate "
            f"(all repetitions): "
            f"{100.0 * len(first) / total:.1f}%"
        )

    if observed:

        print(
            f"1st-break rate among observed "
            f"selected bonds: "
            f"{100.0 * len(first) / len(observed):.1f}%"
        )

    print(
        "=" * 70
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("HFC236 VALIDATION TRAJECTORY ANALYSIS")
    print("=" * 70)

    # ========================================================
    # CHECK DIRECTORIES
    # ========================================================

    print()
    print(
        "Validation directory:"
    )

    print(
        f"    {VALIDATION_ROOT}"
    )

    if not VALIDATION_ROOT.exists():

        raise FileNotFoundError(
            f"Validation directory does not exist:\n"
            f"{VALIDATION_ROOT}"
        )

    if not SELECTED_MOMENTA.exists():

        raise FileNotFoundError(
            f"selected_momenta.txt not found:\n"
            f"{SELECTED_MOMENTA}"
        )

    if not REPETITIONS_DIR.exists():

        raise FileNotFoundError(
            f"repetitions directory not found:\n"
            f"{REPETITIONS_DIR}"
        )

    # ========================================================
    # READ SELECTED BONDS
    # ========================================================

    print()
    print(
        "Reading selected bonds:"
    )

    print(
        f"    {SELECTED_MOMENTA}"
    )

    selected_bonds = (
        parse_selected_momenta()
    )

    print(
        f"Selected bonds found: "
        f"{len(selected_bonds)}"
    )

    # ========================================================
    # SHOW SELECTED BONDS
    # ========================================================

    print()

    for rep_number in sorted(
        selected_bonds
    ):

        bond = selected_bonds[
            rep_number
        ]

        print(
            f"    rep-{rep_number}: "
            f"{plain_bond(bond)}"
        )

    # ========================================================
    # FIND REPS
    # ========================================================

    repetitions = []

    for p in REPETITIONS_DIR.iterdir():

        if not p.is_dir():
            continue

        match = re.fullmatch(
            r"rep-(\d+)",
            p.name
        )

        if match:

            repetitions.append(
                int(
                    match.group(1)
                )
            )

    repetitions.sort()

    print()
    print(
        f"Repetitions found: "
        f"{len(repetitions)}"
    )

    # ========================================================
    # ANALYSE
    # ========================================================

    print()
    print("=" * 70)
    print("SCANNING TRAJECTORIES")
    print("=" * 70)

    results = []

    for rep_number in repetitions:

        selected_bond = (
            selected_bonds.get(
                rep_number
            )
        )

        result = analyse_rep(
            rep_number,
            selected_bond
        )

        results.append(
            result
        )

        print(
            f"{result['rep']:>7} | "
            f"selected: "
            f"{result['selected_bond']:<12} | "
            f"1st: "
            f"{result['actual_1st_break']:<12} "
            f"@ "
            f"{result['1st_break_timestep']:<8} | "
            f"match: "
            f"{result['matched_position']:<5}"
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    csv_file = (
        VALIDATION_ROOT
        / "validation_dissociation_analysis.csv"
    )

    txt_file = (
        VALIDATION_ROOT
        / "validation_dissociation_analysis.txt"
    )

    write_csv(
        results,
        csv_file
    )

    write_txt(
        results,
        txt_file
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print_summary(
        results
    )

    print()
    print(
        "Output CSV:"
    )

    print(
        f"    {csv_file}"
    )

    print()
    print(
        "Output TXT:"
    )

    print(
        f"    {txt_file}"
    )

    print()
    print(
        "DONE."
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
