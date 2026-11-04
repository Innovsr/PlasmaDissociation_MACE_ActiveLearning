"""
setup.py - BOMD initial-condition generation with optional momentum screening.

Selection mode:
    1. Q-Chem calculates the optimised geometry and normal modes.
    2. A large number of independent momentum samples is generated in memory.
    3. For every initially bonded pair, calculate the initial radial velocity:
           v_rad = (p_i/m_i - p_j/m_j) . rhat_ij
    4. For each user category (C-C, C-F, C-H, ...), keep the largest
       POSITIVE v_rad for each candidate sample.
    5. Select the requested number of candidate samples in each category.
       A candidate is used only once, so the selected trajectories are unique.
    6. ONLY the selected samples are written as rep-1, rep-2, ...
    7. Candidate/selection information is written to selection/.

If Trajectory_selection.enabled = 0, the original behaviour is retained:
    HPC.repeats momentum samples are generated and rep-1 ... rep-N are written.
"""

import os
import sys
import json
import csv
import shutil
from pathlib import Path

import numpy as np
import pubchempy as pcp
from scipy.spatial.distance import pdist, squareform

import init


# ======================================================================
# GENERAL UTILITIES
# ======================================================================

def log(message):
    """Print immediately so progress is visible in the Slurm .out file."""
    print(message, flush=True)


def write_error(run_root, message):
    """Append a non-fatal warning/error to the selection log."""
    result_dir = Path(run_root) / "selection"
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "trajectory_selection_error.txt", "a") as f:
        f.write(message.rstrip() + "\n")


# ======================================================================
# ORIGINAL MOMENTUM SAMPLING
# ======================================================================

def create_geom(n, nmod, T, modes, m, mom_num):
    """
    Generate mom_num independent initial momentum distributions.

    The existing normal-mode sampling used by the BOMD code is retained.
    rn ~ N(0,1) for each normal mode and candidate sample.
    """
    Ax = modes[:, 0].reshape(n, nmod, order="F")
    Ay = modes[:, 1].reshape(n, nmod, order="F")
    Az = modes[:, 2].reshape(n, nmod, order="F")

    rn = np.random.randn(nmod, mom_num)

    # Kelvin -> atomic units of energy.
    T_au = T * 0.0000031668

    Meff = np.zeros(nmod)
    rv = np.zeros((nmod, mom_num))

    for i in range(nmod):
        for j in range(n):
            Meff[i] += (
                Ax[j, i] ** 2
                + Ay[j, i] ** 2
                + Az[j, i] ** 2
            ) * m[j]

        rv[i, :] = rn[i, :] * np.sqrt(2.0 * T_au / Meff[i])

    Vx = np.dot(Ax, rv)
    Vy = np.dot(Ay, rv)
    Vz = np.dot(Az, rv)

    Px = Vx * m[:, None]
    Py = Vy * m[:, None]
    Pz = Vz * m[:, None]

    return Px, Py, Pz


# ======================================================================
# GEOMETRY
# ======================================================================

def get_geometry_file():
    symbols = []
    coordinates = []

    with open("initial_guess.txt", "r") as file:
        for line in file:
            parts = line.split()
            if len(parts) == 4:
                symbols.append(parts[0])
                x, y, z = map(float, parts[1:])
                coordinates.append(
                    [x * 1.88973, y * 1.88973, z * 1.88973]
                )

    return symbols, coordinates


def get_geometry_pubchem(molecule_name):
    search = pcp.get_compounds(
        molecule_name,
        "name",
        record_type="3d"
    )
    molecule = search[0]

    bonds = molecule.bonds
    atoms = molecule.atoms

    atom_id_to_symbol = {
        atom.aid: atom.element for atom in atoms
    }

    symbols = [atom.element for atom in atoms]
    coordinates = [
        [atom.x * 1.88973, atom.y * 1.88973, atom.z * 1.88973]
        for atom in atoms
    ]

    with open("../results/bondarr.txt", "w") as file:
        for bond in bonds:
            atom1 = bond.aid1
            atom2 = bond.aid2
            order = bond.order

            if order == 1:
                bond_type = "-"
            elif order == 2:
                bond_type = "="
            elif order == 3:
                bond_type = "#"
            else:
                bond_type = "-"

            atom1_symbol = atom_id_to_symbol.get(atom1, "Unknown")
            atom2_symbol = atom_id_to_symbol.get(atom2, "Unknown")

            file.write(
                f"{atom1}-{atom2}:{atom1_symbol}{bond_type}{atom2_symbol}\n"
            )

    return symbols, coordinates


# ======================================================================
# INITIAL BOND DETECTION
# ======================================================================

BOND_THRESHOLDS = {
    ("C", "C"): [(2.7401, "C=C"), (3.25, "C-C")],
    ("C", "F"): [(2.8346, "C-F")],
    ("C", "H"): [(2.1732, "C-H")],
    ("C", "O"): [(2.75, "C-O")],
    ("O", "H"): [(2.0, "O-H")],
}


def generate_bondarr(symbols, coordinates):
    """
    Determine the initially bonded atom pairs from the initial geometry.
    Atom numbers are 1-based, matching the existing dissociation output.
    """
    distances = squareform(pdist(coordinates))
    bonds = []

    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            key = tuple(sorted((symbols[i], symbols[j])))
            d = distances[i, j]

            if key not in BOND_THRESHOLDS:
                continue

            for max_dist, bond_type in BOND_THRESHOLDS[key]:
                if d <= max_dist:
                    bonds.append((i + 1, j + 1, bond_type))
                    break

    return bonds


# ======================================================================
# NORMAL-MODE ORGANISATION
# ======================================================================

def organise_modes(modes, atoms):
    numeric_modes = np.zeros((len(modes) * atoms, 3))

    cnt = 0
    for mode in modes:
        for j in range(atoms):
            numeric_modes[cnt, 0] = float(
                mode["displacement"][j][0]
            )
            numeric_modes[cnt, 1] = float(
                mode["displacement"][j][1]
            )
            numeric_modes[cnt, 2] = float(
                mode["displacement"][j][2]
            )
            cnt += 1

    return numeric_modes



def momenta_checks(Px, Py, Pz, symbols, temperature, output_file):
    """
    Preserve the original momentum sanity check used by setup.py.

    Reports total kinetic energy for every generated trajectory/sample
    and the average momentum magnitude per atom.
    """
    natoms = Px.shape[0]
    repeats = Px.shape[1]
    masses = init.setup_masses(symbols)

    kinetic_energies = np.zeros(repeats)

    for j in range(repeats):
        ke = 0.0
        for i in range(natoms):
            p2 = (
                Px[i, j] ** 2
                + Py[i, j] ** 2
                + Pz[i, j] ** 2
            )
            ke += 0.5 * p2 / masses[i]
        kinetic_energies[j] = ke

    average_ke = np.mean(kinetic_energies)
    expected_ke = (
        (3 * natoms - 6)
        * 3.16881e-6
        * temperature
    )

    avg_momentum = np.mean(
        np.sqrt(Px**2 + Py**2 + Pz**2),
        axis=1
    )

    with open(output_file, "w") as file:
        file.write("Kinetic Energy Checks:\n")
        file.write(
            f"Expected Kinetic Energy: {expected_ke:.6f}\n"
        )
        file.write(
            f"Average Kinetic Energy: {average_ke:.6f}\n"
        )
        file.write(
            "Average Momentum Magnitude per Atom:\n"
        )

        for i in range(natoms):
            file.write(
                f"Atom {i+1} ({symbols[i]}): "
                f"{avg_momentum[i]:.6f}\n"
            )

        file.write("\nIndividual samples:\n")

        for j in range(repeats):
            file.write(
                f"Sample {j+1}: "
                f"Kinetic Energy = "
                f"{kinetic_energies[j]:.6f}\n"
            )

# ======================================================================
# BOND-SPECIFIC RADIAL VELOCITY
# ======================================================================

def bond_vrad(Px, Py, Pz, masses, coordinates, bond):
    """
    Calculate initial bond-directed relative velocity:

       v_i = p_i / m_i
       v_j = p_j / m_j

       rhat_ij = (r_i-r_j)/|r_i-r_j|

       v_rad = (v_i-v_j) . rhat_ij

    Positive v_rad means the two atoms initially move apart.
    """
    i = bond[0] - 1
    j = bond[1] - 1

    rij = np.asarray(coordinates[i]) - np.asarray(coordinates[j])
    distance = np.linalg.norm(rij)

    if distance == 0:
        raise ValueError(
            f"Zero initial distance for bond {bond[0]}-{bond[1]}"
        )

    rhat = rij / distance

    vi = np.column_stack(
        (Px[i, :] / masses[i],
         Py[i, :] / masses[i],
         Pz[i, :] / masses[i])
    )

    vj = np.column_stack(
        (Px[j, :] / masses[j],
         Py[j, :] / masses[j],
         Pz[j, :] / masses[j])
    )

    return np.sum((vi - vj) * rhat[None, :], axis=1)


# ======================================================================
# USER BOND SELECTION
# ======================================================================

def normalise_bond_pair(pair):
    """Accept [2,4], (2,4), or '2-4' and return (2,4)."""
    if isinstance(pair, str):
        parts = pair.replace(" ", "").split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid bond specification: {pair}")
        return tuple(sorted((int(parts[0]), int(parts[1]))))

    if len(pair) != 2:
        raise ValueError(f"Invalid bond specification: {pair}")

    return tuple(sorted((int(pair[0]), int(pair[1]))))


def bond_category_from_label(bond_type):
    """Convert C=C, C-C, C#C, etc. to the user category C-C."""
    label = bond_type.strip().upper()
    for symbol in ("=", "#", "-"):
        if symbol in label:
            left, right = label.split(symbol, 1)
            return "-".join(sorted((left, right)))
    return label


def prepare_categories(selection_cfg, bonds):
    """
    Convert the compact user specification into actual bond lists.

    bonds: [] means ALL initially detected bonds of that type.
    C=C is treated as the C-C category because the selection is based
    on the two atom types, not on the bond order.
    """
    detected = {}

    for i, j, bond_type in bonds:
        key = bond_category_from_label(bond_type)
        detected.setdefault(key, []).append((i, j))

    categories = []

    for category in selection_cfg.get("categories", []):
        name = category["name"].strip().upper()
        requested = int(category.get("select", 0))

        if requested < 1:
            continue

        explicit = category.get("bonds", [])

        if len(explicit) == 0:
            bond_list = detected.get(name, [])
        else:
            bond_list = [
                normalise_bond_pair(pair)
                for pair in explicit
            ]

        categories.append({
            "name": name,
            "select": requested,
            "bonds": bond_list
        })

    return categories


# ======================================================================
# WRITE MOMENTUM SAMPLES
# ======================================================================

def write_selected_geometry(
    repdir,
    opt_coords,
    Px,
    Py,
    Pz,
    sample_index,
    category,
    selected_bond,
    vrad
):
    repdir = Path(repdir)

    repdir.mkdir(parents=True, exist_ok=False)
    (repdir / "output").mkdir()
    (repdir / "tmp").mkdir()

    with open(repdir / "Geometry", "w") as file:
        file.write(opt_coords)
        file.write("momentum\n")

        for atom in range(Px.shape[0]):
            file.write(
                f"{Px[atom, sample_index]:.15g}  "
                f"{Py[atom, sample_index]:.15g}  "
                f"{Pz[atom, sample_index]:.15g}\n"
            )

    with open(repdir / "selection_info.txt", "w") as file:
        file.write(f"Candidate sample: {sample_index + 1}\n")
        file.write(f"Selection category: {category}\n")
        file.write(
            f"Selected bond: "
            f"{selected_bond[0]}-{selected_bond[1]}\n"
        )
        file.write(f"Initial positive Vrad: {vrad:.12g}\n")


# ======================================================================
# SELECTION MODE
# ======================================================================

def run_selection(
    inputs,
    symbols,
    coordinates,
    opt_coords,
    modes,
    masses,
    initial_bonds,
    run_root
):
    result_dir = Path(run_root) / "selection"
    repetitions_dir = Path(run_root) / "repetitions"

    selection_cfg = inputs["Trajectory_selection"]

    candidate_count = int(
        selection_cfg.get("momentum_samples", 0)
    )

    categories = prepare_categories(
        selection_cfg,
        initial_bonds
    )

    requested_total = sum(
        category["select"] for category in categories
    )

    log("")
    log("=" * 70)
    log("TRAJECTORY SELECTION MODE")
    log("=" * 70)
    log(f"Temperature                : "
        f"{inputs['Molecule_data']['Temp']} K")
    log(f"Candidate momentum samples : {candidate_count}")
    log(f"Requested BOMD trajectories : {requested_total}")
    log("")

    for category in categories:
        log(
            f"  {category['name']} : "
            f"{category['select']} selected, "
            f"{len(category['bonds'])} bond(s) screened"
        )

    if candidate_count < requested_total:
        write_error(
            run_root,
            "WARNING: momentum_samples is smaller than the "
            "number of requested selected trajectories. "
            "Selection may be unable to provide unique samples."
        )

    log("-" * 70)
    log("STEP 1: Generating candidate momentum distributions")
    log("-" * 70)

    Px, Py, Pz = create_geom(
        len(symbols),
        len(modes) // len(symbols),
        inputs["Molecule_data"]["Temp"],
        modes,
        masses,
        candidate_count
    )

    log(
        f"Generated {candidate_count} independent momentum samples."
    )

    # ---------------------------------------------------------------
    # Calculate Vrad for every bond.
    # ---------------------------------------------------------------
    log("-" * 70)
    log("STEP 2: Calculating bond-specific initial radial velocities")
    log("-" * 70)

    vrad_data = {}

    all_bonds = []
    for category in categories:
        for bond in category["bonds"]:
            if bond not in all_bonds:
                all_bonds.append(bond)

    for bond in all_bonds:
        log(
            f"Calculating Vrad for bond "
            f"{bond[0]}-{bond[1]} ..."
        )

        vrad_data[bond] = bond_vrad(
            Px,
            Py,
            Pz,
            masses,
            coordinates,
            bond
        )

    # ---------------------------------------------------------------
    # Select candidates.
    #
    # For each category and candidate:
    #     category score = largest POSITIVE Vrad among its bonds.
    #
    # The bond producing this value is recorded.
    #
    # A sample can only be selected once globally.
    # ---------------------------------------------------------------
    log("-" * 70)
    log("STEP 3: Selecting trajectories")
    log("-" * 70)

    selected = []
    used_samples = set()

    for category in categories:

        scores = np.full(candidate_count, -np.inf)
        best_bonds = [None] * candidate_count

        for bond in category["bonds"]:

            values = vrad_data[bond]

            positive = values > 0

            update = positive & (values > scores)

            scores[update] = values[update]
            for idx in np.where(update)[0]:
                best_bonds[idx] = bond

        # Rank candidates by positive Vrad.
        available = [
            idx for idx in range(candidate_count)
            if idx not in used_samples
            and np.isfinite(scores[idx])
        ]

        available.sort(
            key=lambda idx: scores[idx],
            reverse=True
        )

        chosen = available[:category["select"]]

        # If not enough positive candidates exist, write a warning and
        # fill remaining requested slots with the best unused samples.
        # This keeps the job running and makes the issue explicit.
        if len(chosen) < category["select"]:

            warning = (
                f"WARNING: Category {category['name']} requested "
                f"{category['select']} positive-Vrad trajectories, "
                f"but only {len(chosen)} were available. "
                f"The remaining trajectories will be selected from "
                f"the highest available Vrad values, including "
                f"non-positive values."
            )

            log(warning)
            write_error(run_root, warning)

            fallback_scores = np.full(
                candidate_count,
                -np.inf
            )

            fallback_bonds = [None] * candidate_count

            for bond in category["bonds"]:

                values = vrad_data[bond]

                update = (
                    ~np.isin(
                        np.arange(candidate_count),
                        list(used_samples)
                    )
                    & (values > fallback_scores)
                )

                fallback_scores[update] = values[update]

                for idx in np.where(update)[0]:
                    fallback_bonds[idx] = bond

            fallback = [
                idx for idx in range(candidate_count)
                if idx not in used_samples
                and np.isfinite(fallback_scores[idx])
            ]

            fallback.sort(
                key=lambda idx: fallback_scores[idx],
                reverse=True
            )

            needed = category["select"] - len(chosen)

            for idx in fallback:
                if idx not in chosen:
                    chosen.append(idx)

                if len(chosen) >= category["select"]:
                    break

            for idx in chosen:
                if best_bonds[idx] is None:
                    best_bonds[idx] = fallback_bonds[idx]
                    scores[idx] = fallback_scores[idx]

        log(
            f"{category['name']}: requested "
            f"{category['select']} -> selected {len(chosen)}"
        )

        for idx in chosen:
            used_samples.add(idx)

            selected.append({
                "sample_index": idx,
                "category": category["name"],
                "bond": best_bonds[idx],
                "vrad": float(scores[idx])
            })

    # ---------------------------------------------------------------
    # Check actual number.
    # ---------------------------------------------------------------
    if len(selected) != requested_total:
        write_error(
            run_root,
            f"ERROR: Requested {requested_total} trajectories but "
            f"only {len(selected)} could be selected."
        )

    log(f"Total selected trajectories: {len(selected)}")

    # ---------------------------------------------------------------
    # Write detailed CSV containing every candidate sample.
    # ---------------------------------------------------------------
    log("-" * 70)
    log("STEP 4: Writing screening data")
    log("-" * 70)

    bond_columns = [
        f"{bond[0]}-{bond[1]}_Vrad"
        for bond in all_bonds
    ]

    csv_file = result_dir / "trajectory_selection.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "sample_index",
                "selected",
                "selected_category",
                "selected_bond",
                "selected_vrad"
            ] + bond_columns
        )

        selected_by_sample = {
            item["sample_index"]: item
            for item in selected
        }

        for idx in range(candidate_count):

            item = selected_by_sample.get(idx)

            row = [
                idx + 1,
                "YES" if item else "NO",
                item["category"] if item else "",
                (
                    f"{item['bond'][0]}-{item['bond'][1]}"
                    if item else ""
                ),
                item["vrad"] if item else ""
            ]

            for bond in all_bonds:
                row.append(float(vrad_data[bond][idx]))

            writer.writerow(row)

    # ---------------------------------------------------------------
    # Human-readable summary.
    # ---------------------------------------------------------------
    summary_file = (
        result_dir / "trajectory_selection.txt"
    )

    with open(summary_file, "w") as f:

        f.write("TRAJECTORY SELECTION SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(
            f"Temperature: "
            f"{inputs['Molecule_data']['Temp']} K\n"
        )
        f.write(
            f"Candidate momentum samples: "
            f"{candidate_count}\n"
        )
        f.write(
            f"Requested trajectories: "
            f"{requested_total}\n"
        )
        f.write(
            f"Selected trajectories: "
            f"{len(selected)}\n\n"
        )

        for number, item in enumerate(selected, 1):

            f.write(
                f"rep-{number:04d} | "
                f"sample {item['sample_index'] + 1:06d} | "
                f"{item['category']:5s} | "
                f"bond {item['bond'][0]}-{item['bond'][1]} | "
                f"Vrad = {item['vrad']:.12g}\n"
            )

    # ---------------------------------------------------------------
    # Write all candidate momenta.
    #
    # Long-format CSV:
    # one row = one atom in one candidate sample.
    # ---------------------------------------------------------------
    momentum_file = (
        result_dir / "momentum_samples.csv"
    )

    with open(momentum_file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "sample_index",
                "atom_index",
                "element",
                "Px",
                "Py",
                "Pz"
            ]
        )

        for sample in range(candidate_count):
            for atom in range(len(symbols)):

                writer.writerow(
                    [
                        sample + 1,
                        atom + 1,
                        symbols[atom],
                        f"{Px[atom, sample]:.15g}",
                        f"{Py[atom, sample]:.15g}",
                        f"{Pz[atom, sample]:.15g}"
                    ]
                )

    # ---------------------------------------------------------------
    # Write selected momenta in an easy-to-read text file.
    # ---------------------------------------------------------------
    selected_file = (
        result_dir / "selected_momenta.txt"
    )

    with open(selected_file, "w") as f:

        for rep_number, item in enumerate(selected, 1):

            idx = item["sample_index"]

            f.write("=" * 70 + "\n")
            f.write(f"rep-{rep_number}\n")
            f.write(
                f"Original candidate sample: {idx + 1}\n"
            )
            f.write(
                f"Category: {item['category']}\n"
            )
            f.write(
                f"Selected bond: "
                f"{item['bond'][0]}-{item['bond'][1]}\n"
            )
            f.write(
                f"Vrad: {item['vrad']:.15g}\n"
            )
            f.write("\n")
            f.write(
                "Atom Element Px Py Pz\n"
            )

            for atom in range(len(symbols)):

                f.write(
                    f"{atom + 1:4d} "
                    f"{symbols[atom]:>2s} "
                    f"{Px[atom, idx]: .15g} "
                    f"{Py[atom, idx]: .15g} "
                    f"{Pz[atom, idx]: .15g}\n"
                )

    # ---------------------------------------------------------------
    # Create ONLY selected repetition directories.
    # ---------------------------------------------------------------
    log("-" * 70)
    log("STEP 5: Writing selected BOMD trajectories")
    log("-" * 70)

    # Remove any old rep-* directories. This prevents an old 200-run
    # calculation from accidentally being submitted with the new setup.
    for old_rep in repetitions_dir.glob("rep-*"):
        if old_rep.is_dir():
            shutil.rmtree(old_rep)

    for rep_number, item in enumerate(selected, 1):

        repdir = (
            repetitions_dir /
            f"rep-{rep_number}"
        )

        idx = item["sample_index"]

        write_selected_geometry(
            repdir,
            opt_coords,
            Px,
            Py,
            Pz,
            idx,
            item["category"],
            item["bond"],
            item["vrad"]
        )

        log(
            f"rep-{rep_number} <- "
            f"sample {idx + 1} | "
            f"{item['category']} | "
            f"bond {item['bond'][0]}-{item['bond'][1]} | "
            f"Vrad={item['vrad']:.6g}"
        )

    log("")
    log("=" * 70)
    log("SELECTION COMPLETED")
    log("=" * 70)
    log(
        f"Candidate samples generated : {candidate_count}"
    )
    log(
        f"Selected BOMD trajectories  : {len(selected)}"
    )
    log(
        f"Results directory           : {result_dir}"
    )
    log("")
    log("Output files:")
    log(f"  {csv_file}")
    log(f"  {summary_file}")
    log(f"  {momentum_file}")
    log(f"  {selected_file}")
    log("=" * 70)

    return len(selected)


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":

    log("=" * 70)
    log("BOMD SETUP STARTED")
    log("=" * 70)

    #with open("../inputs.json") as f:
    with open("../validation.json") as f:
        inputs = json.load(f)

    run_root = Path("..").resolve()
    result_dir = run_root / "selection"
    repetitions_dir = run_root / "repetitions"

    result_dir.mkdir(exist_ok=True)
    repetitions_dir.mkdir(exist_ok=True)

    # This setup stage always uses Q-Chem to optimise the initial geometry and generate normal modes.
    method = "QChem"
    BASIS = "DEF2-TZVP"

    log(f"Electronic structure method: {method}")
    log(
        f"Temperature: "
        f"{inputs['Molecule_data']['Temp']} K"
    )

    # ---------------------------------------------------------------
    # Geometry.
    # ---------------------------------------------------------------
    log("-" * 70)
    log("Reading initial geometry")
    log("-" * 70)

    # This package always uses the local Initial_Geometry file copied into setup/initial_guess.txt.
    symbols, coordinates = get_geometry_file()

    log(f"Number of atoms: {len(symbols)}")

    # ---------------------------------------------------------------
    # Electronic structure / normal modes.
    # ---------------------------------------------------------------
    log("-" * 70)
    log("Running initial electronic-structure calculation")
    log("-" * 70)

    if method == "QChem":

        import qchem as qc

        opt_coords, modes_result = qc.initial_conditions(
            symbols,
            coordinates,
            inputs["HPC"]["cores"],
            BASIS
        )

        modes = modes_result["modes"]

    elif method == "PySCF":

        import elec_pyscf as pyscf

        pyscf_result = pyscf.initial_conditions(
            symbols,
            coordinates
        )

        if isinstance(pyscf_result, tuple):
            opt_coords, modes_result = pyscf_result
            modes = (
                modes_result["modes"]
                if isinstance(modes_result, dict)
                else modes_result
            )
        else:
            raise RuntimeError(
                "PySCF initial_conditions() did not return "
                "optimised geometry and normal modes."
            )

    else:

        raise ValueError(
            f"Unknown electronic structure method: {method}"
        )

    modes = organise_modes(
        modes,
        len(symbols)
    )

    num_modes = len(modes) // len(symbols)

    masses = init.setup_masses(symbols)

    # ---------------------------------------------------------------
    # Selection switch.
    # ---------------------------------------------------------------
    selection_cfg = inputs.get(
        "Trajectory_selection",
        {}
    )

    selection_enabled = int(
        selection_cfg.get("enabled", 0)
    )

    if selection_enabled == 1:

        log("")
        log(
            "Trajectory selection is ENABLED."
        )

        run_selection(
            inputs,
            symbols,
            coordinates,
            opt_coords,
            modes,
            masses,
            generate_bondarr(
                symbols,
                coordinates
            ),
            run_root
        )

    else:

        # -----------------------------------------------------------
        # ORIGINAL BEHAVIOUR
        # -----------------------------------------------------------
        log("")
        log(
            "Trajectory selection is DISABLED."
        )
        log(
            f"Generating "
            f"{inputs['HPC']['repeats']} momentum samples."
        )

        Px, Py, Pz = create_geom(
            len(symbols),
            num_modes,
            inputs["Molecule_data"]["Temp"],
            modes,
            masses,
            inputs["HPC"]["repeats"]
        )

        # Remove any stale repetitions before writing the normal set.
        for old_rep in repetitions_dir.glob("rep-*"):
            if old_rep.is_dir():
                shutil.rmtree(old_rep)

        for j in range(inputs["HPC"]["repeats"]):

            repdir = repetitions_dir / f"rep-{j + 1}"

            repdir.mkdir()
            (repdir / "output").mkdir()
            (repdir / "tmp").mkdir()

            with open(repdir / "Geometry", "w") as file:

                file.write(opt_coords)
                file.write("momentum\n")

                for atom in range(len(symbols)):

                    file.write(
                        f"{Px[atom, j]:.15g}  "
                        f"{Py[atom, j]:.15g}  "
                        f"{Pz[atom, j]:.15g}\n"
                    )

            log(
                f"Written rep-{j + 1} "
                f"({j + 1}/{inputs['HPC']['repeats']})"
            )

        log(
            "Original setup completed successfully."
        )
