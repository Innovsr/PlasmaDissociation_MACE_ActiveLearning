#import time
#import os
#import numpy as np
#from ase import Atoms
#from mace.calculators import MACECalculator
#
#
## ============================================================
## EDIT THIS PATH
## ============================================================
#
##MODEL_PATH = "/users/lgjs798/MACE_model/model_to_use/AIMD_V01_5000_Cyclo-E52.9meV_F586meV.model"
## ============================================================
## MACE SETTINGS PASSED BY THE VALIDATION SCRIPT
## ============================================================
#
#MODEL_PATH = os.environ.get(
#    "MACE_MODEL_PATH",
#    "/users/lgjs798/MACE_model/model_to_use/AIMD_V01_5000_Cyclo-E52.9meV_F586meV.model"
#)
#
#MACE_DEVICE = os.environ.get(
#    "MACE_DEVICE",
#    "cuda"
#)
#
#MACE_DTYPE = os.environ.get(
#    "MACE_DTYPE",
#    "float64"
#)
#
#
## ============================================================
## Unit conversion
## ============================================================
#
#BOHR_TO_ANG = 0.529177210903
#EV_TO_HARTREE = 1.0 / 27.211386245988
#
## 1 Hartree/Bohr = 51.422067 eV/Angstrom
#EVANG_TO_HARTREEBOHR = 1.0 / 51.422067
#
#
## ============================================================
## Load model ONCE
## ============================================================
#
##calculator = MACECalculator(
##    model_paths=MODEL_PATH,
##    device="cuda",
##    default_dtype="float64"
##)
#
#print(
#    f"Loading dynamic MACE model: {MODEL_PATH}",
#    flush=True
#)
#
#print(
#    f"MACE device: {MACE_DEVICE}",
#    flush=True
#)
#
#print(
#    f"MACE dtype: {MACE_DTYPE}",
#    flush=True
#)
#
#calculator = MACECalculator(
#    model_paths=MODEL_PATH,
#    device=MACE_DEVICE,
#    default_dtype=MACE_DTYPE
#)
#
#def run_mace(molecule):
#
#    start = time.time()
#
#    # --------------------------------------------------------
#    # PlasmaDissociation coordinates are in Bohr.
#    # Convert Bohr -> Angstrom for MACE/ASE.
#    # --------------------------------------------------------
#
#    positions_ang = (
#        np.asarray(molecule.coordinates, dtype=float)
#        * BOHR_TO_ANG
#    )
#
#    atoms = Atoms(
#        symbols=molecule.symbols,
#        positions=positions_ang
#    )
#
#    atoms.calc = calculator
#
#    # --------------------------------------------------------
#    # MACE predictions
#    # Usually:
#    # energy -> eV
#    # forces -> eV / Angstrom
#    # --------------------------------------------------------
#    t0 = time.time()
#
#    energy_ev = atoms.get_potential_energy()
#    forces_ev_ang = atoms.get_forces()
#
#    print(f"MACE inference only: {time.time() - t0:.4f} s", flush=True)
#
#    # --------------------------------------------------------
#    # Convert back to atomic units expected by propagation code
#    # --------------------------------------------------------
#
#    energy_hartree = energy_ev * EV_TO_HARTREE
#
#    forces_hartree_bohr = (
#        forces_ev_ang * EVANG_TO_HARTREEBOHR
#    )
#
#    # --------------------------------------------------------
#    # PlasmaDissociation expects:
#    #
#    # scf_energy shape = (num_states,)
#    # forces shape = (natoms, 3, num_states)
#    #
#    # For your single-state trajectory use state 0.
#    # --------------------------------------------------------
#
#    molecule.scf_energy[0] = energy_hartree
#    molecule.forces[:, :, 0] = forces_hartree_bohr
#
#    elapsed = time.time() - start
#
#    # Keep timing structure compatible with old code
#    if molecule.time is not None:
#        molecule.time[0] += elapsed
#        molecule.time[2] += elapsed
#
#    print(
#        f"MACE: E = {energy_ev:.12f} eV "
#        f"({energy_hartree:.12f} Hartree)"
#    )
#
#    return molecule


import os
import time

import numpy as np
from ase import Atoms
from mace.calculators import MACECalculator


# The validation script passes this environment variable to every dynamic run.
MODEL_PATH = os.environ.get(
    "MACE_MODEL_PATH",
    "/users/lgjs798/MACE_model/model_to_use/"
    "AIMD_V01_5000_Cyclo-E52.9meV_F586meV.model",
)
MACE_DEVICE = os.environ.get("MACE_DEVICE", "cuda")
MACE_DTYPE = os.environ.get("MACE_DTYPE", "float64")

BOHR_TO_ANG = 0.529177210903
EV_TO_HARTREE = 1.0 / 27.211386245988
EVANG_TO_HARTREEBOHR = 1.0 / 51.4220674763


calculator = MACECalculator(
    model_paths=MODEL_PATH,
    device=MACE_DEVICE,
    default_dtype=MACE_DTYPE,
)


def _active_atom_indices(molecule, total_atoms, expected_active_atoms):
    """Mirror the active-atom selection used by qchem.py and elec_pyscf.py."""
    flags = list(molecule.dissociation_flags)
    if len(flags) != total_atoms:
        raise RuntimeError(
            "Dissociation-flag mismatch: "
            f"{len(flags)} flags for {total_atoms} stored atoms."
        )

    active_indices = np.asarray(
        [i for i, flag in enumerate(flags) if flag == "NO"],
        dtype=int,
    )

    if len(active_indices) != expected_active_atoms:
        raise RuntimeError(
            "Active-atom mismatch after applying dissociation_flags: "
            f"flags select {len(active_indices)} atoms, but molecule.forces "
            f"expects {expected_active_atoms}. Flags: {flags}"
        )

    return active_indices


def run_mace(molecule):
    start = time.time()

    all_positions_bohr = np.asarray(molecule.coordinates, dtype=float)
    all_symbols = list(molecule.symbols)

    if all_positions_bohr.ndim != 2 or all_positions_bohr.shape[1] != 3:
        raise ValueError(
            f"Expected coordinates with shape (natoms, 3), got {all_positions_bohr.shape}"
        )
    if len(all_symbols) != len(all_positions_bohr):
        raise ValueError(
            f"Symbol/coordinate mismatch: {len(all_symbols)} symbols and "
            f"{len(all_positions_bohr)} coordinate rows"
        )

    expected_active_atoms = int(molecule.forces.shape[0])
    active_indices = _active_atom_indices(
        molecule,
        total_atoms=len(all_symbols),
        expected_active_atoms=expected_active_atoms,
    )

    active_positions_ang = all_positions_bohr[active_indices] * BOHR_TO_ANG
    active_symbols = [all_symbols[index] for index in active_indices]

    atoms = Atoms(symbols=active_symbols, positions=active_positions_ang)
    atoms.calc = calculator

    t0 = time.time()
    energy_ev = float(atoms.get_potential_energy())
    forces_ev_ang = np.asarray(atoms.get_forces(), dtype=float)
    print(f"MACE inference only: {time.time() - t0:.4f} s", flush=True)

    if forces_ev_ang.shape != (expected_active_atoms, 3):
        raise RuntimeError(
            "Unexpected MACE force shape: "
            f"received {forces_ev_ang.shape}, expected {(expected_active_atoms, 3)}"
        )

    energy_hartree = energy_ev * EV_TO_HARTREE
    forces_hartree_bohr = forces_ev_ang * EVANG_TO_HARTREEBOHR

    molecule.scf_energy[0] = energy_hartree
    molecule.forces[:, :, 0] = forces_hartree_bohr

    elapsed = time.time() - start
    if molecule.time is not None:
        molecule.time[0] += elapsed
        molecule.time[2] += elapsed

    print(
        f"MACE: active atoms = {expected_active_atoms}/{len(all_symbols)} | "
        f"E = {energy_ev:.12f} eV ({energy_hartree:.12f} Hartree)",
        flush=True,
    )

    return molecule

