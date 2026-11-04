# PlasmaDissociation-MACE Active Learning v2

Three-stage workflow with one common scratch Runfolder.

## Stages

1. `./01_validation/run_validation.sh` reads `configs/validation.json`, selects the requested trajectories, runs Q-Chem BOMD, and writes `01_validation/validation/validation.extxyz` when all selected trajectories reach the requested length.
2. `sbatch 02_model_update/run_model_update.sh` reads `configs/update_model.json`, validates the source model, selects high-error configurations, builds the old:new training set, trains/resumes a candidate, and leaves it in scratch for explicit approval.
3. `./03_production/run_production.sh` reads `configs/production.json`, requires an explicitly approved model in `models/`, and runs all requested production trajectories with MACE.

All large run output is under:

`/mnt/scratch/$USER/<Runfolder>/`

with `01_validation/`, `02_model_update/`, and `03_production/`.

Restart is data-driven: `restart_manager.py` checks the common timestep in `xyz.all`, `momenta.all`, `forces.all`, and `time.all`. It compares the physical trajectory timestep against `Tot_timesteps * Timestep`; no completion marker is used.

## Important prerequisite

The included `.model` file is retained as the existing source model, but the package does not invent approval status, training archives, or checkpoints. Before model update, the source model must be installed as an approved model directory with its preserved training archive and checkpoint history.


## What the JSON files do

- `configs/validation.json`: Stage-1 Q-Chem initial-condition generation, trajectory-selection settings, and validation BOMD length/resources.
- `configs/update_model.json`: source model, old:new data ratio, RMSE targets, MACE architecture/training settings, and Stage-2 resources.
- `configs/production.json`: approved model, production molecule/temperature, number of trajectories, propagation length, and Stage-3 resources.

Fixed single-surface BOMD settings are internal: `Initial` geometry, one state, start state 1, spin-flip off, GPU off, branching off, and the DEF2-TZVP setup basis.
