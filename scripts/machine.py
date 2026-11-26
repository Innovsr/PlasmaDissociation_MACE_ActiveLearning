#!/usr/bin/env python3

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MACHINES_FILE = ROOT / "configs" / "machines.json"


def load_machines():
    with open(MACHINES_FILE) as f:
        return json.load(f)


def get_machine(cfg):
    machine = cfg["HPC"].get("machine", "aire").lower()

    machines = load_machines()

    if machine not in machines:
        raise ValueError(
            f"Unknown machine '{machine}'. "
            f"Available machines: {', '.join(machines)}"
        )

    return machine, machines[machine]


def get_scratch_root(cfg):
    machine, settings = get_machine(cfg)

    root = settings["scratch_root"]

    root = root.replace(
        "$USER",
        os.environ.get("USER", os.environ.get("LOGNAME", "unknown"))
    )

    return Path(root)


def get_run_stage(cfg, stage):
    return (
        get_scratch_root(cfg)
        / cfg["HPC"]["Runfolder"]
        / stage
    )
