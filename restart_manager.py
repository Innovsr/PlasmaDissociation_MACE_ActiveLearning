#!/usr/bin/env python3
"""Data-driven restart manager for PlasmaDissociation BOMD trajectories.

A trajectory is considered complete only when the latest common timestep in
xyz.all, momenta.all, forces.all and time.all reaches the requested number of
propagation steps. No completion marker file is used.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path

FILES = ("xyz.all", "momenta.all", "forces.all", "time.all")
TS_RE = re.compile(r"^\s*Timestep:\s*(-?\d+(?:\.\d+)?)\s*$")

def timesteps(path: Path) -> set[float]:
    out=set()
    if not path.exists():
        return out
    with path.open("r", errors="replace") as f:
        for line in f:
            m=TS_RE.match(line)
            if m:
                out.add(float(m.group(1)))
    return out

def latest_common(repdir: Path) -> float | None:
    sets=[]
    for name in FILES:
        s=timesteps(repdir/"output"/name)
        if not s:
            return None
        sets.append(s)
    common=sets[0]
    for s in sets[1:]:
        common &= s
        if not common:
            return None
    return max(common)

def status(repdir: Path, target_steps: float) -> dict:
    current=latest_common(repdir)
    return {
        "rep": repdir.name,
        "current_timestep": current,
        "target_timestep": target_steps,
        "complete": current is not None and current >= target_steps,
        "has_trajectory_files": all((repdir/"output"/n).exists() for n in FILES),
    }

def all_complete(run_root: Path, target_steps: int) -> bool:
    reps=sorted((run_root/"repetitions").glob("rep-*"))
    if not reps:
        return False
    states=[status(r,target_steps) for r in reps]
    return all(x["complete"] for x in states)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-root", required=True)
    p.add_argument("--target-steps", required=True, type=int, help="Number of integration steps")
    p.add_argument("--timestep", required=True, type=float, help="Physical timestep used in trajectory files")
    p.add_argument("--all-complete", action="store_true")
    p.add_argument("--status", action="store_true")
    a=p.parse_args()
    root=Path(a.run_root).resolve()
    if a.status or a.all_complete:
        reps=sorted((root/"repetitions").glob("rep-*"))
        states=[status(r,a.target_steps * a.timestep) for r in reps]
        for x in states:
            print(f"{x['rep']}: current={x['current_timestep']} target={x['target_timestep']} complete={x['complete']}")
        if a.all_complete:
            raise SystemExit(0 if states and all(x["complete"] for x in states) else 1)
if __name__=="__main__":
    main()
