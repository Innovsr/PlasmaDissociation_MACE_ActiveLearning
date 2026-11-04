#!/usr/bin/env python3
"""Convert cleaned BOMD output blocks to ASE EXTXYZ for MACE validation/training."""
from __future__ import annotations
import argparse,re
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import write

BOHR_TO_ANG = 0.529177210903
HARTREE_TO_EV = 27.211386245988
HARTREE_PER_BOHR_TO_EV_PER_ANG = 51.4220674763

def parse_xyz(path):
    blocks={}
    lines=path.read_text(errors="replace").splitlines()
    i=0
    while i<len(lines):
        if lines[i].startswith("Timestep:"):
            ts=int(float(lines[i].split()[1])); i+=1; atoms=[]
            while i<len(lines) and not lines[i].startswith("-"*10):
                p=lines[i].split()
                if len(p)>=5 and p[0].isdigit():
                    atoms.append((p[1],[float(p[2]),float(p[3]),float(p[4])]))
                i+=1
            blocks[ts]=atoms
        i+=1
    return blocks

def parse_forces(path):
    blocks={}
    lines=path.read_text(errors="replace").splitlines()
    i=0
    while i<len(lines):
        if lines[i].startswith("Timestep:"):
            ts=int(float(lines[i].split()[1])); i+=1; energy=None; forces={}
            while i<len(lines) and not lines[i].startswith("-"*10):
                line=lines[i].strip()
                if line.startswith("SCF Energy:"):
                    val=line.split(":",1)[1].strip().strip("[]")
                    energy=float(val.split(",")[0])
                elif line.startswith("Atom "):
                    atom=int(re.search(r"Atom\s+(\d+)",line).group(1))
                    i+=1
                    while i<len(lines) and not lines[i].startswith("-"*10) and not lines[i].startswith("Atom "):
                        st=lines[i].strip()
                        if st.startswith("State 1:"):
                            v=st.split(":",1)[1].split()
                            forces[atom]=[float(v[0]),float(v[1]),float(v[2])]
                        i+=1
                    continue
                i+=1
            blocks[ts]=(energy,forces)
        i+=1
    return blocks

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--stride",type=int,default=1)
    a=p.parse_args()
    root=Path(a.run_root); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    frames=[]
    for rep in sorted((root/"repetitions").glob("rep-*"), key=lambda x:int(x.name.split("-")[1])):
        xyzp=rep/"output/xyz.all"; fp=rep/"output/forces.all"
        if not xyzp.exists() or not fp.exists(): continue
        xyz=parse_xyz(xyzp); frc=parse_forces(fp)
        common=sorted(set(xyz)&set(frc))
        for j,ts in enumerate(common):
            if j % a.stride: continue
            atoms_data=xyz[ts]; energy, fdict=frc[ts]
            if energy is None or len(fdict)!=len(atoms_data): continue
            symbols=[x[0] for x in atoms_data]
            # BOMD stores coordinates in Bohr, energy in Hartree and forces in Hartree/Bohr.
            # ASE/MACE training data use Angstrom, eV and eV/Angstrom.
            pos=np.array([x[1] for x in atoms_data],float) * BOHR_TO_ANG
            forces=np.array([fdict.get(k+1,[np.nan]*3) for k in range(len(symbols))],float) * HARTREE_PER_BOHR_TO_EV_PER_ANG
            if not np.isfinite(forces).all(): continue
            at=Atoms(symbols=symbols,positions=pos)
            at.info["REF_energy"]=energy * HARTREE_TO_EV
            at.arrays["REF_forces"]=forces
            at.info["rep"]=rep.name
            at.info["timestep"]=ts
            frames.append(at)
    if not frames: raise SystemExit("No complete configurations could be converted.")
    write(out,frames,format="extxyz")
    print(f"Wrote {len(frames)} configurations to {out}")
if __name__=="__main__": main()
