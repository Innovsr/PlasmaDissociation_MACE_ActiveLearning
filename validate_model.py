#!/usr/bin/env python3
import argparse,csv,json
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator
EV2MEV=1000.0
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True); p.add_argument("--validation",required=True)
    p.add_argument("--output",default="validation_results.csv")
    p.add_argument("--device",default="cpu"); p.add_argument("--default-dtype",default="float64"); p.add_argument("--summary",default="")
    a=p.parse_args()
    xs=read(a.validation,index=":")
    calc=MACECalculator(model_paths=a.model,device=a.device,default_dtype=a.default_dtype)
    rows=[]; er=[]; ep=[]; fr=[]; fp=[]
    for i,x in enumerate(xs):
        if "REF_energy" not in x.info or "REF_forces" not in x.arrays:
            raise KeyError(f"Missing REF_energy/REF_forces in configuration {i}")
        x.calc=calc
        re=float(x.info["REF_energy"])/len(x); pe=float(x.get_potential_energy())/len(x)
        rf=np.asarray(x.arrays["REF_forces"]); pf=np.asarray(x.get_forces())
        ferr=np.sqrt(np.mean((pf-rf)**2))*EV2MEV
        eerr=(pe-re)*EV2MEV
        rows.append({"configuration":i,"n_atoms":len(x),"energy_error_mev_atom":eerr,
                     "force_rmse_mev_A":ferr,"max_force_error_mev_A":np.max(np.abs(pf-rf))*EV2MEV})
        er.append(re); ep.append(pe); fr.append(rf.ravel()); fp.append(pf.ravel())
    er=np.array(er); ep=np.array(ep); fr=np.concatenate(fr); fp=np.concatenate(fp)
    erm=np.sqrt(np.mean((ep-er)**2))*EV2MEV
    frm=np.sqrt(np.mean((fp-fr)**2))*EV2MEV
    with open(a.output,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summary={"energy_rmse_mev_atom":float(erm),"force_rmse_mev_A":float(frm),
             "n_configurations":len(xs)}
    if a.summary:
        with open(a.summary,"w") as sf:
            json.dump(summary,sf,indent=4)
    print(f"Energy RMSE: {erm:.3f} meV/atom")
    print(f"Force RMSE : {frm:.3f} meV/A")
    print(f"Per-configuration results: {a.output}")
if __name__=="__main__": main()
