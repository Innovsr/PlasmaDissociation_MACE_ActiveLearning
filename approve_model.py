#!/usr/bin/env python3
"""Promote a candidate model to models/ after explicit user approval."""
from __future__ import annotations
import argparse,csv,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--attempt",required=True)
 p.add_argument("--version",required=True)
 p.add_argument("--T",required=True)
 p.add_argument("--Mol",required=True); p.add_argument("--best-epoch",default="")
 a=p.parse_args()
 att=Path(a.attempt)
 cand=att/"candidate.model"
 train=att/"training.extxyz"
 val=Path(a.attempt)/"validation.extxyz"
 if not cand.exists(): raise SystemExit(f"Candidate model missing: {cand}")
 out=ROOT/"models"/a.version
 if out.exists(): raise SystemExit(f"{out} already exists")
 out.mkdir(parents=True)
 shutil.copy2(cand,out/"model.model")
 # Archive training data.
 import zipfile
 with zipfile.ZipFile(out/"training_data.zip","w",zipfile.ZIP_DEFLATED) as z:
  z.write(train,train.name)
 if val.exists():
  with zipfile.ZipFile(out/"validation_data.zip","w",zipfile.ZIP_DEFLATED) as z: z.write(val,val.name)
 # Preserve the candidate training checkpoints so a later update can resume
 # from the selected/best epoch rather than starting from scratch.
 ckpt=Path(a.attempt)/"mace_work"/"checkpoints"
 if ckpt.exists(): shutil.copytree(ckpt,out/"checkpoints")
 source=(att/"source_model.txt").read_text().splitlines()[0] if (att/"source_model.txt").exists() else ""
 comp=json.loads((att/"comparison.json").read_text()) if (att/"comparison.json").exists() else {}
 epoch=int(a.best_epoch) if a.best_epoch else None
 meta={"model_id":a.version,"best_epoch":epoch,
       "force_rmse_mev_A":comp.get("candidate_force_rmse_mev_A"),
       "energy_rmse_mev_atom":comp.get("candidate_energy_rmse_mev_atom"),
       "training_data":"training_data.zip",
       "comment":{"T":a.T,"Mol":a.Mol,"Source_model":source}}
 (out/"metadata.json").write_text(json.dumps(meta,indent=4)+"\n")
 # append/update registry
 reg=ROOT/"registry"/"models.csv"
 fields=["model_id","parent_model","best_epoch","molecules","temperature_range","train_size","validation_size","energy_rmse_mev_atom","force_rmse_mev_A","relative_force_rmse_percent","training_archive","validation_archive","model_file","status","best_for","notes"]
 rows=[]
 if reg.exists():
  with reg.open() as f: rows=list(csv.DictReader(f))
 rows.append({k:"" for k in fields})
 rows[-1].update({"model_id":a.version,"parent_model":source,"best_epoch":str(epoch or ""),"molecules":a.Mol,"temperature_range":a.T,
                  "training_archive":str(out/"training_data.zip"),"validation_archive":str(out/"validation_data.zip"),
                  "model_file":str(out/"model.model"),"status":"approved","best_for":f"{a.Mol} @ {a.T}",
                  "notes":"Explicitly approved by user."})
 with reg.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 print(f"Approved {a.version}: T={a.T}; Mol={a.Mol}; Source model={source}")
if __name__=="__main__": main()
