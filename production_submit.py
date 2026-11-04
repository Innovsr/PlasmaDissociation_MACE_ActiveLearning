#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, getpass, json, os, shutil, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BOMD=ROOT/"bomd"
RESTART=ROOT/"restart_manager.py"

def load(path): return json.loads(Path(path).read_text())

def approved_model(model_id):
    d=ROOT/"models"/model_id
    if not (d/"model.model").exists(): raise SystemExit(f"Approved model not found: {d}/model.model")
    reg=ROOT/"registry/models.csv"
    if not reg.exists(): raise SystemExit(f"Model registry missing: {reg}")
    with reg.open() as f: rows=list(csv.DictReader(f))
    matches=[r for r in rows if r.get("model_id")==model_id and r.get("status","").lower()=="approved"]
    if not matches: raise SystemExit(f"Model {model_id} is not marked approved in {reg}")
    return d

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); a=ap.parse_args()
    c=load(a.input); model_dir=approved_model(c["model"])
    h=c["HPC"]; user=getpass.getuser(); parent=Path("/mnt/scratch")/user/h["Runfolder"]; stage=parent/"03_production"
    stage.mkdir(parents=True,exist_ok=True); (stage/"setup").mkdir(exist_ok=True); (stage/"setup/tmp").mkdir(parents=True,exist_ok=True); (stage/"repetitions").mkdir(exist_ok=True); (stage/"logs").mkdir(exist_ok=True)
    shutil.copy2(a.input,stage/"inputs.json")
    geom=BOMD/"Initial_Geometry"/(c["Molecule_data"]["Molecule"]+".txt")
    if not geom.exists(): raise SystemExit(f"Initial geometry not found: {geom}")
    shutil.copy2(geom,stage/"setup/initial_guess.txt")
    setup_py=BOMD/"code/setup.py"; main_py=BOMD/"code/main.py"; py=os.sys.executable; runname=h["Runfolder"]
    setup_script=stage/"setup_production.sh"
    setup_script.write_text(f'''#!/bin/bash\n#SBATCH --account={h["account"]}\n#SBATCH --job-name=Setup_{runname}\n#SBATCH --time={h["hours"]}:00:00\n#SBATCH --cpus-per-task={h["cores"]}\n#SBATCH --mem={h["memory"]}G\n#SBATCH --output={stage}/logs/setup_%j.out\n#SBATCH --error={stage}/logs/setup_%j.err\nset -euo pipefail\nmodule unload qchem 2>/dev/null || true\nmodule load qchem/6.0.2\nexport LM_LICENSE_FILE=27000@uol-lnx-lic01.leeds.ac.uk\nexport QC="$QCHEM_HOME"; export QCAUX="$QC/qcaux"; export QCPROG="$QC/exe/qcprog.exe"; export QCPROG_S="$QC/exe/qcprog.exe_s"; export PATH="$PATH:$QC/exe:$QC/bin"\nexport QCSCRATCH="{stage}/setup/tmp"\ncd "{stage}/setup"\nexec "{py}" "{setup_py}"\n'''); setup_script.chmod(0o755)
    target=int(c["Propagation"]["Tot_timesteps"]); dt=float(c["Propagation"]["Timestep"]); n=int(h["repeats"])
    arr=stage/"production_array.sh"
    arr.write_text(f'''#!/bin/bash\n#SBATCH --account={h["account"]}\n#SBATCH --job-name=MACE_{runname}\n#SBATCH --time={h["hours"]}:00:00\n#SBATCH --cpus-per-task={h["cores"]}\n#SBATCH --mem={h["memory"]}G\n#SBATCH --array=1-{n}\n#SBATCH --output={stage}/logs/mace_%A_%a.out\n#SBATCH --error={stage}/logs/mace_%A_%a.err\nset -euo pipefail\nexport MACE_MODEL_PATH="{model_dir}/model.model"\nexport MACE_DEVICE="{c.get("MACE",{}).get("device","cpu")}"\nexport MACE_DTYPE="{c.get("MACE",{}).get("default_dtype","float64")}"\nif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then\n  if "{py}" "{RESTART}" --run-root "{stage}" --target-steps {target} --timestep {dt} --all-complete; then exit 0; fi\n  sbatch --parsable --dependency=afterany:${{SLURM_ARRAY_JOB_ID}} "$0" > "{stage}/logs/next_job_id.txt"\nfi\ncd "{stage}/repetitions/rep-$SLURM_ARRAY_TASK_ID"\nexec "{py}" "{main_py}"\n'''); arr.chmod(0o755)
    reps=list((stage/"repetitions").glob("rep-*"))
    if reps and len(reps)>=n: setup_job=None
    else: setup_job=subprocess.check_output(["sbatch","--parsable",str(setup_script)]).decode().strip().split(";")[0]
    if setup_job: jid=subprocess.check_output(["sbatch","--parsable","--dependency=afterok:"+setup_job,str(arr)]).decode().strip().split(";")[0]
    else: jid=subprocess.check_output(["sbatch","--parsable",str(arr)]).decode().strip().split(";")[0]
    print(f"Production array job: {jid}"); print(f"Production directory: {stage}")
if __name__=="__main__": main()
