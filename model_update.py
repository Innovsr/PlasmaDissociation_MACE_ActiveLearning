#!/usr/bin/env python3
"""One complete model-update attempt. No interactive prompts."""
from __future__ import annotations
import argparse, csv, json, os, shutil, subprocess, zipfile, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def load(p): return json.loads(Path(p).read_text())

def latest_model(root, requested):
    d=root/"models"/requested
    if not d.exists(): raise SystemExit(f"Approved model not found: {requested}")
    mf=d/"model.model"
    if not mf.exists(): raise SystemExit(f"Missing {mf}")
    return d

def extract_training(zip_path, out):
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members=[n for n in z.namelist() if n.endswith(".xyz") or n.endswith(".extxyz")]
        if not members: raise SystemExit("Training archive contains no EXTXYZ/XYZ dataset.")
        # Prefer the largest/most likely dataset file.
        member=max(members,key=lambda n:z.getinfo(n).file_size)
        z.extract(member,out)
        return out/member

def registry_row(root, model_id):
    p=root/"registry"/"models.csv"
    if not p.exists(): return {}
    with p.open() as f:
        for r in csv.DictReader(f):
            if r.get("model_id")==model_id: return r
    return {}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default=str(ROOT/"configs/update_model.json"))
    ap.add_argument("--validation",default="")
    ap.add_argument("--source-model",default=None)
    a=ap.parse_args()
    c=load(a.config)
    root=ROOT
    source=a.source_model or c["source_model"]
    validation_path = Path(a.validation) if a.validation else (
        Path("/mnt/scratch") / os.environ.get("USER", os.environ.get("LOGNAME","unknown"))
        / c["HPC"]["Runfolder"] / "01_validation" / "validation" / "validation.extxyz"
    )
    if not validation_path.exists():
        raise SystemExit(f"Validation dataset not found: {validation_path}. Complete Stage 1 first.")
    # Keep an exact copy of the validation dataset with this update attempt.
    model_dir=latest_model(root,source)
    meta=registry_row(root,source)
    source_epoch=int(meta.get("best_epoch") or c.get("old_epoch",0))
    scratch_root=Path("/mnt/scratch") / os.environ.get("USER", os.environ.get("LOGNAME","unknown")) / c["HPC"]["Runfolder"]
    attempt_dir=scratch_root/"02_model_update"/c.get("attempt_id", "attempt_001")
    attempt_dir.mkdir(parents=True,exist_ok=True)
    status_file=attempt_dir/"STATUS"
    if status_file.exists() and status_file.read_text().strip() in {"CANDIDATE_READY_FOR_USER_APPROVAL","NO_HIGH_ERROR_CONFIGURATIONS"}:
        print(f"Attempt already terminal: {status_file.read_text().strip()}")
        return 0
    (attempt_dir/"source_model.txt").write_text(f"{source}\nsource_epoch={source_epoch}\n")
    shutil.copy2(validation_path, attempt_dir/"validation.extxyz")

    # Validate source.
    source_results=attempt_dir/"source_validation.csv"
    subprocess.run([sys.executable,str(root/"validate_model.py"),
                    "--model",str(model_dir/"model.model"),
                    "--validation",str(validation_path),"--output",str(source_results),
                    "--device",c["mace"]["device"],"--default-dtype",c["mace"]["default_dtype"],"--summary",str(attempt_dir/"source_summary.json")],check=True)
    source_summary=json.loads((attempt_dir/"source_summary.json").read_text())
    if (source_summary["force_rmse_mev_A"] <= c["validation"]["force_rmse_target_mev_A"] and
        source_summary["energy_rmse_mev_atom"] <= c["validation"]["energy_rmse_target_mev_atom"]):
        (attempt_dir/"STATUS").write_text("MODEL_OK_USE_EXISTING\n")
        print("Source model meets both validation targets; no update required.")
        return 0

    bad=attempt_dir/"high_error.csv"
    subprocess.run([sys.executable,str(root/"select_high_error.py"),
                    "--results",str(source_results),"--output",str(bad),
                    "--max-configurations",str(c["validation"]["max_new_configurations"]),
                    "--force-threshold",str(c["validation"]["high_error_force_threshold_mev_A"])],check=True)

    # The converter creates a full validation EXTXYZ. The current high-error
    # selector records configuration indices; extract those indices here.
    from ase.io import read, write
    val=read(str(validation_path),index=":")
    with bad.open() as f:
        badrows=list(csv.DictReader(f))
    idx=[int(r["configuration"]) for r in badrows]
    new=[val[i] for i in idx if 0 <= i < len(val)]
    new_path=attempt_dir/"new_problematic.extxyz"
    write(new_path,new,format="extxyz")
    if not new:
        (attempt_dir/"STATUS").write_text("NO_HIGH_ERROR_CONFIGURATIONS\n")
        return 0

    old_zip=model_dir/"training_data.zip"
    if not old_zip.exists():
        raise SystemExit(f"Approved model {source} has no training_data.zip")
    old_dir=attempt_dir/"old_training"
    old_ext=extract_training(old_zip,old_dir)

    total=c["training_update"].get("total_configurations")
    if total is None:
        # Preserve the size of the source training set by default.
        total=len(read(old_ext,index=":"))
    train_path=attempt_dir/"training.extxyz"
    subprocess.run([sys.executable,str(root/"build_training_set.py"),
                    "--old",str(old_ext),"--new",str(new_path),
                    "--output",str(train_path),
                    "--old-percent",str(c["training_update"]["old_data_percent"]),
                    "--new-percent",str(c["training_update"]["new_data_percent"]),
                    "--total",str(total),"--seed",str(c["mace"]["seed"])],check=True)

    work=attempt_dir/"mace_work"
    source_ckpt=model_dir/"checkpoints"
    if not source_ckpt.exists() and not (work/"checkpoints").exists():
        raise SystemExit(
            f"Source model {source} has no preserved MACE checkpoints. "
            "The active-learning workflow requires the previous optimised epoch "
            "to be preserved; archive the source checkpoints before updating."
        )
    # train_mace handles checkpoint reuse when restart-from points to the
    # source model work/checkpoints. For a new attempt, if the source model
    # archive contains checkpoints, restore them before training.
    if source_ckpt.exists() and not (work/"checkpoints").exists():
        shutil.copytree(source_ckpt,work/"checkpoints")

    candidate=f"{source}_candidate_{attempt_dir.name}"
    subprocess.run([sys.executable,str(root/"train_mace.py"),
                    "--train",str(train_path),"--valid",str(validation_path),
                    "--test",str(validation_path),"--name",candidate,
                    "--work-dir",str(work),"--restart-from",str(work),
                    "--device",c["mace"]["device"]],check=True)

    # Find compiled model produced by MACE.
    candidates=list(work.rglob("*.model"))
    if not candidates:
        raise SystemExit("No candidate .model file found after MACE training.")
    model=max(candidates,key=lambda p:p.stat().st_mtime)
    candidate_model=attempt_dir/"candidate.model"
    shutil.copy2(model,candidate_model)

    cand_results=attempt_dir/"candidate_validation.csv"
    subprocess.run([sys.executable,str(root/"validate_model.py"),
                    "--model",str(candidate_model),"--validation",str(validation_path),
                    "--output",str(cand_results),
                    "--device",c["mace"]["device"],"--default-dtype",c["mace"]["default_dtype"],"--summary",str(attempt_dir/"candidate_summary.json")],check=True)
    src_summary=json.loads((attempt_dir/"source_summary.json").read_text())
    cand_summary=json.loads((attempt_dir/"candidate_summary.json").read_text())
    src_f=float(src_summary["force_rmse_mev_A"])
    cand_f=float(cand_summary["force_rmse_mev_A"])
    src_e=float(src_summary["energy_rmse_mev_atom"])
    cand_e=float(cand_summary["energy_rmse_mev_atom"])
    comparison={"source_model":source,"candidate":candidate,
                "source_force_rmse_mev_A":src_f,"candidate_force_rmse_mev_A":cand_f,
                "source_energy_rmse_mev_atom":src_e,"candidate_energy_rmse_mev_atom":cand_e,
                "force_improvement_percent":(src_f-cand_f)/src_f*100 if src_f else 0.0,
                "energy_improvement_percent":(src_e-cand_e)/src_e*100 if src_e else 0.0,
                "target_force_met":cand_f <= c["validation"]["force_rmse_target_mev_A"],
                "target_energy_met":cand_e <= c["validation"]["energy_rmse_target_mev_atom"]}
    (attempt_dir/"comparison.json").write_text(json.dumps(comparison,indent=4)+"\n")
    (attempt_dir/"STATUS").write_text("CANDIDATE_READY_FOR_USER_APPROVAL\n")
    print(json.dumps(comparison,indent=2))
    print(f"Candidate stored in {attempt_dir}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
