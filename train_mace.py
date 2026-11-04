#!/usr/bin/env python3
import argparse,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument("--train",required=True);p.add_argument("--valid",required=True);p.add_argument("--test",required=True)
p.add_argument("--name",required=True);p.add_argument("--work-dir",required=True)
p.add_argument("--restart-from",default="");p.add_argument("--device",default="cpu")
a=p.parse_args();c=json.load(open(ROOT/"configs/update_model.json"));m=c["mace"]
work=Path(a.work_dir);work.mkdir(parents=True,exist_ok=True)
if a.restart_from:
    src=Path(a.restart_from)/"checkpoints"
    dst=work/"checkpoints"
    if src.exists() and src.resolve() != dst.resolve():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src,dst)
cmd=[m["command"],f"--name={a.name}",f"--work_dir={work}",f"--train_file={a.train}",
f"--valid_file={a.valid}",f"--test_file={a.test}",f"--energy_key={m['energy_key']}",
f"--forces_key={m['forces_key']}","--model=MACE","--loss=weighted","--E0s=average",
f"--r_max={m['r_max']}",f"--radial_type={m['radial_type']}",f"--num_radial_basis={m['num_radial_basis']}",
f"--num_cutoff_basis={m['num_cutoff_basis']}",f"--max_ell={m['max_ell']}",
f"--correlation={m['correlation']}",f"--num_interactions={m['num_interactions']}",
f"--hidden_irreps={m['hidden_irreps']}",f"--batch_size={m['batch_size']}",
f"--valid_batch_size={m['valid_batch_size']}",f"--energy_weight={m['energy_weight']}",
f"--forces_weight={m['forces_weight']}",f"--lr={m['lr']}",f"--max_num_epochs={m['max_num_epochs']}",
f"--patience={m['patience']}",f"--eval_interval={m['eval_interval']}",
f"--device={a.device}",f"--default_dtype={m['default_dtype']}",f"--seed={m['seed']}"]
if a.restart_from:cmd.append("--restart_latest")
subprocess.run(cmd,check=True)
