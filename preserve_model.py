#!/usr/bin/env python3
import argparse,json,shutil
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--version",required=True);p.add_argument("--model",required=True)
p.add_argument("--training-zip",required=True);p.add_argument("--validation-zip",required=True)
p.add_argument("--best-epoch",required=True);p.add_argument("--metadata",required=True);p.add_argument("--root",default="models");p.add_argument("--checkpoint-dir",default="")
a=p.parse_args();d=Path(a.root)/a.version
if d.exists():raise SystemExit(f"{d} exists; refusing to overwrite")
d.mkdir(parents=True)
shutil.copy2(a.model,d/"model.model");shutil.copy2(a.training_zip,d/"training_data.zip")
shutil.copy2(a.validation_zip,d/"validation_data.zip")
if a.checkpoint_dir:
    src=Path(a.checkpoint_dir)
    if src.exists(): shutil.copytree(src,d/"checkpoints")
meta=json.load(open(a.metadata));meta["model_id"]=a.version;meta["best_epoch"]=int(a.best_epoch)
json.dump(meta,open(d/"metadata.json","w"),indent=4)
print(d)
