#!/usr/bin/env python3
import argparse,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output",required=True)
a=p.parse_args();root=Path(a.input)
with zipfile.ZipFile(a.output,"w",zipfile.ZIP_DEFLATED) as z:
    for f in root.rglob("*"):
        if f.is_file():z.write(f,f.relative_to(root.parent))
print(a.output)
