#!/usr/bin/env python3
import argparse,random
from ase.io import read,write
p=argparse.ArgumentParser()
p.add_argument("--old",required=True);p.add_argument("--new",required=True);p.add_argument("--output",required=True)
p.add_argument("--old-percent",type=float,required=True);p.add_argument("--new-percent",type=float,required=True)
p.add_argument("--total",type=int,required=True);p.add_argument("--seed",type=int,default=42)
a=p.parse_args()
if abs(a.old_percent+a.new_percent-100)>1e-9:raise SystemExit("old+new must equal 100")
old=read(a.old,index=":");new=read(a.new,index=":")
nold=round(a.total*a.old_percent/100);nnew=a.total-nold
if nold>len(old) or nnew>len(new):raise SystemExit("Not enough data for requested ratio/total")
random.seed(a.seed);out=random.sample(old,nold)+random.sample(new,nnew);random.shuffle(out)
write(a.output,out,format="extxyz")
print(f"Created {a.output}: old={nold}, new={nnew}, total={len(out)}")
