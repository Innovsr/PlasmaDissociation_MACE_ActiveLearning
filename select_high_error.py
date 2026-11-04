#!/usr/bin/env python3
import argparse,csv
p=argparse.ArgumentParser()
p.add_argument("--results",required=True);p.add_argument("--output",required=True)
p.add_argument("--max-configurations",type=int,default=2000)
p.add_argument("--force-threshold",type=float,default=100)
a=p.parse_args()
rs=list(csv.DictReader(open(a.results)))
for r in rs:r["force_rmse_mev_A"]=float(r["force_rmse_mev_A"])
rs=[r for r in sorted(rs,key=lambda r:r["force_rmse_mev_A"],reverse=True)
    if r["force_rmse_mev_A"]>=a.force_threshold][:a.max_configurations]
if rs:
    with open(a.output,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rs[0].keys());w.writeheader();w.writerows(rs)
print(f"Selected {len(rs)} high-error configurations")
