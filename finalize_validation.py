#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from bomd_to_extxyz import main as convert_main

# This wrapper is intentionally tiny; the converter itself is the authoritative parser.
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-root",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    # Invoke converter programmatically by temporarily constructing argv.
    import sys
    old=sys.argv
    sys.argv=["bomd_to_extxyz.py","--run-root",a.run_root,"--output",a.output]
    try: convert_main()
    finally: sys.argv=old
if __name__=="__main__": main()
