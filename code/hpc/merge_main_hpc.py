#!/usr/bin/env python3
"""
merge_main_hpc.py
-----------------
Merge per-task CSV files from the main_hpc PBS array job into a single file.

Usage (run locally after downloading results from HPC):
    python code/hpc/merge_main_hpc.py
"""
import os
import glob
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
IN_DIR   = os.path.join(DATA_DIR, "main_hpc")
OUT_FILE = os.path.join(DATA_DIR, "coal_rho2.csv")

files = sorted(glob.glob(os.path.join(IN_DIR, "coal_task*.csv")))
if not files:
    raise FileNotFoundError(f"No task CSV files found in {IN_DIR}")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(OUT_FILE, index=False)
print(f"Merged {len(files)} files → {OUT_FILE}  ({len(df)} rows)")
