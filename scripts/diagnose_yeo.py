#!/usr/bin/env python3
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path

# --- 0. Test Yeo atlas loading ---
from nilearn.datasets import fetch_atlas_yeo_2011
yeo = fetch_atlas_yeo_2011()
print(f"Yeo keys: {list(yeo.keys())}")
print(f"Yeo maps: {yeo.maps}")
print(f"Yeo labels: {yeo.labels}")

# --- 1. Find first available aseg ---
fmriprep = Path("/scratch/hyruuk/neuromod/mario_data/mario.fmriprep")
aseg_files = sorted(fmriprep.glob("sub-*/ses-*/func/*desc-aseg_dseg.nii.gz"))
real = [f for f in aseg_files if f.resolve().exists() and f.resolve().stat().st_size > 0]

if not real:
    print("ERROR: no aseg file found or all are datalad stubs")
else:
    path = real[0]
    print(f"Aseg file: {path.name}")
    d = nib.load(path).get_fdata().astype(int)
    labels, counts = np.unique(d, return_counts=True)
    lmap = dict(zip(labels.tolist(), counts.tolist()))
    for lid, name in [
        (3,  "L-Cerebral-Cortex"),
        (42, "R-Cerebral-Cortex"),
        (8,  "L-Cereb-Cortex"),
        (47, "R-Cereb-Cortex"),
    ]:
        print(f"  Label {lid:2d} ({name}): {lmap.get(lid, 0):,} voxels")
    print(f"  All labels present: {sorted(lmap.keys())}")

# --- 2. Check qc_runs.csv for yeo columns (search broadly) ---
candidates = sorted(Path(".").glob("**/qc_runs.csv")) + sorted(Path("/scratch/hyruuk").glob("**/qc_runs.csv"))
if not candidates:
    print("\nqc_runs.csv not found anywhere — has the pipeline been run with the updated code?")
else:
    for csv in candidates:
        print(f"\nChecking: {csv}")
        df = pd.read_csv(csv)
        yeo_cols = [c for c in df.columns if c.startswith("yeo_")]
        print(f"  Yeo columns: {yeo_cols if yeo_cols else 'NONE'}")
        if yeo_cols:
            print(df[yeo_cols].describe().to_string())
