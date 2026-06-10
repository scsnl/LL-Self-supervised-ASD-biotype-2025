#!/usr/bin/env python3
"""
Create a smoke-test dataset by randomly sampling 100 participants from ABIDE.

Produces a self-contained pickle file (same format as the full ABIDE data) that
can be used to verify the pipeline runs end-to-end without access to the full
dataset.  The sample is stratified by DX_GROUP (50 ASD / 50 TDC) and seeded
for reproducibility.

The output file can be used in place of the full ABIDE pickle for any script
that accepts --abide-pklz.

Usage:
  python create_smoke_test.py \\
    --abide-pklz  <path/to/combined_ABIDE_information_with_fMRI.pklz> \\
    --out-pklz    smoke_test_abide100.pkl \\
    --n-per-group 50 \\
    --seed        42

Output columns retained:
  data, DX_GROUP, SEX, AGE_AT_SCAN, SITE_ID, SITE, ABIDE,
  mean_fd, percentofvolsrepaired, (all other columns preserved)
"""

import os
import argparse
import pickle
import random
import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--abide-pklz', required=True,
                    help='Path to full ABIDE combined pickle / pklz file')
    ap.add_argument('--out-pklz', default='smoke_test_abide100.pkl',
                    help='Output path for the smoke-test pickle (default: smoke_test_abide100.pkl)')
    ap.add_argument('--n-per-group', type=int, default=50,
                    help='Number of participants per group (ASD / TDC); default 50')
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def load_pklz(path):
    import gzip, bz2, lzma
    ext = os.path.splitext(path)[1].lower()
    openers = {'.gz': gzip.open, '.bz2': bz2.open, '.xz': lzma.open, '.lzma': lzma.open}
    opener = openers.get(ext, open)
    with opener(path, 'rb') as f:
        return pickle.load(f)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Loading ABIDE data from: {args.abide_pklz}")
    df = load_pklz(args.abide_pklz)
    print(f"  Raw rows: {len(df)}")

    # QC filters (mirror 02_train_vicreg.py)
    df = df[(df["percentofvolsrepaired"] <= 10) & (df["mean_fd"] <= 0.5)].copy()
    df["data"] = df["data"].apply(lambda x: np.array(x) if isinstance(x, list) else x)

    def is_valid(x):
        a = np.array(x)
        return a.ndim == 2 and a.shape[0] >= 120 and np.isfinite(a).all()

    df = df[df["data"].apply(is_valid)].reset_index(drop=True)
    print(f"  After QC: {len(df)}")

    # Keep original ABIDE encoding (1=ASD, 2=TDC) so downstream scripts work correctly.
    # We only use the remapped values here to split by group; the saved file keeps originals.
    dx_vals = df["DX_GROUP"].unique()
    if set(dx_vals).issubset({1, 2}):
        asd_idx = df.index[df["DX_GROUP"] == 1].tolist()
        tdc_idx = df.index[df["DX_GROUP"] == 2].tolist()
    else:
        # Already remapped: 0=ASD, 1=TDC
        asd_idx = df.index[df["DX_GROUP"] == 0].tolist()
        tdc_idx = df.index[df["DX_GROUP"] == 1].tolist()

    n = args.n_per_group
    if len(asd_idx) < n or len(tdc_idx) < n:
        raise SystemExit(
            f"Not enough participants after QC: {len(asd_idx)} ASD, {len(tdc_idx)} TDC. "
            f"Requested {n} per group."
        )

    sel_asd = rng.choice(asd_idx, size=n, replace=False).tolist()
    sel_tdc = rng.choice(tdc_idx, size=n, replace=False).tolist()
    sel_idx = sorted(sel_asd + sel_tdc)

    smoke_df = df.loc[sel_idx].reset_index(drop=True)
    print(f"  Sampled {n} ASD + {n} TDC = {len(smoke_df)} participants")

    # ── Anonymization ───────────────────────────────────────────────────────
    # Replace real participant identifiers with sequential fake IDs and drop
    # free-text fields that may contain identifying or sensitive information,
    # so the resulting smoke-test file carries no real subject identifiers.
    fake_ids = [f"SUB_{i:04d}" for i in range(1, len(smoke_df) + 1)]
    for id_col in ("SUB_ID", "subject_id", "SUB_ID2", "FILE_ID"):
        if id_col in smoke_df.columns:
            smoke_df[id_col] = fake_ids
    for drop_col in ("MEDICATION_NAME",):
        if drop_col in smoke_df.columns:
            smoke_df = smoke_df.drop(columns=[drop_col])
    print(f"  Anonymized identifiers -> SUB_0001 ... SUB_{len(smoke_df):04d}; "
          f"dropped free-text PII columns where present")

    # Save
    out_path = args.out_pklz
    with open(out_path, 'wb') as f:
        pickle.dump(smoke_df, f)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Saved to: {out_path}  ({size_mb:.1f} MB)")
    print()
    print("Smoke test dataset summary:")
    dx_vals = smoke_df['DX_GROUP'].unique()
    if set(dx_vals).issubset({1, 2}):
        print(f"  ASD : {(smoke_df['DX_GROUP'] == 1).sum()}")
        print(f"  TDC : {(smoke_df['DX_GROUP'] == 2).sum()}")
    else:
        print(f"  ASD : {(smoke_df['DX_GROUP'] == 0).sum()}")
        print(f"  TDC : {(smoke_df['DX_GROUP'] == 1).sum()}")
    if "SITE_ID" in smoke_df.columns:
        print(f"  Sites: {smoke_df['SITE_ID'].nunique()}")
    if "AGE_AT_SCAN" in smoke_df.columns:
        ages = smoke_df["AGE_AT_SCAN"].dropna()
        print(f"  Age : {ages.mean():.1f} ± {ages.std():.1f}  [{ages.min():.0f}-{ages.max():.0f}]")


if __name__ == '__main__':
    main()
