#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Step 6: Demographics Summary
==========================================

Goal:
    Generate a demographic summary table for all datasets used:
    - ABIDE
    - CMI
    - Stanford
    - GENDAAR (if available)

    Metrics:
    - N (Count)
    - Age (Mean ± SD, Range)
    - Sex (Male/Female Count)

Outputs:
    - results/tables/supp_table6_demographics.csv
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import gzip
import bz2
import lzma
import types

# AGGRESSIVE PATCH for NumPy 2.0 -> 1.x pickle compatibility
try:
    import numpy._core
except ImportError:
    _core = types.ModuleType('numpy._core')
    sys.modules['numpy._core'] = _core
    if hasattr(np.core, 'numeric'):
        sys.modules['numpy._core.numeric'] = np.core.numeric
        _core.numeric = np.core.numeric
    if hasattr(np.core, 'multiarray'):
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
        _core.multiarray = np.core.multiarray
    if hasattr(np.core, 'umath'):
        sys.modules['numpy._core.umath'] = np.core.umath
        _core.umath = np.core.umath
    if hasattr(np.core, '_multiarray_umath'):
        sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
        _core._multiarray_umath = np.core._multiarray_umath

def load_pklz(path):
    print(f"Loading {path}...")
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return None
    
    for opener in (gzip.open, bz2.open, lzma.open):
        try:
            with opener(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  Failed to load pklz: {e}")
        return None

def load_gendaar_csv(root):
    csv_path = os.path.join(root, "DATA-GENDAAR", "gendaar_behavior.csv")
    if not os.path.exists(csv_path):
        return None
    
    print(f"  Fallback: Loading CSV {csv_path}...")
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        return df
    except Exception as e:
        print(f"  Failed to load CSV: {e}")
        return None

def get_groups(df, name):
    # Returns (ASD_df, TD_df)
    if df is None: return None, None
    
    cols = df.columns
    
    # 1. ABIDE Style (DX_GROUP: 1=ASD, 2=TD)
    if 'DX_GROUP' in cols:
        vals = sorted(df['DX_GROUP'].dropna().unique())
        if 1 in vals and 2 in vals:
            return df[df['DX_GROUP']==1], df[df['DX_GROUP']==2]
        elif 0 in vals and 1 in vals:
            return df[df['DX_GROUP']==0], df[df['DX_GROUP']==1]
            
    # 2. CMI/Stanford Style (label: 'asd', 'td')
    if 'label' in cols:
        s = df['label'].astype(str).str.lower()
        return df[s=='asd'], df[s=='td']
        
    # 3. GENDAAR CSV Style
    if name == 'GENDAAR':
        # Debug columns
        # print(f"GENDAAR Columns: {list(cols)}")
        # Look for specific columns
        diag_col = None
        if 'srs201_phenotype' in cols:
            diag_col = 'srs201_phenotype'
        elif 'demographics02_diagnosis' in cols:
            diag_col = 'demographics02_diagnosis'
        
        if diag_col:
            print(f"  Using diagnosis column: {diag_col}")
            s = df[diag_col].astype(str).str.lower()
            asd = df[s.str.contains('autism') | s.str.contains('asd')]
            td = df[s.str.contains('non spectrum') | s.str.contains('control') | s.str.contains('td') | s.str.contains('typical')]
            print(f"  Found {len(asd)} ASD, {len(td)} TD")
            return asd, td
            
    print(f"  Warning: Diagnosis column not found for {name}.")
    return pd.DataFrame(), pd.DataFrame()

def analyze_subgroup(df, group_name, name):
    if df is None or len(df) == 0:
        return {
            'N': 0,
            'Age_Mean': np.nan, 'Age_SD': np.nan, 'Age_Min': np.nan, 'Age_Max': np.nan,
            'Male': 0, 'Female': 0
        }
    
    # Age
    age_col = None
    age_factor = 1.0
    
    # Priority: AgeAtScan, AGE_AT_SCAN (Years)
    # GENDAAR: demographics02_interview_age (Months)
    candidates = ['AGE_AT_SCAN', 'age', 'Age', 'interview_age', 'AgeAtScan', 
                  'demographics02_interview_age', 'srs201_interview_age']
    
    for c in candidates:
        if c in df.columns:
            # Check if mostly NaN
            if df[c].notna().sum() > 0:
                age_col = c
                # Heuristic: if max > 120, assume months -> years
                sample = pd.to_numeric(df[c], errors='coerce').dropna()
                if len(sample) > 0 and sample.max() > 100: 
                    age_factor = 1.0 / 12.0
                break
    
    if age_col:
        # print(f"  Using age column: {age_col} (factor={age_factor})")
        ages = pd.to_numeric(df[age_col], errors='coerce').dropna() * age_factor
        age_stats = {
            'Age_Mean': ages.mean(),
            'Age_SD': ages.std(),
            'Age_Min': ages.min(),
            'Age_Max': ages.max()
        }
    else:
        print(f"  No age column found for {group_name}")
        age_stats = {'Age_Mean': np.nan, 'Age_SD': np.nan, 'Age_Min': np.nan, 'Age_Max': np.nan}
        
    # Sex
    sex_col = None
    sex_candidates = ['SEX', 'gender', 'Gender', 'Sex', 
                      'demographics02_demo_sex_tgender', 'srs201_sex', 'demographics02_sex']
    for c in sex_candidates:
        if c in df.columns:
            # Check valid values
            if df[c].notna().sum() > 0:
                sex_col = c
                break
            
    male = 0
    female = 0
    if sex_col:
        if name == 'GENDAAR':
            # print(f"  Using sex column: {sex_col}")
            # print(f"  Unique sex values: {df[sex_col].unique()}")
            pass
            
        s = df[sex_col].astype(str).str.upper()
        male = s.isin(['1', '1.0', 'M', 'MALE']).sum()
        female = s.isin(['2', '2.0', 'F', 'FEMALE']).sum()
    else:
        print(f"  No sex column found for {group_name}")
        
    return {
        'N': len(df),
        **age_stats,
        'Male': male,
        'Female': female
    }

def format_age(stats):
    if np.isnan(stats['Age_Mean']): return "N/A"
    return f"{stats['Age_Mean']:.2f} ± {stats['Age_SD']:.2f} [{stats['Age_Min']:.1f}-{stats['Age_Max']:.1f}]"

def main():
    project_root = os.getcwd()
    
    datasets = {
        'ABIDE': os.path.join(project_root, "DATA", "combined_ABIDE_information_with_fMRI.pklz"),
        'CMI': os.path.join(project_root, "CMI-DATA", "combined_asd_td_rest_run1_data.pklz"),
        'Stanford': os.path.join(project_root, "DATA-Stanford", "stanford_brainnetome_mean_regMov-6param_dt1_bpf008-09_246ROIs.pklz"),
        'GENDAAR': os.path.join(project_root, "DATA-GENDAAR", "combined_GENDAAR.pklz")
    }
    
    results = []
    
    for name, path in datasets.items():
        print(f"\nProcessing {name}...")
        df = load_pklz(path)
        
        if df is None and name == 'GENDAAR':
            df = load_gendaar_csv(project_root)
            
        if df is None:
            continue
            
        asd, td = get_groups(df, name)
        
        # Analyze ASD
        stats_asd = analyze_subgroup(asd, 'ASD', name)
        results.append({
            'Dataset': name,
            'Group': 'ASD',
            'N': stats_asd['N'],
            'Age': format_age(stats_asd),
            'Sex (M/F)': f"{stats_asd['Male']}/{stats_asd['Female']}"
        })
        
        # Analyze TD
        stats_td = analyze_subgroup(td, 'TD', name)
        results.append({
            'Dataset': name,
            'Group': 'TD', 
            'N': stats_td['N'],
            'Age': format_age(stats_td),
            'Sex (M/F)': f"{stats_td['Male']}/{stats_td['Female']}"
        })
        
    # Create DataFrame
    res_df = pd.DataFrame(results)
    
    # Save
    out_dir = "results/tables"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "supp_table6_demographics.csv")
    res_df.to_csv(out_path, index=False)
    
    print("\n=== Demographics Summary ===")
    print(res_df.to_string(index=False))
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
