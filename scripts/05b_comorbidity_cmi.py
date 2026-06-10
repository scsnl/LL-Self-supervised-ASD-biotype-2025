#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5b: CMI-HBN Broad Psychiatric Comorbidity Enrichment Across ASD Subtypes

Uses the CMI-HBN Clinician Consensus Diagnostic Protocol records to compute
fold-enrichment of confirmed/presumptive psychiatric diagnoses per ASD subtype.

Usage:
  python 05b_comorbidity_cmi.py \\
    --abide-pklz   <DATA/combined_ABIDE_information_with_fMRI.pklz> \\
    --cmi-pklz     <CMI-DATA/combined_asd_td_rest_run1_data.pklz> \\
    --behavior-csv <CMI-DATA/labeled_asd_td_adhd.csv> \\
    --step2-outdir <path/to/step3_output/cmi> \\
    --emb-root     <path/to/cmi_embeddings> \\
    --outdir       <output_dir>
"""

import os
import argparse
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import binomtest
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def import_step1_module():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '02_train_vicreg.py')
    spec = importlib.util.spec_from_file_location('step1_module', base)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def robust_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, engine='python')
    df.columns = [str(c).replace('\n', ' ').replace('\r', ' ').strip() for c in df.columns]
    if 'subject_id' not in df.columns:
        cand = [c for c in df.columns if c.lower() in ('id','subject_id','identifiers')]
        if cand:
            df.rename(columns={cand[0]: 'subject_id'}, inplace=True)
    return df

def extract_comorbidity_diseases(df: pd.DataFrame) -> pd.DataFrame:
    print("Extracting comorbidity diseases from CSV...")
    
    cat_cols = [col for col in df.columns if 'ConsensusDx,DX_' in col and '_Cat' in col]
    sub_cols = [col for col in df.columns if 'ConsensusDx,DX_' in col and '_Sub' in col]
    
    print(f"Found {len(cat_cols)} category columns and {len(sub_cols)} subcategory columns")
    
    comorbidity_data = []
    
    for idx, row in df.iterrows():
        if 'Identifiers' in df.columns:
            subject_id = row['Identifiers']
        elif 'subject_id' in df.columns:
            subject_id = row['subject_id']
        elif 'id' in df.columns:
            subject_id = row['id']
        else:
            print("Warning: No valid ID column found")
            continue
        diseases = []
        
        for i in range(1, 11):
            cat_col = f'ConsensusDx,DX_{i:02d}_Cat'
            sub_col = f'ConsensusDx,DX_{i:02d}_Sub'
            
            if cat_col in df.columns and pd.notna(row[cat_col]) and str(row[cat_col]).strip():
                disease_category = str(row[cat_col]).strip()
                
                if disease_category == 'Neurodevelopmental Disorders':
                    if sub_col in df.columns and pd.notna(row[sub_col]) and str(row[sub_col]).strip():
                        disease_name = str(row[sub_col]).strip()
                    else:
                        disease_name = disease_category
                else:
                    disease_name = disease_category
                
                disease_name = disease_name.replace('Disorders', '').replace('Disorder', '').strip()
                if disease_name and disease_name not in diseases:
                    diseases.append(disease_name)
        
        if diseases:
            comorbidity_data.append({
                'subject_id': subject_id,
                'diseases': diseases,
                'num_diseases': len(diseases)
            })
    
    comorbidity_df = pd.DataFrame(comorbidity_data)
    print(f"Extracted comorbidity data for {len(comorbidity_df)} subjects")
    
    return comorbidity_df

def standardize_disease_status(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.lower().str.strip()
        mapping = {
            '1': 1, 'true': 1, 'yes': 1, 'y': 1, 'present': 1, 'positive': 1,
            '0': 0, 'false': 0, 'no': 0, 'n': 0, 'absent': 0, 'negative': 0,
            'nan': np.nan, 'none': np.nan, '': np.nan, '.': np.nan
        }
        series = series.map(mapping)
    
    series = series.fillna(0).astype(int)
    return series

def calculate_disease_enrichment(merged_df: pd.DataFrame, 
                               disease_matrix: pd.DataFrame, 
                               labels: np.ndarray,
                               min_samples: int = 5,
                               baseline: str = "overall") -> pd.DataFrame:
    merged_df = merged_df.copy()
    merged_df['subtype'] = labels.astype(int)
    
    results = []
    
    for disease in disease_matrix.columns:
        disease_status = disease_matrix[disease]
        
        if baseline == "overall":
            p_base_all = float(disease_status.mean())
        
        for k in sorted(merged_df['subtype'].unique()):
            mask_in = merged_df['subtype'] == k
            mask_out = ~mask_in
            
            x_bin = disease_status[mask_in].dropna()
            y_bin = disease_status[mask_out].dropna()
            
            if len(x_bin) >= min_samples and (len(y_bin) if baseline == "rest" else len(disease_status.dropna())) >= min_samples:
                prop_in = float(x_bin.mean())
                p_base = float(y_bin.mean()) if baseline == "rest" else p_base_all
                
                if not np.isfinite(p_base) or p_base <= 0:
                    p_base = 1e-9
                
                p_greater = binomtest(int(x_bin.sum()), n=int(len(x_bin)), p=p_base, alternative="greater").pvalue
                p_less = binomtest(int(x_bin.sum()), n=int(len(x_bin)), p=p_base, alternative="less").pvalue
                
                if prop_in >= p_base:
                    pval = p_greater
                    direction = "enriched"
                    fold = prop_in / p_base
                    signed_fold = max(fold, 1.0)
                else:
                    pval = p_less
                    direction = "depleted"
                    fold = prop_in / p_base
                    signed_fold = -max(1.0/fold, 1.0) if fold > 0 else -np.inf
                
                results.append({
                    'disease': disease,
                    'subtype': int(k),
                    'direction': direction,
                    'prop_in': prop_in,
                    'p_base': p_base,
                    'fold': float(fold),
                    'signed_fold': float(signed_fold),
                    'pval': float(pval),
                    'n_in': int(len(x_bin)),
                    'n_base': int(len(y_bin) if baseline == "rest" else len(disease_status.dropna())),
                    'baseline': baseline
                })
    
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results['qval'] = df_results['pval']
        df_results = df_results.sort_values(['disease', 'subtype', 'pval'])
    
    return df_results

def fit_pca_on_ref_and_transform(FitX: np.ndarray, X: np.ndarray):
    pca = PCA(n_components=2, random_state=42)
    pca.fit(FitX)
    return pca.transform(X)

def plot_comorbidity_heatmap(df_results: pd.DataFrame, out_png: str, alpha_q: float = 0.05):
    if df_results.empty:
        print("No enrichment results to plot.")
        return
    
    pivot_data = df_results.pivot_table(
        index='disease', 
        columns='subtype', 
        values='signed_fold', 
        fill_value=0
    )
    
    sig_data = df_results.pivot_table(
        index='disease', 
        columns='subtype', 
        values='qval', 
        fill_value=1
    ) < alpha_q
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(pivot_data, 
                annot=True, 
                fmt='.2f',
                cmap='RdBu_r', 
                center=0,
                cbar_kws={'label': 'Signed Fold Enrichment'},
                square=True)
    
    plt.title('Disease Comorbidity Enrichment by Subtype', fontsize=16, fontweight='bold')
    plt.xlabel('ASD Subtype', fontsize=14, fontweight='bold')
    plt.ylabel('Disease', fontsize=14, fontweight='bold')
    
    for i, disease in enumerate(pivot_data.index):
        for j, subtype in enumerate(pivot_data.columns):
            if sig_data.loc[disease, subtype]:
                plt.text(j + 0.5, i + 0.5, '*', ha='center', va='center', 
                        fontsize=16, fontweight='bold', color='black')
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved:', out_png)

def plot_comorbidity_pca(merged_df: pd.DataFrame, X: np.ndarray, 
                        disease_matrix: pd.DataFrame, 
                        out_png: str, emb_root: str):
    abide_emb_path = os.path.join(emb_root, 'abide_asd_emb.npy')
    if not os.path.exists(abide_emb_path):
        print(f"Warning: ABIDE embeddings not found: {abide_emb_path}")
        return
    
    E_abide_asd = np.load(abide_emb_path)
    XY = fit_pca_on_ref_and_transform(E_abide_asd, X)
    
    plt.rcParams.update({
        'font.size': 12,
        'font.weight': 'bold',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    disease_counts = disease_matrix.sum().sort_values(ascending=False)
    top_diseases = disease_counts.head(6).index.tolist()
    
    n_diseases = len(top_diseases)
    n_cols = min(3, n_diseases)
    n_rows = (n_diseases + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_diseases == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, disease in enumerate(top_diseases):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        disease_status = disease_matrix[disease]
        
        mask_disease = disease_status == 1
        ax.scatter(XY[~mask_disease, 0], XY[~mask_disease, 1], 
                  c='#bbbbbb', s=30, alpha=0.6, label=f'No {disease}')
        ax.scatter(XY[mask_disease, 0], XY[mask_disease, 1], 
                  c='#e41a1c', s=40, alpha=0.8, label=f'{disease}')
        
        ax.set_title(f'CMI-ASD PCA: {disease} Comorbidity', fontweight='bold')
        ax.set_xlabel('PC1', fontweight='bold')
        ax.set_ylabel('PC2', fontweight='bold')
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_diseases, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved:', out_png)

def plot_comorbidity_rates(df_results: pd.DataFrame, out_png: str):
    if df_results.empty:
        print("No enrichment results to plot.")
        return
    
    rate_data = df_results.groupby(['disease', 'subtype']).agg({
        'prop_in': 'first',
        'n_in': 'first'
    }).reset_index()
    
    pivot_rates = rate_data.pivot_table(
        index='disease', 
        columns='subtype', 
        values='prop_in', 
        fill_value=0
    )
    
    plt.figure(figsize=(12, 8))
    pivot_rates.plot(kind='bar', ax=plt.gca(), width=0.8)
    
    plt.title('Disease Comorbidity Rates by Subtype', fontsize=16, fontweight='bold')
    plt.xlabel('Disease', fontsize=14, fontweight='bold')
    plt.ylabel('Comorbidity Rate', fontsize=14, fontweight='bold')
    plt.legend(title='ASD Subtype', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved:', out_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--abide-pklz', type=str, default='DATA/combined_ABIDE_information_with_fMRI.pklz')
    ap.add_argument('--cmi-pklz', type=str, default='CMI-DATA/combined_asd_td_rest_run1_data.pklz')
    ap.add_argument('--behavior-csv', type=str, required=True, 
                   help='CMICSV')
    ap.add_argument('--step2-outdir', type=str, required=True, 
                   help='cmi_asd_labels.npy')
    ap.add_argument('--emb-root', type=str, required=True, 
                   help='Step1CMI embeddings')
    ap.add_argument('--outdir', type=str, default='unsup_results/cmi_comorbidity_enrichment')
    ap.add_argument('--min-samples', type=int, default=5, 
                   help='')
    ap.add_argument('--baseline', type=str, default='overall', 
                   choices=['overall', 'rest'],
                   help='overall=/rest=')
    ap.add_argument('--alpha-q', type=float, default=0.05, 
                   help='')
    
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print("=== CMI ===")
    
    print("Loading CMI-ASD data...")
    step1 = import_step1_module()
    abide_df, cmi_df = step1.load_and_preprocess_data(args.abide_pklz, args.cmi_pklz)
    cmi_asd = cmi_df[cmi_df['DX_GROUP'] == 0].copy().reset_index(drop=True)
    
    print("Loading subtype labels...")
    lab_path = os.path.join(args.step2_outdir, 'cmi_asd_labels.npy')
    if not os.path.exists(lab_path):
        raise SystemExit(f': {lab_path}')
    labels = np.load(lab_path)
    if len(labels) != len(cmi_asd):
        raise SystemExit(f'({len(labels)})CMI-ASD({len(cmi_asd)})')
    cmi_asd['subtype'] = labels.astype(int)
    
    print("Loading embeddings...")
    emb_path = os.path.join(args.emb_root, 'cmi_asd_emb.npy')
    if not os.path.exists(emb_path):
        raise SystemExit(f'embeddings: {emb_path}')
    X = np.load(emb_path)
    if X.shape[0] != len(cmi_asd):
        raise SystemExit(f'Emb({X.shape[0]})CMI-ASD({len(cmi_asd)})')
    
    print("Loading behavioral data...")
    beh = robust_read_csv(args.behavior_csv)
    
    comorbidity_df = extract_comorbidity_diseases(beh)
    
    full_comorbidity_csv = os.path.join(args.outdir, 'full_comorbidity_diseases.csv')
    comorbidity_df.to_csv(full_comorbidity_csv, index=False)
    print(f"Saved full comorbidity diseases: {full_comorbidity_csv}")
    
    cmi_asd['subject_id'] = cmi_asd['subject_id'].astype(str).str.strip()
    comorbidity_df['subject_id'] = comorbidity_df['subject_id'].astype(str).str.strip()
    merged = cmi_asd.merge(comorbidity_df, on='subject_id', how='left')
    
    import ast
    def parse_diseases(diseases_str):
        try:
            if pd.isna(diseases_str):
                return []
            if isinstance(diseases_str, list):
                return diseases_str
            if isinstance(diseases_str, str):
                return ast.literal_eval(diseases_str)
            return []
        except (ValueError, TypeError, SyntaxError):
            return []
    
    merged['diseases'] = merged['diseases'].apply(parse_diseases)
    
    print("Creating comprehensive comorbidity table...")
    
    comorbidity_table = []
    num_subjects_with_diseases = 0
    num_subjects_without_diseases = 0
    
    for idx, row in merged.iterrows():
        subject_id = row['subject_id']
        subtype = row['subtype']
        diseases_list = row['diseases']
        
        has_diseases = False
        if diseases_list is not None and isinstance(diseases_list, list) and len(diseases_list) > 0:
            for disease in diseases_list:
                comorbidity_table.append({
                    'subject_id': subject_id,
                    'subtype': subtype,
                    'disease': disease
                })
                has_diseases = True
        
        if not has_diseases:
            comorbidity_table.append({
                'subject_id': subject_id,
                'subtype': subtype,
                'disease': 'No Diagnosis'
            })
            num_subjects_without_diseases += 1
        else:
            num_subjects_with_diseases += 1
    
    comorbidity_table_df = pd.DataFrame(comorbidity_table)
    comorbidity_csv_path = os.path.join(args.outdir, 'comprehensive_comorbidity_table.csv')
    comorbidity_table_df.to_csv(comorbidity_csv_path, index=False)
    print(f"Saved comprehensive comorbidity table: {comorbidity_csv_path}")
    
    print(f"Total records: {len(comorbidity_table_df)}")
    print(f"Total subjects: {len(merged)}")
    print(f"Subjects with disease diagnoses: {num_subjects_with_diseases}")
    print(f"Subjects without disease diagnoses: {num_subjects_without_diseases}")
    unique_diseases = comorbidity_table_df[comorbidity_table_df['disease'] != 'No Diagnosis']['disease'].unique()
    print(f"Unique diseases: {len(unique_diseases)}")
    print("Disease list:")
    for disease in sorted(unique_diseases):
        count = len(comorbidity_table_df[comorbidity_table_df['disease'] == disease])
        print(f"  - {disease}: {count} records")
    
    print("Creating disease matrix for enrichment analysis...")
    all_diseases = set()
    for diseases_list in merged['diseases'].dropna():
        all_diseases.update(diseases_list)
    
    all_diseases = sorted(list(all_diseases))
    print(f"Found {len(all_diseases)} unique diseases:")
    for disease in all_diseases:
        print(f"  - {disease}")
    
    disease_matrix = pd.DataFrame(0, index=merged.index, columns=all_diseases)
    for idx, diseases_list in merged['diseases'].items():
        try:
            if pd.notna(diseases_list) and isinstance(diseases_list, list):
                for disease in diseases_list:
                    if disease in disease_matrix.columns:
                        disease_matrix.loc[idx, disease] = 1
        except (ValueError, TypeError):
            continue
    
    print("Calculating comorbidity enrichment...")
    df_results = calculate_disease_enrichment(
        merged, disease_matrix, labels, 
        min_samples=args.min_samples, baseline=args.baseline
    )
    
    results_path = os.path.join(args.outdir, 'comorbidity_enrichment_results.csv')
    df_results.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")
    
    print("Generating visualizations...")
    
    heatmap_path = os.path.join(args.outdir, 'comorbidity_enrichment_heatmap.png')
    plot_comorbidity_heatmap(df_results, heatmap_path, alpha_q=args.alpha_q)
    
    pca_path = os.path.join(args.outdir, 'comorbidity_pca_plots.png')
    plot_comorbidity_pca(merged, X, disease_matrix, pca_path, args.emb_root)
    
    rates_path = os.path.join(args.outdir, 'comorbidity_rates.png')
    plot_comorbidity_rates(df_results, rates_path)
    
    print("Generating summary report...")
    summary_path = os.path.join(args.outdir, 'comorbidity_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=== CMI ===\n\n")
        f.write(f": {len(merged)}\n")
        f.write(f": {len(merged['subtype'].unique())}\n")
        f.write(f": {len(all_diseases)}\n\n")
        
        f.write(":\n")
        for disease in all_diseases:
            f.write(f"  {disease}\n")
        
        f.write(f"\n:\n")
        f.write(f"  : {len(df_results)}\n")
        f.write(f"  : {len(df_results[df_results['qval'] < args.alpha_q])}\n")
        
        f.write(f"\n:\n")
        for disease in all_diseases:
            f.write(f"\n{disease}:\n")
            disease_results = df_results[df_results['disease'] == disease]
            for _, row in disease_results.iterrows():
                f.write(f"  {row['subtype']}: {row['prop_in']:.3f} (n={row['n_in']})\n")
    
    print(f"Saved: {summary_path}")
    print("===  ===")

if __name__ == '__main__':
    main()
