#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Step 5: Baseline Comparison (VICReg vs ALFF/FC)
=============================================================

Goal:
    Compare the clustering structure derived from Deep Embedding (VICReg)
    against traditional brain features:
    1. ALFF (Amplitude of Low Frequency Fluctuations)
    2. FC (Functional Connectivity - Whole Brain)
    
    Question: Does VICReg find a structure that simple metrics miss?

Inputs:
    - VICReg Results: step1 run directory
    - Baseline Data: unsup_results/step7_abide/*.npy

Outputs:
    - results/figures/supp_fig5_baseline_comparison.png
    - results/figures/supp_table5_baseline_metrics.csv
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.dpi': 300,
    'axes.grid': False
})

def find_latest_run(base_dir="asd_unsup_runs"):
    if not os.path.exists(base_dir):
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), base_dir)
    if not os.path.exists(base_dir):
        base_dir = "./asd_unsup_runs"
    if not os.path.exists(base_dir):
        return None
    runs = sorted(glob.glob(os.path.join(base_dir, "*")))
    runs = [r for r in runs if os.path.isdir(r)]
    if not runs:
        return None
    runs.sort(key=os.path.getmtime)
    return runs[-1]

def load_vicreg_results(run_dir):
    print(f"Loading VICReg results from {run_dir}")
    emb_path = os.path.join(run_dir, "abide_asd_emb.npy")
    lab_path = os.path.join(run_dir, "abide_asd_labels.npy")
    
    if not (os.path.exists(emb_path) and os.path.exists(lab_path)):
        return None, None
        
    E = np.load(emb_path)
    # Sanitize
    mask = np.isfinite(E).all(axis=1)
    E = E[mask]
    n = np.linalg.norm(E, axis=1, keepdims=True); n[n==0]=1.0
    E = E/n
    
    L = np.load(lab_path)
    if len(L) > len(E):
        E_orig = np.load(emb_path)
        mask_orig = np.isfinite(E_orig).all(axis=1)
        L = L[mask_orig]
        
    return E, L

def get_triu_vector(fcs):
    n, r, _ = fcs.shape
    mask = np.triu_indices(r, k=1)
    vecs = []
    for i in range(n):
        vecs.append(fcs[i][mask])
    return np.stack(vecs)

def load_baselines():
    data = {}
    
    # 1. ALFF (Combat)
    p_alff = "unsup_results/step7_abide/alffs_asd_combat.npy"
    if os.path.exists(p_alff):
        print("Loading ALFF...")
        data['ALFF'] = np.nan_to_num(np.load(p_alff))
        
    # 2. FC (GSR) - Flattened
    p_fc = "unsup_results/step7_abide/fcs_asd_gsr.npy"
    if os.path.exists(p_fc):
        print("Loading FC...")
        fcs = np.load(p_fc)
        data['FC'] = np.nan_to_num(get_triu_vector(fcs))
        
    return data

def align_labels(labels_ref, labels_target):
    K = 3
    cm = confusion_matrix(labels_ref, labels_target, labels=range(K))
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {c: r for r, c in zip(row_ind, col_ind)}
    return np.array([mapping.get(x, x) for x in labels_target])

def analyze_feature(name, X, L_ref):
    print(f"--- Analyzing {name} ---")
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    
    # Clustering
    km = KMeans(n_clusters=3, random_state=123, n_init=20).fit(X_norm)
    L_pred = km.labels_
    
    # Align
    L_aligned = align_labels(L_ref, L_pred)
    
    # Metrics
    ari = adjusted_rand_score(L_ref, L_aligned)
    sil = silhouette_score(X_norm, L_aligned)
    
    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_norm)
    
    return {
        'name': name,
        'ari': ari,
        'sil': sil,
        'labels': L_aligned,
        'X_2d': X_2d
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, default=None)
    args = parser.parse_args()

    target_dir = args.run_dir if args.run_dir else find_latest_run()
    if not target_dir:
        sys.exit("Run dir not found")
        
    # 1. VICReg Results
    E_vic, L_vic = load_vicreg_results(target_dir)
    if E_vic is None:
        sys.exit("VICReg results not found")
    
    # Calculate VICReg stats
    sil_vic = silhouette_score(E_vic, L_vic)
    pca_vic = PCA(n_components=2, random_state=42)
    E_vic_2d = pca_vic.fit_transform(E_vic)
    
    # 2. Load Baselines
    baselines = load_baselines()
    if not baselines:
        sys.exit("No baseline features found.")
        
    # 3. Analyze each
    results = []
    results.append({
        'name': 'VICReg (Ours)',
        'ari': 1.0,
        'sil': sil_vic,
        'labels': L_vic,
        'X_2d': E_vic_2d
    })
    
    for name, X in baselines.items():
        # Ensure length match
        n_min = min(len(X), len(E_vic))
        X_curr = X[:n_min]
        L_vic_curr = L_vic[:n_min]
        
        res = analyze_feature(name, X_curr, L_vic_curr)
        results.append(res)
        
    # Save Metrics Table
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['labels', 'X_2d']} for r in results])
    out_dir = "results/figures"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "supp_table5_baseline_metrics.csv"), index=False)
    print("\nSummary Table:")
    print(df)
    
    # 4. Plot 1x3 Grid
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    custom_palette = ['#2ca02c', '#ff7f0e', '#1f77b4'] # Green, Orange, Blue
    
    # Order: VICReg, ALFF, FC
    ordered_names = ['VICReg (Ours)', 'ALFF', 'FC']
    
    for i, target_name in enumerate(ordered_names):
        if i >= len(axes): break
        
        # Find result
        res = next((r for r in results if r['name'] == target_name), None)
        ax = axes[i]
        
        if res is None:
            ax.text(0.5, 0.5, f"{target_name}\nNot Available", 
                    ha='center', va='center', fontsize=16)
            continue
            
        X_2d = res['X_2d']
        labels = res['labels']
        sil = res['sil']
        ari = res['ari']
        
        title = f"{target_name}\n"
        if target_name == 'VICReg (Ours)':
            title += f"(Sil={sil:.3f})"
        else:
            title += f"(Sil={sil:.3f}, ARI={ari:.3f})"
            
        for c in range(3):
            mask = labels == c
            color = custom_palette[c]
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, 
                       label=f'Subtype {c+1}', alpha=0.6, s=20, edgecolor='w', linewidth=0.1)
            
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend(loc='lower right')
        
    plt.tight_layout()
    out_path = os.path.join(out_dir, "supp_fig5_baseline_comparison.png")
    plt.savefig(out_path)
    print(f"Saved 3-panel figure to {out_path}")

if __name__ == "__main__":
    main()
