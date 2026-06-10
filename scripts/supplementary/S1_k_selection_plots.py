#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Step 1: K Selection Rationale (Enhanced)
======================================================

Goal:
    Visualize multiple clustering metrics to justify K choice:
    - Elbow Method (Inertia/SSE)
    - Silhouette Score
    - Calinski-Harabasz Score
    - Davies-Bouldin Index
    - BIC (Approximate)
    - Stability (Bootstrap AMI)

Inputs:
    - Root directory of a completed step1 run.
    - Data paths (ABIDE/CMI pklz).

Outputs:
    - results/figures/supp_fig1_k_metrics_comprehensive.png
    - results/figures/supp_table1_k_metrics_comprehensive.csv
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_mutual_info_score

# Add current directory to path to import step1
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import step1_train_unsup_vicreg as step1
except ImportError:
    print("Error: step1_train_unsup_vicreg.py not found in the same directory.")
    sys.exit(1)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.alpha': 0.3
})

def find_latest_run(base_dir="asd_unsup_runs"):
    # Handle relative paths
    if not os.path.exists(base_dir):
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), base_dir)
    
    if not os.path.exists(base_dir):
        # Try searching in current dir
        base_dir = "./asd_unsup_runs"
    
    if not os.path.exists(base_dir):
        return None
        
    runs = sorted(glob.glob(os.path.join(base_dir, "*")))
    runs = [r for r in runs if os.path.isdir(r)]
    if not runs:
        return None
    runs.sort(key=os.path.getmtime)
    return runs[-1]

def compute_bic(kmeans, X):
    """
    Computes the BIC metric for a given clusters
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * kmeans.inertia_
    
    const_term = 0.5 * m * np.log(N) * (d+1)
    
    if cl_var == 0:
        return np.nan

    BIC = N * np.log(cl_var) + const_term
    return BIC

def calculate_comprehensive_metrics(E, K_range, stability_repeats=20, seed=123):
    results = {}
    n = E.shape[0]
    rng = np.random.RandomState(seed)
    
    print(f"Calculating metrics for K={list(K_range)} on {n} samples...")
    
    for K in K_range:
        print(f"  Analyzing K={K}...")
        # 1. Main Clustering
        km = KMeans(n_clusters=K, random_state=seed, n_init=20).fit(E)
        labels = km.labels_
        
        # Basic Metrics
        inertia = km.inertia_
        sil = silhouette_score(E, labels) if K > 1 else np.nan
        ch = calinski_harabasz_score(E, labels) if K > 1 else np.nan
        db = davies_bouldin_score(E, labels) if K > 1 else np.nan
        bic = compute_bic(km, E)
        
        # 2. Stability (Bootstrap)
        boot_labels = []
        for r in range(stability_repeats):
            idx = rng.choice(n, size=int(0.8*n), replace=False)
            km_boot = KMeans(n_clusters=K, random_state=seed+r, n_init=5).fit(E[idx])
            # Assign full
            lab_pred = km_boot.predict(E)
            boot_labels.append(lab_pred)
            
        # Calculate pairwise AMI
        ami_vals = []
        for i in range(len(boot_labels)):
            for j in range(i+1, len(boot_labels)):
                ami_vals.append(adjusted_mutual_info_score(boot_labels[i], boot_labels[j]))
        stability = np.mean(ami_vals)
        
        results[K] = {
            'Inertia (SSE)': inertia,
            'Silhouette': sil,
            'Calinski-Harabasz': ch,
            'Davies-Bouldin': db,
            'BIC': bic,
            'Stability (AMI)': stability
        }
        
    return results

def main_pipeline(run_dir, abide_path, cmi_path):
    summary_path = os.path.join(run_dir, "best_overall.json")
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found.")
        return None
        
    with open(summary_path, 'r') as f:
        best_info = json.load(f)
    
    fold_dir_name = os.path.basename(best_info['best']['fold_dir'])
    fold_dir = os.path.join(run_dir, fold_dir_name)
    model_path = os.path.join(fold_dir, "best_model.pth")
    
    summary_fold = os.path.join(fold_dir, "fold_best_summary.json")
    if not os.path.exists(summary_fold):
        cfg = best_info['best_cfg']
    else:
        with open(summary_fold, 'r') as f:
            summ = json.load(f)
        cfg = summ['best_cfg']
        
    print("Loading data...")
    abide_df, _ = step1.load_and_preprocess_data(abide_path, cmi_path)
    abide_asd = abide_df[abide_df["DX_GROUP"]==0].copy()
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Gender Encoder
    ge = step1.OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    try:
        ge.fit(abide_asd[["SEX"]])
    except TypeError:
        ge = step1.OneHotEncoder(sparse=False, handle_unknown="ignore")
        ge.fit(abide_asd[["SEX"]])
        
    meta_dim = ge.transform(abide_asd[["SEX"]]).shape[1] + 1
    age_fill = float(abide_asd["AGE_AT_SCAN"].dropna().mean())
    
    model = step1.stDNN_Embedding_WithProj(246, meta_dim, 512, cfg['dropout_rate']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    print("Embedding ALL ABIDE ASD subjects...")
    # Embed ALL ASD
    E_all, site_all = step1.embed_dataframe(model, abide_asd, ge, age_fill, device, views=4, target_len=cfg['target_len'])
    
    # Calculate Metrics
    K_range = range(2, 9)
    results = calculate_comprehensive_metrics(E_all, K_range)
    
    return results

def plot_comprehensive(results, output_path):
    df = pd.DataFrame(results).T
    Ks = df.index
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # 1. Elbow (Inertia)
    axes[0].plot(Ks, df['Inertia (SSE)'], 'o-', color='#34495e')
    axes[0].set_title('Elbow Method (SSE)')
    axes[0].set_ylabel('Inertia')
    
    # 2. Silhouette (Max is best)
    axes[1].plot(Ks, df['Silhouette'], 'o-', color='#27ae60')
    axes[1].set_title('Silhouette Score (Higher is better)')
    
    # 3. Calinski-Harabasz (Max is best)
    axes[2].plot(Ks, df['Calinski-Harabasz'], 'o-', color='#2980b9')
    axes[2].set_title('Calinski-Harabasz (Higher is better)')
    
    # 4. Davies-Bouldin (Min is best)
    axes[3].plot(Ks, df['Davies-Bouldin'], 'o-', color='#c0392b')
    axes[3].set_title('Davies-Bouldin (Lower is better)')
    
    # 5. BIC (Min is best)
    axes[4].plot(Ks, df['BIC'], 'o-', color='#8e44ad')
    axes[4].set_title('BIC (Lower is better)')
    
    # 6. Stability
    axes[5].plot(Ks, df['Stability (AMI)'], 'o-', color='#f39c12')
    axes[5].set_title('Stability (AMI) (Higher is better)')

    for ax in axes:
        ax.set_xlabel('Number of Clusters (K)')
        ax.grid(True, alpha=0.3)
        # Highlight K=3
        ax.axvline(3, color='r', linestyle='--', alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, default=None)
    parser.add_argument('--abide-path', type=str, default="DATA/combined_ABIDE_information_with_fMRI.pklz")
    parser.add_argument('--cmi-path', type=str, default="CMI-DATA/combined_asd_td_rest_run1_data.pklz")
    args = parser.parse_args()

    target_dir = args.run_dir if args.run_dir else find_latest_run()
    if not target_dir:
        sys.exit("Run dir not found")
        
    # Path fix
    if not os.path.exists(args.abide_path):
        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt = os.path.join(proj_root, args.abide_path)
        if os.path.exists(alt): args.abide_path = alt
        
    if not os.path.exists(args.cmi_path):
         proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
         alt = os.path.join(proj_root, args.cmi_path)
         if os.path.exists(alt): args.cmi_path = alt

    results = main_pipeline(target_dir, args.abide_path, args.cmi_path)
    
    if results:
        out_dir = "results/figures"
        os.makedirs(out_dir, exist_ok=True)
        plot_comprehensive(results, os.path.join(out_dir, "supp_fig1_k_metrics_comprehensive.png"))
        pd.DataFrame(results).T.to_csv(os.path.join(out_dir, "supp_table1_k_metrics_comprehensive.csv"))
