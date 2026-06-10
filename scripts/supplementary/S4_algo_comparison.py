#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Step 4: Independence from Clustering Algorithm
============================================================

Goal:
    Verify if the subtype structure (K=3) is robust across different clustering algorithms.
    - K-Means (Baseline)
    - Spectral Clustering (Nearest Neighbors affinity)
    - Agglomerative Clustering (Ward Linkage)

    Visualizes the decision boundaries on PCA space.

Inputs:
    - Root directory of a completed step1 run (ABIDE embeddings).

Outputs:
    - results/figures/supp_fig4_algo_comparison.png
    - results/figures/supp_table4_algo_ari.csv
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

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

def load_embeddings(run_dir):
    path = os.path.join(run_dir, "abide_asd_emb.npy")
    if not os.path.exists(path):
        return None
    E = np.load(path)
    # Sanitize
    mask = np.isfinite(E).all(axis=1)
    E = E[mask]
    n = np.linalg.norm(E, axis=1, keepdims=True); n[n==0]=1.0
    E = E/n
    return E

def align_labels(labels_ref, labels_target):
    """Align target labels to reference labels using Hungarian algorithm."""
    K = len(np.unique(labels_ref))
    # Ensure labels are in 0..K-1
    unique_ref = sorted(np.unique(labels_ref))
    unique_tgt = sorted(np.unique(labels_target))
    
    # Create confusion matrix
    cm = confusion_matrix(labels_ref, labels_target, labels=range(K))
    # Hungarian algorithm to find best match
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {c: r for r, c in zip(row_ind, col_ind)}
    
    return np.array([mapping.get(x, x) for x in labels_target])

def run_comparison(E, output_dir):
    print(f"Comparing algorithms on {len(E)} samples (K=3)...")
    
    # 1. K-Means (Baseline)
    print("Running K-Means...")
    km = KMeans(n_clusters=3, random_state=123, n_init=50).fit(E)
    labels_km = km.labels_
    
    # 2. Spectral Clustering
    print("Running Spectral Clustering...")
    # Using nearest_neighbors affinity is often better for manifold structures
    sc = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=123, n_init=20, eigen_tol=1e-4)
    labels_sc = sc.fit_predict(E)
    
    # 3. Agglomerative Clustering
    print("Running Agglomerative Clustering (Ward)...")
    ac = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels_ac = ac.fit_predict(E)
    
    # Align to K-Means
    print("Aligning labels...")
    labels_sc_aligned = align_labels(labels_km, labels_sc)
    labels_ac_aligned = align_labels(labels_km, labels_ac)
    
    # Metrics
    ari_km_sc = adjusted_rand_score(labels_km, labels_sc_aligned)
    ari_km_ac = adjusted_rand_score(labels_km, labels_ac_aligned)
    print(f"ARI (K-Means vs Spectral): {ari_km_sc:.4f}")
    print(f"ARI (K-Means vs Agglomerative): {ari_km_ac:.4f}")
    
    # Save Stats
    pd.DataFrame({
        'Comparison': ['K-Means vs Spectral', 'K-Means vs Agglomerative'],
        'ARI': [ari_km_sc, ari_km_ac]
    }).to_csv(os.path.join(output_dir, "supp_table4_algo_ari.csv"), index=False)
    
    # Visualization
    print("Running PCA for visualization...")
    pca = PCA(n_components=2, random_state=42)
    E_vis = pca.fit_transform(E)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    custom_palette = ['#2ca02c', '#ff7f0e', '#1f77b4'] # Green, Orange, Blue (Consistent with previous steps)
    
    configs = [
        ('K-Means (Baseline)', labels_km, axes[0]),
        (f'Spectral Clustering\n(ARI={ari_km_sc:.2f})', labels_sc_aligned, axes[1]),
        (f'Agglomerative (Ward)\n(ARI={ari_km_ac:.2f})', labels_ac_aligned, axes[2])
    ]
    
    for title, labels, ax in configs:
        for c in range(3):
            mask = labels == c
            if mask.sum() == 0: continue
            color = custom_palette[c] if c < 3 else 'gray'
            ax.scatter(E_vis[mask, 0], E_vis[mask, 1], c=color, 
                       label=f'Subtype {c+1}', alpha=0.6, s=25, edgecolor='w', linewidth=0.2)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        if configs.index((title, labels, ax)) == 0:
            ax.legend(loc='lower right', frameon=True)
            
    plt.tight_layout()
    out_path = os.path.join(output_dir, "supp_fig4_algo_comparison.png")
    plt.savefig(out_path)
    print(f"Saved comparison plot to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, default=None)
    args = parser.parse_args()

    target_dir = args.run_dir if args.run_dir else find_latest_run()
    if not target_dir:
        sys.exit("Run dir not found")
        
    print(f"Target Run Directory: {target_dir}")
    
    E = load_embeddings(target_dir)
    if E is None:
        sys.exit("Embeddings not found.")
        
    out_dir = "results/figures"
    os.makedirs(out_dir, exist_ok=True)
    
    run_comparison(E, out_dir)

if __name__ == "__main__":
    main()




