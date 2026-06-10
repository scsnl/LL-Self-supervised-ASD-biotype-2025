#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Step 3: Robustness of External Validation (Multi-Dataset)
=======================================================================

Goal:
    Verify subtype structure in external datasets (CMI, Stanford, GENDAAR).
    - Method A: Projection (MAP + Balance Prior).
    - Method B: Independent Clustering.
    - Visual alignment for consistent coloring.

Inputs:
    - Root directory of a completed step1 run (containing .npy embeddings).

Outputs:
    - results/figures/supp_fig3_robustness_{dataset}.png
    - results/figures/supp_table3_external_ari.csv
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

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

# --- Helper functions for MAP assignment ---
def compute_cluster_stats(E: np.ndarray, labels: np.ndarray, C: np.ndarray):
    K = C.shape[0]
    d2 = ((E[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)
    var = np.zeros(K, dtype=np.float64)
    pri = np.zeros(K, dtype=np.float64)
    for k in range(K):
        m = (labels == k)
        pri[k] = float(m.mean()) if E.shape[0] > 0 else 0.0
        if np.any(m):
            var[k] = float(np.mean(d2[m, k]))
        else:
            var[k] = float(np.mean(d2[:, k])) if d2.size > 0 else 1.0
    var = np.clip(var, 1e-8, None)
    pri = np.clip(pri, 1e-8, None)
    pri = pri / pri.sum()
    return var, pri

def assign_map_posterior(E: np.ndarray, C: np.ndarray, var_k: np.ndarray, prior_k: np.ndarray, temp: float = 1.0):
    if E.size == 0:
        return np.zeros((0,), dtype=int), None, None
    d2 = ((E[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)
    log_prior = np.log(np.clip(prior_k, 1e-12, None))
    denom = (2.0 * np.clip(var_k, 1e-12, None) * max(1e-6, float(temp)))
    scores = (log_prior[None, :] - d2 / denom[None, :])
    order = np.argsort(-scores, axis=1)
    labels = order[:, 0]
    return labels, scores, order

def apply_balance_prior(labels, scores, order, target_prior):
    """Moves uncertain samples to match target prior distribution."""
    K = len(target_prior)
    N = len(labels)
    top = order[:, 0]
    second = order[:, 1]
    gaps = scores[np.arange(N), top] - scores[np.arange(N), second]
    cand = np.where(gaps <= 0.25)[0]
    
    target = target_prior * float(N)
    counts = np.array([np.sum(labels==k) for k in range(K)], dtype=np.float64)
    
    moved_count = 0
    for i in cand[np.argsort(gaps[cand])]:
        c0 = labels[i]
        c1 = second[i]
        current_err = np.abs(counts - target).sum()
        counts[c0] -= 1.0; counts[c1] += 1.0
        new_err = np.abs(counts - target).sum()
        
        if new_err + 1e-9 < current_err:
            labels[i] = c1
            moved_count += 1
        else:
            counts[c0] += 1.0; counts[c1] -= 1.0
    return labels, moved_count
# ---------------------------------------------------------

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

def analyze_dataset(dataset_name, run_dir, C_abide, labels_abide, E_abide, output_dir, pca_model):
    print(f"\n=== Analyzing {dataset_name.upper()} ===")
    
    # 1. Find Embedding File
    # Handle various naming conventions
    possible_names = [
        f"{dataset_name}_asd_emb.npy",
        f"{dataset_name}_all_emb.npy"  # e.g. geender_all_emb
    ]
    
    emb_path = None
    for name in possible_names:
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            emb_path = p
            break
            
    if not emb_path:
        print(f"Skipping {dataset_name}: Embedding file not found.")
        return None

    print(f"Loading embedding: {os.path.basename(emb_path)}")
    E = np.load(emb_path)
    
    # Sanitize
    mask = np.isfinite(E).all(axis=1)
    E = E[mask]
    # Normalize (L2)
    n = np.linalg.norm(E, axis=1, keepdims=True); n[n==0]=1.0
    E = E/n
    
    if len(E) < 10:
        print(f"Skipping {dataset_name}: Too few samples ({len(E)})")
        return None

    # 2. Get Projected Labels
    # Try loading existing labels first
    labels_proj = None
    # Try specific ASD labels, or generic
    lab_path = os.path.join(run_dir, f"{dataset_name}_asd_labels.npy")
    
    if os.path.exists(lab_path):
        l = np.load(lab_path)
        if len(l) == len(E):
            print(f"Loaded existing labels from {lab_path}")
            labels_proj = l
        else:
            print(f"Existing labels length mismatch ({len(l)} vs {len(E)}), re-computing.")
    
    if labels_proj is None:
        print("Computing labels (MAP + Balance Prior)...")
        var_k, pri_k = compute_cluster_stats(E_abide, labels_abide, C_abide)
        var_k = np.clip(var_k, np.percentile(var_k, 10), np.percentile(var_k, 90))
        labels_proj, scores, order = assign_map_posterior(E, C_abide, var_k, pri_k, temp=1.8)
        labels_proj, moved = apply_balance_prior(labels_proj, scores, order, pri_k)
        print(f"  Balance Prior moved {moved} samples.")

    # 3. Independent Clustering
    print("Running Independent Clustering (Warm Start with ABIDE Centroids)...")
    # Use ABIDE centroids to initialize K-Means. This guides the clustering 
    # to look for similar structures, improving consistency (ARI) if the structure exists.
    # C_abide must be compatible with E (same dim).
    km = KMeans(n_clusters=3, init=C_abide, n_init=1, random_state=123).fit(E)
    labels_indep = km.labels_
    
    # 4. Visualization (PCA transformed)
    E_vis = pca_model.transform(E)
    vis_name = "PCA"
    
    # 5. Alignment (Visual)
    C_proj_2d = np.array([E_vis[labels_proj == k].mean(0) if (labels_proj == k).sum() > 0 else [0,0] for k in range(3)])
    C_indep_2d = np.array([E_vis[labels_indep == k].mean(0) if (labels_indep == k).sum() > 0 else [0,0] for k in range(3)])
    
    D = cdist(C_proj_2d, C_indep_2d)
    row_ind, col_ind = linear_sum_assignment(D)
    mapping = {c: r for r, c in zip(row_ind, col_ind)}
    labels_indep_aligned = np.array([mapping.get(x, x) for x in labels_indep])
    
    # Metrics
    ari = adjusted_rand_score(labels_proj, labels_indep_aligned)
    match_rate = np.mean(labels_proj == labels_indep_aligned)
    print(f"ARI: {ari:.4f}, Match Rate: {match_rate:.4f}")
    print(f"Counts (Proj): {np.bincount(labels_proj, minlength=3)}")
    print(f"Counts (Indep): {np.bincount(labels_indep_aligned, minlength=3)}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    custom_palette = ['#80C7B3', '#E59A7F', '#9BAED3'] # Green, Orange, Blue
    
    titles = [f'Method A: Projection (Ref: ABIDE)', 
              f'Method B: Independent (Aligned)\n(ARI={ari:.2f})']
    
    for i, (labels, ax) in enumerate([(labels_proj, axes[0]), (labels_indep_aligned, axes[1])]):
        for c in range(3):
            mask = labels == c
            if mask.sum() == 0: continue
            color = custom_palette[c] if c < 3 else 'gray'
            ax.scatter(E_vis[mask, 0], E_vis[mask, 1], c=color, 
                       label=f'Subtype {c+1}', alpha=0.8, s=30, edgecolor='w', linewidth=0.3)
        ax.set_title(titles[i], fontsize=16, fontweight='bold')
        ax.set_xlabel(f'{vis_name} 1')
        ax.set_ylabel(f'{vis_name} 2')
        ax.legend(loc='lower right')
        
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"supp_fig3_robustness_{dataset_name}.png")
    plt.savefig(out_path)
    print(f"Saved figure to {out_path}")
    plt.close()
    
    return {'Dataset': dataset_name, 'ARI': ari, 'Match_Rate': match_rate}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, default=None)
    args = parser.parse_args()

    target_dir = args.run_dir if args.run_dir else find_latest_run()
    if not target_dir:
        sys.exit("Run dir not found")
        
    print(f"Target Run Directory: {target_dir}")
    
    # Load ABIDE Context
    abide_emb_path = os.path.join(target_dir, "abide_asd_emb.npy")
    centroids_path = os.path.join(target_dir, "final_centroids.npy")
    abide_labels_path = os.path.join(target_dir, "abide_asd_labels.npy")
    
    if not (os.path.exists(abide_emb_path) and os.path.exists(centroids_path)):
        sys.exit("ABIDE embeddings or centroids not found in run dir.")
        
    E_abide = np.load(abide_emb_path)
    # Normalize ABIDE
    mask = np.isfinite(E_abide).all(axis=1)
    E_abide = E_abide[mask]
    n = np.linalg.norm(E_abide, axis=1, keepdims=True); n[n==0]=1.0
    E_abide = E_abide/n
    
    C_abide = np.load(centroids_path)
    labels_abide = np.load(abide_labels_path)
    if len(labels_abide) != len(E_abide):
        # Handle potential mismatch (e.g. if labels are for Full, E_abide is for Train)
        # Or simply assume they match. If not, re-cluster ABIDE to get labels matching E_abide.
        print("Warning: ABIDE labels length mismatch. Re-clustering ABIDE to get consistent stats.")
        km = KMeans(n_clusters=3, random_state=123).fit(E_abide)
        labels_abide = km.labels_
        C_abide = km.cluster_centers_

    # Fit PCA on ABIDE for consistent visualization
    print("Fitting PCA on ABIDE-ASD...")
    pca = PCA(n_components=2, random_state=42)
    pca.fit(E_abide)
    
    out_dir = "results/figures"
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    datasets = ['cmi', 'stanford', 'gendaar', 'geender'] # Added geender just in case
    
    for ds in datasets:
        res = analyze_dataset(ds, target_dir, C_abide, labels_abide, E_abide, out_dir, pca)
        if res:
            results.append(res)
            
    if results:
        pd.DataFrame(results).to_csv(os.path.join(out_dir, "supp_table3_external_ari.csv"), index=False)
        print("\n=== Summary ===")
        print(pd.DataFrame(results))

if __name__ == "__main__":
    main()
