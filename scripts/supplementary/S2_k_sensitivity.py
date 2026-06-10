#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Step 2: Sensitivity Analysis of K (Fixed)
=======================================================

Goal:
    Analyze how cluster structure changes with K=2, 3, 4.
    **Crucial Update**: Loads existing embeddings and labels for K=3 to ensure consistency 
    with previous results. Only re-computes K=2/4.

Inputs:
    - Root directory of a completed step1 run (must contain .npy outputs).

Outputs:
    - results/figures/supp_fig2_k_sensitivity_pca.png
    - results/figures/supp_fig2_k_transitions.png
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
from sklearn.metrics import confusion_matrix

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
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

def load_existing_results(run_dir):
    print(f"Loading artifacts from {run_dir}...")
    
    emb_path = os.path.join(run_dir, "abide_asd_emb.npy")
    lab_path = os.path.join(run_dir, "abide_asd_labels.npy")
    
    if not os.path.exists(emb_path):
        print(f"Error: {emb_path} not found. Please run step1 with --final-train-on-all first.")
        return None, None

    E = np.load(emb_path)
    
    if os.path.exists(lab_path):
        print("Found existing K=3 labels.")
        labels_k3 = np.load(lab_path)
    else:
        print("Warning: Existing labels not found, will re-cluster K=3 (might color swap).")
        km = KMeans(n_clusters=3, random_state=123, n_init=20).fit(E)
        labels_k3 = km.labels_
        
    return E, labels_k3

def plot_pca_sensitivity(E, labels_k3, output_path):
    # 1. Run PCA
    print("Running PCA on embeddings...")
    pca = PCA(n_components=2, random_state=42)
    E_pca = pca.fit_transform(E)
    
    # 2. Run KMeans for K=2,4 (K=3 is fixed)
    print("Clustering for K=2 and K=4...")
    kms = {}
    
    # K=2
    kms[2] = KMeans(n_clusters=2, random_state=123, n_init=20).fit(E)
    
    # K=3 (Use existing)
    class MockKM:
        def __init__(self, labs): self.labels_ = labs
    kms[3] = MockKM(labels_k3)
    
    # K=4
    kms[4] = KMeans(n_clusters=4, random_state=123, n_init=20).fit(E)
    
    # 3. Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Colors: Adjusted based on user feedback (S1=Green, S2=Orange, S3=Blue)
    # 0: Green, 1: Orange, 2: Blue, 3: Purple
    custom_palette = ['#2ca02c', '#ff7f0e', '#1f77b4', '#9467bd']
    
    palette_map = {
        2: [custom_palette[0], custom_palette[1]],
        3: [custom_palette[0], custom_palette[1], custom_palette[2]],
        4: [custom_palette[0], custom_palette[1], custom_palette[2], custom_palette[3]]
    }
    
    titles = ['K=2 (Bisection)', 'K=3 (Original Selected)', 'K=4 (Further Split)']
    
    for i, k in enumerate([2, 3, 4]):
        ax = axes[i]
        labels = kms[k].labels_
        n_clusters = len(np.unique(labels)) # Should match k
        
        # To make colors consistent, we might need to match labels to K=3 logic?
        # For now, just plot distinct colors.
        
        for c in range(n_clusters):
            mask = labels == c
            color = palette_map[k][c] if c < len(palette_map[k]) else 'gray'
            ax.scatter(E_pca[mask, 0], E_pca[mask, 1], 
                       c=color, label=f'Cluster {c+1}',
                       alpha=0.7, s=20, edgecolor='w', linewidth=0.3)
        
        ax.set_title(titles[i], fontsize=16, fontweight='bold')
        ax.set_xlabel('PC1')
        if i == 0: ax.set_ylabel('PC2')
        ax.legend(loc='lower right')
        
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved PCA comparison to {output_path}")
    return kms

def plot_transitions(kms, output_path):
    """
    Plot Confusion Matrices
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # K=2 -> K=3
    ax = axes[0]
    cm = confusion_matrix(kms[2].labels_, kms[3].labels_)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=[f'K3-C{i+1}' for i in range(3)],
                yticklabels=[f'K2-C{i+1}' for i in range(2)])
    ax.set_title('Flow: K=2 → K=3', fontsize=16, fontweight='bold')
    
    # K=3 -> K=4
    ax = axes[1]
    cm = confusion_matrix(kms[3].labels_, kms[4].labels_)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax, cbar=False,
                xticklabels=[f'K4-C{i+1}' for i in range(4)],
                yticklabels=[f'K3-C{i+1}' for i in range(3)])
    ax.set_title('Flow: K=3 → K=4', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved Transition plot to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, default=None)
    args = parser.parse_args()

    target_dir = args.run_dir if args.run_dir else find_latest_run()
    if not target_dir:
        sys.exit("Run dir not found")
        
    print(f"Target Run Directory: {target_dir}")
    
    E, labels_k3 = load_existing_results(target_dir)
    
    if E is not None:
        out_dir = "results/figures"
        os.makedirs(out_dir, exist_ok=True)
        kms = plot_pca_sensitivity(E, labels_k3, os.path.join(out_dir, "supp_fig2_k_sensitivity_pca.png"))
        plot_transitions(kms, os.path.join(out_dir, "supp_fig2_k_transitions.png"))
