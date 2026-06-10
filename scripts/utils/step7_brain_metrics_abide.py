#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step7: ABIDE brain metrics pipeline
- Static FC (246x246 Fisher-z)
- Network intra/inter strength (Yeo7/17 or custom mapping)
- Graph metrics: GBC, degree/eigenvector centrality, clustering coeff, local/global efficiency,
                 participation coefficient, within-module Z
- CAP (Co-Activation Patterns): patterns, occurrence, duration, amplitude
- ALFF / fALFF (0.01-0.08) + optional bands
- Gradients (diffusion maps) first/second component

Inputs:
  --abide-pklz  DATA/combined_ABIDE_information_with_fMRI.pklz
  --subset      asd/td/all (default: asd)
  --outdir      output directory
  --tr          repetition time (default: 2.0)
  --n-cap       CAP K clusters (default: 6)
  --net-map     CSV mapping ROI->network label (columns: roi,network)

Outputs under outdir:
  - fc_subject_mean.npy (ROI x ROI)
  - fc_group_mean.npy   (ROI x ROI)
  - caps_*.npy/.csv, cap_occurrence.csv, cap_duration.csv
  - alff.csv, falff.csv
  - gradients.npy (ROI x 2)
  - graphs/*.csv (node-level metrics), graph_global.csv
  - figures/*.png
"""

from __future__ import annotations

import os
import json
import argparse
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List


def build_site_keys(df, dataset_tag=None):
    df = df.copy()
    if dataset_tag is not None:
        dataset_series = pd.Series(dataset_tag, index=df.index)
    elif "ABIDE" in df.columns:
        dataset_series = df["ABIDE"].map({1: "ABIDE1", 2: "ABIDE2"}).fillna("ABIDE1").astype(str)
    else:
        dataset_series = pd.Series("ABIDE1", index=df.index)
    if "SITE_ID" in df.columns:
        site_id = df["SITE_ID"].astype(str)
    elif "SITE" in df.columns:
        site_id = df["SITE"].astype(str)
    else:
        site_id = pd.Series("UNKNOWN", index=df.index)
    if "SITE" in df.columns:
        site_base = df["SITE"].astype(str)
    else:
        site_base = site_id.str.replace(r"_\d+$", "", regex=True)
    df["DATASET"] = dataset_series
    df["SITE_NAME"] = site_base
    df["SITE_KEY_STRICT"] = df["DATASET"] + "/" + site_id
    df["SITE_KEY_BASE"]   = df["DATASET"] + "/" + site_base
    return df


def load_and_preprocess_data(ABIDE_PATH, CMI_PATH=None):
    """Load and QC ABIDE (and optionally CMI) pickle data."""
    print("Loading ABIDE data ...")
    with open(ABIDE_PATH, "rb") as f:
        original_df = pickle.load(f)
    original_df = original_df[(original_df["percentofvolsrepaired"] <= 10) & (original_df["mean_fd"] <= 0.5)]
    original_df["data"] = original_df["data"].apply(lambda x: np.array(x) if isinstance(x, list) else x)
    def is_valid_array(x):
        x = np.array(x); return x.ndim == 2 and x.shape[0] >= 120 and np.isfinite(x).all()
    original_df = original_df[original_df["data"].apply(is_valid_array)].copy()
    def norm_abide(x):
        x = np.array(x); x = (x - np.mean(x)) / (np.std(x) + 1e-6); return x[:120, :]
    original_df["data"] = original_df["data"].apply(norm_abide)
    original_df["DX_GROUP"] = original_df["DX_GROUP"].map({1: 0, 2: 1})  # 0=ASD, 1=TD
    original_df = build_site_keys(original_df)
    abide_df = original_df[original_df["DATASET"].isin(["ABIDE1", "ABIDE2"])].copy()

    cmi_df = None
    if CMI_PATH is not None:
        with open(CMI_PATH, "rb") as f:
            cmi_df = pickle.load(f)
        def norm_cmi(x):
            x = np.array(x); return (x - np.mean(x)) / (np.std(x) + 1e-6)
        def is_valid_cmi(x):
            x = np.array(x); return x.ndim == 2 and x.shape[0] == 375 and np.isfinite(x).all()
        cmi_df["data"] = cmi_df["data"].apply(norm_cmi)
        cmi_df = cmi_df[cmi_df["data"].apply(is_valid_cmi)].copy()
        cmi_df["DX_GROUP"] = cmi_df["label"].map({"asd": 0, "td": 1})
        cmi_df["SEX"] = cmi_df["gender"]
        cmi_df["AGE_AT_SCAN"] = cmi_df["age"]
        if "SITE_ID" not in cmi_df.columns:
            cmi_df["SITE_ID"] = cmi_df.get("site", "CMI_VIRTUAL_SITE")
        if "SITE" not in cmi_df.columns:
            cmi_df["SITE"] = cmi_df["SITE_ID"].astype(str).str.replace(r"_\d+$", "", regex=True)
        cmi_df = build_site_keys(cmi_df, dataset_tag='CMI')
        print(f"ABIDE: {len(abide_df)} | CMI: {len(cmi_df)}")
    else:
        print(f"ABIDE: {len(abide_df)}")

    return abide_df, cmi_df

from scipy import signal
from nilearn.signal import clean
from nilearn.connectome import ConnectivityMeasure
import networkx as nx
from brainspace.gradient import GradientMaps
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from nilearn import datasets as nl_datasets, surface as nl_surface, plotting as nl_plot
import nibabel as nib

# Step2 labels loader
def load_step2_labels(step2_outdir: str) -> np.ndarray:
    p = os.path.join(step2_outdir, 'abide_asd_labels.npy')
    if not os.path.exists(p):
        raise SystemExit(f'Missing labels: {p}')
    return np.load(p).astype(int)


def preprocess_ts(ts: np.ndarray, tr: float, low: float = 0.01, high: float = 0.08,
                  confounds: np.ndarray | None = None) -> np.ndarray:
    return clean(ts, t_r=tr, detrend=True, standardize=True,
                 low_pass=high, high_pass=low,
                 confounds=confounds, standardize_confounds=True)


def build_confounds_gsr(ts: np.ndarray) -> np.ndarray:
    T = ts.shape[0]
    gsr  = ts.mean(axis=1, keepdims=True)
    lin  = np.linspace(-1, 1, T)[:, None]
    quad = lin**2
    return np.hstack([gsr, lin, quad])


def compute_fc(ts: np.ndarray, use_partial: bool = False) -> np.ndarray:
    eps = 1e-8
    ts_std = (ts - ts.mean(0)) / (ts.std(0) + eps)
    if not use_partial:
        fc_r = np.corrcoef(ts_std, rowvar=False)
    else:
        cov  = np.cov(ts_std, rowvar=False)
        prec = np.linalg.pinv(cov + 1e-6*np.eye(cov.shape[0]))
        d = np.sqrt(np.diag(prec) + eps)
        fc_r = -prec / (d[:, None] * d[None, :])
    np.fill_diagonal(fc_r, 0.0)
    return np.arctanh(np.clip(fc_r, -0.999999, 0.999999))


def global_brain_connectivity(fc_z: np.ndarray) -> np.ndarray:
    fc_r = np.tanh(fc_z)
    return fc_r.mean(axis=1)


def _pos_weighted_graph(fc_z: np.ndarray, density: float = 0.15) -> nx.Graph:
    fc_r = np.tanh(fc_z).copy()
    np.fill_diagonal(fc_r, 0.0)
    w = np.clip(fc_r, 0, None)

    vals = w[np.triu_indices_from(w, k=1)]
    kth = np.quantile(vals[vals>0], 1.0 - density) if (vals>0).any() else np.inf
    w_thr = np.where(w >= kth, w, 0.0)

    G = nx.from_numpy_array(w_thr)
    for i, j in zip(*np.where(np.triu(w_thr, 1) > 0)):
        G[i][j]['weight'] = float(w_thr[i, j])
    return G

def _node_strength(G: nx.Graph) -> np.ndarray:
    n = G.number_of_nodes()
    s = np.array([sum(d.get('weight',1.0) for _, _, d in G.edges(nbr, data=True)) for nbr in G.nodes()])
    return s / max(n-1, 1)

def _weighted_efficiency(G: nx.Graph) -> tuple[float, np.ndarray]:
    """
    """
    import math
    H = G.copy()
    for u, v, d in H.edges(data=True):
        w = d.get('weight', 0.0)
        if w <= 0: d['length'] = math.inf
        else:      d['length'] = 1.0 / w

    lengths = dict(nx.all_pairs_dijkstra_path_length(H, weight='length'))
    n = G.number_of_nodes()
    eff_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            dij = lengths[i].get(j, math.inf)
            if math.isfinite(dij) and dij > 0:
                eff_pairs.append(1.0/dij)
    geff = (2.0 / (n*(n-1))) * np.sum(eff_pairs) if eff_pairs else 0.0

    leffs = []
    for u in range(n):
        nbrs = list(G.neighbors(u))
        if len(nbrs) < 2:
            leffs.append(0.0); continue
        S = H.subgraph(nbrs).copy()
        lengths_s = dict(nx.all_pairs_dijkstra_path_length(S, weight='length'))
        m = len(nbrs)
        eff_pairs_s = []
        for a in range(m):
            for b in range(a+1, m):
                dij = lengths_s[nbrs[a]].get(nbrs[b], math.inf)
                if math.isfinite(dij) and dij > 0:
                    eff_pairs_s.append(1.0/dij)
        leff = (2.0 / (m*(m-1))) * np.sum(eff_pairs_s) if eff_pairs_s else 0.0
        leffs.append(leff)
    return geff, np.array(leffs)

def graph_metrics(fc_z: np.ndarray, density: float = 0.15) -> Dict[str, np.ndarray]:
    G = _pos_weighted_graph(fc_z, density=density)

    strength = _node_strength(G)

    try:
        eig = np.array(list(nx.eigenvector_centrality_numpy(G, weight='weight').values()))
    except nx.AmbiguousSolution:
        largest_cc = max(nx.connected_components(G), key=len)
        G_cc = G.subgraph(largest_cc)
        eig_full = np.zeros(G.number_of_nodes())
        if len(largest_cc) > 1:
            eig_cc = np.array(list(nx.eigenvector_centrality_numpy(G_cc, weight='weight').values()))
            for i, node in enumerate(largest_cc):
                eig_full[node] = eig_cc[i]
        eig = eig_full

    clust = np.array(list(nx.clustering(G, weight='weight').values()))

    geff, leff_vec = _weighted_efficiency(G)

    return dict(strength=strength, eig=eig, clust=clust, geff=geff, leff=leff_vec)


def module_metrics(fc_z: np.ndarray, net_labels: np.ndarray, density: float = 0.15) -> Dict[str, np.ndarray]:
    """
    Participation coefficient / Within-module degree z-score
    """
    try:
        import bct  # pip install bctpy
        fc_r = np.tanh(fc_z); np.fill_diagonal(fc_r, 0)
        W = np.clip(fc_r, 0, None)
        vals = W[np.triu_indices_from(W, 1)]
        kth = np.quantile(vals[vals>0], 1-density) if (vals>0).any() else np.inf
        W = np.where(W >= kth, W, 0.0)

        comm = net_labels.astype(int)
        if comm.min() == 0: comm = comm + 1

        Pc = bct.participation_coef(W, comm, 'und')          # [N,]
        Z  = bct.module_degree_zscore(W, comm, 'und')        # [N,]
        return dict(participation=Pc, moduleZ=Z)
    except ImportError:
        print("Warning: bctpy not installed, skipping module metrics")
        return dict(participation=np.zeros(len(net_labels)), moduleZ=np.zeros(len(net_labels)))


def load_network_map(path: str, num_rois: int) -> np.ndarray:
    if path is None or not os.path.exists(path):
        return np.zeros((num_rois,), dtype=int)
    df = pd.read_csv(path)
    # expect columns roi (1..C) and network (int or string -> category)
    nets = pd.Categorical(df.sort_values('roi')['network']).codes
    return nets.astype(int)


def intra_inter_strength(fc_z: np.ndarray, net_labels: np.ndarray) -> Dict[str, float]:
    fc_r = np.tanh(fc_z)
    C = fc_r.shape[0]
    nets = np.unique(net_labels)
    intra_vals = []
    inter_vals = []
    for a in nets:
        idx = np.where(net_labels == a)[0]
        if len(idx) < 2:
            continue
        sub = fc_r[np.ix_(idx, idx)]
        m = sub[np.triu_indices_from(sub, k=1)].mean()
        intra_vals.append(m)
    for i, a in enumerate(nets):
        for b in nets[i+1:]:
            ia = np.where(net_labels == a)[0]
            ib = np.where(net_labels == b)[0]
            if len(ia) == 0 or len(ib) == 0:
                continue
            inter_vals.append(fc_r[np.ix_(ia, ib)].mean())
    return dict(intra=np.nanmean(intra_vals) if intra_vals else np.nan,
                inter=np.nanmean(inter_vals) if inter_vals else np.nan)


def compute_cap(ts: np.ndarray, n_clusters: int = 6):
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(ts)
    caps = km.cluster_centers_
    labels = km.labels_
    occ = np.bincount(labels, minlength=n_clusters) / len(labels)
    return caps, labels, occ


def find_optimal_k(ts: np.ndarray, max_k: int = 10) -> int:
    """Find optimal k using elbow method"""
    from sklearn.metrics import silhouette_score
    inertias = []
    sil_scores = []
    k_range = range(2, min(max_k + 1, len(ts) // 10))  # Ensure enough samples per cluster
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(ts)
        inertias.append(km.inertia_)
        if k > 1:  # Silhouette score needs at least 2 clusters
            sil_scores.append(silhouette_score(ts, km.labels_))
        else:
            sil_scores.append(0)
    
    # Elbow method: find the point where decrease in inertia starts to level off
    if len(inertias) > 2:
        # Calculate second derivative to find elbow
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        elbow_k = k_range[np.argmax(second_diffs) + 1] if len(second_diffs) > 0 else k_range[0]
    else:
        elbow_k = k_range[0]
    
    # Also consider silhouette score
    best_sil_k = k_range[np.argmax(sil_scores)] if sil_scores else k_range[0]
    
    # Return the k that balances both methods
    return min(elbow_k, best_sil_k)


def compute_transition_matrix(labels: np.ndarray, n_states: int) -> np.ndarray:
    """Compute state transition probability matrix"""
    trans_matrix = np.zeros((n_states, n_states))
    for i in range(len(labels) - 1):
        trans_matrix[labels[i], labels[i + 1]] += 1
    
    # Normalize rows to get probabilities
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    trans_matrix = np.divide(trans_matrix, row_sums, out=np.zeros_like(trans_matrix), where=row_sums != 0)
    
    return trans_matrix


def compute_cap_metrics(ts: np.ndarray, n_clusters: int = None) -> Dict:
    """Compute comprehensive CAP metrics"""
    if n_clusters is None:
        n_clusters = find_optimal_k(ts)
    
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(ts)
    caps = km.cluster_centers_
    labels = km.labels_
    
    # Occurrence probabilities
    occ_probs = np.bincount(labels, minlength=n_clusters) / len(labels)
    
    # Transition probabilities
    trans_matrix = compute_transition_matrix(labels, n_clusters)
    
    # Duration (average consecutive frames per state)
    durations = []
    for state in range(n_clusters):
        state_mask = (labels == state)
        if np.any(state_mask):
            # Find consecutive runs
            diff = np.diff(np.concatenate([[False], state_mask, [False]]).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            if len(starts) > 0 and len(ends) > 0:
                durations.append(np.mean(ends - starts))
            else:
                durations.append(0)
        else:
            durations.append(0)
    
    # Amplitude (mean activation strength per state)
    amplitudes = []
    for state in range(n_clusters):
        state_frames = ts[labels == state]
        if len(state_frames) > 0:
            amplitudes.append(np.mean(np.linalg.norm(state_frames, axis=1)))
        else:
            amplitudes.append(0)
    
    return {
        'optimal_k': n_clusters,
        'caps': caps,
        'labels': labels,
        'occurrence_probs': occ_probs,
        'transition_matrix': trans_matrix,
        'durations': np.array(durations),
        'amplitudes': np.array(amplitudes)
    }


def compute_alff(ts: np.ndarray, tr: float, band=(0.01, 0.08), full_band=(0.0, 0.25)):
    fs = 1.0 / tr
    T, R = ts.shape
    nperseg = min(256, T//2)
    freqs, psd = signal.welch(ts, fs=fs, nperseg=nperseg, axis=0)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    
    full_hi = min(full_band[1], 0.95*fs/2)
    full_mask = (freqs >= full_band[0]) & (freqs <= full_hi)
    
    alff = np.sqrt(np.trapz(psd[band_mask, :], freqs[band_mask], axis=0))
    falff = alff / (np.sqrt(np.trapz(psd[full_mask, :], freqs[full_mask], axis=0)) + 1e-8)
    return alff, falff


def compute_gradients_group(fc_z: np.ndarray, n_components: int = 2, density: float = 0.1) -> np.ndarray:
    from brainspace.gradient import GradientMaps
    from sklearn.metrics.pairwise import cosine_similarity
    
    fc_r = np.tanh(fc_z)
    np.fill_diagonal(fc_r, 0)
    A = cosine_similarity(fc_r)
    n = A.shape[0]
    k = max(1, int(np.floor(density * n)))
    thr = np.partition(A, -k, axis=1)[:, -k][:, None]
    A = np.where(A >= thr, A, 0)
    A = np.maximum(A, A.T)
    
    row_sums = A.sum(axis=1)
    A = A / (row_sums[:, np.newaxis] + 1e-8)

    gm = GradientMaps(n_components=n_components, approach='dm', random_state=0)
    gm.fit(A)
    return gm.gradients_


def save_fig_bar(values: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(10, 3))
    plt.bar(range(len(values)), values)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_group_fc_heatmaps(fcs: np.ndarray, labels: np.ndarray, outdir: str):
    # fcs: N x C x C, labels: N
    os.makedirs(outdir, exist_ok=True)
    K = int(labels.max() + 1)
    for k in range(K):
        mask = (labels == k)
        if mask.sum() == 0:
            continue
        fc_mean_z = np.nanmean(fcs[mask], axis=0)
        fc_mean_r = np.tanh(fc_mean_z)
        np.fill_diagonal(fc_mean_r, 1.0)
        plt.figure(figsize=(8, 6))
        im = plt.imshow(fc_mean_r, cmap='RdBu_r', vmin=-1.0, vmax=1.0)
        plt.title(f'Functional Connectivity - Subtype {k+1}\n(r, n={mask.sum()})', fontsize=14)
        plt.xlabel('ROI', fontsize=12)
        plt.ylabel('ROI', fontsize=12)
        
        # Add colorbar with label
        cbar = plt.colorbar(im, shrink=0.8)
        cbar.set_label('r', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'fc_mean_subtype{k}.png'), dpi=200, bbox_inches='tight')
        plt.close()


def violin_by_subtype(values: np.ndarray, labels: np.ndarray, title: str, out_path: str):
    # values: N or N x R (flatten to summary if needed). Here assume N vector.
    import seaborn as sns
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    K = int(labels.max() + 1)
    df = pd.DataFrame({'value': values, 'subtype': labels})
    plt.figure(figsize=(6, 4))
    sns.violinplot(x='subtype', y='value', data=df, inner='box', palette='Set2')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_volume_from_parcels(atlas_img: nib.Nifti1Image, roi_values: np.ndarray) -> nib.Nifti1Image:
    atlas_data = atlas_img.get_fdata()
    out = np.zeros_like(atlas_data, dtype=np.float32)
    for i, v in enumerate(roi_values):
        mask = (atlas_data == (i + 1))
        if np.any(mask):
            out[mask] = float(v)
    return nib.Nifti1Image(out, affine=atlas_img.affine, header=atlas_img.header)


def plot_surface_quad(stat_img: nib.Nifti1Image, out_path: str, cmap: str = 'PuOr', vmax: float | None = None, title: str = ""):
    fsavg = nl_datasets.fetch_surf_fsaverage('fsaverage5')
    tex_lh = nl_surface.vol_to_surf(stat_img, fsavg.pial_left)
    tex_rh = nl_surface.vol_to_surf(stat_img, fsavg.pial_right)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), subplot_kw={'projection': '3d'})
    
    # Plot with colorbar
    nl_plot.plot_surf_stat_map(fsavg.infl_left, tex_lh, hemi='left', view='lateral', colorbar=True,
                                bg_map=fsavg.sulc_left, cmap=cmap, vmax=vmax, axes=axes[0, 0])
    nl_plot.plot_surf_stat_map(fsavg.infl_left, tex_lh, hemi='left', view='medial', colorbar=True,
                                bg_map=fsavg.sulc_left, cmap=cmap, vmax=vmax, axes=axes[1, 0])
    nl_plot.plot_surf_stat_map(fsavg.infl_right, tex_rh, hemi='right', view='lateral', colorbar=True,
                                bg_map=fsavg.sulc_right, cmap=cmap, vmax=vmax, axes=axes[0, 1])
    nl_plot.plot_surf_stat_map(fsavg.infl_right, tex_rh, hemi='right', view='medial', colorbar=True,
                                bg_map=fsavg.sulc_right, cmap=cmap, vmax=vmax, axes=axes[1, 1])
    
    # Add titles
    axes[0, 0].set_title("Left Lateral", fontsize=12)
    axes[0, 1].set_title("Right Lateral", fontsize=12)
    axes[1, 0].set_title("Left Medial", fontsize=12)
    axes[1, 1].set_title("Right Medial", fontsize=12)
    
    # Add overall title
    if title:
        fig.suptitle(title, fontsize=14, y=0.95)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_cap_heatmaps(cap_metrics: Dict, group_name: str, outdir: str):
    """Plot CAP-related heatmaps"""
    import seaborn as sns
    
    # Transition matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cap_metrics['transition_matrix'], annot=True, fmt='.3f', cmap='Blues',
                xticklabels=[f'State {i}' for i in range(cap_metrics['optimal_k'])],
                yticklabels=[f'State {i}' for i in range(cap_metrics['optimal_k'])])
    plt.title(f'{group_name} - State Transition Probabilities')
    plt.xlabel('Next State')
    plt.ylabel('Current State')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{group_name.lower()}_transition_matrix.png'), dpi=200)
    plt.close()
    
    # Occurrence probabilities bar plot
    plt.figure(figsize=(10, 4))
    states = range(cap_metrics['optimal_k'])
    plt.bar(states, cap_metrics['occurrence_probs'], color='skyblue', alpha=0.7)
    plt.title(f'{group_name} - State Occurrence Probabilities')
    plt.xlabel('State')
    plt.ylabel('Probability')
    plt.xticks(states)
    for i, v in enumerate(cap_metrics['occurrence_probs']):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{group_name.lower()}_occurrence_probs.png'), dpi=200)
    plt.close()
    
    # Duration and amplitude comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.bar(states, cap_metrics['durations'], color='lightcoral', alpha=0.7)
    ax1.set_title(f'{group_name} - Average State Duration')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Duration (frames)')
    ax1.set_xticks(states)
    
    ax2.bar(states, cap_metrics['amplitudes'], color='lightgreen', alpha=0.7)
    ax2.set_title(f'{group_name} - Average State Amplitude')
    ax2.set_xlabel('State')
    ax2.set_ylabel('Amplitude')
    ax2.set_xticks(states)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{group_name.lower()}_duration_amplitude.png'), dpi=200)
    plt.close()


def plot_cap_comparison_heatmap(cap_metrics_dict: Dict[str, Dict], outdir: str):
    """Plot comparison heatmaps across groups"""
    import seaborn as sns
    
    groups = list(cap_metrics_dict.keys())
    max_k = max(metrics['optimal_k'] for metrics in cap_metrics_dict.values())
    
    # Create comparison matrices
    occ_matrix = np.zeros((len(groups), max_k))
    trans_matrices = {}
    
    for i, (group, metrics) in enumerate(cap_metrics_dict.items()):
        k = metrics['optimal_k']
        occ_matrix[i, :k] = metrics['occurrence_probs']
        trans_matrices[group] = metrics['transition_matrix']
    
    # Occurrence probabilities comparison
    plt.figure(figsize=(max_k, len(groups)))
    sns.heatmap(occ_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=[f'State {i}' for i in range(max_k)],
                yticklabels=groups)
    plt.title('State Occurrence Probabilities Comparison')
    plt.xlabel('State')
    plt.ylabel('Group')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'cap_occurrence_comparison.png'), dpi=200)
    plt.close()
    
    # Transition matrices comparison (if all groups have same k)
    if len(set(metrics['optimal_k'] for metrics in cap_metrics_dict.values())) == 1:
        k = list(cap_metrics_dict.values())[0]['optimal_k']
        fig, axes = plt.subplots(1, len(groups), figsize=(6*len(groups), 5))
        if len(groups) == 1:
            axes = [axes]
        
        for i, (group, metrics) in enumerate(cap_metrics_dict.items()):
            sns.heatmap(metrics['transition_matrix'], annot=True, fmt='.3f', cmap='Blues',
                       xticklabels=[f'S{j}' for j in range(k)],
                       yticklabels=[f'S{j}' for j in range(k)],
                       ax=axes[i])
            axes[i].set_title(f'{group} Transitions')
            axes[i].set_xlabel('Next State')
            axes[i].set_ylabel('Current State')
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'cap_transition_comparison.png'), dpi=200)
        plt.close()


def plot_cap_state_brain_maps(cap_metrics: Dict, group_name: str, outdir: str, atlas_img):
    """Plot brain maps for each CAP state"""
    try:
        caps = cap_metrics['caps']  # Shape: [n_states, n_rois]
        
        # Global scaling across all states
        vmax_global = float(np.percentile(np.abs(caps), 95))
        
        for state_idx in range(cap_metrics['optimal_k']):
            state_pattern = caps[state_idx, :]  # [n_rois]
            
            # Create NIfTI image from ROI values
            img = make_volume_from_parcels(atlas_img, state_pattern)
            
            # Generate quad-panel brain map with title
            out_path = os.path.join(outdir, f'{group_name.lower()}_state_{state_idx}_brain_map.png')
            plot_surface_quad(img, out_path, cmap='RdBu_r', vmax=vmax_global,
                            title=f'CAP State {state_idx} - {group_name}')
            
        print(f'Generated {cap_metrics["optimal_k"]} state brain maps for {group_name}')
        
    except Exception as e:
        print(f'CAP state brain map generation failed for {group_name}: {e}')


def plot_cap_state_comparison_brain_maps(cap_metrics_dict: Dict[str, Dict], outdir: str, atlas_img):
    """Plot comparison brain maps for corresponding states across groups"""
    try:
        
        # Find the minimum number of states across all groups
        min_states = min(metrics['optimal_k'] for metrics in cap_metrics_dict.values())
        
        for state_idx in range(min_states):
            # Collect state patterns from all groups
            state_patterns = {}
            for group_name, metrics in cap_metrics_dict.items():
                state_patterns[group_name] = metrics['caps'][state_idx, :]
            
            # Global scaling across all groups for this state
            all_patterns = np.vstack(list(state_patterns.values()))
            vmax_global = float(np.percentile(np.abs(all_patterns), 95))
            
            # Create comparison figure
            n_groups = len(cap_metrics_dict)
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})
            axes = axes.flatten()
            
            for i, (group_name, pattern) in enumerate(state_patterns.items()):
                if i >= 4:  # Limit to 4 groups for 2x2 grid
                    break
                    
                img = make_volume_from_parcels(atlas_img, pattern)
                
                # Get surface textures
                fsavg = nl_datasets.fetch_surf_fsaverage('fsaverage5')
                tex_lh = nl_surface.vol_to_surf(img, fsavg.pial_left)
                tex_rh = nl_surface.vol_to_surf(img, fsavg.pial_right)
                
                # Plot left hemisphere
                nl_plot.plot_surf_stat_map(fsavg.infl_left, tex_lh, hemi='left', view='lateral', 
                                          colorbar=False, bg_map=fsavg.sulc_left, cmap='RdBu_r', 
                                          vmax=vmax_global, axes=axes[i])
                axes[i].set_title(f'{group_name} State {state_idx}')
            
            # Hide unused subplots
            for j in range(i+1, 4):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f'state_{state_idx}_comparison_brain_maps.png'), dpi=300)
            plt.close()
            
        print(f'Generated comparison brain maps for {min_states} states across {len(cap_metrics_dict)} groups')
        
    except Exception as e:
        print(f'CAP state comparison brain map generation failed: {e}')


def plot_gradient_brain_maps(gradients: np.ndarray, group_name: str, outdir: str, atlas_img):
    """Plot brain maps for functional gradients (Gradient 1 and Gradient 2)"""
    try:
        
        # Global scaling across both gradients
        vmax_global = float(np.percentile(np.abs(gradients), 95))
        
        # Plot Gradient 1
        grad1_pattern = gradients[:, 0]  # First gradient
        img1 = make_volume_from_parcels(atlas_img, grad1_pattern)
        out_path1 = os.path.join(outdir, f'{group_name.lower()}_gradient1_quad.png')
        plot_surface_quad(img1, out_path1, cmap='RdBu_r', vmax=vmax_global, 
                         title=f'Functional Gradient 1 - {group_name}')
        
        # Plot Gradient 2
        grad2_pattern = gradients[:, 1]  # Second gradient
        img2 = make_volume_from_parcels(atlas_img, grad2_pattern)
        out_path2 = os.path.join(outdir, f'{group_name.lower()}_gradient2_quad.png')
        plot_surface_quad(img2, out_path2, cmap='RdBu_r', vmax=vmax_global,
                         title=f'Functional Gradient 2 - {group_name}')
        
        print(f'Generated gradient brain maps for {group_name}')
        
    except Exception as e:
        print(f'Gradient brain map generation failed for {group_name}: {e}')


def _neg_edge_ratio(fc_z: np.ndarray) -> float:
    r = np.tanh(fc_z); iu = np.triu_indices_from(r, 1); vals = r[iu]
    return float((vals < 0).mean())

def print_neg_qc(fcs: np.ndarray, tag: str):
    ratios = [_neg_edge_ratio(m) for m in fcs]
    print(f"[QC] {tag} neg-edge ratio: mean={np.mean(ratios):.3f}, sd={np.std(ratios):.3f}")


def plot_gradient_comparison_brain_maps(gradients_dict: Dict[str, np.ndarray], outdir: str, atlas_img):
    """Plot comparison brain maps for gradients across groups"""
    try:
        
        # Plot Gradient 1 comparison
        grad1_patterns = {group: grads[:, 0] for group, grads in gradients_dict.items()}
        all_grad1 = np.vstack(list(grad1_patterns.values()))
        vmax_grad1 = float(np.percentile(np.abs(all_grad1), 95))
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})
        axes = axes.flatten()
        
        for i, (group_name, pattern) in enumerate(grad1_patterns.items()):
            if i >= 4:  # Limit to 4 groups for 2x2 grid
                break
                
            img = make_volume_from_parcels(atlas_img, pattern)
            
            # Get surface textures
            fsavg = nl_datasets.fetch_surf_fsaverage('fsaverage5')
            tex_lh = nl_surface.vol_to_surf(img, fsavg.pial_left)
            tex_rh = nl_surface.vol_to_surf(img, fsavg.pial_right)
            
            # Plot left hemisphere
            nl_plot.plot_surf_stat_map(fsavg.infl_left, tex_lh, hemi='left', view='lateral', 
                                      colorbar=False, bg_map=fsavg.sulc_left, cmap='RdBu_r', 
                                      vmax=vmax_grad1, axes=axes[i])
            axes[i].set_title(f'{group_name} Gradient 1')
        
        # Hide unused subplots
        for j in range(i+1, 4):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'gradient1_comparison_brain_maps.png'), dpi=300)
        plt.close()
        
        # Plot Gradient 2 comparison
        grad2_patterns = {group: grads[:, 1] for group, grads in gradients_dict.items()}
        all_grad2 = np.vstack(list(grad2_patterns.values()))
        vmax_grad2 = float(np.percentile(np.abs(all_grad2), 95))
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})
        axes = axes.flatten()
        
        for i, (group_name, pattern) in enumerate(grad2_patterns.items()):
            if i >= 4:  # Limit to 4 groups for 2x2 grid
                break
                
            img = make_volume_from_parcels(atlas_img, pattern)
            
            # Get surface textures
            fsavg = nl_datasets.fetch_surf_fsaverage('fsaverage5')
            tex_lh = nl_surface.vol_to_surf(img, fsavg.pial_left)
            tex_rh = nl_surface.vol_to_surf(img, fsavg.pial_right)
            
            # Plot left hemisphere
            nl_plot.plot_surf_stat_map(fsavg.infl_left, tex_lh, hemi='left', view='lateral', 
                                      colorbar=False, bg_map=fsavg.sulc_left, cmap='RdBu_r', 
                                      vmax=vmax_grad2, axes=axes[i])
            axes[i].set_title(f'{group_name} Gradient 2')
        
        # Hide unused subplots
        for j in range(i+1, 4):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'gradient2_comparison_brain_maps.png'), dpi=300)
        plt.close()
        
        print(f'Generated gradient comparison brain maps for {len(gradients_dict)} groups')
        
    except Exception as e:
        print(f'Gradient comparison brain map generation failed: {e}')


def compute_and_save_fc(args) -> None:
    """Compute FC matrices from raw ABIDE pickle and save as fcs_asd.npy / fcs_td.npy.

    args requires: abide_pklz, step2_outdir, outdir.
    If bna_atlas is provided and the atlas NIfTI exists, it is used for extraction;
    otherwise the 'data' column (pre-extracted time series) is used directly.
    """
    import pickle
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.abide_pklz, 'rb') as f:
        df = pickle.load(f)

    # DX_GROUP normalisation: 1=ASD -> 0, 2=TDC -> 1
    if set(df['DX_GROUP'].unique()).issubset({1, 2}):
        df = df.copy()
        df['DX_GROUP'] = df['DX_GROUP'].map({1: 0, 2: 1})

    labels = load_step2_labels(args.step2_outdir)
    df_asd = df[df['DX_GROUP'] == 0].reset_index(drop=True)
    df_td  = df[df['DX_GROUP'] == 1].reset_index(drop=True)

    if labels.shape[0] != len(df_asd):
        raise SystemExit(
            f"Label length {labels.shape[0]} != ABIDE-ASD rows {len(df_asd)}. "
            "Ensure smoke_test_abide100.pkl and step2-outdir/abide_asd_labels.npy are aligned."
        )

    def ts_to_fc(df_grp: pd.DataFrame) -> np.ndarray:
        fcs = []
        for _, row in df_grp.iterrows():
            ts = np.array(row['data'], dtype=np.float32)  # (T, 246)
            # Normalise then compute Pearson + Fisher-z
            ts = (ts - ts.mean(0)) / (ts.std(0) + 1e-6)
            fc = np.corrcoef(ts.T)  # (246, 246)
            fc = np.clip(fc, -0.9999, 0.9999)
            fcs.append(np.arctanh(fc).astype(np.float32))
        return np.stack(fcs)  # (N, 246, 246)

    print("  Computing FC for ASD...")
    fcs_asd = ts_to_fc(df_asd)
    print("  Computing FC for TDC...")
    fcs_td  = ts_to_fc(df_td)

    np.save(os.path.join(args.outdir, 'fcs_asd.npy'), fcs_asd)
    np.save(os.path.join(args.outdir, 'fcs_td.npy'),  fcs_td)
    np.save(os.path.join(args.outdir, 'labels_asd.npy'), labels)
    print(f"  Saved fcs_asd {fcs_asd.shape}, fcs_td {fcs_td.shape} -> {args.outdir}")

    # GBC: shape (N, 246, 3) where axis-2 = [all, positive, negative]
    def compute_gbc(df_grp: pd.DataFrame) -> np.ndarray:
        gbcs = []
        for _, row in df_grp.iterrows():
            ts = np.array(row['data'], dtype=np.float32)  # (T, 246)
            ts = (ts - ts.mean(0)) / (ts.std(0) + 1e-6)
            fc = np.corrcoef(ts.T)  # (246, 246)
            np.fill_diagonal(fc, 0)
            gbc_all = fc.mean(1)
            gbc_pos = np.where(fc > 0, fc, 0).mean(1)
            gbc_neg = np.where(fc < 0, fc, 0).mean(1)
            gbcs.append(np.stack([gbc_all, gbc_pos, gbc_neg], axis=1))  # (246, 3)
        return np.stack(gbcs)  # (N, 246, 3)

    print("  Computing GBC...")
    gbcs_asd = compute_gbc(df_asd)
    gbcs_td  = compute_gbc(df_td)
    np.save(os.path.join(args.outdir, 'gbcs_asd_gsr.npy'), gbcs_asd)
    np.save(os.path.join(args.outdir, 'gbcs_td_gsr.npy'),  gbcs_td)

    # ALFF = RMS of bandpass-filtered (0.01-0.1 Hz) signal per ROI
    def compute_alff(df_grp: pd.DataFrame, tr: float = 2.0) -> np.ndarray:
        from scipy.signal import butter, filtfilt
        alffs = []
        nyq = 0.5 / tr
        b, a = butter(4, [0.01 / nyq, 0.1 / nyq], btype='band')
        for _, row in df_grp.iterrows():
            ts = np.array(row['data'], dtype=np.float32)  # (T, 246)
            try:
                filtered = filtfilt(b, a, ts, axis=0)
                alff = np.sqrt(np.mean(filtered ** 2, axis=0))
            except Exception:
                alff = np.full(ts.shape[1], np.nan)
            alffs.append(alff)
        return np.stack(alffs)  # (N, 246)

    print("  Computing ALFF...")
    alffs_asd = compute_alff(df_asd)
    alffs_td  = compute_alff(df_td)
    np.save(os.path.join(args.outdir, 'alffs_asd_gsr.npy'), alffs_asd)
    np.save(os.path.join(args.outdir, 'alffs_td_gsr.npy'),  alffs_td)
    print(f"  Saved GBC ({gbcs_asd.shape}) and ALFF ({alffs_asd.shape}) -> {args.outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--abide-pklz', type=str, default='DATA/combined_ABIDE_information_with_fMRI.pklz')
    ap.add_argument('--subset', type=str, choices=['asd', 'td', 'all'], default='asd')
    ap.add_argument('--outdir', type=str, default='unsup_results/step7_abide')
    ap.add_argument('--tr', type=float, default=2.0)
    ap.add_argument('--n-cap', type=int, default=6)
    ap.add_argument('--net-map', type=str, default=None)
    ap.add_argument('--step2-outdir', type=str, required=True)
    ap.add_argument('--bna-atlas', type=str, default='atlas/brainnetome/BN_Atlas_246_1mm.nii.gz')
    ap.add_argument('--export-gsr', action='store_true',
                    help='Also export a branch with global signal regression (GSR).')
    ap.add_argument('--partial', action='store_true',
                    help='Use partial correlation (precision-based) instead of Pearson.')
    args = ap.parse_args()
    
    BNA_ATLAS_PATH = args.bna_atlas
    atlas_img = nib.load(BNA_ATLAS_PATH)

    os.makedirs(args.outdir, exist_ok=True)
    figdir = os.path.join(args.outdir, 'figures'); os.makedirs(figdir, exist_ok=True)
    graphdir = os.path.join(args.outdir, 'graphs'); os.makedirs(graphdir, exist_ok=True)

    # Load ABIDE via step1 to ensure consistent order and labels compatibility
    print('Loading ABIDE & CMI ...')
    try:
        import sys as _sys, os as _os
        _cur = _os.path.dirname(_os.path.abspath(__file__))
        _root = _os.path.dirname(_cur)
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from utils.step7_brain_metrics_abide import load_and_preprocess_data
        abide_df, _ = load_and_preprocess_data(args.abide_pklz, 'CMI-DATA/combined_asd_td_rest_run1_data.pklz')
        abide_asd = abide_df[abide_df['DX_GROUP'] == 0].copy().reset_index(drop=True)
        abide_td = abide_df[abide_df['DX_GROUP'] == 1].copy().reset_index(drop=True)
        print(f"ABIDE-ASD: {len(abide_asd)} | ABIDE-TD: {len(abide_td)}")
    except Exception as e:
        raise SystemExit(f'Failed to load ABIDE via step1: {e}')

    # Network map
    C = np.asarray(abide_asd.iloc[0]['data']).shape[1]
    net_labels = load_network_map(args.net_map, C)

    # Load subtype labels aligned to abide_asd order
    labels = load_step2_labels(args.step2_outdir)
    if labels.shape[0] != len(abide_asd):
        raise SystemExit(f'label length {labels.shape[0]} != ABIDE-ASD rows {len(abide_asd)} from step1 order')

    # Helper to process a dataframe of subjects
    def process_group(df_group: pd.DataFrame, regress_gsr: bool, tr: float, use_partial: bool):
        fcs = []
        alffs = []
        falffs = []
        gbcs = []
        degs = []
        eics = []
        clsts = []
        geffs = []  # Global efficiency per subject
        leffs = []  # Local efficiency per subject
        for i in range(len(df_group)):
            ts = np.asarray(df_group.iloc[i]['data'], dtype=np.float32)  # [T, ROI]
            conf = build_confounds_gsr(ts) if regress_gsr else None
            ts = preprocess_ts(ts, tr=tr, confounds=conf)
            fc = compute_fc(ts, use_partial=use_partial)
            fcs.append(fc)
            gm = graph_metrics(fc)
            gbcs.append(global_brain_connectivity(fc))
            degs.append(gm['strength'])
            eics.append(gm['eig'])
            clsts.append(gm['clust'])
            geffs.append(gm['geff'])  # Scalar global efficiency
            leffs.append(gm['leff'])  # Scalar local efficiency
            a, fa = compute_alff(ts, tr=tr)
            alffs.append(a)
            falffs.append(fa)
            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{len(df_group)} subjects  (GSR={regress_gsr})")
        return (np.stack(fcs, axis=0), np.stack(alffs, axis=0), np.stack(falffs, axis=0),
                np.stack(gbcs, axis=0), np.stack(degs, axis=0), np.stack(eics, axis=0), np.stack(clsts, axis=0),
                np.array(geffs), np.array(leffs))

    print('Processing ASD/TD without GSR ...')
    fcs_asd, alffs_asd, falffs_asd, gbcs_asd, degs_asd, eics_asd, clsts_asd, geffs_asd, leffs_asd = \
        process_group(abide_asd, regress_gsr=False, tr=args.tr, use_partial=args.partial)
    fcs_td,  alffs_td,  falffs_td,  gbcs_td,  degs_td,  eics_td,  clsts_td,  geffs_td,  leffs_td  = \
        process_group(abide_td,  regress_gsr=False, tr=args.tr, use_partial=args.partial)

    # Save all individual-level metrics
    print('Saving individual-level metrics...')
    print('  - Saving FC matrices...')
    np.save(os.path.join(args.outdir, 'fcs_asd.npy'), fcs_asd)
    np.save(os.path.join(args.outdir, 'fcs_td.npy'), fcs_td)
    print('    ✓ FC matrices saved')
    
    print('  - Saving ALFF/fALFF values...')
    np.save(os.path.join(args.outdir, 'alffs_asd.npy'), alffs_asd)
    np.save(os.path.join(args.outdir, 'alffs_td.npy'), alffs_td)
    np.save(os.path.join(args.outdir, 'falffs_asd.npy'), falffs_asd)
    np.save(os.path.join(args.outdir, 'falffs_td.npy'), falffs_td)
    print('    ✓ ALFF/fALFF values saved')
    
    print('  - Saving graph metrics...')
    np.save(os.path.join(args.outdir, 'gbcs_asd.npy'), gbcs_asd)
    np.save(os.path.join(args.outdir, 'gbcs_td.npy'), gbcs_td)
    np.save(os.path.join(args.outdir, 'degs_asd.npy'), degs_asd)
    np.save(os.path.join(args.outdir, 'degs_td.npy'), degs_td)
    np.save(os.path.join(args.outdir, 'eics_asd.npy'), eics_asd)
    np.save(os.path.join(args.outdir, 'eics_td.npy'), eics_td)
    np.save(os.path.join(args.outdir, 'clsts_asd.npy'), clsts_asd)
    np.save(os.path.join(args.outdir, 'clsts_td.npy'), clsts_td)
    print('    ✓ Graph metrics saved')
    
    print('  - Saving efficiency metrics...')
    np.save(os.path.join(args.outdir, 'geffs_asd.npy'), geffs_asd)
    np.save(os.path.join(args.outdir, 'geffs_td.npy'), geffs_td)
    np.save(os.path.join(args.outdir, 'leffs_asd.npy'), leffs_asd)
    np.save(os.path.join(args.outdir, 'leffs_td.npy'), leffs_td)
    print('    ✓ Efficiency metrics saved')

    fc_group = np.nanmean(fcs_asd, axis=0)
    fc_group = np.clip(fc_group, -1, 1)
    np.save(os.path.join(args.outdir, 'fc_group_mean_asd.npy'), fc_group)
    
    fc_td_mean = np.nanmean(fcs_td, axis=0)
    fc_td_mean = np.clip(fc_td_mean, -1, 1)
    np.save(os.path.join(args.outdir, 'fc_group_mean_td.npy'), fc_td_mean)

    print_neg_qc(fcs_asd, "ASD no-GSR")
    print_neg_qc(fcs_td, "TD no-GSR")

    if args.export_gsr:
        print('Processing ASD/TD WITH GSR ...')
        fcs_asd_g, alffs_asd_g, falffs_asd_g, gbcs_asd_g, degs_asd_g, eics_asd_g, clsts_asd_g, geffs_asd_g, leffs_asd_g = \
            process_group(abide_asd, regress_gsr=True, tr=args.tr, use_partial=args.partial)
        fcs_td_g,  alffs_td_g,  falffs_td_g,  gbcs_td_g,  degs_td_g,  eics_td_g,  clsts_td_g,  geffs_td_g,  leffs_td_g  = \
            process_group(abide_td,  regress_gsr=True, tr=args.tr, use_partial=args.partial)

        print('Saving GSR branch metrics...')
        np.save(os.path.join(args.outdir, 'fcs_asd_gsr.npy'), fcs_asd_g)
        np.save(os.path.join(args.outdir, 'fcs_td_gsr.npy'),  fcs_td_g)
        np.save(os.path.join(args.outdir, 'alffs_asd_gsr.npy'), alffs_asd_g)
        np.save(os.path.join(args.outdir, 'alffs_td_gsr.npy'),  alffs_td_g)
        np.save(os.path.join(args.outdir, 'falffs_asd_gsr.npy'), falffs_asd_g)
        np.save(os.path.join(args.outdir, 'falffs_td_gsr.npy'),  falffs_td_g)

        fc_asd_mean_gsr = np.nanmean(fcs_asd_g, axis=0)
        fc_td_mean_gsr  = np.nanmean(fcs_td_g,  axis=0)
        np.save(os.path.join(args.outdir, 'fc_group_mean_asd_gsr.npy'), fc_asd_mean_gsr)
        np.save(os.path.join(args.outdir, 'fc_group_mean_td_gsr.npy'),  fc_td_mean_gsr)

        plt.figure(figsize=(8,6))
        fc_plot_r = np.tanh(fc_asd_mean_gsr)
        vmax = np.percentile(np.abs(fc_plot_r), 99)
        im = plt.imshow(fc_plot_r, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        cbar = plt.colorbar(im, shrink=0.8); cbar.set_label('r', fontsize=12)
        plt.title('ASD mean FC (with GSR)', fontsize=14); plt.xlabel('ROI'); plt.ylabel('ROI')
        plt.tight_layout(); plt.savefig(os.path.join(figdir, 'fc_mean_asd_gsr.png'), dpi=200); plt.close()

        print_neg_qc(fcs_asd_g, "ASD GSR")
        print_neg_qc(fcs_td_g, "TD GSR")
        print('    ✓ GSR branch metrics saved')

    # Compute gradients for each subtype and TD group
    print('Computing functional gradients...')
    
    # Gradients for each subtype
    gradients_by_subtype = {}
    for k in range(labels.max() + 1):
        mask = (labels == k)
        if mask.sum() == 0:
            continue
        
        print(f'  - Computing gradients for Subtype {k}...')
        # Compute group mean FC for this subtype
        subtype_fc_mean = np.nanmean(fcs_asd[mask], axis=0)
        grads = compute_gradients_group(subtype_fc_mean, n_components=2)
        gradients_by_subtype[f'Subtype_{k}'] = grads
        np.save(os.path.join(args.outdir, f'gradients_subtype_{k}.npy'), grads)
        print(f'    ✓ Gradients for Subtype {k} computed and saved')
    
    # Gradients for TD group
    print('  - Computing gradients for TD group...')
    td_fc_mean = np.nanmean(fcs_td, axis=0)
    grads_td = compute_gradients_group(td_fc_mean, n_components=2)
    gradients_by_subtype['TD'] = grads_td
    np.save(os.path.join(args.outdir, 'gradients_td.npy'), grads_td)
    print('    ✓ Gradients for TD group computed and saved')
    
    # Also save overall ASD gradients (for backward compatibility)
    print('  - Computing overall ASD gradients...')
    grads = compute_gradients_group(fc_group, n_components=2)
    gradients_by_subtype['ASD_Overall'] = grads
    np.save(os.path.join(args.outdir, 'gradients.npy'), grads)
    print('    ✓ Overall ASD gradients computed and saved')

    # CAP analysis with optimal k selection and comprehensive metrics
    print('Computing CAP metrics...')
    
    # For each subtype, compute CAP metrics
    cap_metrics_by_subtype = {}
    for k in range(labels.max() + 1):
        mask = (labels == k)
        if mask.sum() == 0:
            continue
        
        print(f'  - Computing CAP metrics for Subtype {k}...')
        subtype_subjects = abide_asd[mask]
        ts_list = []
        for _, row in subtype_subjects.iterrows():
            ts = preprocess_ts(np.asarray(row['data'], dtype=np.float32), tr=args.tr)
            ts_list.append(ts)
        
        ts_concat = np.vstack(ts_list)
        cap_metrics = compute_cap_metrics(ts_concat, n_clusters=args.n_cap)
        cap_metrics_by_subtype[f'Subtype_{k}'] = cap_metrics
        
        # Save individual metrics
        np.save(os.path.join(args.outdir, f'caps_subtype_{k}.npy'), cap_metrics['caps'])
        np.save(os.path.join(args.outdir, f'cap_labels_subtype_{k}.npy'), cap_metrics['labels'])
        
        # Save detailed metrics
        pd.DataFrame({
            'state': range(cap_metrics['optimal_k']),
            'occurrence_prob': cap_metrics['occurrence_probs'],
            'duration': cap_metrics['durations'],
            'amplitude': cap_metrics['amplitudes']
        }).to_csv(os.path.join(args.outdir, f'cap_metrics_subtype_{k}.csv'), index=False)
        
        # Save transition matrix
        pd.DataFrame(cap_metrics['transition_matrix'], 
                    index=[f'State_{i}' for i in range(cap_metrics['optimal_k'])],
                    columns=[f'State_{i}' for i in range(cap_metrics['optimal_k'])]
        ).to_csv(os.path.join(args.outdir, f'transition_matrix_subtype_{k}.csv'))
        
        print(f'    ✓ Subtype {k}: optimal k={cap_metrics["optimal_k"]}, states analyzed and saved')
    
    # Also compute for TD group
    print('  - Computing CAP metrics for TD group...')
    ts_td_list = []
    for _, row in abide_td.iterrows():
        ts = preprocess_ts(np.asarray(row['data'], dtype=np.float32), tr=args.tr)
        ts_td_list.append(ts)
    
    ts_td_concat = np.vstack(ts_td_list)
    cap_metrics_td = compute_cap_metrics(ts_td_concat, n_clusters=args.n_cap)
    cap_metrics_by_subtype['TD'] = cap_metrics_td
    
    # Save TD metrics
    np.save(os.path.join(args.outdir, 'caps_td.npy'), cap_metrics_td['caps'])
    np.save(os.path.join(args.outdir, 'cap_labels_td.npy'), cap_metrics_td['labels'])
    pd.DataFrame({
        'state': range(cap_metrics_td['optimal_k']),
        'occurrence_prob': cap_metrics_td['occurrence_probs'],
        'duration': cap_metrics_td['durations'],
        'amplitude': cap_metrics_td['amplitudes']
    }).to_csv(os.path.join(args.outdir, 'cap_metrics_td.csv'), index=False)
    
    pd.DataFrame(cap_metrics_td['transition_matrix'],
                index=[f'State_{i}' for i in range(cap_metrics_td['optimal_k'])],
                columns=[f'State_{i}' for i in range(cap_metrics_td['optimal_k'])]
    ).to_csv(os.path.join(args.outdir, 'transition_matrix_td.csv'))
    
    print(f'    ✓ TD: optimal k={cap_metrics_td["optimal_k"]}, states analyzed and saved')
    
    # Generate CAP heatmaps for each group
    print('Generating CAP heatmaps...')
    cap_figdir = os.path.join(figdir, 'cap_analysis')
    os.makedirs(cap_figdir, exist_ok=True)
    
    for group_name, metrics in cap_metrics_by_subtype.items():
        print(f'  - Generating CAP plots for {group_name}...')
        plot_cap_heatmaps(metrics, group_name, cap_figdir)
        print(f'    ✓ CAP plots for {group_name} generated')
    
    # Generate comparison heatmaps
    print('  - Generating CAP comparison plots...')
    plot_cap_comparison_heatmap(cap_metrics_by_subtype, cap_figdir)
    print('    ✓ CAP comparison plots generated')
    
    # Generate CAP state brain maps for each group
    print('Generating CAP state brain maps...')
    for group_name, metrics in cap_metrics_by_subtype.items():
        print(f'  - Generating state brain maps for {group_name}...')
        plot_cap_state_brain_maps(metrics, group_name, cap_figdir, atlas_img)
        print(f'    ✓ State brain maps for {group_name} generated')
    
    # Generate comparison brain maps across groups
    print('  - Generating CAP state comparison brain maps...')
    plot_cap_state_comparison_brain_maps(cap_metrics_by_subtype, cap_figdir, atlas_img)
    print('    ✓ CAP state comparison brain maps generated')
    
    # Generate gradient brain maps for each group
    print('Generating gradient brain maps...')
    for group_name, grads in gradients_by_subtype.items():
        print(f'  - Generating gradient brain maps for {group_name}...')
        plot_gradient_brain_maps(grads, group_name, figdir, atlas_img)
        print(f'    ✓ Gradient brain maps for {group_name} generated')
    
    # Generate gradient comparison brain maps
    print('  - Generating gradient comparison brain maps...')
    plot_gradient_comparison_brain_maps(gradients_by_subtype, figdir, atlas_img)
    print('    ✓ Gradient comparison brain maps generated')

    # Save subject-level summary CSVs
    print('Creating subject-level summary CSVs...')
    print('  - Creating ASD summary...')
    # ASD subjects with subtype labels
    asd_summary = pd.DataFrame({
        'subject_id': abide_asd['subject_id'].values,
        'subtype': labels,
        'mean_gbc': gbcs_asd.mean(axis=1),
        'mean_degree': degs_asd.mean(axis=1),
        'mean_eig': eics_asd.mean(axis=1),
        'mean_clust': clsts_asd.mean(axis=1),
        'global_efficiency': geffs_asd,
        'local_efficiency': leffs_asd,
        'mean_alff': alffs_asd.mean(axis=1),
        'mean_falff': falffs_asd.mean(axis=1)
    })
    asd_summary.to_csv(os.path.join(args.outdir, 'subject_summary_asd.csv'), index=False)
    print('    ✓ ASD summary CSV created')
    
    print('  - Creating TD summary...')
    # TD subjects
    td_summary = pd.DataFrame({
        'subject_id': abide_td['subject_id'].values,
        'group': 'TD',
        'mean_gbc': gbcs_td.mean(axis=1),
        'mean_degree': degs_td.mean(axis=1),
        'mean_eig': eics_td.mean(axis=1),
        'mean_clust': clsts_td.mean(axis=1),
        'global_efficiency': geffs_td,
        'local_efficiency': leffs_td,
        'mean_alff': alffs_td.mean(axis=1),
        'mean_falff': falffs_td.mean(axis=1)
    })
    td_summary.to_csv(os.path.join(args.outdir, 'subject_summary_td.csv'), index=False)
    print('    ✓ TD summary CSV created')

    # Save node-level summaries (means across subjects)
    pd.DataFrame(gbcs_asd.mean(axis=0)).to_csv(os.path.join(graphdir, 'gbc_mean_asd.csv'), index=False, header=['gbc'])
    pd.DataFrame(degs_asd.mean(axis=0)).to_csv(os.path.join(graphdir, 'degree_mean_asd.csv'), index=False, header=['degree'])
    pd.DataFrame(eics_asd.mean(axis=0)).to_csv(os.path.join(graphdir, 'eig_mean_asd.csv'), index=False, header=['eig'])
    pd.DataFrame(clsts_asd.mean(axis=0)).to_csv(os.path.join(graphdir, 'clust_mean_asd.csv'), index=False, header=['clust'])
    pd.DataFrame(gbcs_td.mean(axis=0)).to_csv(os.path.join(graphdir, 'gbc_mean_td.csv'), index=False, header=['gbc'])
    pd.DataFrame(degs_td.mean(axis=0)).to_csv(os.path.join(graphdir, 'degree_mean_td.csv'), index=False, header=['degree'])
    pd.DataFrame(eics_td.mean(axis=0)).to_csv(os.path.join(graphdir, 'eig_mean_td.csv'), index=False, header=['eig'])
    pd.DataFrame(clsts_td.mean(axis=0)).to_csv(os.path.join(graphdir, 'clust_mean_td.csv'), index=False, header=['clust'])

    # Intra/Inter strength
    nets_summary = dict(
        asd=intra_inter_strength(np.nanmean(fcs_asd, axis=0), net_labels),
        td=intra_inter_strength(np.nanmean(fcs_td, axis=0), net_labels)
    )
    # Cast numpy types to native Python types for JSON serialization
    nets_summary_py = {g: {k: float(v) if hasattr(v, 'item') else v for k, v in d.items()} for g, d in nets_summary.items()}
    with open(os.path.join(args.outdir, 'network_strength.json'), 'w') as f:
        json.dump(nets_summary_py, f, indent=2)

    # Quick plots
    save_fig_bar(gbcs_asd.mean(axis=0), 'GBC (ASD mean)', os.path.join(figdir, 'gbc_mean_asd.png'))
    save_fig_bar(gbcs_td.mean(axis=0), 'GBC (TD mean)', os.path.join(figdir, 'gbc_mean_td.png'))

    # Group-level FC heatmaps per subtype and TD
    print('Generating FC heatmaps...')
    plot_group_fc_heatmaps(fcs_asd, labels, os.path.join(figdir, 'fc_by_subtype'))
    print('  - Generated FC heatmaps for all subtypes')
    
    print('  - Generating FC heatmap for TD group...')
    td_mean_z = np.nanmean(fcs_td, axis=0)
    td_mean_r = np.tanh(td_mean_z)
    np.fill_diagonal(td_mean_r, 1.0)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(td_mean_r, cmap='RdBu_r', vmin=-1.0, vmax=1.0)
    plt.title(f'Functional Connectivity - TD Group\n(r, n={len(fcs_td)})', fontsize=14)
    plt.xlabel('ROI', fontsize=12)
    plt.ylabel('ROI', fontsize=12)
    
    # Add colorbar with label
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('r', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, 'fc_mean_td.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('    ✓ FC heatmap for TD group generated')

    subj_gbc_mean_asd = gbcs_asd.mean(axis=1)
    subj_labels_combo = np.concatenate([labels, np.full(len(gbcs_td), fill_value=labels.max()+1)])
    subj_values_combo = np.concatenate([subj_gbc_mean_asd, gbcs_td.mean(axis=1)])
    violin_by_subtype(subj_values_combo, subj_labels_combo, 'GBC (subject mean) ASD subtypes vs TD', os.path.join(figdir, 'gbc_violin_asd_td.png'))
    
    # Violin: Global efficiency
    geff_combo = np.concatenate([geffs_asd, geffs_td])
    violin_by_subtype(geff_combo, subj_labels_combo, 'Global Efficiency ASD subtypes vs TD', os.path.join(figdir, 'global_efficiency_violin_asd_td.png'))
    
    # Violin: Local efficiency  
    leff_combo = np.concatenate([leffs_asd, leffs_td])
    violin_by_subtype(leff_combo, subj_labels_combo, 'Local Efficiency ASD subtypes vs TD', os.path.join(figdir, 'local_efficiency_violin_asd_td.png'))

    print('Generating ALFF/fALFF brain maps...')
    try:
        atlas_img = nib.load('atlas/brainnetome/BN_Atlas_246_1mm.nii.gz')
        vmax_global = float(np.percentile(np.abs(np.concatenate([alffs_asd, alffs_td], axis=0)), 95))
        
        for k in range(labels.max() + 1):
            mask = (labels == k)
            if mask.sum() == 0:
                continue
            print(f'  - Generating ALFF brain map for Subtype {k}...')
            alff_mean = alffs_asd[mask].mean(axis=0)
            img = make_volume_from_parcels(atlas_img, alff_mean)
            plot_surface_quad(img, os.path.join(figdir, f'alff_mean_subtype{k}_quad.png'), 
                            cmap='PuOr', vmax=vmax_global, title=f'ALFF - Subtype {k}')
            print(f'    ✓ ALFF brain map for Subtype {k} generated')
        
        print('  - Generating ALFF brain map for TD...')
        alff_td_mean = alffs_td.mean(axis=0)
        img_td = make_volume_from_parcels(atlas_img, alff_td_mean)
        plot_surface_quad(img_td, os.path.join(figdir, 'alff_mean_td_quad.png'), 
                        cmap='PuOr', vmax=vmax_global, title='ALFF - TD Group')
        print('    ✓ ALFF brain map for TD generated')
        
        # fALFF brain maps
        print('Generating fALFF brain maps...')
        vmax_falff_global = float(np.percentile(np.abs(np.concatenate([falffs_asd, falffs_td], axis=0)), 95))
        
        for k in range(labels.max() + 1):
            mask = (labels == k)
            if mask.sum() == 0:
                continue
            print(f'  - Generating fALFF brain map for Subtype {k}...')
            falff_mean = falffs_asd[mask].mean(axis=0)
            img = make_volume_from_parcels(atlas_img, falff_mean)
            plot_surface_quad(img, os.path.join(figdir, f'falff_mean_subtype{k}_quad.png'), 
                            cmap='PuOr', vmax=vmax_falff_global, title=f'fALFF - Subtype {k}')
            print(f'    ✓ fALFF brain map for Subtype {k} generated')
        
        print('  - Generating fALFF brain map for TD...')
        falff_td_mean = falffs_td.mean(axis=0)
        img_td = make_volume_from_parcels(atlas_img, falff_td_mean)
        plot_surface_quad(img_td, os.path.join(figdir, 'falff_mean_td_quad.png'), 
                        cmap='PuOr', vmax=vmax_falff_global, title='fALFF - TD Group')
        print('    ✓ fALFF brain map for TD generated')
        
    except Exception as e:
        print('ALFF/fALFF brain map export failed:', e)

    print('Step7 ABIDE metrics saved to', os.path.abspath(args.outdir))


if __name__ == '__main__':
    main()


