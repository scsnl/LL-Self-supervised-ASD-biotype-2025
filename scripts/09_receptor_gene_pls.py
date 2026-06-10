#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 9: FC-sum to Receptor/Gene PLS Analysis

Correlates subtype-specific Global Brain Connectivity (FC-sum per ROI) with
spatial maps of neurotransmitter receptor density (Hansen et al. 2022) and
gene expression (Allen Human Brain Atlas) via Partial Least Squares (PLS).

Pipeline:
  1. Load ROI-level FC matrices (step 07) and subtype labels (step 03)
  2. Compute FC-sum per subtype (mean across subjects, per ROI)
  3. Load receptor / gene expression spatial maps
  4. PLS regression with cross-validation
  5. VIP scores for feature ranking
  6. Permutation test for significance
  7. Gene-set enrichment for significant features

Usage:
  python 09_receptor_gene_pls.py \\
    --step7-dir  <path/to/step7_output> \\
    --labels-npy <abide_asd_labels.npy> \\
    --receptor-csv <hansen_receptor_map.csv> \\
    --outdir     <output_dir>
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional

try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False
    print("⚠️ gseapy: pip install gseapy")


DEFAULT_STEP7_DIR = "unsup_results/step7_abide_fixed"
DEFAULT_STEP2_DIR = "results_step2_tuned_retry"
DEFAULT_OUT_DIR = "unsup_results/step9_fc_sum_gene_pls"
DEFAULT_GENE_CSV = "atlas/brainnetome/BN246_geneexpression.csv.gz"
DEFAULT_RECEPTOR_CSV = "atlas/brainnetome/receptor_data_bn246.csv"
DEFAULT_RECEPTOR_NAMES = "atlas/brainnetome/receptor_names_pet.npy"

DEFAULT_MAX_PLS_COMP = 10
DEFAULT_N_PERM = 1000
DEFAULT_RANDOM_STATE = 42



def load_fc_matrices(step7_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    
    Returns:
    --------
    fcs_asd : np.ndarray, shape [N, 246, 246]
    fcs_td : np.ndarray, shape [N_td, 246, 246]
    """
    print("FC...")
    
    possible_asd = [
        "fcs_asd.npy",
        "fcs_asd_gsr.npy"
    ]
    possible_td = [
        "fcs_td.npy",
        "fcs_td_gsr.npy"
    ]
    
    fcs_asd_path = None
    for fname in possible_asd:
        path = os.path.join(step7_dir, fname)
        if os.path.exists(path):
            fcs_asd_path = path
            break
    
    fcs_td_path = None
    for fname in possible_td:
        path = os.path.join(step7_dir, fname)
        if os.path.exists(path):
            fcs_td_path = path
            break
    
    if fcs_asd_path is None:
        raise FileNotFoundError(f"ASD FC: {possible_asd}")
    if fcs_td_path is None:
        raise FileNotFoundError(f"TD FC: {possible_td}")
    
    print(f"  : {os.path.basename(fcs_asd_path)}")
    print(f"  : {os.path.basename(fcs_td_path)}")
    
    fcs_asd = np.load(fcs_asd_path)
    fcs_td = np.load(fcs_td_path)
    
    print(f"  ASD FC: {fcs_asd.shape}")
    print(f"  TD FC: {fcs_td.shape}")
    
    return fcs_asd, fcs_td


def load_subtype_labels(step2_dir: str, labels_file: Optional[str] = None) -> np.ndarray:
    """
    
    Parameters:
    -----------
    step2_dir : str
    labels_file : str, optional
    
    Returns:
    --------
    labels : np.ndarray, shape [N]
    """
    print("\n...")
    
    if labels_file:
        if os.path.exists(labels_file):
            labels_path = labels_file
        else:
            labels_path = os.path.join(step2_dir, labels_file)
    else:
        labels_path = os.path.join(step2_dir, "abide_asd_labels.npy")
        
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f": {labels_path}")
    
    labels = np.load(labels_path).astype(int)
    print(f"  : {os.path.basename(labels_path)}")
    print(f"  : {labels.shape}")
    print(f"  : {len(np.unique(labels))}")
    print(f"  : {np.bincount(labels)}")
    
    return labels


def compute_fc_sum(fc_matrix: np.ndarray, absolute: bool = True) -> np.ndarray:
    """
    
    Parameters:
    -----------
    fc_matrix : np.ndarray, shape [246, 246]
    absolute : bool
    
    Returns:
    --------
    fc_sum : np.ndarray, shape [246]
    """
    fc = fc_matrix.copy()
    np.fill_diagonal(fc, 0)
    
    if absolute:
        fc = np.abs(fc)
    
    fc_sum = np.sum(fc, axis=1)
    
    return fc_sum


def compute_subtype_fc_sum(fcs: np.ndarray, labels: np.ndarray, 
                          absolute: bool = True) -> Dict[int, np.ndarray]:
    """
    
    Parameters:
    -----------
    fcs : np.ndarray, shape [N, 246, 246]
    labels : np.ndarray, shape [N]
    absolute : bool
    
    Returns:
    --------
    subtype_fc_sums : Dict[int, np.ndarray]
    """
    print("\nFC-sum...")
    
    n_subtypes = len(np.unique(labels))
    subtype_fc_sums = {}
    
    for k in range(n_subtypes):
        mask = (labels == k)
        if mask.sum() == 0:
            print(f"  ⚠️ {k+1}")
            continue
        
        fcs_k = fcs[mask]
        
        fc_sums_k = []
        for i in range(fcs_k.shape[0]):
            fc_sum_i = compute_fc_sum(fcs_k[i], absolute=absolute)
            fc_sums_k.append(fc_sum_i)
        
        fc_sum_mean = np.mean(fc_sums_k, axis=0)
        subtype_fc_sums[k] = fc_sum_mean
        
        print(f"  ✓ {k+1}: {mask.sum()} , FC-sum: [{fc_sum_mean.min():.3f}, {fc_sum_mean.max():.3f}]")
    
    return subtype_fc_sums


def load_gene_data(gene_csv: str) -> pd.DataFrame:
    """
    
    Returns:
    --------
    gene_df : pd.DataFrame
    """
    print("\n...")
    
    if not os.path.exists(gene_csv):
        raise FileNotFoundError(f": {gene_csv}")
    
    gene_df = pd.read_csv(gene_csv)
    print(f"  : {gene_df.shape}")
    
    roi_cols = ['ROI', 'roi', 'label', 'Label', 'region', 'Region']
    roi_col = None
    for col in roi_cols:
        if col in gene_df.columns:
            roi_col = col
            break
    
    if roi_col is None:
        print("  ⚠️ ROIROI")
        gene_df.rename(columns={gene_df.columns[0]: 'ROI'}, inplace=True)
    else:
        gene_df.rename(columns={roi_col: 'ROI'}, inplace=True)
    
    if gene_df['ROI'].min() == 1:
        gene_df['ROI'] = gene_df['ROI'] - 1
    
    print(f"  : {gene_df.shape[1] - 1}")
    print(f"  ROI: {len(gene_df)}")
    
    return gene_df


def load_receptor_data(receptor_csv: str, receptor_names_npy: Optional[str] = None) -> pd.DataFrame:
    """
    
    Returns:
    --------
    receptor_df : pd.DataFrame
    """
    print("\n...")
    
    if not os.path.exists(receptor_csv):
        print(f"  ⚠️ : {receptor_csv}")
        return pd.DataFrame()
    
    receptor_df = pd.read_csv(receptor_csv)
    print(f"  : {receptor_df.shape}")
    
    has_roi_col = False
    if 'ROI' in receptor_df.columns or receptor_df.columns[0] in ['roi', 'label', 'Label', 'region', 'Region']:
        has_roi_col = True
    
    if receptor_names_npy and os.path.exists(receptor_names_npy):
        receptor_names = np.load(receptor_names_npy, allow_pickle=True)
        print(f"  : {len(receptor_names)} ")
        
        if has_roi_col:
            receptor_df.columns = ['ROI'] + list(receptor_names)
        else:
            print(f"  ROI (0-{len(receptor_df)-1})")
            receptor_df.columns = list(receptor_names)
            receptor_df.insert(0, 'ROI', np.arange(len(receptor_df)))
    else:
        if not has_roi_col:
            print(f"  ROI (0-{len(receptor_df)-1})")
            receptor_df.insert(0, 'ROI', np.arange(len(receptor_df)))
        else:
            if receptor_df.columns[0] != 'ROI':
                receptor_df.rename(columns={receptor_df.columns[0]: 'ROI'}, inplace=True)
    
    if 'ROI' in receptor_df.columns:
        if receptor_df['ROI'].min() == 1:
            receptor_df['ROI'] = receptor_df['ROI'] - 1
    
    print(f"  : {receptor_df.shape[1] - 1}")
    print(f"  ROI: {len(receptor_df)}")
    
    return receptor_df


def correlation_table(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """
    
    Parameters:
    -----------
    X : pd.DataFrame
    y : np.ndarray
    
    Returns:
    --------
    corr_df : pd.DataFrame
    """
    rows = []
    for col in X.columns:
        x_vals = X[col].values.astype(float)
        valid = ~np.isnan(x_vals) & ~np.isnan(y)
        
        if valid.sum() >= 3:
            r, p = pearsonr(x_vals[valid], y[valid])
        else:
            r, p = np.nan, np.nan
        
        rows.append({'feature': col, 'r': r, 'p_raw': p})
    
    corr_df = pd.DataFrame(rows)
    
    mask = corr_df['p_raw'].notna()
    if mask.sum() > 0:
        _, p_fdr, _, _ = multipletests(corr_df.loc[mask, 'p_raw'], method='fdr_bh')
        corr_df.loc[mask, 'p_FDR'] = p_fdr
    else:
        corr_df['p_FDR'] = np.nan
    
    corr_df = corr_df.sort_values('p_FDR').reset_index(drop=True)
    
    return corr_df


def select_pls_components_cv(X: np.ndarray, y: np.ndarray, 
                             max_comp: int, random_state: int) -> int:
    """
    
    Parameters:
    -----------
    X : np.ndarray, shape [N, p]
    y : np.ndarray, shape [N]
    max_comp : int
    random_state : int
    
    Returns:
    --------
    best_comp : int
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    max_comp = min(max_comp, X.shape[0] - 2, X.shape[1])
    
    best_score = -np.inf
    best_comp = 1
    
    for n_comp in range(1, max_comp + 1):
        scores = []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            
            X_train_s = scaler_x.fit_transform(X_train)
            y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            X_test_s = scaler_x.transform(X_test)
            y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
            
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_train_s, y_train_s)
            y_pred = pls.predict(X_test_s).ravel()
            
            valid = ~np.isnan(y_test_s) & ~np.isnan(y_pred)
            if valid.sum() >= 3:
                r, _ = pearsonr(y_test_s[valid], y_pred[valid])
                scores.append(r)
        
        if len(scores) > 0:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_comp = n_comp
    
    print(f"  PLS: {best_comp} (CV score: {best_score:.3f})")
    
    return best_comp


def pls_fit_and_vip(X: np.ndarray, y: np.ndarray, 
                   n_components: int) -> Tuple[PLSRegression, np.ndarray, np.ndarray]:
    """
    
    Parameters:
    -----------
    X : np.ndarray, shape [N, p]
    y : np.ndarray, shape [N]
    n_components : int
    
    Returns:
    --------
    pls : PLSRegression
    vip : np.ndarray, shape [p]
    y_pred : np.ndarray, shape [N]
    """
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_s = scaler_x.fit_transform(X)
    y_s = scaler_y.fit_transform(y.reshape(-1, 1))
    
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_s, y_s)
    y_pred = scaler_y.inverse_transform(pls.predict(X_s)).ravel()
    
    T = pls.x_scores_  # shape [N, n_components]
    W = pls.x_weights_  # shape [p, n_components]
    Q = pls.y_loadings_  # shape [n_components, 1]
    
    p = X_s.shape[1]
    
    if Q.shape[1] == 1:
        Q_flat = Q.flatten()
    else:
        Q_flat = Q
    
    # SSY explained by each component
    SSY = np.sum((T ** 2) * (Q_flat ** 2), axis=0)
    vip = np.zeros(p)
    denom = np.sum(SSY)
    
    if denom > 1e-12:
        for j in range(p):
            vip_j = 0.0
            for comp in range(n_components):
                vip_j += (W[j, comp] ** 2) * SSY[comp]
            vip[j] = np.sqrt(p * vip_j / denom)
    else:
        vip = np.sqrt(np.sum(W ** 2, axis=1))
    
    return pls, vip, y_pred


def permutation_test_pls(X: np.ndarray, y: np.ndarray, 
                         n_components: int, n_perm: int, 
                         random_state: int) -> Dict[str, float]:
    """
    
    Parameters:
    -----------
    X : np.ndarray
    y : np.ndarray
    n_components : int
    n_perm : int
    random_state : int
    
    Returns:
    --------
    result : dict
    """
    _, _, y_pred_obs = pls_fit_and_vip(X, y, n_components)
    valid = ~np.isnan(y) & ~np.isnan(y_pred_obs)
    
    if valid.sum() < 3:
        return {"r_obs": np.nan, "p_perm_two_sided": np.nan}
    
    r_obs, _ = pearsonr(y[valid], y_pred_obs[valid])
    
    rng = np.random.RandomState(random_state)
    null_rs = []
    
    print(f"   (n={n_perm})...")
    for i in range(n_perm):
        if (i + 1) % 200 == 0:
            print(f"    : {i+1}/{n_perm}")
        
        y_perm = rng.permutation(y)
        _, _, y_pred_perm = pls_fit_and_vip(X, y_perm, n_components)
        
        valid_perm = ~np.isnan(y_perm) & ~np.isnan(y_pred_perm)
        if valid_perm.sum() >= 3:
            r_perm, _ = pearsonr(y_perm[valid_perm], y_pred_perm[valid_perm])
            null_rs.append(r_perm)
    
    null_rs = np.array(null_rs)
    
    p_two = np.mean(np.abs(null_rs) >= np.abs(r_obs))
    
    return {"r_obs": float(r_obs), "p_perm_two_sided": float(p_two)}


def run_pls_analysis(fc_sum: np.ndarray, feature_df: pd.DataFrame, 
                    feature_type: str, subtype_id: int, out_dir: Path,
                    max_pls_comp: int, n_perm: int, random_state: int) -> Optional[pd.DataFrame]:
    """
    
    Parameters:
    -----------
    fc_sum : np.ndarray, shape [246]
    feature_df : pd.DataFrame
    feature_type : str
    subtype_id : int
    out_dir : Path
    max_pls_comp : int
    n_perm : int
    random_state : int
    
    Returns:
    --------
    vip_df : pd.DataFrame
    """
    print(f"\n Subtype{subtype_id} {feature_type}...")
    
    subtype_dir = out_dir / f"subtype{subtype_id}" / feature_type
    subtype_dir.mkdir(parents=True, exist_ok=True)
    
    feature_cols = [c for c in feature_df.columns if c != 'ROI']
    
    fc_df = pd.DataFrame({'ROI': np.arange(246), 'FC_sum': fc_sum})
    
    merged = pd.merge(feature_df, fc_df, on='ROI', how='inner')
    merged = merged.sort_values('ROI').reset_index(drop=True)
    
    if len(merged) < 10:
        print(f"  ⚠️  ({len(merged)}), ")
        return None
    
    X = merged[feature_cols].values
    y = merged['FC_sum'].values.astype(float)
    
    print("  1. ...")
    corr_df = correlation_table(merged[feature_cols], y)
    corr_path = subtype_dir / f"{feature_type}_fc_sum_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    
    n_sig = (corr_df['p_FDR'] < 0.05).sum()
    print(f"      (FDR<0.05): {n_sig}")
    
    print("  2. PLS...")
    
    valid_mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    
    if X_valid.shape[0] < 10:
        print(f"  ⚠️  ({X_valid.shape[0]}), PLS")
        return None
    
    n_comp = select_pls_components_cv(X_valid, y_valid, max_pls_comp, random_state)
    
    pls, vip, y_pred = pls_fit_and_vip(X_valid, y_valid, n_comp)
    
    vip_df = pd.DataFrame({
        'feature': feature_cols,
        'VIP': vip
    }).sort_values('VIP', ascending=False)
    
    vip_path = subtype_dir / f"{feature_type}_pls_vip_scores.csv"
    vip_df.to_csv(vip_path, index=False)
    
    n_vip_sig = (vip_df['VIP'] > 1.0).sum()
    print(f"     VIP>1.0: {n_vip_sig}")
    
    valid_pred = ~np.isnan(y_valid) & ~np.isnan(y_pred)
    if valid_pred.sum() >= 3:
        r_fit, p_fit = pearsonr(y_valid[valid_pred], y_pred[valid_pred])
        print(f"     PLS: r={r_fit:.3f}, p={p_fit:.3e}")
    else:
        r_fit, p_fit = np.nan, np.nan
    
    print("  3. ...")
    perm_result = permutation_test_pls(X_valid, y_valid, n_comp, n_perm, random_state)
    
    print(f"     r_obs={perm_result['r_obs']:.3f}, p_perm={perm_result['p_perm_two_sided']:.3e}")
    
    perf_df = pd.DataFrame([{
        'subtype': subtype_id,
        'feature_type': feature_type,
        'n_components': n_comp,
        'n_features': len(feature_cols),
        'n_samples': X_valid.shape[0],
        'r_fit': r_fit,
        'p_fit': p_fit,
        'r_obs': perm_result['r_obs'],
        'p_perm': perm_result['p_perm_two_sided'],
        'n_perm': n_perm
    }])
    
    perf_path = subtype_dir / f"{feature_type}_pls_performance.csv"
    perf_df.to_csv(perf_path, index=False)
    
    print("  4. ...")
    
    pls1 = PLSRegression(n_components=1)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_valid_s = scaler_x.fit_transform(X_valid)
    y_valid_s = scaler_y.fit_transform(y_valid.reshape(-1, 1)).ravel()
    
    pls1.fit(X_valid_s, y_valid_s)
    x_scores_pls1 = pls1.x_scores_[:, 0]
    
    from scipy.stats import linregress
    r_pls1, p_pls1 = pearsonr(x_scores_pls1, y_valid_s)
    
    scatter_color = 'forestgreen'
    if feature_type == 'receptor':
        scatter_color = 'darkorange'
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    axes[0].hist(vip_df['VIP'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(1.0, color='red', linestyle='--', linewidth=2, label='VIP=1.0')
    axes[0].set_xlabel('VIP Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'VIP Distribution (Subtype{subtype_id} {feature_type})', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if valid_pred.sum() >= 3:
        axes[1].scatter(y_valid[valid_pred], y_pred[valid_pred], 
                       alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
        
        slope, intercept, _, _, _ = linregress(y_valid[valid_pred], y_pred[valid_pred])
        line_x = np.linspace(y_valid[valid_pred].min(), y_valid[valid_pred].max(), 100)
        line_y = slope * line_x + intercept
        axes[1].plot(line_x, line_y, 'r-', linewidth=2, label=f'r={r_fit:.3f}, p={p_fit:.2e}')
        
        axes[1].set_xlabel('Observed FC-sum', fontsize=12)
        axes[1].set_ylabel('Predicted FC-sum', fontsize=12)
        axes[1].set_title(f'PLS Prediction (Subtype{subtype_id} {feature_type})', fontsize=13)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
    axes[2].scatter(x_scores_pls1, y_valid_s, 
                   alpha=0.6, s=50, color=scatter_color, edgecolors='black', linewidth=0.5)
    
    slope1, intercept1, _, _, _ = linregress(x_scores_pls1, y_valid_s)
    line_x1 = np.linspace(x_scores_pls1.min(), x_scores_pls1.max(), 100)
    line_y1 = slope1 * line_x1 + intercept1
    axes[2].plot(line_x1, line_y1, 'r-', linewidth=2, label=f'r={r_pls1:.3f}, p={p_pls1:.2e}')
    
    axes[2].set_xlabel('PLS1 Score (Gene Expression Pattern)', fontsize=12)
    axes[2].set_ylabel('Normalized FC-sum (Z-score)', fontsize=12)
    axes[2].set_title(f'PLS1 Score vs Phenotype', fontsize=13)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = subtype_dir / f"{feature_type}_pls_summary.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(7, 6))
    plt.scatter(x_scores_pls1, y_valid_s, 
               alpha=0.7, s=60, color=scatter_color, edgecolors='black', linewidth=0.5)
    plt.plot(line_x1, line_y1, 'r-', linewidth=2, label=f'r={r_pls1:.3f}, p={p_pls1:.2e}')
    plt.xlabel('PLS1 Score (Gene Expression Pattern)', fontsize=12, fontweight='bold')
    plt.ylabel('Normalized FC-sum (Z-score)', fontsize=12, fontweight='bold')
    plt.title(f'PLS1 Score vs FC-sum (Subtype{subtype_id})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(subtype_dir / f"{feature_type}_pls1_scatter.png", dpi=300)
    plt.close()
    
    print(f"  ✓ : {subtype_dir}")
    
    return vip_df


def run_enrichment_analysis(vip_df: pd.DataFrame, feature_type: str,
                           subtype_id: int, out_dir: Path,
                           vip_threshold: float = 1.5,
                           top_n: int = 200) -> Optional[pd.DataFrame]:
    """
    
    Parameters:
    -----------
    vip_df : pd.DataFrame
    feature_type : str
    subtype_id : int
    out_dir : Path
    vip_threshold : float
    top_n : int
    
    Returns:
    --------
    enrichment_df : pd.DataFrame
    """
    if not GSEAPY_AVAILABLE or feature_type != 'gene':
        return None
    
    print(f"\n (Subtype{subtype_id})...")
    
    enrich_dir = out_dir / f"subtype{subtype_id}" / "enrichment"
    enrich_dir.mkdir(parents=True, exist_ok=True)
    
    sig_genes = vip_df[vip_df['VIP'] > vip_threshold]['feature'].tolist()
    
    if len(sig_genes) < 5:
        print(f"  ⚠️ VIP>{vip_threshold} ({len(sig_genes)}), top{top_n}")
        sig_genes = vip_df.head(top_n)['feature'].tolist()
    
    if len(sig_genes) < 5:
        print(f"  ⚠️ ")
        return None
    
    print(f"  : {len(sig_genes)}")
    
    gene_list_path = enrich_dir / "significant_genes.txt"
    with open(gene_list_path, 'w') as f:
        for gene in sig_genes:
            f.write(f"{gene}\n")
    
    libraries = [
        "GO_Biological_Process_2023",
        "KEGG_2021_Human",
        "Reactome_2022"
    ]
    
    results = []
    
    for lib in libraries:
        print(f"  : {lib}")
        try:
            enr = gp.enrichr(
                gene_list=sig_genes,
                gene_sets=lib,
                outdir=str(enrich_dir / lib),
                cutoff=1.0,
                no_plot=True
            )
            
            if enr.results is not None and len(enr.results) > 0:
                df = enr.results.copy()
                df['Library'] = lib
                results.append(df)
                print(f"    ✓  {len(df)} ")
            else:
                print(f"    ")
                
        except Exception as e:
            print(f"    ✗ : {e}")
    
    if not results:
        print("  ")
        return None
    
    enrichment_df = pd.concat(results, ignore_index=True)
    
    column_mapping = {
        "Term": "term",
        "Adjusted P-value": "padj",
        "P-value": "pval",
        "Odds Ratio": "odds_ratio",
        "Combined Score": "combined_score",
        "Overlap": "overlap",
        "Genes": "genes"
    }
    
    enrichment_df = enrichment_df.rename(columns={
        k: v for k, v in column_mapping.items() if k in enrichment_df.columns
    })
    
    enrichment_df.to_csv(enrich_dir / "enrichment_results.csv", index=False)
    
    n_sig = (enrichment_df['padj'] < 0.05).sum()
    print(f"  ✓  (FDR<0.05): {n_sig}")
    
    if n_sig > 0:
        sig_results = enrichment_df[enrichment_df['padj'] < 0.05].sort_values('padj').head(15)
        
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(sig_results))
        neg_log_padj = -np.log10(sig_results['padj'])
        
        colors = plt.cm.viridis(neg_log_padj / neg_log_padj.max())
        plt.barh(y_pos, neg_log_padj, color=colors)
        
        labels = [f"{row['Library']} | {row['term'][:40]}{'...' if len(row['term']) > 40 else ''}"
                 for _, row in sig_results.iterrows()]
        plt.yticks(y_pos, labels, fontsize=9)
        plt.xlabel('-log10(FDR)', fontsize=12)
        plt.title(f'Top Enriched Pathways (Subtype{subtype_id})', fontsize=13)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(enrich_dir / "top_pathways.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return enrichment_df


def create_summary_report(out_dir: Path, n_subtypes: int):
    """
    
    Parameters:
    -----------
    out_dir : Path
    n_subtypes : int
    """
    print("\n...")
    
    report_path = out_dir / "step9_summary_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Step9 FC-sum/PLS\n")
        f.write("=" * 60 + "\n\n")
        
        for k in range(n_subtypes):
            f.write(f"\n{'=' * 60}\n")
            f.write(f" {k+1}\n")
            f.write(f"{'=' * 60}\n\n")
            
            for feature_type in ['gene', 'receptor']:
                perf_file = out_dir / f"subtype{k+1}" / feature_type / f"{feature_type}_pls_performance.csv"
                
                if perf_file.exists():
                    perf_df = pd.read_csv(perf_file)
                    
                    f.write(f"\n{feature_type.upper()}:\n")
                    f.write(f"  PLS: {perf_df['n_components'].iloc[0]}\n")
                    f.write(f"  : {perf_df['n_features'].iloc[0]}\n")
                    f.write(f"  : {perf_df['n_samples'].iloc[0]}\n")
                    f.write(f"  : {perf_df['r_fit'].iloc[0]:.3f}\n")
                    f.write(f"  : {perf_df['r_obs'].iloc[0]:.3f}\n")
                    f.write(f"  p: {perf_df['p_perm'].iloc[0]:.3e}\n")
                    
                    vip_file = out_dir / f"subtype{k+1}" / feature_type / f"{feature_type}_pls_vip_scores.csv"
                    if vip_file.exists():
                        vip_df = pd.read_csv(vip_file)
                        n_vip1 = (vip_df['VIP'] > 1.0).sum()
                        n_vip15 = (vip_df['VIP'] > 1.5).sum()
                        f.write(f"  VIP>1.0: {n_vip1}\n")
                        f.write(f"  VIP>1.5: {n_vip15}\n")
                else:
                    f.write(f"\n{feature_type.upper()}: \n")
    
    print(f"  ✓ : {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Step9: FC-sum/PLS")
    parser.add_argument("--step7_dir", type=str, default=DEFAULT_STEP7_DIR,
                       help="Step7")
    parser.add_argument("--step2_dir", type=str, default=DEFAULT_STEP2_DIR,
                       help="Step2")
    parser.add_argument("--labels_file", type=str, default=None,
                       help="step2_dir")
    parser.add_argument("--gene_csv", type=str, default=DEFAULT_GENE_CSV,
                       help="CSV")
    parser.add_argument("--receptor_csv", type=str, default=DEFAULT_RECEPTOR_CSV,
                       help="CSV")
    parser.add_argument("--receptor_names", type=str, default=DEFAULT_RECEPTOR_NAMES,
                       help="npy")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR,
                       help="")
    parser.add_argument("--max_pls_comp", type=int, default=DEFAULT_MAX_PLS_COMP,
                       help="PLS")
    parser.add_argument("--n_perm", type=int, default=DEFAULT_N_PERM,
                       help="")
    parser.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE,
                       help="")
    parser.add_argument("--absolute", action="store_true", default=True,
                       help="FC-sum")
    parser.add_argument("--run_enrichment", action="store_true", default=True,
                       help="")
    parser.add_argument("--vip_threshold", type=float, default=1.5,
                       help="VIP")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Step9: FC-sum/PLS")
    print("=" * 60)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        fcs_asd, fcs_td = load_fc_matrices(args.step7_dir)
        labels = load_subtype_labels(args.step2_dir, args.labels_file)
        
        if fcs_asd.shape[0] != len(labels):
            raise ValueError(f"FC ({fcs_asd.shape[0]})  ({len(labels)}) ")
        
        subtype_fc_sums = compute_subtype_fc_sum(fcs_asd, labels, absolute=args.absolute)
        
        for k, fc_sum in subtype_fc_sums.items():
            fc_sum_path = out_dir / f"subtype{k+1}_fc_sum.npy"
            np.save(fc_sum_path, fc_sum)
            print(f"✓ {k+1} FC-sum: {fc_sum_path}")
        
        gene_df = load_gene_data(args.gene_csv)
        receptor_df = load_receptor_data(args.receptor_csv, args.receptor_names)
        
        n_subtypes = len(subtype_fc_sums)
        
        for k, fc_sum in subtype_fc_sums.items():
            print(f"\n{'=' * 60}")
            print(f" {k+1}")
            print(f"{'=' * 60}")
            
            if not gene_df.empty:
                gene_vip_df = run_pls_analysis(
                    fc_sum, gene_df, 'gene', k+1, out_dir,
                    args.max_pls_comp, args.n_perm, args.random_state
                )
                
                if args.run_enrichment and gene_vip_df is not None:
                    run_enrichment_analysis(
                        gene_vip_df, 'gene', k+1, out_dir,
                        vip_threshold=args.vip_threshold
                    )
            
            if not receptor_df.empty:
                run_pls_analysis(
                    fc_sum, receptor_df, 'receptor', k+1, out_dir,
                    args.max_pls_comp, args.n_perm, args.random_state
                )
        
        create_summary_report(out_dir, n_subtypes)
        
        print("\n" + "=" * 60)
        print("✅ Step9")
        print(f": {out_dir.resolve()}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ : {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

