#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supply - Step 7: ABIDE Network Analysis with Motion (FD) Control
----------------------------------------------------------------

Usage:
  python script0925/supply-step7_abide_network_analysis_with_fd_control.py \
    --outdir unsup_results/step7_abide_fixed \
    --step2-outdir results_step2_tuned_retry \
    --net-map atlas/subregion_func_network_Yeo_updated.csv \
    --use-gsr
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler

import sys as _sys, os as _os
_cur = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_cur)
if _root not in _sys.path:
    _sys.path.insert(0, _root)

from ..utils.step7_brain_metrics_abide import load_and_preprocess_data
from ..utils.step7_brain_metrics_abide import load_step2_labels
from ..utils.step7_abide_network_metrics import load_network_map_csv, aggregate_subject_metric_by_network

# Define local aggregation function to keep individual data (Z-values)
def aggregate_fc_by_network_individual_z(fcs_z: np.ndarray, net_labels: np.ndarray) -> np.ndarray:
    """
    Aggregate ROI-level FC (Z) to Network-level FC (Z) for each subject.
    Returns: (N, n_net, n_net)
    """
    n_net = int(np.max(net_labels)) + 1
    N = fcs_z.shape[0]
    out = np.zeros((N, n_net, n_net), dtype=np.float32)
    
    # Optimize: use loop but avoid heavy numpy indexing per subject if possible?
    # Or just simple double loop over networks (outer) then vectorized over subjects?
    # No, fcs_z is (N, R, R). We can slice (:, ia, ib).
    
    for a in range(n_net):
        ia = np.where(net_labels == a)[0]
        for b in range(n_net):
            ib = np.where(net_labels == b)[0]
            
            # Extract sub-block for all subjects: (N, len(ia), len(ib))
            # Advanced indexing: fcs_z[:, ia][:, :, ib] ? No.
            # fcs_z[:, ia[:, None], ib] works.
            
            block = fcs_z[:, ia[:, None], ib] # (N, len(ia), len(ib))
            
            if a == b:
                # For diagonal blocks, we only want upper triangle to avoid self-correlation
                # Since we have N subjects, we need to mask per subject or just use mean excluding diag.
                # However, diagonal of input fcs_z is usually 0 or nan.
                # Safer to extract upper triangle elements.
                # block is (N, k, k).
                k = block.shape[1]
                if k > 1:
                    iu_k = np.triu_indices(k, 1)
                    # block[:, iu_k[0], iu_k[1]] -> (N, n_edges)
                    vals = block[:, iu_k[0], iu_k[1]]
                    out[:, a, b] = np.nanmean(vals, axis=1)
                else:
                    out[:, a, b] = np.nan # Single ROI network has no internal FC
            else:
                # Off-diagonal block: simple mean
                # We can flatten last two dims
                vals = block.reshape(N, -1)
                out[:, a, b] = np.nanmean(vals, axis=1)
                
    return out # Z-values

def compute_gbc_split(fcs_z):
    """
    Compute Positive and Negative GBC (Z-values) from FC matrices.
    fcs_z: (N, R, R)
    Returns: gbc_pos (N, R), gbc_neg (N, R)
    """
    N, R, _ = fcs_z.shape
    gbc_pos = np.zeros((N, R), dtype=np.float32)
    gbc_neg = np.zeros((N, R), dtype=np.float32)
    
    print("Computing GBC (Pos/Neg) from FC matrices...")
    
    for i in range(N):
        mat = fcs_z[i]
        
        # Per ROI
        for r in range(R):
            # Get row excluding self
            # Simple way: create mask
            vals = np.delete(mat[r], r)
            
            # Remove NaNs if any
            vals = vals[np.isfinite(vals)]
            
            # Pos
            pos_vals = vals[vals > 0]
            gbc_pos[i, r] = np.mean(pos_vals) if pos_vals.size > 0 else 0
            
            # Neg
            neg_vals = vals[vals < 0]
            gbc_neg[i, r] = np.mean(neg_vals) if neg_vals.size > 0 else 0
            
    return gbc_pos, gbc_neg

def compute_network_interaction(fcs_z, net_labels):
    """
    Compute Within-Network (Cooperate) and Between-Network (Competitive) FC.
    Returns: within (N, n_net), between (N, n_net)
    """
    N, R, _ = fcs_z.shape
    n_net = int(np.max(net_labels)) + 1
    
    within = np.zeros((N, n_net), dtype=np.float32)
    between = np.zeros((N, n_net), dtype=np.float32)
    
    print("Computing Within (Cooperate) and Between (Competitive) Network FC...")
    
    for i in range(N):
        mat = fcs_z[i].copy()
        np.fill_diagonal(mat, np.nan)
        
        for k in range(n_net):
            idx_k = np.where(net_labels == k)[0]
            idx_other = np.where(net_labels != k)[0]
            
            # Within
            if len(idx_k) > 1:
                block_w = mat[np.ix_(idx_k, idx_k)]
                # Extract upper triangle to strictly avoid diagonal and duplicates
                # (Though mean of full block excluding diag is same)
                # Let's just use nanmean of full block since diag is nan
                within[i, k] = np.nanmean(block_w)
            else:
                within[i, k] = np.nan
            
            # Between
            if len(idx_other) > 0:
                block_b = mat[np.ix_(idx_k, idx_other)]
                between[i, k] = np.nanmean(block_b)
            else:
                between[i, k] = np.nan
                
    return within, between

def compute_tdc_guided_network_metrics(fcs_all, n_td_total, net_labels):
    """
    Compute Network-level Cooperate (Pos in TDC) and Competitive (Neg in TDC) Strength.
    """
    print(f"Computing Network Metrics guided by TDC mean (n_td={n_td_total})...")
    fcs_td = fcs_all[-n_td_total:]
    mean_td = np.nanmean(fcs_td, axis=0) # (R, R)
    np.fill_diagonal(mean_td, 0)
    
    mask_pos = (mean_td > 0)
    mask_neg = (mean_td < 0)
    
    N = fcs_all.shape[0]
    n_net = int(np.max(net_labels)) + 1
    
    coop_net = np.zeros((N, n_net), dtype=np.float32)
    comp_net = np.zeros((N, n_net), dtype=np.float32)
    
    for k in range(n_net):
        idx_k = np.where(net_labels == k)[0]
        
        # Mask for edges connected to network k
        net_mask = np.zeros(mean_td.shape, dtype=bool)
        net_mask[idx_k, :] = True
        net_mask[:, idx_k] = True
        # Ensure symmetry and exclude diagonal logic handled by mask_pos/neg (diag is 0)
        
        # Intersection
        mask_k_pos = mask_pos & net_mask
        mask_k_neg = mask_neg & net_mask
        
        for i in range(N):
            mat = fcs_all[i]
            vals_p = mat[mask_k_pos]
            vals_n = mat[mask_k_neg]
            
            coop_net[i, k] = np.nanmean(vals_p) if vals_p.size else np.nan
            comp_net[i, k] = np.nanmean(vals_n) if vals_n.size else np.nan
            
    return coop_net, comp_net

def compute_tdc_guided_global_metrics(fcs_all, n_td_total):
    """
    Compute Global Cooperate (Positive in TDC) and Competitive (Negative in TDC) Strength.
    Aggregated across ALL networks (Whole Brain).
    """
    print(f"Computing Global Metrics guided by TDC mean (n_td={n_td_total})...")
    fcs_td = fcs_all[-n_td_total:]
    mean_td = np.nanmean(fcs_td, axis=0) # (R, R)
    np.fill_diagonal(mean_td, 0)
    
    mask_pos = (mean_td > 0)
    mask_neg = (mean_td < 0)
    
    print(f"Global TDC Mask: {mask_pos.sum()} pos edges, {mask_neg.sum()} neg edges.")
    
    N = fcs_all.shape[0]
    coop_glob = np.zeros(N, dtype=np.float32)
    comp_glob = np.zeros(N, dtype=np.float32)
    
    for i in range(N):
        mat = fcs_all[i]
        # Full matrix masking
        vals_p = mat[mask_pos]
        vals_n = mat[mask_neg]
        
        coop_glob[i] = np.nanmean(vals_p) if vals_p.size else np.nan
        comp_glob[i] = np.nanmean(vals_n) if vals_n.size else np.nan
        
    return coop_glob, comp_glob

# Plot settings
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150
})

def pretty_subtype_label(x, td_label):
    return 'TD' if x == td_label else f'Subtype {x+1}'

def _apply_yeo7_axis(ax, n):
    names7 = ['VN','SMN','DAN','VAN','LN','FPN','DMN']
    if n == 7:
        labels = names7
    else:
        labels = [str(i+1) for i in range(n)]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

def get_covariates_aligned(n_asd, n_td, abide_pklz, cmi_pklz):
    """
    Load ABIDE phenotypic data and align with step7 data order.
    Step7 data order is: [ASD subjects from step1] then [TD subjects from step1].
    We assume step1 `load_and_preprocess_data` is deterministic.
    """
    print("Loading covariates from pklz...")
    abide_df, _ = load_and_preprocess_data(abide_pklz, cmi_pklz)
    
    # Split ASD/TD as in Step7
    abide_asd = abide_df[abide_df['DX_GROUP'] == 0].copy().reset_index(drop=True)
    abide_td  = abide_df[abide_df['DX_GROUP'] == 1].copy().reset_index(drop=True)
    
    # Verify lengths
    if len(abide_asd) != n_asd:
        print(f"[WARN] ASD covariate length {len(abide_asd)} != feature length {n_asd}. Truncating to min.")
    if len(abide_td) != n_td:
        print(f"[WARN] TD covariate length {len(abide_td)} != feature length {n_td}. Truncating to min.")
        
    n_asd_use = min(len(abide_asd), n_asd)
    n_td_use  = min(len(abide_td), n_td)
    
    abide_asd = abide_asd.iloc[:n_asd_use]
    abide_td  = abide_td.iloc[:n_td_use]
    
    # Concat
    df_cov = pd.concat([abide_asd, abide_td], axis=0, ignore_index=True)
    
    # Extract columns
    # Ensure numeric
    df_cov['AGE_AT_SCAN'] = pd.to_numeric(df_cov['AGE_AT_SCAN'], errors='coerce')
    if 'mean_fd' in df_cov.columns:
        df_cov['mean_fd'] = pd.to_numeric(df_cov['mean_fd'], errors='coerce')
    else:
        print("[WARN] 'mean_fd' column not found! Using zeros.")
        df_cov['mean_fd'] = 0.0
        
    return df_cov, n_asd_use, n_td_use

def run_glm_pairwise_t(y, groups, df_cov, target_group, ref_group='TD'):
    """
    Run GLM for pairwise comparison (Target vs Ref) controlling for covariates.
    Returns T-statistic for the group term.
    """
    # Filter data for only these two groups
    mask = np.isin(groups, [target_group, ref_group])
    y_sub = y[mask]
    groups_sub = groups[mask]
    data_sub = df_cov.loc[mask].copy()
    
    data_sub['y'] = y_sub
    data_sub['group'] = groups_sub
    
    # Enforce order: Ref is base
    data_sub['group'] = pd.Categorical(data_sub['group'], categories=[ref_group, target_group], ordered=True)
    
    try:
        model = ols('y ~ C(group) + AGE_AT_SCAN + C(SEX) + C(SITE_KEY_STRICT) + mean_fd', data=data_sub).fit()
        # T-value for target group (second category)
        term = f'C(group)[T.{target_group}]'
        return model.tvalues[term], model.pvalues[term]
    except:
        return np.nan, np.nan

def regress_out_covariates(y, df_cov, formula='y ~ AGE_AT_SCAN + C(SEX) + C(SITE_KEY_STRICT) + mean_fd'):
    """
    Regress out covariates to get adjusted values (residuals + mean).
    y: target variable (N,)
    df_cov: dataframe with covariates (N, cols)
    """
    data = df_cov.copy()
    data['y'] = y
    
    # Handle NaNs
    valid = data[['y', 'AGE_AT_SCAN', 'mean_fd']].notna().all(axis=1)
    if valid.sum() < 10:
        return np.full_like(y, np.nan)
        
    data_valid = data[valid]
    
    # Fit OLS
    try:
        model = ols(formula, data=data_valid).fit()
        # Residuals
        resid = model.resid
        # Add back global mean to keep scale interpretable
        y_adj_valid = resid + data_valid['y'].mean()
        
        # Fill back
        y_adj = np.full_like(y, np.nan)
        y_adj[valid] = y_adj_valid
        return y_adj
    except Exception as e:
        print(f"Regression failed: {e}")
        return y

def run_glm_stats(y, groups, df_cov, formula='y ~ C(group) + AGE_AT_SCAN + C(SEX) + C(SITE_KEY_STRICT) + mean_fd'):
    """
    Run GLM to test Group effect controlling for FD etc.
    """
    data = df_cov.copy()
    data['y'] = y
    data['group'] = groups
    
    # Ensure TD is reference? Or specific contrasts?
    # If 'group' is categorical strings, statsmodels picks one as ref.
    # We want to test if 'C(group)' is significant overall (ANOVA-like) or specific pairwise.
    # For heatmap F-value, we want the F of the Group term.
    
    try:
        model = ols(formula, data=data).fit()
        aov = sm.stats.anova.anova_lm(model, typ=2)
        if 'C(group)' in aov.index:
            return aov.loc['C(group)', 'F'], aov.loc['C(group)', 'PR(>F)']
        else:
            return np.nan, np.nan
    except:
        return np.nan, np.nan

def plot_adjusted_violin(y_adj, groups, title, out_path, labels_order=None, sig_pairs=None):
    """
    Plot violin of adjusted values with significance annotations.
    sig_pairs: list of (g1, g2, p_val)
    """
    df = pd.DataFrame({'Value (Adj)': y_adj, 'Group': groups})
    df = df.dropna()
    
    if labels_order:
        present = [g for g in labels_order if g in df['Group'].unique()]
    else:
        present = sorted(df['Group'].unique())
        
    plt.figure(figsize=(7, 6)) # Taller for annotations
    ax = sns.violinplot(x='Group', y='Value (Adj)', data=df, order=present, palette='Set2', inner='box')
    plt.title(title)
    plt.xticks(rotation=15)
    sns.despine()
    
    # Annotate significance
    if sig_pairs:
        # Get y range for placing brackets
        y_max = df['Value (Adj)'].max()
        y_range = y_max - df['Value (Adj)'].min()
        offset = y_range * 0.05
        curr_y = y_max + offset
        
        # Map group to x-coordinate
        group_map = {g: i for i, g in enumerate(present)}
        
        for g1, g2, p in sig_pairs:
            if g1 not in group_map or g2 not in group_map: continue
            if p >= 0.05: continue # Skip non-significant
            
            x1, x2 = group_map[g1], group_map[g2]
            
            # Draw bracket
            plt.plot([x1, x1, x2, x2], [curr_y, curr_y+offset, curr_y+offset, curr_y], lw=1.5, c='k')
            
            # Star
            star = '*' if p < 0.05 else ''
            if p < 0.01: star = '**'
            if p < 0.001: star = '***'
            
            plt.text((x1+x2)/2, curr_y+offset, star, ha='center', va='bottom', color='k')
            
            curr_y += offset * 2.5 # Move up for next bracket
            
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, required=True, help='Step7 output dir (input)')
    parser.add_argument('--step2-outdir', type=str, required=True, help='Labels dir')
    parser.add_argument('--net-map', type=str, default='atlas/subregion_func_network_Yeo_updated.csv')
    parser.add_argument('--use-gsr', action='store_true', help='Use GSR versions of metrics')
    parser.add_argument('--abide-pklz', type=str, default='DATA/combined_ABIDE_information_with_fMRI.pklz')
    parser.add_argument('--cmi-pklz', type=str, default='CMI-DATA/combined_asd_td_rest_run1_data.pklz')
    args = parser.parse_args()

    # Directories
    suffix = "_gsr" if args.use_gsr else ""
    res_dir = os.path.join(args.outdir, f'fd_controlled_analysis{suffix}')
    os.makedirs(res_dir, exist_ok=True)
    
    # 1. Load Metrics (Individual Level)
    print("Loading metrics...")
    fcs_asd = np.load(os.path.join(args.outdir, f'fcs_asd{suffix}.npy'))
    fcs_td  = np.load(os.path.join(args.outdir, f'fcs_td{suffix}.npy'))
    alffs_asd = np.load(os.path.join(args.outdir, f'alffs_asd{suffix}.npy'))
    alffs_td  = np.load(os.path.join(args.outdir, f'alffs_td{suffix}.npy'))
    
    # Stack metrics
    # Note: Must truncate before stacking if lengths differ from Step1 alignment
    # But we need n_asd to load labels.
    
    n_asd, n_td = len(fcs_asd), len(fcs_td)
    print(f"ASD: {n_asd}, TD: {n_td}")
    
    # 2. Load Labels
    labels = load_step2_labels(args.step2_outdir) # [0, 1, 2...] for ASD
    if len(labels) != n_asd:
        # Attempt truncate if mismatch is small (sometimes labels are all, but embeddings were filtered)
        # But here fcs_asd comes from compute_only which aligns with labels... wait.
        # load_abide in compute_only aligns.
        # If mismatch, likely path error.
        print(f"[ERROR] Label length {len(labels)} != ASD metric length {n_asd}")
        return

    td_label = labels.max() + 1
    
    # Create group array
    # 0..K-1 for subtypes, K for TD
    groups_num = np.concatenate([labels, np.full(n_td, td_label)])
    groups_str = np.array([pretty_subtype_label(g, td_label) for g in groups_num])
    
    # 3. Load Covariates & FD
    df_cov, n_asd_use, n_td_use = get_covariates_aligned(n_asd, n_td, args.abide_pklz, args.cmi_pklz)
    
    # Truncate metrics if needed (should be matched)
    fcs_asd = fcs_asd[:n_asd_use]; fcs_td = fcs_td[:n_td_use]
    alffs_asd = alffs_asd[:n_asd_use]; alffs_td = alffs_td[:n_td_use]
    groups_str = groups_str[:(n_asd_use + n_td_use)]
    
    # Stack metrics
    fcs_all = np.concatenate([fcs_asd, fcs_td], axis=0)
    alffs_all = np.concatenate([alffs_asd, alffs_td], axis=0)
    
    # Compute GBC (Pos/Neg) locally
    gbc_pos_roi, gbc_neg_roi = compute_gbc_split(fcs_all) # (N, 246)
    
    # 4. Prepare Network Mapping
    C = fcs_all.shape[1]
    net_labels = load_network_map_csv(args.net_map, C)
    n_net = int(np.max(net_labels)) + 1
    
    # Compute Within/Between (Cooperate/Competitive)
    # Needs net_labels
    within_net, between_net = compute_network_interaction(fcs_all, net_labels) # (N, 7)
    
    # Compute TDC-guided Network Metrics
    coop_tdc_net, comp_tdc_net = compute_tdc_guided_network_metrics(fcs_all, n_td, net_labels)
    
    # =========================================================================
    # Analysis A: Network FC (GLM with FD)
    # =========================================================================
    print("\nAnalyzing Network FC (controlling for FD)...")
    
    # Aggregate ROI FC to Network FC (28 edges)
    # Note: aggregate_fc_by_network_z takes (N, R, R) -> (N, Net, Net)
    fcs_net_z = aggregate_fc_by_network_individual_z(fcs_all, net_labels) # shape (N, 7, 7) Z-values
    
    # Extract upper triangle
    iu = np.triu_indices(n_net, 1)
    n_edges = len(iu[0])
    X_edges = np.array([mat[iu] for mat in fcs_net_z]) # (N, 28)
    
    f_vals = []
    p_vals = []
    
    for i in range(n_edges):
        edge_val = X_edges[:, i]
        f, p = run_glm_stats(edge_val, groups_str, df_cov)
        f_vals.append(f)
        p_vals.append(p)
        
    f_vals = np.array(f_vals)
    p_vals = np.array(p_vals)
    
    # FDR
    _, q_vals, _, _ = multipletests(p_vals, method='fdr_bh')
    
    # Plot Heatmap of F-values
    # Reconstruct matrix
    F_mat = np.zeros((n_net, n_net))
    F_mat[iu] = f_vals
    F_mat = F_mat + F_mat.T
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(F_mat, cmap='viridis', square=True, cbar_kws={'label': 'F-statistic'})
    _apply_yeo7_axis(plt.gca(), n_net)
    plt.title(f'Network FC Group Differences\n(Adj for FD, Age, Sex, Site)')
    
    # Add stars
    stars = np.where(q_vals < 0.001, '**', np.where(q_vals < 0.05, '*', ''))
    for idx, (r, c) in enumerate(zip(*iu)):
        s = stars[idx]
        if s:
            plt.text(c, r, s, ha='center', va='center', color='white', fontweight='bold')
            plt.text(r, c, s, ha='center', va='center', color='white', fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, 'net_fc_glm_fd_control_heatmap.png'), dpi=300)
    plt.close()
    print(f"Saved FC heatmap. Significant edges (q<0.05): {(q_vals < 0.05).sum()}")
    
    # -------------------------------------------------------------------------
    # Analysis A2: Pairwise FC Differences (T-maps)
    # -------------------------------------------------------------------------
    print("\nGenerating Pairwise FC Difference Maps (controlling for FD)...")
    subtypes = [g for g in np.unique(groups_str) if g != 'TD']
    
    for sub in subtypes:
        t_vals = []
        p_vals_sub = []
        for i in range(n_edges):
            edge_val = X_edges[:, i]
            t, p = run_glm_pairwise_t(edge_val, groups_str, df_cov, sub, 'TD')
            t_vals.append(t)
            p_vals_sub.append(p)
            
        # Reconstruct T matrix
        T_mat = np.zeros((n_net, n_net))
        T_mat[iu] = t_vals
        T_mat = T_mat + T_mat.T
        
        # Plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(T_mat, cmap='coolwarm', center=0, square=True, vmin=-4, vmax=4, cbar_kws={'label': 'T-statistic'})
        _apply_yeo7_axis(plt.gca(), n_net)
        plt.title(f'FC Diff: {sub} vs TD\n(Adj for FD, Age, Sex, Site)')
        plt.tight_layout()
        safe_sub = sub.replace(' ', '_')
        plt.savefig(os.path.join(res_dir, f'net_fc_diff_{safe_sub}_vs_TD_fd_adj.png'), dpi=300)
        plt.close()

    # =========================================================================
    # Analysis B: Network ALFF (GLM + Adjusted Violin)
    # =========================================================================
    print("\nAnalyzing Network ALFF (controlling for FD)...")
    
    # Aggregate ALFF to Network
    alff_net = aggregate_subject_metric_by_network(alffs_all, net_labels) # (N, 7)
    
    names7 = ['Visual','Somatomotor','Dorsal Attn','Ventral Attn','Limbic','Frontoparietal','Default Mode']
    
    stats_rows = []
    
    for i in range(n_net):
        net_name = names7[i] if i < 7 else f"Net{i}"
        val = alff_net[:, i]
        
        # 1. GLM Stats
        f, p = run_glm_stats(val, groups_str, df_cov)
        stats_rows.append({'Metric': 'ALFF', 'Network': net_name, 'F': f, 'p': p})
        
        # 2. Regress out covariates (Age, Sex, Site, FD) -> Get Residuals
        # Formula: val ~ Age + Sex + Site + FD. (NO Group)
        # Use residuals + mean for plotting
        val_adj = regress_out_covariates(val, df_cov, formula='y ~ AGE_AT_SCAN + C(SEX) + C(SITE_KEY_STRICT) + mean_fd')
        
        # 3. Plot Violin
        plot_title = f"ALFF {net_name} (Adj for FD+Covs)\nGLM Group p={p:.3g}"
        out_name = os.path.join(res_dir, f'alff_violin_adj_{net_name}.png')
        plot_adjusted_violin(val_adj, groups_str, plot_title, out_name, 
                             labels_order=['Subtype 1', 'Subtype 2', 'Subtype 3', 'TD'])
                             
    # Save stats
    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv(os.path.join(res_dir, 'alff_glm_stats.csv'), index=False)
    print("Saved ALFF stats and violin plots.")

    # =========================================================================
    # Analysis C: Network Metrics (GBC Pos/Neg, Cooperate/Competitive)
    # =========================================================================
    print("\nAnalyzing Network Metrics (GBC Pos/Neg, Within/Between) (controlling for FD)...")
    
    # Aggregate ROI GBC to Network GBC
    gbc_pos_net = aggregate_subject_metric_by_network(gbc_pos_roi, net_labels) # (N, 7)
    gbc_neg_net = aggregate_subject_metric_by_network(gbc_neg_roi, net_labels)
    
    # Loop over channels
    metrics = [
        ('GBC_Pos', 'Pos Strength', gbc_pos_net),
        ('GBC_Neg', 'Neg Strength', gbc_neg_net),
        ('Within', 'Within-Net', within_net),
        ('Between', 'Between-Net', between_net),
        ('TDC_Coop', 'Cooperate (TDC)', coop_tdc_net),
        ('TDC_Comp', 'Competitive (TDC)', comp_tdc_net)
    ]
    
    gbc_stats_rows = []

    for met_code, met_name, data_net in metrics:
        for i in range(n_net):
            net_name = names7[i] if i < 7 else f"Net{i}"
            val = data_net[:, i]
            
            # 1. GLM Stats
            f, p = run_glm_stats(val, groups_str, df_cov)
            gbc_stats_rows.append({'Metric': met_code, 'Type': met_name, 'Network': net_name, 'F': f, 'p': p})
            
            # 1b. Pairwise Stats (Subtype vs TD)
            sig_pairs = []
            subtypes = [g for g in np.unique(groups_str) if g != 'TD']
            for sub in subtypes:
                t_sub, p_sub = run_glm_pairwise_t(val, groups_str, df_cov, sub, 'TD')
                sig_pairs.append((sub, 'TD', p_sub))
            
            # 2. Adjust
            val_adj = regress_out_covariates(val, df_cov)
            
            # 3. Plot
            # Title: "Network Cooperate (Within) - DMN"
            plot_title = f"{net_name} {met_name}\n({met_code}) (Adj for FD) GLM p={p:.3g}"
            # Clean filename
            fname = f"metric_{met_code.lower()}_{net_name.replace(' ', '_')}.png"
            out_name = os.path.join(res_dir, fname)
            
            plot_adjusted_violin(val_adj, groups_str, plot_title, out_name,
                                 labels_order=['Subtype 1', 'Subtype 2', 'Subtype 3', 'TD'],
                                 sig_pairs=sig_pairs)
    
    pd.DataFrame(gbc_stats_rows).to_csv(os.path.join(res_dir, 'network_metrics_glm_stats.csv'), index=False)
    print("Saved Network Metrics stats and violin plots.")

    # =========================================================================
    # Analysis D: Global Metrics (TDC-guided)
    # =========================================================================
    print("\nAnalyzing Global Metrics (Coop/Comp) (controlling for FD)...")
    coop_glob, comp_glob = compute_tdc_guided_global_metrics(fcs_all, n_td)
    
    global_metrics = [
        ('Global_Coop', 'Global Cooperate', coop_glob),
        ('Global_Comp', 'Global Competitive', comp_glob)
    ]
    
    for met_code, met_name, val in global_metrics:
        # GLM
        f, p = run_glm_stats(val, groups_str, df_cov)
        
        # Pairwise
        sig_pairs = []
        subtypes = [g for g in np.unique(groups_str) if g != 'TD']
        for sub in subtypes:
            t_sub, p_sub = run_glm_pairwise_t(val, groups_str, df_cov, sub, 'TD')
            sig_pairs.append((sub, 'TD', p_sub))
        
        # Adjust
        val_adj = regress_out_covariates(val, df_cov)
        
        # Plot
        plot_title = f"{met_name} (TDC-guided)\n(Adj for FD) GLM p={p:.3g}"
        fname = f"global_{met_code.lower()}.png"
        out_name = os.path.join(res_dir, fname)
        
        plot_adjusted_violin(val_adj, groups_str, plot_title, out_name,
                             labels_order=['Subtype 1', 'Subtype 2', 'Subtype 3', 'TD'],
                             sig_pairs=sig_pairs)
                             
    print("Saved Global metrics plots.")

    print(f"\nDone. Results in {res_dir}")

if __name__ == '__main__':
    main()

