#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 7: Network-level Functional Connectivity Statistics

Derives per-subject 7×7 (or 6×6) network FC matrices from pre-computed ROI FC
arrays (fcs_asd.npy / fcs_td.npy), performs group comparisons (ANOVA or
Kruskal–Wallis) with BH-FDR correction, and produces:
  - F-statistic heatmap (significant cells annotated)
  - Optional UMAP embedding of 28-D network FC features
  - Global Brain Connectivity (GBC) decomposition

Usage:
  python 07_brain_connectivity.py \\
    --outdir      <output_dir> \\
    --step2-outdir <path/to/step3_output> \\
    --net-map     atlas/subregion_func_network_Yeo_updated.csv \\
    --method anova --use-gsr
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys as _sys, os as _os
_cur = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_cur)
if _root not in _sys.path:
    _sys.path.insert(0, _root)

from utils.step7_abide_network_metrics import load_network_map_csv, aggregate_fc_by_network_z
from utils.step7_brain_metrics_abide import load_and_preprocess_data
from utils.step7_brain_metrics_abide import load_step2_labels


plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})


def define_positive_negative_edges_by_td(td_mean_fc: np.ndarray, net_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    
    Args:
    
    Returns:
    """
    print(f"Input td_mean_fc shape: {td_mean_fc.shape}, ndim: {td_mean_fc.ndim}")
    
    n_net = td_mean_fc.shape[0]
    iu = np.triu_indices(n_net, k=1)
    
    td_upper_values = td_mean_fc[iu]
    
    positive_mask = td_upper_values > 0.1
    negative_mask = td_upper_values < -0.1
    
    print(f"TD-based edge definition:")
    print(f"  Total edges: {len(td_upper_values)}")
    print(f"  Positive edges (TD FC > 0.1): {positive_mask.sum()}")
    print(f"  Negative edges (TD FC < -0.1): {negative_mask.sum()}")
    print(f"  Neutral edges: {len(td_upper_values) - positive_mask.sum() - negative_mask.sum()}")
    
    return positive_mask, negative_mask


def compute_roi_gbc_by_td_definition(fcs_z: np.ndarray, positive_mask: np.ndarray, negative_mask: np.ndarray) -> np.ndarray:
    """
    
    Args:
    
    Returns:
        gbc_roi: [N_subjects, N_ROI, 3] (all, positive, negative)
    """
    n_subjects, n_roi = fcs_z.shape[0], fcs_z.shape[1]
    gbc_roi = np.zeros((n_subjects, n_roi, 3))
    
    for subj in range(n_subjects):
        fc_subj = fcs_z[subj]
        
        for roi_i in range(n_roi):
            connections = []
            for roi_j in range(n_roi):
                if roi_i != roi_j:
                    connections.append(fc_subj[roi_i, roi_j])
            
            if len(connections) > 0:
                connections = np.array(connections)
                
                gbc_roi[subj, roi_i, 0] = np.nanmean(connections)
                
                pos_connections = []
                neg_connections = []
                
                for roi_j in range(n_roi):
                    if roi_i != roi_j:
                        if positive_mask[roi_i, roi_j]:
                            pos_connections.append(fc_subj[roi_i, roi_j])
                        elif negative_mask[roi_i, roi_j]:
                            neg_connections.append(fc_subj[roi_i, roi_j])
                
                if len(pos_connections) > 0:
                    gbc_roi[subj, roi_i, 1] = np.nanmean(pos_connections)
                else:
                    gbc_roi[subj, roi_i, 1] = 0.0
                
                if len(neg_connections) > 0:
                    gbc_roi[subj, roi_i, 2] = np.nanmean(neg_connections)
                else:
                    gbc_roi[subj, roi_i, 2] = 0.0
    
    return gbc_roi


def compute_network_gbc_by_td_definition(fcs_z: np.ndarray, net_labels: np.ndarray, 
                                        positive_mask: np.ndarray, negative_mask: np.ndarray) -> np.ndarray:
    """
    
    Args:
    
    Returns:
        gbc_network: [N_subjects, N_networks, 3] (all, positive, negative)
    """
    n_subjects, n_networks = fcs_z.shape[0], fcs_z.shape[1]
    gbc_network = np.zeros((n_subjects, n_networks, 3))
    
    iu = np.triu_indices(n_networks, k=1)
    
    for subj in range(n_subjects):
        fc_subj = fcs_z[subj]
        
        for net_i in range(n_networks):
            connections = []
            for net_j in range(n_networks):
                if net_i != net_j:
                    if net_i < net_j:
                        edge_idx = np.where((iu[0] == net_i) & (iu[1] == net_j))[0]
                    else:
                        edge_idx = np.where((iu[0] == net_j) & (iu[1] == net_i))[0]
                    
                    if len(edge_idx) > 0:
                        connections.append(fc_subj[net_i, net_j])
            
            if len(connections) > 0:
                connections = np.array(connections)
                
                gbc_network[subj, net_i, 0] = np.nanmean(connections)
                
                pos_connections = []
                neg_connections = []
                
                for net_j in range(n_networks):
                    if net_i != net_j:
                        if net_i < net_j:
                            edge_idx = np.where((iu[0] == net_i) & (iu[1] == net_j))[0]
                        else:
                            edge_idx = np.where((iu[0] == net_j) & (iu[1] == net_i))[0]
                        
                        if len(edge_idx) > 0:
                            edge_idx = edge_idx[0]
                            if positive_mask[edge_idx]:
                                pos_connections.append(fc_subj[net_i, net_j])
                            elif negative_mask[edge_idx]:
                                neg_connections.append(fc_subj[net_i, net_j])
                
                if len(pos_connections) > 0:
                    gbc_network[subj, net_i, 1] = np.nanmean(pos_connections)
                else:
                    gbc_network[subj, net_i, 1] = 0.0
                
                if len(neg_connections) > 0:
                    gbc_network[subj, net_i, 2] = np.nanmean(neg_connections)
                else:
                    gbc_network[subj, net_i, 2] = 0.0
    
    return gbc_network


def extract_network_fc_vector(fcs_z: np.ndarray, net_labels: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray,np.ndarray]]:
    """Vectorize network-level mean FC from [N, ROI, ROI] to [N, n_pairs]."""
    n_net = int(np.max(net_labels)) + 1
    iu = np.triu_indices(n_net, 1)
    feats = []
    for fc in fcs_z:
        fc_net_r = aggregate_fc_by_network_z(np.expand_dims(fc, 0), net_labels)
        feats.append(fc_net_r[iu])
    return np.stack(feats, axis=0), iu


def pretty_group_labels(labels: np.ndarray) -> list[str]:
    td_label = int(np.max(labels))
    out = []
    for g in labels:
        if g == td_label:
            out.append('TD')
        else:
            out.append(f'Subtype {int(g)+1}')
    return out


def stats_by_edge(X_asd: np.ndarray, X_td: np.ndarray, labels_asd: np.ndarray, method: str = 'anova') -> pd.DataFrame:
    from scipy.stats import f_oneway, kruskal
    from statsmodels.stats.multitest import multipletests
    td_lab = labels_asd.max() + 1
    labels_all = np.concatenate([labels_asd, np.full(len(X_td), td_lab)])
    uniq = np.unique(labels_all)
    pvals = []
    stats_val = []
    for i in range(X_asd.shape[1]):
        vals = np.concatenate([X_asd[:, i], X_td[:, i]])
        groups = [vals[labels_all == g] for g in uniq]
        if any(len(g) < 2 for g in groups):
            p = np.nan; statv = np.nan
        else:
            if method == 'kruskal':
                statv, p = kruskal(*groups)
            else:
                statv, p = f_oneway(*groups)
        pvals.append(p)
        stats_val.append(statv)
    pvals = np.array(pvals)
    stats_val = np.array(stats_val)
    reject, qvals, _, _ = multipletests(pvals, method='fdr_bh', is_sorted=False)
    df = pd.DataFrame({
        'edge_idx': np.arange(X_asd.shape[1]),
        'stat': stats_val,
        'p': pvals,
        'q': qvals,
        'sig_fdr': reject,
    })
    return df


def _apply_yeo7_axis(ax, n):
    names7 = ['VN','SMN','DAN','VAN','LN','FPN','DMN']
    if n == 7:
        labels = names7
    else:
        labels = [str(i+1) for i in range(n)]
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)


def plot_f_matrix(stat_vals: np.ndarray, iu: tuple[np.ndarray,np.ndarray], title: str, out_png: str,
                  q_vals: np.ndarray | None = None):
    n = int((1 + np.sqrt(1 + 8*len(stat_vals))) / 2)  # recover matrix size
    M = np.zeros((n, n), dtype=float)
    M[iu] = stat_vals
    M[(iu[1], iu[0])] = stat_vals
    np.fill_diagonal(M, 0.0)
    vmin = np.nanmin(M)
    vmax = np.nanmax(M)
    plt.figure(figsize=(5.6, 4.6))
    im = plt.imshow(M, cmap='hot', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, shrink=0.8); cbar.set_label('F-value', fontsize=12, fontweight='bold')
    plt.title(title)
    plt.xlabel('Network', fontsize=14, fontweight='bold'); plt.ylabel('Network', fontsize=14, fontweight='bold')
    ax = plt.gca(); _apply_yeo7_axis(ax, n)
    if q_vals is not None:
        try:
            stars = np.where(q_vals < 1e-3, '**', np.where(q_vals < 0.05, '*', ''))
            for (y, x), s in zip(zip(iu[0], iu[1]), stars):
                if s:
                    plt.text(x, y, s, ha='center', va='center', color='k', fontsize=10, fontweight='bold')
                    plt.text(y, x, s, ha='center', va='center', color='k', fontsize=10, fontweight='bold')
        except Exception:
            pass
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def cellwise_effect_and_perm(fcs_z_groupA, fcs_z_groupB, net_labels, n_perm=2000, seed=0):
    rng = np.random.default_rng(seed)
    def subj_net_z(fcs_z):
        n_net = int(np.max(net_labels)) + 1
        out = np.zeros((fcs_z.shape[0], n_net, n_net), float)
        for s in range(fcs_z.shape[0]):
            z = fcs_z[s]
            for a in range(n_net):
                ia = np.where(net_labels == a)[0]
                for b in range(n_net):
                    ib = np.where(net_labels == b)[0]
                    sub = z[np.ix_(ia, ib)]
                    vals = sub[np.triu_indices_from(sub, 1)] if a == b else sub.ravel()
                    out[s, a, b] = np.nanmean(vals) if vals.size else np.nan
        return out
    ZA = subj_net_z(fcs_z_groupA)
    ZB = subj_net_z(fcs_z_groupB)
    maskA = ~np.isnan(ZA).any((1, 2)); maskB = ~np.isnan(ZB).any((1, 2))
    ZA, ZB = ZA[maskA], ZB[maskB]
    def cohend(a, b):
        v = ((a.var(ddof=1) + b.var(ddof=1)) / 2.0) + 1e-8
        return (a.mean() - b.mean()) / np.sqrt(v)
    d = np.zeros((ZA.shape[1], ZA.shape[2]))
    p = np.ones_like(d)
    A = ZA.reshape(ZA.shape[0], -1)
    B = ZB.reshape(ZB.shape[0], -1)
    for idx in range(A.shape[1]):
        a = A[:, idx]; b = B[:, idx]
        d.flat[idx] = cohend(a, b)
        pooled = np.concatenate([a, b])
        nA = len(a)
        obs = abs(a.mean() - b.mean())
        cnt = 0
        for _ in range(n_perm):
            rng.shuffle(pooled)
            pa, pb = pooled[:nA], pooled[nA:]
            if abs(pa.mean() - pb.mean()) >= obs:
                cnt += 1
        p.flat[idx] = (cnt + 1) / (n_perm + 1)
    return d, p

def plot_effect_heatmap(mat, title, out_png, vlim=None, cmap='coolwarm', sig_mask: np.ndarray | None = None,
                        q_mat: np.ndarray | None = None):
    import numpy as np
    if vlim is None:
        vlim = 1.0
    plt.figure(figsize=(5.6, 4.6))
    im = plt.imshow(mat, cmap=cmap, vmin=-vlim, vmax=vlim)
    cbar = plt.colorbar(im, shrink=0.8); cbar.set_label('Cohen d (z-space)')
    plt.title(title); plt.xlabel('Yeo7'); plt.ylabel('Yeo7')
    ax = plt.gca(); _apply_yeo7_axis(ax, mat.shape[0])
    if sig_mask is not None:
        try:
            iu = np.triu_indices_from(sig_mask, 1)
            stars = None
            if q_mat is not None:
                stars = np.full(sig_mask.shape, '', dtype=object)
                q_up = q_mat[iu]
                star_up = np.where(q_up < 1e-3, '**', np.where(q_up < 0.05, '*', ''))
                for (y, x), s in zip(zip(iu[0], iu[1]), star_up):
                    if s:
                        plt.text(x, y, s, ha='center', va='center', color='k', fontsize=10, fontweight='bold')
                        plt.text(y, x, s, ha='center', va='center', color='k', fontsize=10, fontweight='bold')
            else:
                yy, xx = iu[0][sig_mask[iu]], iu[1][sig_mask[iu]]
                for y, x in zip(yy, xx):
                    plt.text(x, y, '*', ha='center', va='center', color='k', fontsize=10, fontweight='bold')
                    plt.text(y, x, '*', ha='center', va='center', color='k', fontsize=10, fontweight='bold')
        except Exception:
            pass
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def plot_edge_grouped_bars(X_asd: np.ndarray, X_td: np.ndarray, labels_asd: np.ndarray,
                           iu: tuple[np.ndarray,np.ndarray], out_png: str, title: str):
    import seaborn as sns
    td_lab = labels_asd.max() + 1
    groups = [0, 1, 2, td_lab]
    group_names = {0: 'Subtype 1', 1: 'Subtype 2', 2: 'Subtype 3', td_lab: 'TD'}
    rows = []
    n = iu[0].max() + 1
    edge_names = [f'N{a+1}-N{b+1}' for a, b in zip(iu[0], iu[1])]
    for ei, ename in enumerate(edge_names):
        for g in groups:
            if g == td_lab:
                vals = X_td[:, ei]
            else:
                vals = X_asd[labels_asd == g, ei]
            if len(vals) == 0:
                continue
            rows.append({'edge': ename, 'group': group_names[g], 'value': float(np.nanmean(vals))})
    import pandas as pd
    df_long = pd.DataFrame(rows)
    desired = ['Subtype 1', 'Subtype 2', 'Subtype 3', 'TD']
    df_long['group'] = pd.Categorical(df_long['group'], categories=desired, ordered=True)
    fig_w = max(10.0, 0.6 * df_long['edge'].nunique())
    plt.figure(figsize=(fig_w, 4.6))
    ax = sns.barplot(data=df_long, x='edge', y='value', hue='group', palette='Set2', errorbar=None)
    ax.set_title(title)
    ax.set_xlabel('Network pair (edge)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean r', fontsize=14, fontweight='bold')
    plt.xticks(rotation=60, ha='right')
    ax.legend(title='Group', fontsize=10)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def stats_by_edge_with_covariates(X_asd: np.ndarray, X_td: np.ndarray, labels_asd: np.ndarray,
                                  age_asd: np.ndarray, sex_asd: np.ndarray, site_asd: np.ndarray,
                                  age_td: np.ndarray, sex_td: np.ndarray, site_td: np.ndarray):
    """OLS regression with age/sex/site covariates for each network-pair edge.
    Returns DataFrame with Cohen's d, p-values, and FDR-corrected q-values.
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.multitest import multipletests
    rows = []
    cat_order = ['TD', 'Subtype 1', 'Subtype 2', 'Subtype 3']
    coef_map = {k: [] for k in cat_order if k != 'TD'}
    pval_map = {k: [] for k in cat_order if k != 'TD'}
    td_lab = labels_asd.max() + 1
    group_asd = np.array([f'Subtype {int(g)+1}' for g in labels_asd])
    group_td = np.array(['TD'] * len(X_td))
    for i in range(X_asd.shape[1]):
        val = np.concatenate([X_asd[:, i], X_td[:, i]])
        grp = np.concatenate([group_asd, group_td])
        age = np.concatenate([age_asd, age_td])
        sex = np.concatenate([sex_asd, sex_td]).astype(str)
        site = np.concatenate([site_asd, site_td]).astype(str)
        df = pd.DataFrame({'value': val, 'group': grp, 'age': age, 'sex': sex, 'site': site})
        try:
            df['group'] = pd.Categorical(df['group'], categories=cat_order, ordered=True)
        except Exception:
            pass
        try:
            model = ols('value ~ C(group) + age + C(sex) + C(site)', data=df).fit()
            aov = anova_lm(model, typ=2)
            if 'C(group)' in aov.index:
                F = float(aov.loc['C(group)', 'F']) if pd.notna(aov.loc['C(group)', 'F']) else np.nan
                p = float(aov.loc['C(group)', 'PR(>F)']) if pd.notna(aov.loc['C(group)', 'PR(>F)']) else np.nan
            else:
                F, p = np.nan, np.nan
            for gname in ['Subtype 1', 'Subtype 2', 'Subtype 3']:
                term = f'C(group)[T.{gname}]'
                b = model.params.get(term, np.nan)
                pv = model.pvalues.get(term, np.nan)
                coef_map[gname].append(float(b) if pd.notna(b) else np.nan)
                pval_map[gname].append(float(pv) if pd.notna(pv) else np.nan)
        except Exception:
            F, p = np.nan, np.nan
            for gname in ['Subtype 1', 'Subtype 2', 'Subtype 3']:
                coef_map[gname].append(np.nan)
                pval_map[gname].append(np.nan)
        rows.append((i, F, p))
    df_stats = pd.DataFrame(rows, columns=['edge_idx','stat','p'])
    try:
        rej, q, _, _ = multipletests(df_stats['p'].to_numpy(), method='fdr_bh')
        df_stats['q'] = q; df_stats['sig_fdr'] = rej
    except Exception:
        df_stats['q'] = np.nan; df_stats['sig_fdr'] = False
    for k in list(coef_map.keys()):
        coef_map[k] = np.array(coef_map[k], dtype=float)
    for k in list(pval_map.keys()):
        pval_map[k] = np.array(pval_map[k], dtype=float)
    return df_stats, coef_map, pval_map


def build_edge_matrix(vec: np.ndarray, iu: tuple[np.ndarray,np.ndarray]) -> np.ndarray:
    n = int((1 + np.sqrt(1 + 8*len(vec))) / 2)
    M = np.zeros((n, n), dtype=float)
    M[iu] = vec
    M[(iu[1], iu[0])] = vec
    np.fill_diagonal(M, 0.0)
    return M

def plot_group_vs_td_heatmap(vec: np.ndarray, iu: tuple[np.ndarray,np.ndarray], title: str, out_png: str,
                             sig_vec: np.ndarray | None = None, cmap='coolwarm'):
    M = build_edge_matrix(vec, iu)
    vlim = 1.0
    plt.figure(figsize=(5.6, 4.6))
    im = plt.imshow(M, cmap=cmap, vmin=-vlim, vmax=vlim)
    cbar = plt.colorbar(im, shrink=0.8); cbar.set_label('Adj. diff (vs TD)')
    plt.title(title); plt.xlabel('Yeo7'); plt.ylabel('Yeo7')
    ax = plt.gca(); _apply_yeo7_axis(ax, M.shape[0])
    if sig_vec is not None:
        sigM = build_edge_matrix(sig_vec.astype(float), iu)
        yy, xx = np.where(np.triu(sigM, 1) > 0)
        for y, x in zip(yy, xx):
            plt.scatter([x, y], [y, x], s=10, c='k')
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=str, required=True)
    ap.add_argument('--step2-outdir', type=str, required=True)
    ap.add_argument('--abide-pklz', type=str, default=None,
                    help='ABIDE pickle (required if fcs_asd.npy does not exist yet)')
    ap.add_argument('--bna-atlas', type=str, default='atlas/brainnetome/BN_Atlas_246_1mm.nii.gz',
                    help='Brainnetome 246 atlas NIfTI (used for FC computation)')
    ap.add_argument('--net-map', type=str, default='atlas/subregion_func_network_Yeo_updated.csv')
    ap.add_argument('--use-gsr', action='store_true', help='use fcs_*_gsr.npy instead of fcs_*')
    ap.add_argument('--method', type=str, choices=['anova', 'kruskal'], default='anova')
    ap.add_argument('--umap', action='store_true', help='also compute UMAP of 28D features')
    ap.add_argument(
        '--asd-subtypes-only',
        action='store_true',
        help='run three-subtype-only network figures',
    )
    args = ap.parse_args()

    figdir = os.path.join(args.outdir, 'figures_net'); os.makedirs(figdir, exist_ok=True)
    statdir = os.path.join(args.outdir, 'net_stats'); os.makedirs(statdir, exist_ok=True)

    f_asd = os.path.join(args.outdir, 'fcs_asd_gsr.npy' if args.use_gsr else 'fcs_asd.npy')
    f_td  = os.path.join(args.outdir, 'fcs_td_gsr.npy'  if args.use_gsr else 'fcs_td.npy')

    # Auto-compute FC matrices from raw data if they don't exist
    if not os.path.exists(f_asd) or not os.path.exists(f_td):
        if args.abide_pklz is None:
            raise SystemExit(
                f"FC files not found in {args.outdir}.\n"
                "Supply --abide-pklz to compute them, or run utils/step7_brain_metrics_abide.py first."
            )
        print("FC files not found – computing from raw data...")
        import importlib.util, sys as _sys
        _cur = os.path.dirname(os.path.abspath(__file__))
        _bm_path = os.path.join(_cur, 'utils', 'step7_brain_metrics_abide.py')
        _spec = importlib.util.spec_from_file_location('_bm', _bm_path)
        _bm = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_bm)
        _bm_args = type('Args', (), {
            'abide_pklz': args.abide_pklz, 'cmi_pklz': None,
            'bna_atlas': args.bna_atlas, 'step2_outdir': args.step2_outdir,
            'outdir': args.outdir, 'use_gsr': args.use_gsr,
            'tr': 2.0, 'bandpass': True, 'low': 0.01, 'high': 0.08,
        })()
        _bm.compute_and_save_fc(_bm_args)

    fcs_asd = np.load(f_asd); fcs_td = np.load(f_td)
    labels_asd = load_step2_labels(args.step2_outdir)

    C = fcs_asd.shape[1]
    net_labels = load_network_map_csv(args.net_map, C)

    X_asd, iu = extract_network_fc_vector(fcs_asd, net_labels)
    X_td, _ = extract_network_fc_vector(fcs_td, net_labels)

    if args.asd_subtypes_only:
        from utils.step7_network_fc_stats_three_subtypes_lib import run_figures_net_three_subtypes
        run_figures_net_three_subtypes(
            args, labels_asd, X_asd, X_td, fcs_asd, fcs_td, net_labels, iu
        )
        return

    try:
        abide_df, _ = load_and_preprocess_data('DATA/combined_ABIDE_information_with_fMRI.pklz', 'CMI-DATA/combined_asd_td_rest_run1_data.pklz')
        abide_asd_df = abide_df[abide_df['DX_GROUP'] == 0].copy().reset_index(drop=True)
        abide_td_df  = abide_df[abide_df['DX_GROUP'] == 1].copy().reset_index(drop=True)
        age_asd  = abide_asd_df['AGE_AT_SCAN'].astype(float).to_numpy()
        sex_asd  = abide_asd_df['SEX'].astype(str).to_numpy()
        site_asd = abide_asd_df['SITE_KEY_STRICT'].astype(str).to_numpy()
        age_td   = abide_td_df['AGE_AT_SCAN'].astype(float).to_numpy()
        sex_td   = abide_td_df['SEX'].astype(str).to_numpy()
        site_td  = abide_td_df['SITE_KEY_STRICT'].astype(str).to_numpy()
        age_asd, sex_asd, site_asd = age_asd[:len(X_asd)], sex_asd[:len(X_asd)], site_asd[:len(X_asd)]
        age_td,  sex_td,  site_td  = age_td[:len(X_td)],  sex_td[:len(X_td)],  site_td[:len(X_td)]
        use_cov = True
    except Exception as e:
        print('Covariate load failed, falling back to unadjusted stats:', e)
        use_cov = False

    coef_map = None; pval_map = None
    if use_cov:
        df_stats, coef_map, pval_map = stats_by_edge_with_covariates(
            X_asd, X_td, labels_asd,
            age_asd, sex_asd, site_asd,
            age_td, sex_td, site_td
        )
    else:
        df_stats = stats_by_edge(X_asd, X_td, labels_asd, method=args.method)
    df_stats.to_csv(os.path.join(statdir, f'net_fc_stats_{"gsr" if args.use_gsr else "nogsr"}_{args.method}.csv'), index=False)

    try:
        q_vals = df_stats['q'].to_numpy() if 'q' in df_stats.columns else None
    except Exception:
        q_vals = None
    plot_f_matrix(df_stats['stat'].to_numpy(), iu,
                  title=f'Network FC group diff ({args.method}, {"GSR" if args.use_gsr else "no GSR"})',
                  out_png=os.path.join(figdir, f'net_fc_stats_{"gsr" if args.use_gsr else "nogsr"}_{args.method}.png'),
                  q_vals=q_vals)

    print('\n=== Plotting TD mean network FC matrix ===')
    td_mean_net_fc_vec = np.nanmean(X_td, axis=0)
    td_mean_net_fc_mat = build_edge_matrix(td_mean_net_fc_vec, iu)
    vlim = 1.0
    plt.figure(figsize=(5.6, 4.6))
    im = plt.imshow(td_mean_net_fc_mat, cmap='coolwarm', vmin=-vlim, vmax=vlim)
    cbar = plt.colorbar(im, shrink=0.8); cbar.set_label('Mean FC (z-space)', fontsize=12, fontweight='bold')
    plt.title(f'Network FC (TD, {"GSR" if args.use_gsr else "no GSR"})', fontsize=14, fontweight='bold')
    plt.xlabel('Network', fontsize=14, fontweight='bold'); plt.ylabel('Network', fontsize=14, fontweight='bold')
    ax = plt.gca(); _apply_yeo7_axis(ax, td_mean_net_fc_mat.shape[0])
    _td_mean_png = os.path.join(figdir, f'net_fc_td_mean_{"gsr" if args.use_gsr else "nogsr"}.png')
    plt.tight_layout(); plt.savefig(_td_mean_png, dpi=200); plt.close()
    print(f'Saved TD mean network FC matrix to: {_td_mean_png}')

    sig_idx = np.where(df_stats['sig_fdr'].to_numpy())[0]
    print(f"Significant edges (FDR): {len(sig_idx)}/{len(df_stats)}")

    if args.umap:
        try:
            from umap import UMAP
            td_lab = labels_asd.max() + 1
            labels_all = np.concatenate([labels_asd, np.full(len(X_td), td_lab)])
            X = np.concatenate([X_asd, X_td], axis=0)
            emb = UMAP(n_neighbors=15, min_dist=0.1, metric='correlation', random_state=42).fit_transform(X)
            plt.figure(figsize=(5.2,4.4))
            sc = plt.scatter(emb[:,0], emb[:,1], c=labels_all, cmap='tab10', s=12)
            plt.title('UMAP of network-FC (28D)')
            plt.xlabel('UMAP-1', fontsize=14, fontweight='bold'); plt.ylabel('UMAP-2', fontsize=14, fontweight='bold')
            plt.tight_layout(); plt.savefig(os.path.join(figdir, f'net_fc_umap_{"gsr" if args.use_gsr else "nogsr"}.png'), dpi=200); plt.close()
        except Exception as e:
            print('UMAP skipped:', e)

    print('Network-FC stats completed. Outputs ->', statdir)
    
    print('\n=== Computing ROI-level GBC based on TD-defined positive/negative edges ===')
    
    print(f"Original FC shapes: ASD {fcs_asd.shape}, TD {fcs_td.shape}")
    
    td_mean_fc = np.nanmean(fcs_td, axis=0)
    print(f"TD mean FC shape: {td_mean_fc.shape}")
    print(f"TD mean FC ndim: {td_mean_fc.ndim}")
    
    positive_mask, negative_mask = define_positive_negative_edges_by_td(td_mean_fc, net_labels)
    
    n_roi = td_mean_fc.shape[0]
    positive_mask_2d = np.zeros((n_roi, n_roi), dtype=bool)
    negative_mask_2d = np.zeros((n_roi, n_roi), dtype=bool)
    
    iu = np.triu_indices(n_roi, k=1)
    
    positive_mask_2d[iu] = positive_mask
    negative_mask_2d[iu] = negative_mask
    
    positive_mask_2d = positive_mask_2d + positive_mask_2d.T
    negative_mask_2d = negative_mask_2d + negative_mask_2d.T
    
    print(f"Positive mask 2D shape: {positive_mask_2d.shape}, sum: {positive_mask_2d.sum()}")
    print(f"Negative mask 2D shape: {negative_mask_2d.shape}, sum: {negative_mask_2d.sum()}")
    
    gbc_roi_asd = compute_roi_gbc_by_td_definition(fcs_asd, positive_mask_2d, negative_mask_2d)
    gbc_roi_td = compute_roi_gbc_by_td_definition(fcs_td, positive_mask_2d, negative_mask_2d)
    
    print(f"ROI GBC shapes: ASD {gbc_roi_asd.shape}, TD {gbc_roi_td.shape}")
    
    np.save(os.path.join(statdir, 'gbc_roi_asd_td_defined.npy'), gbc_roi_asd)
    np.save(os.path.join(statdir, 'gbc_roi_td_td_defined.npy'), gbc_roi_td)
    
    for gbc_type, gbc_idx, gbc_name in [(0, 0, 'All')]:
        print(f'\nGenerating ROI GBC {gbc_name} violin plots...')
        
        gbc_asd_type = gbc_roi_asd[:, :, gbc_idx].mean(axis=1)
        gbc_td_type = gbc_roi_td[:, :, gbc_idx].mean(axis=1)
        
        gbc_values = np.concatenate([gbc_asd_type, gbc_td_type])
        gbc_labels = np.concatenate([labels_asd, np.full(len(gbc_td_type), labels_asd.max() + 1)])
        
        from utils.step7_abide_alff_violin_only import perform_ttest_and_fdr
        gbc_stats_with = perform_ttest_and_fdr(gbc_values, gbc_labels, labels_asd.max() + 1)
        
        print(f'ROI GBC {gbc_name} statistical results (with outliers):')
        for subtype, stats in gbc_stats_with.items():
            print(f'  {subtype} vs TD: t={stats["t_statistic"]:.3f}, p={stats["p_value"]:.4f}, p_corrected={stats["p_corrected"]:.4f}')
        
        from utils.step7_abide_alff_violin_only import plot_violin_with_sig
        plot_violin_with_sig(
            gbc_values, 
            gbc_labels, 
            f'ROI GBC {gbc_name} (TD-defined edges) ASD subtypes vs TD (with outliers)', 
            os.path.join(figdir, f'roi_gbc_{gbc_name.lower()}_td_defined_violin_with_outliers.png'),
            perform_stats=True,
            stats_results=gbc_stats_with
        )
        
        from utils.step7_abide_alff_violin_only import remove_outliers_iqr
        gbc_values, gbc_labels, outliers = remove_outliers_iqr(gbc_values, gbc_labels, factor=2.0)
        
        gbc_stats = perform_ttest_and_fdr(gbc_values, gbc_labels, labels_asd.max() + 1)
        
        print(f'ROI GBC {gbc_name} statistical results (outliers removed):')
        for subtype, stats in gbc_stats.items():
            print(f'  {subtype} vs TD: t={stats["t_statistic"]:.3f}, p={stats["p_value"]:.4f}, p_corrected={stats["p_corrected"]:.4f}')
        
        plot_violin_with_sig(
            gbc_values, 
            gbc_labels, 
            f'ROI GBC {gbc_name} (TD-defined edges) ASD subtypes vs TD (outliers removed)', 
            os.path.join(figdir, f'roi_gbc_{gbc_name.lower()}_td_defined_violin.png'),
            perform_stats=True,
            stats_results=gbc_stats
        )
    
    print('\nROI-level GBC based on TD-defined edges completed!')

    # Effect-size + permutation for Subtype1-3 vs TD
    try:
        net_labels_local = net_labels
        td_lab = labels_asd.max() + 1
        figdir_local = figdir
        for k in [0, 1, 2]:
            sub_mask = (labels_asd == k)
            if sub_mask.sum() == 0:
                continue
            d, p = cellwise_effect_and_perm(fcs_asd[sub_mask], fcs_td, net_labels_local, n_perm=2000, seed=42)
            np.save(os.path.join(statdir, f'effect_cohend_subtype{k+1}_vs_td.npy'), d)
            np.save(os.path.join(statdir, f'perm_p_subtype{k+1}_vs_td.npy'), p)
            plot_effect_heatmap(d, f'Subtype{k+1} vs TD (Cohen d)', os.path.join(figdir_local, f'effect_subtype{k+1}_vs_td.png'))
            try:
                from statsmodels.stats.multitest import multipletests
                n = d.shape[0]
                iu = np.triu_indices(n, 1)
                rej, q, _, _ = multipletests(p[iu].ravel(), method='fdr_bh')
                sig_mask = np.zeros_like(d, dtype=bool)
                sig_mask[iu] = rej
                sig_mask[(iu[1], iu[0])] = rej
                q_mat = np.full_like(d, np.nan, dtype=float)
                q_mat[iu] = q
                q_mat[(iu[1], iu[0])] = q
            except Exception:
                sig_mask = None; q_mat = None
            plot_effect_heatmap(d, f'Hotspot: Subtype{k+1} vs TD (Cohen d)', os.path.join(figdir_local, f'net_fc_hotspot_subtype{k+1}_vs_td.png'), sig_mask=sig_mask, q_mat=q_mat)
    except Exception as e:
        print('Effect-size/permutation plotting skipped:', e)

    try:
        if coef_map is not None and pval_map is not None:
            from statsmodels.stats.multitest import multipletests
            for k_name in ['Subtype 1', 'Subtype 2', 'Subtype 3']:
                vec = coef_map.get(k_name, None)
                pv  = pval_map.get(k_name, None)
                if vec is None or pv is None:
                    continue
                # FDR on per-edge p-values
                try:
                    rej, q, _, _ = multipletests(pv, method='fdr_bh')
                except Exception:
                    rej = np.zeros_like(pv, dtype=bool)
                plot_group_vs_td_heatmap(vec, iu,
                    title=f'{k_name} vs TD (covariate-adjusted)',
                    out_png=os.path.join(figdir, f'net_fc_cov_adj_{k_name.replace(" ", "_").lower()}_vs_td.png'),
                    sig_vec=rej.astype(int))
    except Exception as e:
        print('Covariate-adjusted heatmaps skipped:', e)

    try:
        plot_edge_grouped_bars(
            X_asd, X_td, labels_asd, iu,
            out_png=os.path.join(figdir, f'net_fc_grouped_edges_{"gsr" if args.use_gsr else "nogsr"}_{args.method}.png'),
            title=f'Network FC (mean r) by edge and group ({"GSR" if args.use_gsr else "no GSR"})'
        )
    except Exception as e:
        print('Grouped bars plotting skipped:', e)


if __name__ == '__main__':
    main()


