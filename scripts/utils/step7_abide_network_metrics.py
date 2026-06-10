#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

python script0925/step7_network_metrics.py \
  --outdir unsup_results/step7_abide_fixed \
  --step2-outdir results_step2_tuned_retry \
  --net-map mapping/bna246_to_yeo7.csv
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import sys as _sys, os as _os
_cur = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_cur)
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import importlib.util as _ilu, os as _os2
_bm_path = _os2.path.join(_os2.path.dirname(_os2.path.abspath(__file__)), 'step7_brain_metrics_abide.py')
_bm_spec = _ilu.spec_from_file_location('step7_brain_metrics_abide', _bm_path)
_bm_mod = _ilu.module_from_spec(_bm_spec)
_bm_spec.loader.exec_module(_bm_mod)
load_step2_labels = _bm_mod.load_step2_labels
preprocess_ts = _bm_mod.preprocess_ts
build_confounds_gsr = _bm_mod.build_confounds_gsr

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

def pretty_subtype_label(x: int, td_label: int) -> str:
    return 'TD' if x == td_label else f'Subtype {x+1}'

def load_network_map_csv(path: str, num_rois: int) -> np.ndarray:
    """Load ROI-to-network mapping from CSV (Label, Yeo_7network columns).
    Returns int array of length num_rois; -1 for unmapped ROIs.
    """
    df = pd.read_csv(path, header=1)
    if 'Label' not in df.columns or 'Yeo_7network' not in df.columns:
        raise ValueError(f"CSV : Label / Yeo_7network ({path})")
    df = df[['Label', 'Yeo_7network']].copy()
    df = df.dropna()
    df['Label'] = df['Label'].astype(int)
    df = df.sort_values('Label')
    yeo = pd.to_numeric(df['Yeo_7network'], errors='coerce').fillna(0).astype(int).to_numpy()
    if len(yeo) != num_rois:
        raise ValueError(f"{len(yeo)}ROI{num_rois}")
    mapped = np.where(yeo > 0, yeo - 1, -1)
    return mapped


def aggregate_subject_metric_by_network(values: np.ndarray, net_labels: np.ndarray) -> np.ndarray:
    """Aggregate per-ROI values to per-network means.

    values: [N_subjects, N_roi]
    """
    n_net = int(np.max(net_labels)) + 1
    out = np.zeros((values.shape[0], n_net), dtype=np.float32)
    for k in range(n_net):
        idx = np.where(net_labels == k)[0]
        if len(idx) == 0:
            out[:, k] = np.nan
        else:
            out[:, k] = np.nanmean(values[:, idx], axis=1)
    return out


def aggregate_fc_by_network_z(fcs_z: np.ndarray, net_labels: np.ndarray) -> np.ndarray:
    """Aggregate ROI-level Fisher-z FC matrices to network-level.

    fcs_z: [N_subjects, C, C] Fisher-z FC matrices
    Returns: [N_subjects, n_net, n_net]
    """
    n_net = int(np.max(net_labels)) + 1
    nets_z = np.zeros((fcs_z.shape[0], n_net, n_net), dtype=np.float32)
    for s in range(fcs_z.shape[0]):
        z = fcs_z[s]
        for a in range(n_net):
            ia = np.where(net_labels == a)[0]
            for b in range(n_net):
                ib = np.where(net_labels == b)[0]
                sub = z[np.ix_(ia, ib)]
                if a == b:
                    vals = sub[np.triu_indices_from(sub, 1)]
                else:
                    vals = sub.ravel()
                nets_z[s, a, b] = np.nanmean(vals) if vals.size else np.nan
    group_mean_z = np.nanmean(nets_z, axis=0)
    group_mean_r = np.tanh(group_mean_z)
    return group_mean_r


def violin_with_sig(values: np.ndarray, labels: np.ndarray, title: str, out_path: str):
    from itertools import combinations
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    td_label = int(np.nanmax(labels))
    group_display = [pretty_subtype_label(int(g), td_label) for g in labels]
    desired_order = ['Subtype 1', 'Subtype 2', 'Subtype 3', 'TD']
    present = [g for g in desired_order if g in set(group_display)]
    df = pd.DataFrame({'value': values, 'group': group_display})
    plt.figure(figsize=(6.8, 4.2))
    ax = sns.violinplot(x='group', y='value', data=df, inner='box', palette='Set2', order=present)
    ax.set_title(title)
    ax.set_xlabel('Group', fontsize=14, fontweight='bold')
    ax.set_ylabel('Value', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()


def plot_network_fc_heatmap(fc_net_r: np.ndarray, title: str, out_png: str):
    plt.figure(figsize=(5.6, 4.6))
    im = plt.imshow(fc_net_r, cmap='RdBu_r', vmin=-1.0, vmax=1.0)
    cbar = plt.colorbar(im, shrink=0.8); cbar.set_label('r', fontsize=12, fontweight='bold')
    plt.title(title)
    plt.xlabel('Yeo7 network', fontsize=14, fontweight='bold');
    plt.ylabel('Yeo7 network', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def network_fc_from_ts(subject_ts: np.ndarray, net_labels: np.ndarray) -> np.ndarray:
    nets = np.unique(net_labels[net_labels >= 0])
    net_ts = []
    for k in nets:
        idx = np.where(net_labels == k)[0]
        if len(idx) == 0:
            net_ts.append(np.zeros((subject_ts.shape[0],), dtype=float))
        else:
            net_ts.append(subject_ts[:, idx].mean(axis=1))
    net_ts = np.column_stack(net_ts)  # [T, 7]
    r = np.corrcoef(net_ts, rowvar=False)
    np.fill_diagonal(r, 1.0)
    return r


def group_network_fc_from_ts(list_of_subject_ts, net_labels) -> np.ndarray:
    nets = np.unique(net_labels[net_labels >= 0])
    z_mats = []
    for ts in list_of_subject_ts:
        r = network_fc_from_ts(ts, net_labels)
        z = np.arctanh(np.clip(r, -0.999999, 0.999999))
        z_mats.append(z)
    z_mean = np.nanmean(z_mats, axis=0)
    r_mean = np.tanh(z_mean)
    np.fill_diagonal(r_mean, 1.0)
    return r_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=str, required=True, help='step7  fcs_*.npy ')
    ap.add_argument('--step2-outdir', type=str, required=True)
    ap.add_argument('--net-map', type=str, default='atlas/subregion_func_network_Yeo_updated.csv',
                    help='CSV mapping BNA246 to Yeo-7 (Label, Yeo_7network)')
    ap.add_argument('--fc-network-ts', action='store_true',
                    help='Use network-mean time series to compute 7x7 FC (recommended)')
    ap.add_argument('--abide-pklz', type=str, default='DATA/combined_ABIDE_information_with_fMRI.pklz')
    ap.add_argument('--tr', type=float, default=2.0)
    args = ap.parse_args()

    figdir = os.path.join(args.outdir, 'figures_net'); os.makedirs(figdir, exist_ok=True)

    fcs_asd = np.load(os.path.join(args.outdir, 'fcs_asd.npy'))      # [N_asd, C, C]
    fcs_td  = np.load(os.path.join(args.outdir, 'fcs_td.npy'))       # [N_td,  C, C]
    alffs_asd = np.load(os.path.join(args.outdir, 'alffs_asd.npy'))  # [N_asd, C]
    alffs_td  = np.load(os.path.join(args.outdir, 'alffs_td.npy'))   # [N_td,  C]
    falffs_asd = np.load(os.path.join(args.outdir, 'falffs_asd.npy'))
    falffs_td  = np.load(os.path.join(args.outdir, 'falffs_td.npy'))
    grad_by_group = {}
    for k in [0, 1, 2]:
        p = os.path.join(args.outdir, f'gradients_subtype_{k}.npy')
        if os.path.exists(p):
            grad_by_group[f'Subtype{k}'] = np.load(p)  # [C,2]
    if os.path.exists(os.path.join(args.outdir, 'gradients_td.npy')):
        grad_by_group['TD'] = np.load(os.path.join(args.outdir, 'gradients_td.npy'))

    C = alffs_asd.shape[1]
    net_labels = load_network_map_csv(args.net_map, C)
    n_net = int(np.max(net_labels)) + 1

    labels = load_step2_labels(args.step2_outdir)
    td_label = labels.max() + 1


    alff_net_asd = aggregate_subject_metric_by_network(alffs_asd, net_labels)
    alff_net_td  = aggregate_subject_metric_by_network(alffs_td,  net_labels)
    falff_net_asd = aggregate_subject_metric_by_network(falffs_asd, net_labels)
    falff_net_td  = aggregate_subject_metric_by_network(falffs_td,  net_labels)
    for net_k in range(n_net):
        val = np.concatenate([alff_net_asd[:, net_k], alff_net_td[:, net_k]])
        lab = np.concatenate([labels, np.full(alff_net_td.shape[0], td_label)])
        violin_with_sig(val, lab, f'ALFF (Yeo7-{net_k})', os.path.join(figdir, f'violin_alff_yeo{net_k}.png'))
        val = np.concatenate([falff_net_asd[:, net_k], falff_net_td[:, net_k]])
        lab = np.concatenate([labels, np.full(falff_net_td.shape[0], td_label)])
        violin_with_sig(val, lab, f'fALFF (Yeo7-{net_k})', os.path.join(figdir, f'violin_falff_yeo{net_k}.png'))

    if args.fc_network_ts:
        from utils.step7_brain_metrics_abide import load_and_preprocess_data
        abide_df, _ = load_and_preprocess_data(args.abide_pklz, 'CMI-DATA/combined_asd_td_rest_run1_data.pklz')
        abide_asd = abide_df[abide_df['DX_GROUP'] == 0].copy().reset_index(drop=True)
        abide_td  = abide_df[abide_df['DX_GROUP'] == 1].copy().reset_index(drop=True)

        for k in range(labels.max() + 1):
            mask = (labels == k)
            if mask.sum() == 0:
                continue
            ts_list = []
            for _, row in abide_asd[mask].iterrows():
                ts = preprocess_ts(np.asarray(row['data'], dtype=np.float32), tr=args.tr, confounds=None)
                ts_list.append(ts)
            fc_net_r = group_network_fc_from_ts(ts_list, net_labels)
            plot_network_fc_heatmap(fc_net_r, f'Network FC (Subtype {k+1})', os.path.join(figdir, f'net_fc_subtype{k+1}.png'))
        ts_list_td = []
        for _, row in abide_td.iterrows():
            ts = preprocess_ts(np.asarray(row['data'], dtype=np.float32), tr=args.tr, confounds=None)
            ts_list_td.append(ts)
        fc_net_r_td = group_network_fc_from_ts(ts_list_td, net_labels)
        plot_network_fc_heatmap(fc_net_r_td, 'Network FC (TD)', os.path.join(figdir, 'net_fc_td.png'))

        # WITH GSR
        for k in range(labels.max() + 1):
            mask = (labels == k)
            if mask.sum() == 0:
                continue
            ts_list = []
            for _, row in abide_asd[mask].iterrows():
                raw = np.asarray(row['data'], dtype=np.float32)
                conf = build_confounds_gsr(raw)
                ts = preprocess_ts(raw, tr=args.tr, confounds=conf)
                ts_list.append(ts)
            fc_net_r = group_network_fc_from_ts(ts_list, net_labels)
            plot_network_fc_heatmap(fc_net_r, f'Network FC (Subtype {k+1}, GSR)', os.path.join(figdir, f'net_fc_subtype{k+1}_gsr.png'))
        ts_list_td = []
        for _, row in abide_td.iterrows():
            raw = np.asarray(row['data'], dtype=np.float32)
            conf = build_confounds_gsr(raw)
            ts = preprocess_ts(raw, tr=args.tr, confounds=conf)
            ts_list_td.append(ts)
        fc_net_r_td = group_network_fc_from_ts(ts_list_td, net_labels)
        plot_network_fc_heatmap(fc_net_r_td, 'Network FC (TD, GSR)', os.path.join(figdir, 'net_fc_td_gsr.png'))
    else:
        for k in range(labels.max() + 1):
            mask = (labels == k)
            if mask.sum() == 0:
                continue
            fc_net_r = aggregate_fc_by_network_z(fcs_asd[mask], net_labels)
            plot_network_fc_heatmap(fc_net_r, f'Network FC (Subtype {k+1})', os.path.join(figdir, f'net_fc_subtype{k+1}.png'))
        fc_net_r_td = aggregate_fc_by_network_z(fcs_td, net_labels)
        plot_network_fc_heatmap(fc_net_r_td, 'Network FC (TD)', os.path.join(figdir, 'net_fc_td.png'))
        f_asd_g = os.path.join(args.outdir, 'fcs_asd_gsr.npy')
        f_td_g  = os.path.join(args.outdir, 'fcs_td_gsr.npy')
        if os.path.exists(f_asd_g) and os.path.exists(f_td_g):
            fcs_asd_g = np.load(f_asd_g)
            fcs_td_g  = np.load(f_td_g)
            for k in range(labels.max() + 1):
                mask = (labels == k)
                if mask.sum() == 0:
                    continue
                fc_net_r_g = aggregate_fc_by_network_z(fcs_asd_g[mask], net_labels)
                plot_network_fc_heatmap(fc_net_r_g, f'Network FC (Subtype {k+1}, GSR)', os.path.join(figdir, f'net_fc_subtype{k+1}_gsr.png'))
            fc_net_r_td_g = aggregate_fc_by_network_z(fcs_td_g, net_labels)
            plot_network_fc_heatmap(fc_net_r_td_g, 'Network FC (TD, GSR)', os.path.join(figdir, 'net_fc_td_gsr.png'))

    for gname, grads in grad_by_group.items():
        grad1 = grads[:, 0]
        grad2 = grads[:, 1]
        net1 = np.zeros(n_net, dtype=np.float32)
        net2 = np.zeros(n_net, dtype=np.float32)
        for k in range(n_net):
            idx = np.where(net_labels == k)[0]
            net1[k] = float(np.nanmean(grad1[idx])) if len(idx) else np.nan
            net2[k] = float(np.nanmean(grad2[idx])) if len(idx) else np.nan
        display_name = gname
        if gname.startswith('Subtype') and gname[-1].isdigit():
            try:
                idx = int(gname[-1])
                display_name = f'Subtype {idx+1}'
            except Exception:
                display_name = gname
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
        ax[0].bar(np.arange(n_net), net1, color=sns.color_palette('Set2')[:n_net])
        ax[0].set_title(f'{display_name} - Gradient 1 (network mean)')
        ax[0].set_xlabel('Yeo7', fontsize=14, fontweight='bold'); ax[0].set_ylabel('Grad1', fontsize=14, fontweight='bold')
        ax[1].bar(np.arange(n_net), net2, color=sns.color_palette('Set2')[:n_net])
        ax[1].set_title(f'{display_name} - Gradient 2 (network mean)')
        ax[1].set_xlabel('Yeo7', fontsize=14, fontweight='bold'); ax[1].set_ylabel('Grad2', fontsize=14, fontweight='bold')
        plt.tight_layout(); plt.savefig(os.path.join(figdir, f'grad_net_{display_name.replace(" ", "_")}.png'), dpi=200); plt.close()

    pd.DataFrame(alff_net_asd, columns=[f'Yeo7_{i}' for i in range(n_net)]).assign(subtype=labels).to_csv(
        os.path.join(figdir, 'alff_net_subjects_asd.csv'), index=False)
    pd.DataFrame(alff_net_td, columns=[f'Yeo7_{i}' for i in range(n_net)]).assign(group='TD').to_csv(
        os.path.join(figdir, 'alff_net_subjects_td.csv'), index=False)
    pd.DataFrame(falff_net_asd, columns=[f'Yeo7_{i}' for i in range(n_net)]).assign(subtype=labels).to_csv(
        os.path.join(figdir, 'falff_net_subjects_asd.csv'), index=False)
    pd.DataFrame(falff_net_td, columns=[f'Yeo7_{i}' for i in range(n_net)]).assign(group='TD').to_csv(
        os.path.join(figdir, 'falff_net_subjects_td.csv'), index=False)

    print('Step7 network-level metrics done. Figures ->', os.path.abspath(figdir))


if __name__ == '__main__':
    main()


