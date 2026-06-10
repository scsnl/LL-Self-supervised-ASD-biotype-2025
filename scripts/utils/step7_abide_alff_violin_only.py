#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

python script0925/step7_alff_violin_only.py \
  --outdir unsup_results/step7_abide_fixed \
  --step2-outdir results_step2_tuned_retry \
  --net-map atlas/subregion_func_network_Yeo_updated.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_cur = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_cur)
if _root not in _sys.path:
    _sys.path.insert(0, _root)
if _cur not in _sys.path:
    _sys.path.insert(0, _cur)

from step7_violin_three_subtypes_lib import (
    figdir_three_subtypes,
    load_alff_asd_masked_for_three_subtypes,
    plot_violin_three_subtypes,
    kruskal_three,
)


def load_labels(step2_outdir: str) -> np.ndarray:
    p = os.path.join(step2_outdir, 'abide_asd_labels.npy')
    if not os.path.exists(p):
        raise SystemExit(f'Missing labels: {p}')
    return np.load(p).astype(int)


def pretty_subtype_label(x: int, td_label: int) -> str:
    return 'TD' if x == td_label else f'Subtype {x+1}'


def load_network_map_csv(path: str, num_rois: int) -> np.ndarray:
    """
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
    """Aggregate ROI-level values to network-level means.

    values: [N_subjects, N_roi]
    """
    valid_net_labels = net_labels[net_labels >= 0]
    if len(valid_net_labels) == 0:
        raise ValueError("No valid network labels found")
    
    n_net = int(np.max(valid_net_labels)) + 1
    out = np.zeros((values.shape[0], n_net), dtype=np.float32)
    
    for k in range(n_net):
        idx = np.where(net_labels == k)[0]
        if len(idx) == 0:
            out[:, k] = np.nan
        else:
            out[:, k] = np.nanmean(values[:, idx], axis=1)
    
    return out


def remove_outliers_iqr(values: np.ndarray, labels: np.ndarray, factor: float = 1.5) -> tuple:
    """Remove IQR outliers per group.

    Args:
        values: 1-D metric array
        labels: integer group labels (one per subject)
        factor: IQR multiplier (default 1.5)

    Returns:
        (filtered_values, filtered_labels, outlier_mask)
    """
    outlier_mask = np.zeros(len(values), dtype=bool)
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = (labels == label)
        group_values = values[mask]
        
        if len(group_values) > 4:
            Q1 = np.percentile(group_values, 25)
            Q3 = np.percentile(group_values, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            group_outliers = (group_values < lower_bound) | (group_values > upper_bound)
            outlier_mask[mask] = group_outliers
    
    filtered_values = values[~outlier_mask]
    filtered_labels = labels[~outlier_mask]
    
    print(f"Removed {outlier_mask.sum()} outliers ({outlier_mask.sum()/len(values)*100:.1f}%)")
    
    return filtered_values, filtered_labels, outlier_mask


def get_yeo7_network_names() -> list:
    return ['Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention', 
            'Limbic', 'Frontoparietal', 'Default Mode']


def perform_ttest_and_fdr(values: np.ndarray, labels: np.ndarray, td_label: int) -> dict:
    """Two-sample t-test between each subtype and TDC, with BH-FDR correction.

    Args:
        values: 1-D float array of metric values (one per subject)
        labels: integer group labels (td_label marks TDC)
        td_label: integer label used for the TDC group

    Returns:
        dict with keys: pvalues, qvalues, significant
    """
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests
    
    td_values = values[labels == td_label]
    td_values = td_values[~np.isnan(td_values)]
    
    results = {}
    p_values = []
    test_names = []
    
    unique_labels = np.unique(labels[labels != td_label])
    for subtype_label in unique_labels:
        subtype_values = values[labels == subtype_label]
        subtype_values = subtype_values[~np.isnan(subtype_values)]
        
        if len(subtype_values) > 0 and len(td_values) > 0:
            t_stat, p_val = ttest_ind(subtype_values, td_values)
            
            subtype_name = pretty_subtype_label(subtype_label, td_label)
            results[subtype_name] = {
                'subtype_values': subtype_values,
                'td_values': td_values,
                't_statistic': t_stat,
                'p_value': p_val,
                'mean_subtype': np.mean(subtype_values),
                'mean_td': np.mean(td_values),
                'std_subtype': np.std(subtype_values),
                'std_td': np.std(td_values),
                'n_subtype': len(subtype_values),
                'n_td': len(td_values)
            }
            p_values.append(p_val)
            test_names.append(subtype_name)
    
    if len(p_values) > 0:
        _, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
        
        for i, test_name in enumerate(test_names):
            results[test_name]['p_corrected'] = p_corrected[i]
            results[test_name]['significant_uncorrected'] = p_values[i] < 0.05
            results[test_name]['significant_corrected'] = p_corrected[i] < 0.05
    
    return results


def get_significance_stars(p_uncorrected: float, p_corrected: float) -> str:
    if p_corrected < 0.001:
        return '***'
    elif p_corrected < 0.01:
        return '**'
    elif p_corrected < 0.05:
        return '*'
    elif p_uncorrected < 0.05:
        return '†'
    else:
        return 'ns'


def plot_violin_with_sig(values: np.ndarray, labels: np.ndarray, title: str, out_path: str, 
                        perform_stats: bool = True, stats_results: dict = None):
    import seaborn as sns
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    td_label = int(np.nanmax(labels))
    group_display = [pretty_subtype_label(int(g), td_label) for g in labels]
    df = pd.DataFrame({'value': values, 'group': group_display})
    
    plt.figure(figsize=(7, 5))
    desired_order = ['Subtype 1', 'Subtype 2', 'Subtype 3', 'TD']
    present = [g for g in desired_order if g in set(group_display)]
    ax = sns.violinplot(x='group', y='value', data=df, palette='Set2', order=present)
    
    ax.set_title(title)
    ax.set_xlabel('Group', fontsize=14, fontweight='bold')
    ax.set_ylabel('Value', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    
    if perform_stats and stats_results:
        y_max = df['value'].max()
        y_min = df['value'].min()
        y_range = y_max - y_min
        base_y = y_max + 0.12 * y_range
        
        x_ticks = ax.get_xticks()
        group_positions = {group: x_ticks[i] for i, group in enumerate(present)}
        td_pos = group_positions.get('TD', -1)
        
        if td_pos >= 0:
            comparisons = []
            for subtype_name, stats in stats_results.items():
                if stats.get('significant_corrected', False) or stats.get('significant_uncorrected', False):
                    if subtype_name in group_positions:
                        stars = get_significance_stars(stats['p_value'], stats['p_corrected'])
                        if stars != 'ns':
                            comparisons.append((subtype_name, stats, stars))
            
            comparisons.sort(key=lambda x: present.index(x[0]) if x[0] in present else 999)
            
            for i, (subtype_name, stats, stars) in enumerate(comparisons):
                subtype_pos = group_positions[subtype_name]
                
                y_offset = i * 0.05 * y_range
                y_current = base_y + y_offset
                
                x_start = min(subtype_pos, td_pos)
                x_end = max(subtype_pos, td_pos)
                x_center = (x_start + x_end) / 2
                
                ax.plot([x_start, x_end], [y_current, y_current], 'k-', linewidth=1.5)
                ax.plot([subtype_pos, subtype_pos], [y_current - 0.03 * y_range, y_current], 'k-', linewidth=1.5)
                ax.plot([td_pos, td_pos], [y_current - 0.03 * y_range, y_current], 'k-', linewidth=1.5)
                
                ax.text(x_center, y_current, stars, 
                       ha='center', va='center', fontsize=12, fontweight='bold', color='black')
        
        legend_text = "Significance: *** p<0.001, ** p<0.01, * p<0.05 (FDR corrected), † p<0.05 (uncorrected)"
        plt.figtext(0.5, 0.02, legend_text, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.85)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _pairwise_stats_to_rows(
    metric_name: str,
    variant: str,
    stats_pairs: dict,
    p_kw: float,
) -> list[dict]:
    rows = []
    for _k, st in stats_pairs.items():
        rows.append({
            "metric": metric_name,
            "variant": variant,
            "comparison": f"{st['group_a']} vs {st['group_b']}",
            "U_statistic": st["U_statistic"],
            "p_value": st["p_value"],
            "p_corrected_fdr_bh": st["p_corrected"],
            "significant_uncorrected": st["significant_uncorrected"],
            "significant_corrected": st["significant_corrected"],
            "kruskal_wallis_p": p_kw,
        })
    return rows


def main_alff_three_subtypes_only(args) -> None:
    figdir_base = os.path.join(args.outdir, "figures")
    os.makedirs(figdir_base, exist_ok=True)
    figdir = figdir_three_subtypes(figdir_base)

    labels = load_labels(args.step2_outdir)
    m_asd = np.isin(labels, [0, 1, 2])
    labels_3 = labels[m_asd].astype(int)
    if len(labels_3) == 0:
        raise SystemExit(" (0/1/2)")

    print("Loading ALFF data (three subtypes only, no TD)...")
    alffs_asd = load_alff_asd_masked_for_three_subtypes(args.outdir, m_asd)

    print(f"ALFF ASD shape (filtered)={alffs_asd.shape}, n_labels={len(labels_3)}")

    alff_asd_means = np.nanmean(alffs_asd, axis=1)
    all_rows: list[dict] = []

    def run_whole(variant: str, vals: np.ndarray, labs: np.ndarray) -> None:
        H, p_kw = kruskal_three(vals.astype(float), labs.astype(int))
        stats_pairs = plot_violin_three_subtypes(
            vals.astype(float),
            labs.astype(int),
            f"ALFF (subject mean) three subtypes ({variant})",
            os.path.join(
                figdir,
                f"alff_violin_three_subtypes_whole_brain_{variant.replace(' ', '_')}.png",
            ),
            ylabel="ALFF (mean)",
        )
        all_rows.extend(_pairwise_stats_to_rows("Whole-brain ALFF", variant, stats_pairs, p_kw))
        for st in stats_pairs.values():
            print(
                f"  {st['group_a']} vs {st['group_b']}: p={st['p_value']:.4f}, "
                f"q={st['p_corrected']:.4f}"
            )
        print(f"  Kruskal-Wallis p={p_kw:.4g}")

    print("Whole-brain (with outliers)...")
    run_whole("with outliers", alff_asd_means, labels_3)
    v_clean, l_clean, _ = remove_outliers_iqr(alff_asd_means, labels_3, factor=2.0)
    print("Whole-brain (outliers removed)...")
    run_whole("outliers removed", v_clean, l_clean)

    C = alffs_asd.shape[1]
    net_labels = load_network_map_csv(args.net_map, C)
    valid_rois = net_labels >= 0
    valid_net_labels = net_labels[valid_rois]
    n_net = int(np.max(valid_net_labels)) + 1
    network_names = get_yeo7_network_names()
    alff_asd_valid = alffs_asd[:, valid_rois]
    alff_net_asd = aggregate_subject_metric_by_network(alff_asd_valid, valid_net_labels)

    for net_k in range(n_net):
        net_name = network_names[net_k] if net_k < len(network_names) else f"Network_{net_k}"
        val = alff_net_asd[:, net_k]
        lab = labels_3
        ok = ~np.isnan(val)
        val, lab = val[ok], lab[ok]
        if len(val) == 0:
            print(f"  Skip network {net_name}: no data")
            continue
        print(f"  Network {net_name}...")

        def run_net(variant: str, vv: np.ndarray, ll: np.ndarray) -> None:
            H, p_kw = kruskal_three(vv.astype(float), ll.astype(int))
            stats_pairs = plot_violin_three_subtypes(
                vv.astype(float),
                ll.astype(int),
                f"ALFF ({net_name}) three subtypes ({variant})",
                os.path.join(
                    figdir,
                    f"alff_violin_three_subtypes_{net_name.lower().replace(' ', '_')}_{variant.replace(' ', '_')}.png",
                ),
                ylabel="ALFF (network mean)",
            )
            all_rows.extend(
                _pairwise_stats_to_rows(f"ALFF ({net_name})", variant, stats_pairs, p_kw)
            )

        run_net("with outliers", val, lab)
        v2, l2, _ = remove_outliers_iqr(val, lab, factor=2.0)
        run_net("outliers removed", v2, l2)

    pd.DataFrame(all_rows).to_csv(
        os.path.join(figdir, "alff_three_subtypes_pairwise_stats.csv"), index=False
    )
    print(f" ALFF : {os.path.abspath(figdir)}")


def main():
    ap = argparse.ArgumentParser(description='Generate ALFF violin plots for ASD subtypes vs TD (whole-brain and network-level)')
    ap.add_argument('--outdir', type=str, required=True, help='Output directory containing ALFF data')
    ap.add_argument('--step2-outdir', type=str, required=True, help='Step2 output directory containing labels')
    ap.add_argument('--net-map', type=str, default='atlas/subregion_func_network_Yeo_updated.csv',
                    help='CSV mapping BNA246 to Yeo-7 (Label, Yeo_7network)')
    ap.add_argument(
        '--asd-subtypes-only',
        action='store_true',
        help=' TD figures/violin_three_subtypes_only/',
    )
    args = ap.parse_args()

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelsize': 14,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    labels = load_labels(args.step2_outdir)
    
    figdir = os.path.join(args.outdir, 'figures')
    os.makedirs(figdir, exist_ok=True)

    if args.asd_subtypes_only:
        main_alff_three_subtypes_only(args)
        return

    print('Loading ALFF data...')
    alff_combat_path = os.path.join(args.outdir, 'alffs_asd_combat.npy')
    alff_td_combat_path = os.path.join(args.outdir, 'alffs_td_combat.npy')
    
    if os.path.exists(alff_combat_path) and os.path.exists(alff_td_combat_path):
        print("ComBatALFF")
        alffs_asd = np.load(alff_combat_path)
        alffs_td = np.load(alff_td_combat_path)
    else:
        print("GSRALFF")
        alffs_asd = np.load(os.path.join(args.outdir, 'alffs_asd_gsr.npy'))
        alffs_td = np.load(os.path.join(args.outdir, 'alffs_td_gsr.npy'))
    
    print(f'ALFF data loaded: ASD shape={alffs_asd.shape}, TD shape={alffs_td.shape}')

    print('Generating whole-brain ALFF violin plot...')
    alff_asd_means = np.nanmean(alffs_asd, axis=1)  # [n_asd_subjects]
    alff_td_means = np.nanmean(alffs_td, axis=1)    # [n_td_subjects]
    
    print(f'Whole-brain ALFF means: ASD mean={np.nanmean(alff_asd_means):.4f}, TD mean={np.nanmean(alff_td_means):.4f}')

    td_label = labels.max() + 1
    subj_labels_combo = np.concatenate([labels, np.full(len(alff_td_means), fill_value=td_label)])
    subj_values_combo = np.concatenate([alff_asd_means, alff_td_means])
    
    print(f'Combined data: {len(subj_values_combo)} subjects, {len(np.unique(subj_labels_combo))} groups')

    print('Performing statistical tests for whole-brain ALFF (with outliers)...')
    stats_results_whole_with = perform_ttest_and_fdr(subj_values_combo, subj_labels_combo, td_label)
    
    print('Whole-brain ALFF statistical results (with outliers):')
    for subtype, stats in stats_results_whole_with.items():
        print(f'  {subtype} vs TD: t={stats["t_statistic"]:.3f}, p={stats["p_value"]:.4f}, p_corrected={stats["p_corrected"]:.4f}')
    
    plot_violin_with_sig(
        subj_values_combo, 
        subj_labels_combo, 
        'ALFF (subject mean) ASD subtypes vs TD (with outliers)', 
        os.path.join(figdir, 'alff_violin_asd_td_with_outliers.png'),
        perform_stats=True,
        stats_results=stats_results_whole_with
    )

    print('Removing outliers from whole-brain ALFF...')
    subj_values_combo_clean, subj_labels_combo_clean, outliers_whole = remove_outliers_iqr(subj_values_combo, subj_labels_combo, factor=2.0)

    print('Performing statistical tests for whole-brain ALFF (outliers removed)...')
    stats_results_whole_clean = perform_ttest_and_fdr(subj_values_combo_clean, subj_labels_combo_clean, td_label)
    
    print('Whole-brain ALFF statistical results (outliers removed):')
    for subtype, stats in stats_results_whole_clean.items():
        print(f'  {subtype} vs TD: t={stats["t_statistic"]:.3f}, p={stats["p_value"]:.4f}, p_corrected={stats["p_corrected"]:.4f}')
    
    plot_violin_with_sig(
        subj_values_combo_clean, 
        subj_labels_combo_clean, 
        'ALFF (subject mean) ASD subtypes vs TD (outliers removed)', 
        os.path.join(figdir, 'alff_violin_asd_td.png'),
        perform_stats=True,
        stats_results=stats_results_whole_clean
    )

    print(f'Whole-brain ALFF violin plot saved to: {os.path.abspath(os.path.join(figdir, "alff_violin_asd_td.png"))}')

    print('Generating network-level ALFF violin plots...')
    
    C = alffs_asd.shape[1]
    net_labels = load_network_map_csv(args.net_map, C)
    valid_rois = net_labels >= 0
    valid_net_labels = net_labels[valid_rois]
    n_net = int(np.max(valid_net_labels)) + 1
    network_names = get_yeo7_network_names()
    
    print(f'Network mapping loaded: {n_net} networks, {np.sum(valid_rois)} valid ROIs (excluding {np.sum(~valid_rois)} undefined)')
    
    alff_asd_valid = alffs_asd[:, valid_rois]
    alff_td_valid = alffs_td[:, valid_rois]
    alff_net_asd = aggregate_subject_metric_by_network(alff_asd_valid, valid_net_labels)
    alff_net_td = aggregate_subject_metric_by_network(alff_td_valid, valid_net_labels)
    
    print(f'Network-level ALFF calculated: ASD shape={alff_net_asd.shape}, TD shape={alff_net_td.shape}')
    
    all_network_stats = {}
    for net_k in range(n_net):
        net_name = network_names[net_k] if net_k < len(network_names) else f'Network_{net_k}'
        
        val = np.concatenate([alff_net_asd[:, net_k], alff_net_td[:, net_k]])
        lab = np.concatenate([labels, np.full(alff_net_td.shape[0], labels.max()+1)])
        
        valid_mask = ~np.isnan(val)
        val_clean = val[valid_mask]
        lab_clean = lab[valid_mask]
        
        if len(val_clean) > 0:
            print(f'  Generating violin plot for {net_name}: {len(val_clean)} valid subjects')
            
            stats_results_net_with = perform_ttest_and_fdr(val_clean, lab_clean, td_label)
            all_network_stats[f'{net_name}_with_outliers'] = stats_results_net_with
            
            print(f'  {net_name} statistical results (with outliers):')
            for subtype, stats in stats_results_net_with.items():
                print(f'    {subtype} vs TD: t={stats["t_statistic"]:.3f}, p={stats["p_value"]:.4f}, p_corrected={stats["p_corrected"]:.4f}')
            
            plot_violin_with_sig(
                val_clean, 
                lab_clean, 
                f'ALFF ({net_name}) ASD subtypes vs TD (with outliers)', 
                os.path.join(figdir, f'alff_violin_{net_name.lower().replace(" ", "_")}_asd_td_with_outliers.png'),
                perform_stats=True,
                stats_results=stats_results_net_with
            )
            
            val_clean_no_outliers, lab_clean_no_outliers, outliers_net = remove_outliers_iqr(val_clean, lab_clean, factor=2.0)
            
            stats_results_net_clean = perform_ttest_and_fdr(val_clean_no_outliers, lab_clean_no_outliers, td_label)
            all_network_stats[net_name] = stats_results_net_clean
            
            print(f'  {net_name} statistical results (outliers removed):')
            for subtype, stats in stats_results_net_clean.items():
                print(f'    {subtype} vs TD: t={stats["t_statistic"]:.3f}, p={stats["p_value"]:.4f}, p_corrected={stats["p_corrected"]:.4f}')
            
            plot_violin_with_sig(
                val_clean_no_outliers, 
                lab_clean_no_outliers, 
                f'ALFF ({net_name}) ASD subtypes vs TD (outliers removed)', 
                os.path.join(figdir, f'alff_violin_{net_name.lower().replace(" ", "_")}_asd_td.png'),
                perform_stats=True,
                stats_results=stats_results_net_clean
            )
        else:
            print(f'  Skipping {net_name}: no valid data')

    print('Saving network-level data to CSV...')
    network_cols = [network_names[i] if i < len(network_names) else f'Network_{i}' for i in range(n_net)]
    
    alff_net_asd_df = pd.DataFrame(alff_net_asd, columns=network_cols)
    alff_net_asd_df['subtype'] = labels
    alff_net_asd_df.to_csv(os.path.join(figdir, 'alff_net_subjects_asd.csv'), index=False)
    
    alff_net_td_df = pd.DataFrame(alff_net_td, columns=network_cols)
    alff_net_td_df['group'] = 'TD'
    alff_net_td_df.to_csv(os.path.join(figdir, 'alff_net_subjects_td.csv'), index=False)
    
    print(f'Network-level CSV files saved to: {os.path.abspath(figdir)}')
    
    print('Saving statistical results...')
    
    whole_brain_stats_df = []
    for subtype, stats in stats_results_whole_clean.items():
        whole_brain_stats_df.append({
            'metric': 'Whole-brain ALFF',
            'comparison': f'{subtype} vs TD',
            't_statistic': stats['t_statistic'],
            'p_value': stats['p_value'],
            'p_corrected': stats['p_corrected'],
            'significant_uncorrected': stats['significant_uncorrected'],
            'significant_corrected': stats['significant_corrected'],
            'mean_subtype': stats['mean_subtype'],
            'mean_td': stats['mean_td'],
            'std_subtype': stats['std_subtype'],
            'std_td': stats['std_td'],
            'n_subtype': stats['n_subtype'],
            'n_td': stats['n_td']
        })
    
    pd.DataFrame(whole_brain_stats_df).to_csv(
        os.path.join(figdir, 'alff_statistical_results_whole_brain.csv'), index=False)
    
    network_stats_df = []
    for net_name, net_stats in all_network_stats.items():
        for subtype, stats in net_stats.items():
            network_stats_df.append({
                'metric': f'ALFF ({net_name})',
                'network': net_name,
                'comparison': f'{subtype} vs TD',
                't_statistic': stats['t_statistic'],
                'p_value': stats['p_value'],
                'p_corrected': stats['p_corrected'],
                'significant_uncorrected': stats['significant_uncorrected'],
                'significant_corrected': stats['significant_corrected'],
                'mean_subtype': stats['mean_subtype'],
                'mean_td': stats['mean_td'],
                'std_subtype': stats['std_subtype'],
                'std_td': stats['std_td'],
                'n_subtype': stats['n_subtype'],
                'n_td': stats['n_td']
            })
    
    pd.DataFrame(network_stats_df).to_csv(
        os.path.join(figdir, 'alff_statistical_results_networks.csv'), index=False)
    
    print(f'Statistical results saved to: {os.path.abspath(figdir)}')
    print('All ALFF violin plots, data files, and statistical results generated successfully!')


if __name__ == '__main__':
    main()
