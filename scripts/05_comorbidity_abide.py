#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5a: ABIDE ADHD Comorbidity Distribution Across ASD Subtypes

Quantifies the prevalence of comorbid ADHD across the three ASD subtypes in
the ABIDE cohort and produces PCA embedding plots colored by ADHD status.

Usage:
  python 05_comorbidity_abide.py \\
    --abide-pklz  <DATA/combined_ABIDE_information_with_fMRI.pklz> \\
    --step2-outdir <path/to/step3_output> \\
    --outdir      <output_dir>
"""

import os
import pickle
import argparse
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def is_valid_array(x):
    x = np.array(x)
    return x.ndim == 2 and x.shape[0] >= 120 and np.isfinite(x).all()


def build_abide_asd_with_step1(abide_pklz: str, cmi_pklz: str = None) -> pd.DataFrame:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '02_train_vicreg.py')
    spec = importlib.util.spec_from_file_location('step1_module', base)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    cmi_arg = cmi_pklz if (cmi_pklz and __import__("os").path.exists(cmi_pklz)) else None
    abide_df, _ = mod.load_and_preprocess_data(abide_pklz, cmi_arg)
    asd_df = abide_df[abide_df['DX_GROUP'] == 0].copy().reset_index(drop=True)
    return asd_df


def extract_embeddings(asd_df: pd.DataFrame) -> np.ndarray:
    feats = []
    for i in range(len(asd_df)):
        x = np.array(asd_df.iloc[i]['data'])
        x = (x - x.mean()) / (x.std() + 1e-6)
        feats.append(x.mean(axis=0))
    return np.vstack(feats)


def fit_pca_2d(X: np.ndarray):
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X)


def has_adhd(comorb):
    if pd.isna(comorb):
        return 0
    return int('adhd' in str(comorb).lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--abide-pklz', type=str, default='DATA/combined_ABIDE_information_with_fMRI.pklz')
    ap.add_argument('--step2-outdir', type=str, required=True, help=' abide_asd_labels.npy ')
    ap.add_argument('--cmi-pklz', type=str, default=None, help="CMI pklz (optional)")
    ap.add_argument('--outdir', type=str, default='unsup_results/abide_adhd_plots')
    ap.add_argument('--emb-root', type=str, default=None, help='Step1  embeddings  abide_asd_emb.npy')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    asd_df = build_abide_asd_with_step1(args.abide_pklz, args.cmi_pklz)

    lab_path = os.path.join(args.step2_outdir, 'abide_asd_labels.npy')
    if not os.path.exists(lab_path):
        raise SystemExit(f': {lab_path}')
    labels = np.load(lab_path)
    if len(labels) != len(asd_df):
        raise SystemExit(f'({len(labels)}) ABIDE-ASD ({len(asd_df)})')
    asd_df = asd_df.copy()
    asd_df['subtype'] = labels.astype(int)

    asd_df['ADHD_ANY'] = asd_df['COMORBIDITY'].apply(has_adhd).astype(int)

    stats = []
    for k in sorted(asd_df['subtype'].unique()):
        sub = asd_df[asd_df['subtype'] == k]
        n = len(sub)
        adhd_n = int(sub['ADHD_ANY'].sum())
        stats.append({'subtype': int(k), 'n': n, 'ADHD_any': adhd_n, 'ADHD_rate': adhd_n / max(1, n)})
    stats_df = pd.DataFrame(stats).sort_values('subtype')
    stats_path = os.path.join(args.outdir, 'abide_adhd_counts.csv')
    stats_df.to_csv(stats_path, index=False)

    if args.emb_root:
        emb_path = os.path.join(args.emb_root, 'abide_asd_emb.npy')
        if not os.path.exists(emb_path):
            raise SystemExit(f' embeddings: {emb_path}')
        X = np.load(emb_path)
        if X.shape[0] != len(asd_df):
            raise SystemExit(f'Emb ({X.shape[0]}) ABIDE-ASD ({len(asd_df)})')
    else:
        X = extract_embeddings(asd_df)
    XY = fit_pca_2d(X)

    plt.rcParams.update({
        'font.size': 20,
        'font.weight': 'bold',
        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18
    })
    
    plt.figure(figsize=(12, 8))
    mask_adhd = asd_df['ADHD_ANY'] == 1
    plt.scatter(XY[~mask_adhd, 0], XY[~mask_adhd, 1], c='#bbbbbb', s=40, alpha=0.5, label='ASD (non-ADHD)')
    plt.scatter(XY[mask_adhd, 0], XY[mask_adhd, 1], c='#e41a1c', edgecolors='black', linewidths=0.4, s=50, alpha=0.7, label='ASD + ADHD')
    
    plt.title('ABIDE-ASD PCA (colored by ADHD comorbidity)', fontweight='bold', fontsize=24)
    plt.xlabel('PC1', fontweight='bold', fontsize=20)
    plt.ylabel('PC2', fontweight='bold', fontsize=20)
    plt.legend(frameon=False, fontsize=14)
    
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    plt.tight_layout()
    fig_path = os.path.join(args.outdir, 'abide_asd_pca_adhd.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print('Saved:', fig_path)

    plt.figure(figsize=(6, 4))
    plt.bar(stats_df['subtype'] + 1, stats_df['ADHD_rate'], color='#377eb8', alpha=0.8)
    plt.xticks(stats_df['subtype'] + 1, [f'Subtype {i+1}' for i in stats_df['subtype']])
    plt.ylabel('ADHD rate')
    plt.title('ADHD comorbidity rate by subtype (ABIDE-ASD)')
    plt.tight_layout()
    bar_path = os.path.join(args.outdir, 'abide_subtype_adhd_rate.png')
    plt.savefig(bar_path, dpi=220)
    print('Saved:', bar_path)

    adhd_only = asd_df[asd_df['ADHD_ANY'] == 1].copy()
    if len(adhd_only) > 0:
        cnt = adhd_only['subtype'].value_counts().sort_index()
        prop = (cnt / cnt.sum()).rename('proportion')
        adhd_subtype_df = pd.DataFrame({'subtype': cnt.index.astype(int), 'count': cnt.values, 'proportion': prop.values})
        adhd_csv = os.path.join(args.outdir, 'abide_adhd_only_subtype_distribution.csv')
        adhd_subtype_df.to_csv(adhd_csv, index=False)
        plt.figure(figsize=(6, 4))
        plt.bar(adhd_subtype_df['subtype'] + 1, adhd_subtype_df['proportion'], color='#e41a1c', alpha=0.85)
        plt.xticks(adhd_subtype_df['subtype'] + 1, [f'Subtype {i+1}' for i in adhd_subtype_df['subtype']])
        plt.ylabel('Proportion within ADHD-only cohort')
        plt.title('Subtype distribution among ADHD comorbid (ABIDE-ASD)')
        plt.tight_layout()
        adhd_bar = os.path.join(args.outdir, 'abide_adhd_only_subtype_distribution.png')
        plt.savefig(adhd_bar, dpi=220)
        print('Saved:', adhd_bar)


if __name__ == '__main__':
    main()
