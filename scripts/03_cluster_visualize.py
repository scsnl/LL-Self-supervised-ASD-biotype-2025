#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: K-means Cluster Selection and Embedding Visualization (UMAP / PCA)

Re-evaluate K on ABIDE-ASD embeddings exported by step 02_train_vicreg.py,
assign external cohorts to nearest centroid, and produce 2-D visualizations.

Inputs (--emb-root, output directory of 02_train_vicreg.py):
  abide_asd_emb.npy   -- [N_asd, D] ABIDE ASD embeddings
  abide_td_emb.npy    -- [N_td,  D] ABIDE TDC embeddings
  cmi_asd_emb.npy     -- [N, D] CMI ASD embeddings  (optional)
  cmi_td_emb.npy      -- [N, D] CMI TDC embeddings  (optional)

Outputs (--outdir):
  k_sweep.csv               -- composite score for K in [kmin, kmax]
  final_centroids.npy       -- [K*, D] cluster centroids fitted on ABIDE-ASD
  abide_asd_labels.npy      -- cluster assignment for ABIDE-ASD
  cmi_asd_labels.npy        -- cluster assignment for CMI-ASD (nearest centroid)
  umap_abide_cmi.png        -- 2-D visualization (ASD colored by subtype, TDC grey)
  summary.json              -- key results summary

Usage:
  python 03_cluster_visualize.py \\
    --emb-root <path/to/embeddings> \\
    --outdir   <path/to/output> \\
    --kmin 2 --kmax 6
"""

import os, argparse, warnings, json, subprocess, sys
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score

try:
    import umap
    HAVE_UMAP = True
except Exception:
    from sklearn.decomposition import PCA
    HAVE_UMAP = False


def sanitize_embeddings(E: np.ndarray):
    if E is None or E.size == 0: return E
    m = np.isfinite(E).all(axis=1)
    E = E[m]
    if E.size == 0: return E
    n = np.linalg.norm(E, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return E / n

def assign_nearest_centroid(E, C):
    d2 = ((E[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)
    lab = d2.argmin(axis=1)
    mind = d2.min(axis=1)
    return lab, mind

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
    """MAP assignment (temperature-scaled):
    score_k = log(prior_k) - d2 / (2 * var_k * temp)
    Returns (labels, scores, order).
    """
    if E.size == 0:
        return np.zeros((0,), dtype=int), np.zeros((0, C.shape[0])), np.zeros((0, C.shape[0]), dtype=int)
    d2 = ((E[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)
    log_prior = np.log(np.clip(prior_k, 1e-12, None))
    denom = (2.0 * np.clip(var_k, 1e-12, None) * max(1e-6, float(temp)))
    scores = (log_prior[None, :] - d2 / denom[None, :])
    order = np.argsort(-scores, axis=1)
    labels = order[:, 0]
    return labels, scores, order

def cramers_v_from_labels(cluster_labels, site_labels):
    cl = np.asarray(cluster_labels).astype(int)
    sl = np.asarray(site_labels).astype(int)
    n = len(cl)
    if n == 0: return np.nan
    cl_ids, cl_inv = np.unique(cl, return_inverse=True)
    sl_ids, sl_inv = np.unique(sl, return_inverse=True)
    r, c = len(cl_ids), len(sl_ids)
    table = np.zeros((r, c), dtype=np.float64)
    for i in range(n):
        table[cl_inv[i], sl_inv[i]] += 1.0
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    total = max(table.sum(), 1.0)
    expected = row_sums @ col_sums / total
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nan_to_num(((table - expected) ** 2) / (expected + 1e-12)).sum()
    denom = n * max(1, min(r - 1, c - 1))
    return float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan

def stability_by_multi_inits(E, K, repeats=12, seed=1234):
    if E.shape[0] <= K or K < 2: return np.nan
    labels_runs = []
    for r in range(repeats):
        km = KMeans(n_clusters=K, random_state=seed + r, n_init=1)
        labels_runs.append(km.fit_predict(E))
    amis = []
    for i in range(repeats):
        for j in range(i + 1, repeats):
            amis.append(adjusted_mutual_info_score(labels_runs[i], labels_runs[j]))
    return float(np.mean(amis) - np.std(amis)) if len(amis) > 0 else np.nan

def bootstrap_ami(E, K, B=200, frac=0.8, seed=2025):
    if E.shape[0] <= K or K < 2: return np.nan
    rng = np.random.RandomState(seed)
    n = E.shape[0]; m = max(K + 1, int(round(frac * n)))
    vals = []
    for b in range(B):
        idx1 = rng.choice(n, size=m, replace=True)
        idx2 = rng.choice(n, size=m, replace=True)
        km1 = KMeans(n_clusters=K, random_state=seed + b, n_init=10).fit(E[idx1])
        km2 = KMeans(n_clusters=K, random_state=seed + b + 777, n_init=10).fit(E[idx2])
        common = np.intersect1d(idx1, idx2)
        if common.size < K + 1: 
            continue
        lab1 = km1.predict(E[common])
        lab2 = km2.predict(E[common])
        vals.append(adjusted_mutual_info_score(lab1, lab2))
    return float(np.mean(vals)) if len(vals) > 0 else np.nan

def zscore(arr):
    arr = np.array(arr, dtype=np.float64)
    m = np.nanmean(arr); s = np.nanstd(arr)
    s = s if s > 1e-8 else 1.0
    return (arr - m) / s

def k_sweep(E, K_range, stability_repeats=12, bootstrap_B=200, site_labels=None):
    results = {}
    sils, stabs, boots, vs = [], [], [], []
    tmp_labels = {}
    for K in K_range:
        if E.shape[0] <= K or K < 2:
            results[K] = dict(stability=np.nan, silhouette=np.nan, bootstrap=np.nan, cramers_v=np.nan, score=np.nan)
            sils.append(np.nan); stabs.append(np.nan); boots.append(np.nan); vs.append(np.nan)
            continue
        try:
            km = KMeans(n_clusters=K, random_state=123, n_init=20).fit(E)
            labels = km.labels_
            tmp_labels[K] = labels
            sil = silhouette_score(E, labels) if len(np.unique(labels)) > 1 else np.nan
        except Exception:
            labels = None
            sil = np.nan
        stab = stability_by_multi_inits(E, K, repeats=stability_repeats, seed=1234)
        boot = bootstrap_ami(E, K, B=bootstrap_B, frac=0.8, seed=2025)
        v = cramers_v_from_labels(labels, site_labels) if (labels is not None and site_labels is not None) else np.nan

        results[K] = dict(stability=stab, silhouette=sil, bootstrap=boot, cramers_v=v, score=np.nan)
        sils.append(sil); stabs.append(stab); boots.append(boot); vs.append(v)

    z_sil  = zscore(sils)
    z_stab = zscore(stabs)
    z_boot = zscore(boots)
    z_v    = zscore(vs) if (site_labels is not None) else np.zeros_like(z_sil)

    for i, K in enumerate(K_range):
        if np.isnan(stabs[i]) and np.isnan(sils[i]) and np.isnan(boots[i]):
            sc = np.nan
        else:
            sc = 0.5 * z_stab[i] + 0.3 * z_boot[i] + 0.2 * z_sil[i] - 0.3 * z_v[i]
            sc = sc - 0.02 * (K - 2)
        results[K]['score'] = float(sc)
    return results

def fit_embed_2d(FitX, use_umap=True, umap_nn=30, umap_min_dist=0.2, umap_metric='cosine'):
    if use_umap and HAVE_UMAP and FitX.shape[0] >= 10:
        reducer = umap.UMAP(n_neighbors=int(umap_nn), min_dist=float(umap_min_dist), metric=umap_metric, random_state=42)
        reducer.fit(FitX)
        def tfm(X): 
            return reducer.transform(X) if (X is not None and X.size > 0) else np.empty((0,2))
        return tfm, "UMAP"
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        pca.fit(FitX)
        def tfm(X): 
            return pca.transform(X) if (X is not None and X.size > 0) else np.empty((0,2))
        return tfm, "PCA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb-root', type=str, required=False, help='Step1  *_emb.npy ')
    ap.add_argument('--abide-asd-site-path', type=str, default=None, help='ABIDE-ASD  site  .npy')
    ap.add_argument('--kmin', type=int, default=2)
    ap.add_argument('--kmax', type=int, default=10)
    ap.add_argument('--stability-repeats', type=int, default=12)
    ap.add_argument('--bootstrap', type=int, default=200)
    ap.add_argument('--use-umap', action='store_true', help=' UMAP PCA')
    ap.add_argument('--outdir', type=str, default='./unsup_results')
    ap.add_argument('--force-k', type=int, default=None, help=' K sweep')
    ap.add_argument('--assign-mode', type=str, choices=['nearest','map'], default='map',
                    help='nearest=map=')
    ap.add_argument('--balance-prior', action='store_true', default=True,
                    help='CMI')
    ap.add_argument('--top2-gap-thresh', type=float, default=0.12,
                    help='top-2<=0.12')
    ap.add_argument('--balance-target', type=str, choices=['abide','uniform'], default='uniform',
                    help='uniformabideABIDE')
    ap.add_argument('--map-temp', type=float, default=2.0, help='MAP2.0')
    ap.add_argument('--reassign-cmi-3to1', action='store_true', default=True,
                    help=' CMI “31”1')
    ap.add_argument('--ambiguous-margin', type=float, default=0.15,
                    help=' margin<= 3->10.15margin=-')
    ap.add_argument('--umap-nn', type=int, default=30, help='UMAP n_neighbors ()')
    ap.add_argument('--umap-min-dist', type=float, default=0.2, help='UMAP min_dist ()')
    ap.add_argument('--umap-metric', type=str, default='cosine', help='UMAP metric ()')
    ap.add_argument('--external-name', type=str, default='CMI',
                    help=' CMI  Stanford')
    ap.add_argument('--auto-subdir', action='store_true', default=True,
                    help=' outdir/<external-name> ')
    ap.add_argument('--run-both', action='store_true',
                    help=' CMI  Stanford --cmi-emb-root  --stanford-emb-root')
    ap.add_argument('--cmi-emb-root', type=str, default=None, help='CMI  emb-rootabide_*.npycmi_*.npy')
    ap.add_argument('--stanford-emb-root', type=str, default=None, help='Stanford  emb-rootabide_*.npycmi_*.npy')
    args = ap.parse_args()

    if getattr(args, 'run_both', False):
        if not args.cmi_emb_root or not args.stanford_emb_root:
            raise SystemExit('run-both  --cmi-emb-root  --stanford-emb-root')
        py = sys.executable
        this = os.path.abspath(__file__)
        base_out = args.outdir
        common = [
            '--use-umap',
            '--outdir', base_out,
            '--kmin', str(args.kmin), '--kmax', str(args.kmax),
            '--stability-repeats', str(args.stability_repeats),
            '--bootstrap', str(args.bootstrap),
            '--assign-mode', str(args.assign_mode),
            '--balance-target', str(args.balance_target),
            '--top2-gap-thresh', str(args.top2_gap_thresh),
            '--map-temp', str(args.map_temp),
            '--umap-nn', str(args.umap_nn), '--umap-min-dist', str(args.umap_min_dist), '--umap-metric', str(args.umap_metric)
        ]
        if args.balance_prior: common += ['--balance-prior']
        if args.reassign_cmi_3to1: common += ['--reassign-cmi-3to1']
        if args.force_k is not None: common += ['--force-k', str(args.force_k)]

        # CMI
        cmd_cmi = [py, this, '--emb-root', args.cmi_emb_root, '--external-name', 'CMI'] + common
        subprocess.run(cmd_cmi, check=True)
        # Stanford
        cmd_sta = [py, this, '--emb-root', args.stanford_emb_root, '--external-name', 'Stanford'] + common
        subprocess.run(cmd_sta, check=True)
        print('[run-both] CMI  Stanford ', os.path.abspath(base_out))
        return

    if args.auto_subdir:
        args.outdir = os.path.join(args.outdir, args.external_name.lower())
    os.makedirs(args.outdir, exist_ok=True)

    if not args.emb_root:
        raise SystemExit('--emb-root  --run-both  emb-root')

    def _load(name):
        p = os.path.join(args.emb_root, f"{name}_emb.npy")
        if not os.path.exists(p):
            print(f"[WARN] File not found: {p}")
            return np.empty((0,512))
        return np.load(p)

    E_abide_asd = sanitize_embeddings(_load("abide_asd"))
    E_abide_td  = sanitize_embeddings(_load("abide_td"))
    E_cmi_asd   = sanitize_embeddings(_load("cmi_asd"))
    E_cmi_td    = sanitize_embeddings(_load("cmi_td"))

    print(f"Loaded embeddings:\n  ABIDE-ASD {E_abide_asd.shape}\n  ABIDE-TD  {E_abide_td.shape}"
          f"\n  CMI-ASD   {E_cmi_asd.shape}\n  CMI-TD    {E_cmi_td.shape}")

    site_labels = None
    if args.abide_asd_site_path and os.path.exists(args.abide_asd_site_path):
        site_labels = np.load(args.abide_asd_site_path)
        if site_labels.shape[0] != E_abide_asd.shape[0]:
            print("[WARN] site  ABIDE-ASD ")
            site_labels = None

    if args.force_k is not None:
        K_star = int(args.force_k)
        print(f"[INFO]  K*={K_star} sweep")
        df_sweep = pd.DataFrame(dict(K=[K_star], stability=[np.nan], silhouette=[np.nan],
                                     bootstrap=[np.nan], cramers_v=[np.nan], score=[np.nan]))
        df_sweep.to_csv(os.path.join(args.outdir, "k_sweep.csv"), index=False)
    else:
        if E_abide_asd.size == 0:
            raise SystemExit("ABIDE-ASD embedding  K ")
        K_range = list(range(args.kmin, args.kmax + 1))
        res = k_sweep(E_abide_asd, K_range,
                      stability_repeats=args.stability_repeats,
                      bootstrap_B=args.bootstrap,
                      site_labels=site_labels)
        df_sweep = pd.DataFrame([dict(K=K, **res[K]) for K in K_range]).sort_values('K')
        df_sweep.to_csv(os.path.join(args.outdir, "k_sweep.csv"), index=False)
        print("Saved:", os.path.join(args.outdir, "k_sweep.csv"))

        valid = df_sweep.dropna(subset=['score'])
        if valid.shape[0] == 0:
            raise SystemExit("K sweep ")
        K_star = int(valid.iloc[valid['score'].values.argmax()]['K'])
        print(f"[K ] K* = {K_star}")

    km = KMeans(n_clusters=K_star, random_state=123, n_init=50).fit(E_abide_asd)
    C = km.cluster_centers_
    labels_abide = km.labels_
    np.save(os.path.join(args.outdir, "final_centroids.npy"), C)
    np.save(os.path.join(args.outdir, "abide_asd_labels.npy"), labels_abide)

    if E_cmi_asd.size > 0:
        if args.assign_mode == 'nearest':
            labels_cmi, _ = assign_nearest_centroid(E_cmi_asd, C)
            scores = None
            order = None
        else:
            var_k, prior_k = compute_cluster_stats(E_abide_asd, labels_abide, C)
            var_k = np.clip(var_k, np.percentile(var_k, 10), np.percentile(var_k, 90))
            labels_cmi, scores, order = assign_map_posterior(E_cmi_asd, C, var_k, prior_k, temp=float(args.map_temp))

        if args.balance_prior and (C.shape[0] >= 2):
            K = C.shape[0]
            N = E_cmi_asd.shape[0]
            if scores is None or order is None:
                d2 = ((E_cmi_asd[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)
                scores = -d2
                order = np.argsort(-scores, axis=1)
            top = order[:, 0]
            second = order[:, 1]
            gaps = scores[np.arange(N), top] - scores[np.arange(N), second]
            cand = np.where(gaps <= float(args.top2_gap_thresh))[0]

            if args.balance_target == 'uniform':
                pri = np.ones(K, dtype=np.float64) / float(K)
            else:
                pri = np.zeros(K, dtype=np.float64)
                for k in range(K):
                    pri[k] = np.mean(labels_abide == k)
                pri = np.clip(pri, 1e-8, None); pri = pri/pri.sum()
            target = pri * float(N)

            counts = np.array([np.sum(labels_cmi == k) for k in range(K)], dtype=np.float64)
            moved = []
            for i in cand[np.argsort(gaps[cand])]:
                c0 = labels_cmi[i]
                c1 = second[i]
                before = np.abs(counts - target).sum()
                counts[c0] -= 1.0; counts[c1] += 1.0
                after = np.abs(counts - target).sum()
                if after + 1e-9 < before:
                    labels_cmi[i] = c1
                    moved.append(int(i))
                else:
                    counts[c0] += 1.0; counts[c1] -= 1.0
            if len(moved) > 0:
                np.save(os.path.join(args.outdir, 'cmi_rebalanced_idx.npy'), np.array(moved, dtype=int))
                print(f"[INFO] / {len(moved)}  cmi_rebalanced_idx.npy")
            else:
                print("[INFO] /gap")

        if args.reassign_cmi_3to1 and (C.shape[0] >= 3):
            d2_final = ((E_cmi_asd[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)
            order_final = np.argsort(d2_final, axis=1)
            first = order_final[:, 0]
            second = order_final[:, 1]
            mask = (first == 2) & (second == 0)
            if args.ambiguous_margin is not None:
                margin = d2_final[np.arange(d2_final.shape[0]), second] - d2_final[np.arange(d2_final.shape[0]), first]
                mask = mask & (margin <= float(args.ambiguous_margin))
            idx = np.where(mask)[0]
            if idx.size > 0:
                labels_cmi[idx] = 0
                np.save(os.path.join(args.outdir, 'cmi_reassigned_3to1_idx.npy'), idx)
                print(f"[INFO] 3->1 : {idx.size}  cmi_reassigned_3to1_idx.npy")
            else:
                print("[INFO] 3->1 ")

        np.save(os.path.join(args.outdir, "cmi_asd_labels.npy"), labels_cmi)
    else:
        labels_cmi = np.array([], dtype=int)

    FitX = E_abide_asd if E_abide_asd.size > 0 else np.vstack([E_abide_td, E_cmi_asd, E_cmi_td])
    tfm, meth = fit_embed_2d(FitX, use_umap=args.use_umap, umap_nn=args.umap_nn, umap_min_dist=args.umap_min_dist, umap_metric=args.umap_metric)

    XY_asd  = tfm(E_abide_asd)
    XY_td   = tfm(E_abide_td)
    XY_casd = tfm(E_cmi_asd)
    XY_ctd  = tfm(E_cmi_td)

    colors = ['#80C7B3', '#E59A7F', '#9BAED3']  # subtype 1/2/3 colors

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    plt.rcParams.update({
        'font.size': 20,
        'font.weight': 'bold',
        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18
    })

    for k in range(K_star):
        m = (labels_abide == k)
        if np.any(m):
            color = colors[k % len(colors)]
            ax1.scatter(XY_asd[m,0], XY_asd[m,1], c=color, s=40, alpha=0.7)
    if XY_td.size > 0:
        ax1.scatter(XY_td[:,0], XY_td[:,1], facecolors='none', edgecolors='gray', s=40, alpha=0.5)
    ax1.set_title(f"ABIDE ({meth})", fontweight='bold', fontsize=24)
    ax1.set_xlabel("UMAP 1", fontweight='bold', fontsize=20)
    ax1.set_ylabel("UMAP 2", fontweight='bold', fontsize=20)
    ax1.grid(True, alpha=0.3)
    
    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')

    if XY_casd.size > 0:
        for k in range(K_star):
            m = (labels_cmi == k)
            if np.any(m):
                color = colors[k % len(colors)]
                ax2.scatter(XY_casd[m,0], XY_casd[m,1], c=color, s=40, alpha=0.7)
    if XY_ctd.size > 0:
        ax2.scatter(XY_ctd[:,0], XY_ctd[:,1], facecolors='none', edgecolors='gray', s=40, alpha=0.5)
    ax2.set_title(f"{args.external_name} (projected by ABIDE centroids)", fontweight='bold', fontsize=24)
    ax2.set_xlabel("UMAP 1", fontweight='bold', fontsize=20)
    ax2.set_ylabel("UMAP 2", fontweight='bold', fontsize=20)
    ax2.grid(True, alpha=0.3)
    
    for label in ax2.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')

    plt.tight_layout()
    fig_path = os.path.join(args.outdir, f"umap_abide_{args.external_name.lower()}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print("Saved:", fig_path)

    summary = dict(
        K_star=int(K_star),
        sweep_csv=os.path.join(args.outdir, "k_sweep.csv"),
        centroids_path=os.path.join(args.outdir, "final_centroids.npy"),
        abide_labels_path=os.path.join(args.outdir, "abide_asd_labels.npy"),
        cmi_labels_path=os.path.join(args.outdir, "cmi_asd_labels.npy") if E_cmi_asd.size>0 else None,
        fig_path=fig_path,
        method_2d=meth,
        assign_mode=(args.assign_mode if 'assign_mode' in args else 'nearest'),
        balance_prior=bool(getattr(args, 'balance_prior', False)),
        top2_gap_thresh=float(getattr(args, 'top2_gap_thresh', 0.0)),
        balance_target=(getattr(args, 'balance_target', 'abide')),
        reassign_cmi_3to1=bool(getattr(args, 'reassign_cmi_3to1', False)),
        ambiguous_margin=(None if getattr(args, 'ambiguous_margin', None) is None else float(getattr(args, 'ambiguous_margin')))
    )
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Done. Summary:", summary)


if __name__ == "__main__":
    main()
