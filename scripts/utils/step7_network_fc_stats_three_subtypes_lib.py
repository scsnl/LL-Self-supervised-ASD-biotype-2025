#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multitest import multipletests

_cur = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_cur)
if _root not in sys.path:
    sys.path.insert(0, _root)
if _cur not in sys.path:
    sys.path.insert(0, _cur)


import importlib.util as _ilu_lib, os as _os_lib

def _load_sibling(modname):
    fp = _os_lib.path.join(_os_lib.path.dirname(_os_lib.path.abspath(__file__)), modname + '.py')
    spec = _ilu_lib.spec_from_file_location(modname, fp)
    mod = _ilu_lib.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_fc_stats_mod = _load_sibling('step7_abide_network_fc_stats')
define_positive_negative_edges_by_td = _fc_stats_mod.define_positive_negative_edges_by_td
aggregate_fc_by_network = getattr(_fc_stats_mod, 'aggregate_fc_by_network', None)
load_network_map_csv_from_fc_stats = getattr(_fc_stats_mod, 'load_network_map_csv', None)

_alff_mod = _load_sibling('step7_abide_alff_violin_only')
remove_outliers_iqr = _alff_mod.remove_outliers_iqr

_violin_mod = _load_sibling('step7_violin_three_subtypes_lib')
_violin_attrs = ['plot_violin_group', 'SUBTYPE_COLORS', 'TD_COLOR']
for _attr in _violin_attrs:
    if hasattr(_violin_mod, _attr):
        globals()[_attr] = getattr(_violin_mod, _attr)


def stats_by_edge_three_subtypes(
    X_asd: np.ndarray, labels_asd: np.ndarray, method: str = "kruskal"
) -> pd.DataFrame:
    m = np.isin(labels_asd, [0, 1, 2])
    X = X_asd[m]
    lab = labels_asd[m].astype(int)
    n_edges = X.shape[1]
    pvals: list[float] = []
    stats_val: list[float] = []
    for i in range(n_edges):
        groups = []
        for g in (0, 1, 2):
            v = X[lab == g, i].astype(float)
            v = v[np.isfinite(v)]
            groups.append(v)
        if any(len(g) < 2 for g in groups):
            pvals.append(float("nan"))
            stats_val.append(float("nan"))
        else:
            if method == "kruskal":
                statv, p = kruskal(*groups)
            else:
                statv, p = f_oneway(*groups)
            pvals.append(float(p))
            stats_val.append(float(statv))
    p_arr = np.array(pvals, dtype=float)
    s_arr = np.array(stats_val, dtype=float)
    ok = np.isfinite(p_arr)
    q = np.full_like(p_arr, np.nan, dtype=float)
    sig = np.zeros_like(p_arr, dtype=bool)
    if ok.any():
        rej, qv, _, _ = multipletests(p_arr[ok], method="fdr_bh")
        q[ok] = qv
        sig[ok] = rej
    return pd.DataFrame(
        {
            "edge_idx": np.arange(n_edges),
            "stat": s_arr,
            "p": p_arr,
            "q": q,
            "sig_fdr": sig,
        }
    )


def plot_stat_matrix_edges(
    stat_vals: np.ndarray,
    iu: tuple[np.ndarray, np.ndarray],
    title: str,
    out_png: str,
    q_vals: np.ndarray | None = None,
    cbar_label: str = "Statistic",
) -> None:
    n = int((1 + np.sqrt(1 + 8 * len(stat_vals))) / 2)
    M = np.zeros((n, n), dtype=float)
    M[iu] = stat_vals
    M[(iu[1], iu[0])] = stat_vals
    np.fill_diagonal(M, 0.0)
    vmin = float(np.nanmin(M))
    vmax = float(np.nanmax(M))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0
    plt.figure(figsize=(5.6, 4.6))
    im = plt.imshow(M, cmap="hot", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=12, fontweight="bold")
    plt.title(title)
    plt.xlabel("Network", fontsize=14, fontweight="bold")
    plt.ylabel("Network", fontsize=14, fontweight="bold")
    ax = plt.gca()
    _apply_yeo7_axis(ax, n)
    if q_vals is not None and len(q_vals) == len(stat_vals):
        try:
            stars = np.where(q_vals < 1e-3, "**", np.where(q_vals < 0.05, "*", ""))
            for (y, x), s in zip(zip(iu[0], iu[1]), stars):
                if s:
                    plt.text(
                        x, y, s, ha="center", va="center", color="k", fontsize=10, fontweight="bold"
                    )
                    plt.text(
                        y, x, s, ha="center", va="center", color="k", fontsize=10, fontweight="bold"
                    )
        except Exception:
            pass
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_edge_grouped_bars_three_subtypes(
    X_asd: np.ndarray,
    labels_asd: np.ndarray,
    iu: tuple[np.ndarray, np.ndarray],
    out_png: str,
    title: str,
) -> None:
    m = np.isin(labels_asd, [0, 1, 2])
    X = X_asd[m]
    lab = labels_asd[m].astype(int)
    group_names = {0: "Subtype 1", 1: "Subtype 2", 2: "Subtype 3"}
    edge_names = [f"N{a+1}-N{b+1}" for a, b in zip(iu[0], iu[1])]
    rows: list[dict] = []
    for ei, ename in enumerate(edge_names):
        for g in (0, 1, 2):
            vals = X[lab == g, ei].astype(float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            rows.append(
                {"edge": ename, "group": group_names[g], "value": float(np.nanmean(vals))}
            )
    df_long = pd.DataFrame(rows)
    if df_long.empty:
        print("plot_edge_grouped_bars_three_subtypes: no data, skip")
        return
    desired = ["Subtype 1", "Subtype 2", "Subtype 3"]
    df_long["group"] = pd.Categorical(df_long["group"], categories=desired, ordered=True)
    fig_w = max(10.0, 0.6 * df_long["edge"].nunique())
    plt.figure(figsize=(fig_w, 4.6))
    ax = sns.barplot(data=df_long, x="edge", y="value", hue="group", palette="Set2", errorbar=None)
    ax.set_title(title)
    ax.set_xlabel("Network pair (edge)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean (network FC feature)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=60, ha="right")
    ax.legend(title="Group", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def run_figures_net_three_subtypes(
    args,
    labels_asd: np.ndarray,
    X_asd: np.ndarray,
    X_td: np.ndarray,
    fcs_asd: np.ndarray,
    fcs_td: np.ndarray,
    net_labels: np.ndarray,
    iu: tuple[np.ndarray, np.ndarray],
) -> None:
    figdir = os.path.join(args.outdir, "figures_net", "three_subtypes_only")
    statdir = os.path.join(args.outdir, "net_stats", "three_subtypes_only")
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(statdir, exist_ok=True)
    gsr_tag = "gsr" if args.use_gsr else "nogsr"
    method = args.method
    cbar = "Kruskal–Wallis H" if method == "kruskal" else "ANOVA F"

    print("=== figures_net three subtypes only (no TD in group tests) ===")
    df_stats = stats_by_edge_three_subtypes(X_asd, labels_asd, method=method)
    df_stats.to_csv(
        os.path.join(statdir, f"net_fc_stats_{gsr_tag}_{method}_three_subtypes.csv"),
        index=False,
    )
    q_vals = df_stats["q"].to_numpy()
    plot_stat_matrix_edges(
        df_stats["stat"].to_numpy(),
        iu,
        title=f"Network FC: three ASD subtypes ({method}, {gsr_tag})",
        out_png=os.path.join(figdir, f"net_fc_stats_{gsr_tag}_{method}_three_subtypes.png"),
        q_vals=q_vals,
        cbar_label=cbar,
    )

    if args.umap:
        try:
            from umap import UMAP

            m = np.isin(labels_asd, [0, 1, 2])
            X = X_asd[m]
            lab = labels_asd[m].astype(int)
            if len(X) >= 5:
                emb = UMAP(
                    n_neighbors=min(15, len(X) - 1),
                    min_dist=0.1,
                    metric="correlation",
                    random_state=42,
                ).fit_transform(X)
                plt.figure(figsize=(5.2, 4.4))
                plt.scatter(emb[:, 0], emb[:, 1], c=lab, cmap="tab10", s=12, vmin=0, vmax=9)
                plt.colorbar(label="Subtype (0–2)")
                plt.title(f"UMAP network-FC ({X.shape[1]}D), ASD subtypes only")
                plt.xlabel("UMAP-1", fontsize=14, fontweight="bold")
                plt.ylabel("UMAP-2", fontsize=14, fontweight="bold")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(figdir, f"net_fc_umap_{gsr_tag}_three_subtypes.png"),
                    dpi=200,
                )
                plt.close()
        except Exception as e:
            print("UMAP (three subtypes) skipped:", e)

    plot_edge_grouped_bars_three_subtypes(
        X_asd,
        labels_asd,
        iu,
        out_png=os.path.join(figdir, f"net_fc_grouped_edges_{gsr_tag}_three_subtypes.png"),
        title=f"Network FC by edge — three subtypes only ({gsr_tag})",
    )

    print("\n=== ROI GBC (TD-defined edges), violin: three subtypes only ===")
    td_mean_fc = np.nanmean(fcs_td, axis=0)
    positive_mask, negative_mask = define_positive_negative_edges_by_td(td_mean_fc, net_labels)
    n_roi = td_mean_fc.shape[0]
    iu_roi = np.triu_indices(n_roi, k=1)
    positive_mask_2d = np.zeros((n_roi, n_roi), dtype=bool)
    negative_mask_2d = np.zeros((n_roi, n_roi), dtype=bool)
    positive_mask_2d[iu_roi] = positive_mask
    negative_mask_2d[iu_roi] = negative_mask
    positive_mask_2d = positive_mask_2d + positive_mask_2d.T
    negative_mask_2d = negative_mask_2d + negative_mask_2d.T

    gbc_roi_asd = compute_roi_gbc_by_td_definition(fcs_asd, positive_mask_2d, negative_mask_2d)
    np.save(os.path.join(statdir, "gbc_roi_asd_td_defined.npy"), gbc_roi_asd)

    m3 = np.isin(labels_asd, [0, 1, 2])
    lab3 = labels_asd[m3].astype(int)

    for _gbc_type, gbc_idx, gbc_name in [(0, 0, "All")]:
        vec_full = gbc_roi_asd[:, :, gbc_idx].mean(axis=1)
        vals = vec_full[m3].astype(float)
        labs = lab3

        def _one(variant: str, v: np.ndarray, l: np.ndarray) -> None:
            H, p_kw = kruskal_three(v, l)
            plot_violin_three_subtypes(
                v,
                l,
                f"ROI GBC {gbc_name} (TD-defined edges), three subtypes ({variant})",
                os.path.join(
                    figdir,
                    f"roi_gbc_{gbc_name.lower()}_td_defined_three_subtypes_{variant.replace(' ', '_')}.png",
                ),
                ylabel="GBC (ROI mean)",
            )

        _one("with outliers", vals, labs)
        v2, l2, _ = remove_outliers_iqr(vals, labs, factor=2.0)
        _one("outliers removed", v2, l2)

    print("three_subtypes figures_net done ->", figdir)
