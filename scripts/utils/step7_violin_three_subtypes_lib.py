#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests

THREE_GROUPS = ["Subtype 1", "Subtype 2", "Subtype 3"]


def load_alff_asd_masked_for_three_subtypes(outdir: str, mask_asd: np.ndarray) -> np.ndarray:
    """
    """
    combat_a = os.path.join(outdir, "alffs_asd_combat.npy")
    combat_td = os.path.join(outdir, "alffs_td_combat.npy")
    gsr_a = os.path.join(outdir, "alffs_asd_gsr.npy")
    if os.path.exists(combat_a) and os.path.exists(combat_td):
        cand = np.load(combat_a)[mask_asd]
        if np.any(np.isfinite(cand)):
            print(" ComBat  ALFF ASD ")
            return cand
        print("ComBat ALFF  GSR")
    if not os.path.exists(gsr_a):
        raise FileNotFoundError(f" ALFF : {gsr_a}")
    print(" GSR  ALFF ASD ")
    return np.load(gsr_a)[mask_asd]


def load_gbcs_asd_masked_for_three_subtypes(outdir: str, mask_asd: np.ndarray) -> np.ndarray:
    combat_a = os.path.join(outdir, "gbcs_asd_combat.npy")
    combat_td = os.path.join(outdir, "gbcs_td_combat.npy")
    gsr_a = os.path.join(outdir, "gbcs_asd_gsr.npy")
    if os.path.exists(combat_a) and os.path.exists(combat_td):
        cand = np.load(combat_a)[mask_asd]
        if np.any(np.isfinite(cand)):
            print(" ComBat  GBC ASD")
            return cand
        print("ComBat GBC  GSR")
    if not os.path.exists(gsr_a):
        raise FileNotFoundError(f" GBC : {gsr_a}")
    print(" GSR  GBC ASD")
    return np.load(gsr_a)[mask_asd]


def int_label_to_name(i: int) -> str:
    return f"Subtype {int(i) + 1}"


def pairwise_mannwhitney_fdr(values: np.ndarray, labels: np.ndarray) -> dict[str, dict]:
    pairs = [
        (0, 1, "Subtype 1", "Subtype 2"),
        (0, 2, "Subtype 1", "Subtype 3"),
        (1, 2, "Subtype 2", "Subtype 3"),
    ]
    p_raw: list[float] = []
    keys: list[str] = []
    meta: list[tuple[str, str, float]] = []
    for ka, kb, na, nb in pairs:
        a = values[labels == ka]
        b = values[labels == kb]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        key = f"{na} vs {nb}"
        keys.append(key)
        if len(a) < 2 or len(b) < 2:
            p_raw.append(1.0)
            meta.append((na, nb, np.nan))
            continue
        U, p = mannwhitneyu(a, b, alternative="two-sided")
        p_raw.append(float(p))
        meta.append((na, nb, float(U)))
    _, q, _, _ = multipletests(np.array(p_raw), method="fdr_bh")
    out: dict[str, dict] = {}
    for i, key in enumerate(keys):
        na, nb, Uv = meta[i]
        out[key] = {
            "group_a": na,
            "group_b": nb,
            "U_statistic": Uv,
            "p_value": p_raw[i],
            "p_corrected": float(q[i]),
            "significant_uncorrected": p_raw[i] < 0.05,
            "significant_corrected": float(q[i]) < 0.05,
        }
    return out


def kruskal_three(values: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    g0 = values[labels == 0]
    g1 = values[labels == 1]
    g2 = values[labels == 2]
    g0 = g0[np.isfinite(g0)]
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]
    if any(len(g) < 2 for g in (g0, g1, g2)):
        return float("nan"), float("nan")
    H, p = kruskal(g0, g1, g2)
    return float(H), float(p)


def _sig_stars(p: float, q: float) -> str:
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    if p < 0.05:
        return "†"
    return "ns"


def plot_violin_three_subtypes(
    values: np.ndarray,
    labels_int: np.ndarray,
    title: str,
    out_path: str,
    ylabel: str = "Value",
) -> dict:
    """
    """
    m = np.isfinite(values) & np.isin(labels_int, [0, 1, 2])
    v = values[m].astype(float)
    lab = labels_int[m].astype(int)
    if len(v) == 0:
        raise ValueError("")

    group_disp = [int_label_to_name(int(g)) for g in lab]
    df = pd.DataFrame({"value": v, "group": group_disp})
    stats_pairs = pairwise_mannwhitney_fdr(v, lab)
    H, p_kw = kruskal_three(v, lab)

    plt.figure(figsize=(7, 5))
    ax = sns.violinplot(
        x="group", y="value", data=df, palette="Set2", order=THREE_GROUPS
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Group", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")

    y_max = float(df["value"].max())
    y_min = float(df["value"].min())
    y_range = y_max - y_min + 1e-12
    x_ticks = ax.get_xticks()
    pos = {THREE_GROUPS[i]: float(x_ticks[i]) for i in range(len(THREE_GROUPS))}

    base_y = y_max + 0.08 * y_range
    step = 0.045 * y_range
    comp_idx = 0
    for _key, st in stats_pairs.items():
        stars = _sig_stars(st["p_value"], st["p_corrected"])
        if stars == "ns":
            continue
        na, nb = st["group_a"], st["group_b"]
        xa, xb = pos[na], pos[nb]
        y = base_y + comp_idx * step * 2.2
        comp_idx += 1
        ax.plot([xa, xb], [y, y], "k-", lw=1.5)
        ax.plot([xa, xa], [y - 0.02 * y_range, y], "k-", lw=1.5)
        ax.plot([xb, xb], [y - 0.02 * y_range, y], "k-", lw=1.5)
        ax.text(
            (xa + xb) / 2,
            y,
            stars,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    kw_txt = ""
    if np.isfinite(p_kw):
        kw_txt = f"Kruskal-Wallis p={p_kw:.3g}. "
    leg = (
        kw_txt
        + "Pairwise: Mann-Whitney U, FDR-BH (3 tests). "
        "***q<0.001 **q<0.01 *q<0.05 †p<0.05"
    )
    plt.figtext(0.5, 0.02, leg, ha="center", fontsize=8, style="italic")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14, top=0.88)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return stats_pairs


def figdir_three_subtypes(base_figdir: str) -> str:
    d = os.path.join(base_figdir, "violin_three_subtypes_only")
    os.makedirs(d, exist_ok=True)
    return d
