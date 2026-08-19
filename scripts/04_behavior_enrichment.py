#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4: Behavioral Domain Enrichment Analysis

Maps phenotypic measures from ABIDE to six harmonized behavioral domains and
computes fold enrichment for each ASD subtype relative to the overall ASD
sample.

Domains:
  SCL     -- Social Communication
  RRB     -- Restricted/Repetitive Behaviors
  COG     -- Cognitive Ability (reverse-scored)
  BEH-EMO -- Behavioral/Emotional Dysregulation
  PHY     -- Physical/Motor
  AGE     -- Age (developmental proxy)

Standardization: robust Z-score (median / MAD); directionality harmonized so
higher values = greater symptom burden.

Enrichment: binary threshold at robust-Z > 1.0; fold enrichment is the ratio of
within-subtype prevalence to overall ASD prevalence, plotted on a log scale
(values > 1 indicate enrichment, values < 1 indicate depletion, midline = 1).
Significance uses a directional (upper- or lower-tail) binomial test.

Usage:
  python 04_behavior_enrichment.py \\
    --abide-pheno  <phenotype_csv>  \\
    --labels-npy   <abide_asd_labels.npy> \\
    --outdir       <output_dir>
"""
import os, argparse, importlib.util
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import ttest_ind, binomtest
import matplotlib.pyplot as plt

DOMAIN_COLUMNS: Dict[str, List[str]] = {
    "SCL": [
        "ADI_R_SOCIAL_TOTAL_A", "ADI_R_VERBAL_TOTAL_BV",
        "CBCL_6-18_SOCIAL_PROBLEM_T", "CBCL_6-18_SOCIAL_T",
        "ADOS_2_SOCAFFECT", "ADOS_2_TOTAL", "ADOS_2_SEVERITY_TOTAL",
        "ADOS_GOTHAM_TOTAL", "ADOS_GOTHAM_SEVERITY",
        "SCQ_TOTAL", "SRS_RAW_TOTAL", "AQ_TOTAL",
    ],
    "RRB": [
        "ADI_RRB_TOTAL_C", "ADOS_2_RRB", "ADOS_GOTHAM_RRB",
        "RBSR_5SUBSCALE_COMPULSIVE", "RBSR_5SUBSCALE_RESTRICTED",
        "RBSR_5SUBSCALE_RITUALISTIC", "RBSR_5SUBSCALE_STEREOTYPIC",
        "RBSR_5SUBSCALE_SELF-INJURIOUS", "RBSR_5SUBSCALE_TOTAL",
        "RBSR_6SUBSCALE_COMPULSIVE", "RBSR_6SUBSCALE_RESTRICTED",
        "RBSR_6SUBSCALE_RITUALISTIC", "RBSR_6SUBSCALE_SAMENESS",
        "RBSR_6SUBSCALE_SELF-INJURIOUS", "RBSR_6SUBSCALE_STEREOTYPED",
        "RBSR_6SUBSCALE_TOTAL", "CBCL_6-18_OBSESSIVE_T",
    ],
    "COG": ["FIQ", "PIQ", "VIQ"],
    "BEH-EMO": [

        # CBCL 1.5–5
        "CBCL_1.5-5_AFFECTIVE_T", "CBCL_1.5-5_AGGRESSIVE_T", "CBCL_1.5-5_ANXIOUS_T",
        "CBCL_1.5-5_ATTENTION_DEFICIT_T", "CBCL_1.5-5_ATTENTION_PROBLEM_T",
        "CBCL_1.5-5_EMOTION_T", "CBCL_1.5-5_EXTERNAL_T", "CBCL_1.5-5_INTERNAL_T",
        "CBCL_1.5-5_OPPOSITIONAL_T", "CBCL_1.5-5_PERVASIVE_T",
        "CBCL_1.5-5_SLEEP_T", "CBCL_1.5-5_SOMANTIC_T",
        "CBCL_1.5-5_TOTAL_T", "CBCL_1.5-5_WITHDRAWN_T",
        "CBCL_6-18_AFFECTIVE_T", "CBCL_6-18_AGGRESSIVE_T", "CBCL_6-18_ANXIOUS_T",
        "CBCL_6-18_ATTENTION_DEFICIT_T", "CBCL_6-18_ATTENTION_T",
        "CBCL_6-18_CONDUCT_T", "CBCL_6-18_EXTERNAL_T", "CBCL_6-18_INTERNAL_T",
        "CBCL_6-18_POST_TRAUMATIC_T", "CBCL_6-18_RULE_T",
        "CBCL_6-18_SLUGGISH_T", "CBCL_6-18_SOMATIC_COMPAINT_T",
        "CBCL_6-18_SOMATIC_PROBLEM_T", "CBCL_6-18_THOUGHT_T",
        "CBCL_6-18_TOTAL_PROBLEM_T", "CBCL_6-18_WITHDRAWN_T",
        "CBCL_6-18_ACTIVITIES_T", "CBCL_6-18_SCHOOL_T", "CBCL_6-18_TOTAL_COMPETENCE_T",
    ],
    "PHY": ["BMI"],
    "AGE": ["AGE_AT_SCAN", "AGE_AT_MPRAGE"],
}

REVERSE_COLUMNS = set([
    "FIQ","PIQ","VIQ",
    "CBCL_6-18_SOCIAL_T",
    "CBCL_6-18_ACTIVITIES_T","CBCL_6-18_SCHOOL_T","CBCL_6-18_TOTAL_COMPETENCE_T",
])

EXCLUDE_COLUMNS = set([
    "ABIDE","ADI_R_ONSET_TOTAL_D","ADI_R_RSRCH_RELIABLE",
    "ASD_DSM_5","DSM_IV_TR","EYE_STATUS_AT_SCAN",
    "BRIEF_INFORMANT","BRIEF_VERSION","SRS_VERSION",
])

def _import_step1(cur_file: str):
    base = os.path.dirname(os.path.abspath(cur_file))
    path = os.path.join(base, '02_train_vicreg.py')
    spec = importlib.util.spec_from_file_location('step1_module', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def rebuild_abide_asd(abide_pklz: str, cmi_pklz: str, cur_file: str) -> pd.DataFrame:
    step1 = _import_step1(cur_file)
    cmi_arg = cmi_pklz if (cmi_pklz and os.path.exists(cmi_pklz)) else None
    abide_df, _ = step1.load_and_preprocess_data(abide_pklz, cmi_arg)
    asd_df = abide_df[abide_df["DX_GROUP"] == 0].copy()
    asd_df.reset_index(drop=True, inplace=True)
    return asd_df

def z_standard(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors='coerce')
    m, s = x.mean(skipna=True), x.std(skipna=True, ddof=1)
    if s and np.isfinite(s) and s > 0: return (x - m) / s
    return pd.Series(np.nan, index=x.index)

def z_robust(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors='coerce')
    med = x.median(skipna=True)
    mad = (x - med).abs().median(skipna=True)
    if mad and np.isfinite(mad) and mad > 0:
        return (x - med) / (1.4826 * mad)  # MAD→σ
    return z_standard(x)

def z_minmax(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors='coerce')
    xmin, xmax = x.min(skipna=True), x.max(skipna=True)
    if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
        z = (x - xmin) / (xmax - xmin)  # 0~1
        return (z - 0.5) * 6.0
    return pd.Series(np.nan, index=x.index)

def standardize_column(x: pd.Series, method: str) -> pd.Series:
    if method == "robust":   return z_robust(x)
    if method == "standard": return z_standard(x)
    if method == "minmax":   return z_minmax(x)
    raise ValueError("z-method must be 'robust' | 'standard' | 'minmax'")

def stouffer_aggregate(Zmat: pd.DataFrame) -> pd.Series:
    if Zmat.shape[1] == 0: return pd.Series(np.nan, index=Zmat.index)
    def _one(r: pd.Series) -> float:
        v = r.to_numpy()
        m = np.isfinite(v)
        k = m.sum()
        if k == 0: return np.nan
        return float(v[m].sum() / np.sqrt(k))
    return Zmat.apply(_one, axis=1)

def build_domain_scores(asd_df: pd.DataFrame, z_method: str) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    dom_scores = {}
    used_cols: Dict[str, List[str]] = {}
    for dom, cols in DOMAIN_COLUMNS.items():
        if dom == "AGE":
            cols = [c for c in ["AGE_AT_SCAN", "AGE_AT_MPRAGE"] if c in cols]
        keep = [c for c in cols if (c in asd_df.columns and c not in EXCLUDE_COLUMNS)]
        used_cols[dom] = keep
        if len(keep) == 0:
            dom_scores[dom] = pd.Series(np.nan, index=asd_df.index); continue
        Zs = []
        for c in keep:
            z = standardize_column(asd_df[c], z_method)
            if c in REVERSE_COLUMNS: z = -z
            Zs.append(z.rename(c))
        Zmat = pd.concat(Zs, axis=1)
        dom_scores[dom] = stouffer_aggregate(Zmat)
    dom_df = pd.DataFrame(dom_scores, index=asd_df.index)
    return dom_df, used_cols

# --------------------- FDR ---------------------
def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if mask.sum() == 0: return q
    pv = p[mask]; m = pv.size
    order = np.argsort(pv); ranks = np.empty_like(order); ranks[order] = np.arange(1, m+1)
    qv = pv * m / ranks
    qv_sorted = np.minimum.accumulate(qv[order[::-1]])[::-1]
    q[mask] = np.clip(qv_sorted, 0, 1)
    return q

def per_subtype_fold_enrichment(asd_df: pd.DataFrame,
                                dom_df: pd.DataFrame,
                                labels: np.ndarray,
                                z_thr: float = 1.0,
                                min_n: int = 5,
                                baseline: str = "overall") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fold enrichment of elevated behavioral domains within each subtype.

    Returns two data frames:
      - enrichment: one row per domain x subtype, with 'fold' = within-subtype
        prevalence / baseline prevalence (>1 enrichment, <1 depletion) and a
        directional binomial p value.
      - continuous: per-domain group mean comparison (Welch t test).

    The column 'signed_fold' is retained as an internal plotting convenience
    (depletion stored as -1/fold); after the log transform applied in
    plot_fold_enrichment it is numerically identical to log10(fold), so the
    figure axis is a plain log fold-enrichment scale centred on 1.
    """
    asd_df = asd_df.copy()
    asd_df["subtype"] = labels.astype(int)
    Ks = np.sort(asd_df["subtype"].unique())

    endorsed = {}
    for d in dom_df.columns:
        z = dom_df[d]
        endorsed[d] = (z.abs() > z_thr).astype(float) if d == "PHY" else (z > z_thr).astype(float)

    rows_signed, rows_cont = [], []
    for d in dom_df.columns:
        binvec = endorsed[d]
        z = dom_df[d]

        if baseline == "overall":
            p_base_all = float(binvec.mean(skipna=True))

        for k in Ks:
            m_in = asd_df["subtype"] == k
            m_out = ~m_in

            x_bin = binvec[m_in].dropna()
            y_bin = binvec[m_out].dropna()

            if len(x_bin) >= min_n and (len(y_bin) if baseline=="rest" else len(binvec.dropna())) >= min_n:
                prop_in = float(x_bin.mean())
                p_base  = float(y_bin.mean()) if baseline == "rest" else p_base_all
                if not np.isfinite(p_base) or p_base <= 0: p_base = 1e-9

                p_greater = binomtest(int(x_bin.sum()), n=int(len(x_bin)), p=p_base, alternative="greater").pvalue
                p_less    = binomtest(int(x_bin.sum()), n=int(len(x_bin)), p=p_base, alternative="less").pvalue

                if prop_in >= p_base:
                    pval = p_greater
                    direction = "enriched"
                    fold = prop_in / p_base
                    signed_fold = max(fold, 1.0)
                else:
                    pval = p_less
                    direction = "depleted"
                    fold = prop_in / p_base
                    signed_fold = -max(1.0/fold, 1.0) if fold>0 else -np.inf

                rows_signed.append(dict(
                    domain=d, subtype=int(k), direction=direction,
                    prop_in=prop_in, p_base=p_base, fold=float(fold),
                    signed_fold=float(signed_fold), pval=float(pval),
                    n_in=int(len(x_bin)),
                    n_base=int(len(y_bin) if baseline=="rest" else len(binvec.dropna())),
                    baseline=baseline
                ))

            x = z[m_in].dropna(); y = z[m_out].dropna()
            if len(x) >= min_n and len(y) >= min_n:
                stat, p = ttest_ind(x, y, equal_var=False)
                vx, vy = x.var(ddof=1), y.var(ddof=1)
                s = np.sqrt((vx + vy) / 2.0) if np.isfinite(vx) and np.isfinite(vy) else np.nan
                d_eff = (x.mean() - y.mean()) / s if s and s > 0 else np.nan
                rows_cont.append(dict(
                    domain=d, subtype=int(k),
                    mean_in=float(x.mean()), mean_out=float(y.mean()),
                    cohens_d=float(d_eff), pval=float(p)
                ))

    df_sig = pd.DataFrame(rows_signed)
    df_cont = pd.DataFrame(rows_cont)

    if not df_sig.empty:
        df_sig["qval"] = df_sig["pval"]
        df_sig = df_sig.sort_values(["domain","subtype","pval"])
    if not df_cont.empty:
        df_cont["qval"] = df_cont["pval"]
        df_cont = df_cont.sort_values(["domain","subtype","pval"])

    return df_sig, df_cont

def plot_fold_enrichment(df_sig: pd.DataFrame, out_png: str, alpha_q: float = 0.05, point_size: int = 300):
    if df_sig.empty:
        print("No enrichment results to plot."); return

    subtype_colors = {
        0: "#5CB85C",
        1: "#F0AD4E",
        2: "#7E57C2",
        3: "#17A2B8",
        4: "#A3A3A3",
    }

    dfp = df_sig.copy()

    # x = sign * log10(|fold|)
    def x_coord(sf):
        if not np.isfinite(sf) or sf == 0: return np.nan
        sgn = 1.0 if sf > 0 else -1.0
        val = abs(sf)
        if not np.isfinite(val) or val <= 0: return np.nan
        return sgn * np.log10(val)

    dfp["x"] = dfp["signed_fold"].apply(x_coord)
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna(subset=["x"])
    dfp["point_size"] = point_size

    dom_order = ["PHY","BEH-EMO","COG","RRB","SCL","AGE"]
    dfp["domain"] = pd.Categorical(dfp["domain"], categories=dom_order, ordered=True)

    plt.figure(figsize=(10, 7), dpi=200)
    ax = plt.gca()
    
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 12

    for domain in dom_order:
        domain_data = dfp[dfp["domain"] == domain]
        if domain_data.empty:
            continue
            
        for k, subdf in domain_data.groupby("subtype"):
            c = subtype_colors.get(int(k), "#888888")
            sig = subdf["qval"] < alpha_q
            sub_sig = subdf[sig]
            sub_nsig = subdf[~sig]

            if not sub_sig.empty:
                ax.scatter(sub_sig["x"], sub_sig["domain"],
                           s=sub_sig["point_size"],
                           facecolors=c, edgecolors=c, alpha=0.95, label=None)
            if not sub_nsig.empty:
                ax.scatter(sub_nsig["x"], sub_nsig["domain"],
                           s=sub_nsig["point_size"],
                           facecolors="none", edgecolors=c, linewidths=2.0, alpha=0.95, label=None)

    ax.axvline(0.0, ls="--", lw=2.0, color="gray")
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    xticks = [-2,-1,-0.5,0,0.5,1,2]  # 0.01,0.1,~0.32,1,~3.16,10,100
    xticklabels = ["1/100","1/10","1/3.2","1","3.2","10","100"]
    ax.set_xticks(xticks); ax.set_xticklabels(xticklabels, fontsize=12, fontweight='bold')
    
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=12)
    
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(14)

    handles = []
    for k in sorted(dfp["subtype"].unique()):
        cc = subtype_colors.get(int(k), "#888")
        h = plt.scatter([], [], s=80, facecolors=cc, edgecolors=cc)
        handles.append(h)
    ax.legend(handles, [f"ASD Subtype {int(k)+1}" for k in sorted(dfp["subtype"].unique())],
              frameon=False, loc="upper left", bbox_to_anchor=(1.02,1.0), fontsize=12)

    ax.set_xlabel("Fold enrichment (log scale; <1 = depletion, >1 = enrichment)", 
                  fontsize=14, fontweight='bold')
    ax.set_ylabel("", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print("Saved figure:", out_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abide-pklz", type=str, required=True)
    ap.add_argument("--cmi-pklz", type=str, default=None,
                    help="Path to CMI pklz (optional; omit for ABIDE-only run)")
    ap.add_argument("--step2-outdir", type=str, required=True, help=" abide_asd_labels.npy")
    ap.add_argument("--outdir", type=str, default="unsup_results/abide_behavior_enrichment_domains")
    ap.add_argument("--z-method", type=str, default="robust",
                    choices=["robust","minmax","standard"],
                    help="robust=Median/MAD() | minmax | standard")
    ap.add_argument("--z-thr", type=float, default=1.0, help=" endorsed  z PHY  |z|>thr")
    ap.add_argument("--baseline", type=str, default="overall", choices=["overall","rest"],
                    help="fold  p overall=/ rest=")
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--alpha-q", type=float, default=0.05, help="")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cur_file = __file__
    asd_df = rebuild_abide_asd(args.abide_pklz, args.cmi_pklz, cur_file)
    lab_path = os.path.join(args.step2_outdir, "abide_asd_labels.npy")
    if not os.path.exists(lab_path):
        raise SystemExit(f": {lab_path}")
    labels = np.load(lab_path)
    if len(labels) != len(asd_df):
        raise SystemExit(f"({len(labels)}) ABIDE-ASD ({len(asd_df)})")

    dom_df, used_cols = build_domain_scores(asd_df, z_method=args.z_method)
    dom_df.to_csv(os.path.join(args.outdir, "domain_composite_scores.csv"), index=False)

    df_sig, df_cont = per_subtype_fold_enrichment(
        asd_df, dom_df, labels,
        z_thr=args.z_thr, min_n=args.min_samples, baseline=args.baseline
    )
    out_sig = os.path.join(args.outdir, "enrichment_fold.csv")
    out_cont = os.path.join(args.outdir, "enrichment_continuous.csv")
    df_sig.to_csv(out_sig, index=False)
    df_cont.to_csv(out_cont, index=False)
    print("Saved:", out_sig)
    print("Saved:", out_cont)

    plot_fold_enrichment(df_sig, os.path.join(args.outdir, "fold_enrichment.png"),
                         alpha_q=args.alpha_q)

    with open(os.path.join(args.outdir, "domain_matched_columns.txt"), "w") as f:
        for d, cols in used_cols.items():
            f.write(f"[{d}] ({len(cols)})\n")
            for c in cols: f.write(f"  - {c}\n")
            f.write("\n")

if __name__ == "__main__":
    main()
