#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 8: Neurobiological Gradient Axis Analysis
================================================
Constructs a one-dimensional neurobiological gradient axis by:
  1. Computing pairwise normalized mean differences (delta-mu / SD_all) between
     groups (S1, S2, S3, TDC) for each metric
  2. Projecting onto 1-D via MDS on the absolute normalized mean differences
  3. Omnibus Kruskal-Wallis (S1 vs S2 vs S3)
  4. Pairwise Mann-Whitney U + FDR correction (S1-S2, S1-S3, S2-S3)
  5. Visualize: each metric = one row, one column per cohort

Metrics (30 total per subject):
  - PC1, PC2, PC3 (embedding PCA projection)
  - ALFF: global + per DevAtlas6 network (7)
  - GBC: All, Positive, Negative (3)
  - Cooperative FC, Competitive FC (2)
  - Network-pair FC: all 15 pairs among 6 DevAtlas6 networks (15)

NOTE: Step6 IG scores are group-level prototypes (one value per subtype),
      not per-subject measurements. They cannot be used in the normalized mean difference / MDS
      analysis (zero within-group variance). They are excluded here.

Usage:
    python3 08_gradient_axis.py
    python3 08_gradient_axis.py --gendaar-pklz DATA-GENDAAR/combined_GENDAAR.pklz

    gradient_axis_01_embedding_pc.png
    gradient_axis_02_alff.png
    gradient_axis_03_gbc.png
    gradient_axis_04_coop_comp_fc.png
    gradient_axis_05_net6_pairwise_fc.png
    gradient_axis_all_metrics_stats.csv

    --pca-dir-override Stanford=unsup_results/other_datasets_pca1_3_group_comparison_final_clean/stanford_pca1_3_group_comparison
"""

from __future__ import annotations

import argparse
import io
import os
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import kruskal, mannwhitneyu
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)

# ─────────────────────────────────────────────
# 0. DATASET CONFIG
# ─────────────────────────────────────────────
# key: display name
# Each dataset must have step7 npy files aligned with subtype label files.
_DATASET_CONFIG = {
    "ABIDE": {
        "step7_dir":   "unsup_results/step7_abide",
        "asd_labels":  "unsup_results/pca_asd_only_plots_final/abide/abide_asd_labels.npy",
        "pca_dir":     "unsup_results/pca_asd_only_plots_final/abide",
        "asd_emb":     "unsup_results/cmi_embroot_fixed_all/abide_asd_emb.npy",
        "td_emb":      "unsup_results/cmi_embroot_fixed_all/abide_td_emb.npy",
    },
    "CMI-HBN": {
        "step7_dir":   "unsup_results/step7_cmi",
        "asd_labels":  "unsup_results/pca_asd_only_plots_final/abide/cmi_asd_labels.npy",
        "pca_dir":     "unsup_results/pca_asd_only_plots_final/cmi",
        "asd_emb":     "unsup_results/cmi_embroot_fixed_all/cmi_asd_emb.npy",
        "td_emb":      "unsup_results/cmi_embroot_fixed_all/cmi_td_emb.npy",
    },
    "GENDAAR": {
        "step7_dir":   "unsup_results/step7_gendaar",
        "asd_labels":  "unsup_results/step7_gendaar/gendaar_asd_labels_aligned.npy",
        "full_asd_labels": "unsup_results/gendaar/cmi_asd_labels.npy",
        "pca_dir":     "unsup_results/pca_asd_only_plots_final/gendaar",
        "asd_emb":     "unsup_results/gendaar_embroot_fixed_all/cmi_asd_emb.npy",
        "td_emb":      "unsup_results/gendaar_embroot_fixed_all/cmi_td_emb.npy",
    },
    "Stanford": {
        "step7_dir":   "unsup_results/step7_stanford",
        "asd_labels":  "unsup_results/step7_stanford/stanford_asd_labels_aligned.npy",
        "pca_dir":     "unsup_results/pca_asd_only_plots_final/stanford",
        "asd_emb":     "unsup_results/stanford_embroot_fixed_all/cmi_asd_emb.npy",
        "td_emb":      "unsup_results/stanford_embroot_fixed_all/cmi_td_emb.npy",
    },
}

NET6_NAMES = ["DMN", "SMN", "VIS", "DAN", "VAN", "CON"]

COLORS = {
    "S1":  "#7BC8A4",
    "S2":  "#E8885A",
    "S3":  "#7B9EC8",
    "TDC": "#AAAAAA",
}
MARKERS = {"S1": "o", "S2": "o", "S3": "o", "TDC": "D"}
GROUPS     = ["S1", "S2", "S3", "TDC"]
ASD_GROUPS = ["S1", "S2", "S3"]
PAIRS      = [("S1", "S2"), ("S1", "S3"), ("S2", "S3")]

# ─────────────────────────────────────────────
# 1. DevAtlas6 ROI MAPPING
# ─────────────────────────────────────────────
def _build_devatlas6_roi_map(root: str) -> np.ndarray:
    """Build ROI(246) -> DevAtlas6 mapping from BN246-to-DevAtlas24 CSV + parent labels.

    The 24 DevAtlas RSNs are collapsed into 6 parent systems:
    DefaultMode, Sensorimotor, Visual, Control, DorsalAttention, VentralAttention.
    """
    map_csv = os.path.join(root, "atlas", "devatlas",
                           "BN246_to_DevAtlas24_mapping.csv")
    lbl_txt = os.path.join(root, "atlas", "devatlas",
                           "DevAtlas_24RSNS_Labels.txt")

    df_map = pd.read_csv(map_csv)
    roi_to_rsn = np.zeros(246, dtype=int)
    for _, row in df_map.iterrows():
        rid = int(row["roi_id"])
        nid = int(row["assigned_net_id"])
        if 1 <= rid <= 246:
            roi_to_rsn[rid - 1] = nid

    rsn_to_parent: dict[int, str] = {}
    parent_order: list[str] = []
    with open(lbl_txt, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = [p.strip() for p in line.strip().split(",") if p.strip()]
            if len(parts) < 2:
                continue
            parent = parts[0]
            if parent not in parent_order:
                parent_order.append(parent)
            for p in parts[1:]:
                try:
                    rsn_to_parent[int(p)] = parent
                except ValueError:
                    pass

    parent_to_idx = {p: i for i, p in enumerate(parent_order)}
    roi_to_net6 = np.full(246, -1, dtype=int)
    for i in range(246):
        rsn = roi_to_rsn[i]
        parent = rsn_to_parent.get(rsn)
        if parent is not None:
            roi_to_net6[i] = parent_to_idx.get(parent, -1)
    return roi_to_net6


def _devatlas_file_index_to_static6(roi_file6: np.ndarray) -> np.ndarray:
    """Remap DevAtlas file-order index (0=DMN,1=SMN,2=VIS,3=CON,4=DAN,5=VAN) to
    the canonical 6-network axis order used in gradient-axis plots.
    """
    file_to_static = np.array([0, 1, 2, 5, 3, 4], dtype=np.int32)
    out = np.full(roi_file6.shape, -1, dtype=np.int32)
    m = roi_file6 >= 0
    out[m] = file_to_static[np.clip(roi_file6[m], 0, 5)]
    return out


# ─────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────
def _abs(root: str, rel: str) -> str:
    return os.path.join(root, rel)


def _safe_load(path: str) -> np.ndarray | None:
    if os.path.isfile(path):
        return np.load(path, allow_pickle=True)
    return None


def _td_guided_global_coop_comp(
    fc_asd: np.ndarray,
    fc_td: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if fc_asd is None or fc_td is None or fc_td.shape[0] < 1:
        return None
    if fc_asd.shape[1:] != fc_td.shape[1:]:
        return None
    mean_td = np.nanmean(fc_td.astype(np.float64), axis=0)
    np.fill_diagonal(mean_td, 0.0)
    mpos = mean_td > 0
    mneg = mean_td < 0
    flat_p = mpos.ravel()
    flat_n = mneg.ravel()
    if flat_p.sum() < 1 or flat_n.sum() < 1:
        return None

    def _batch(fc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = fc.astype(np.float64)
        n = x.shape[0]
        m2 = x.reshape(n, -1)
        coop = np.nanmean(m2[:, flat_p], axis=1)
        comp = np.nanmean(m2[:, flat_n], axis=1)
        return coop, comp

    ca, na = _batch(fc_asd)
    ct, nt = _batch(fc_td)
    return ca, na, ct, nt


def _spread_display_x(
    coords: dict[str, float],
    keys: list[str],
    *,
    x_span: float,
) -> dict[str, float]:
    items = [(g, float(coords[g])) for g in keys if g in coords]
    if len(items) <= 1:
        return dict(items)
    span = max(float(x_span), 1e-6)
    min_sep = max(0.045, 0.018 * span)
    items.sort(key=lambda t: t[1])
    gs = [t[0] for t in items]
    xv = [t[1] for t in items]
    adj = xv[:]
    for i in range(1, len(adj)):
        if adj[i] - adj[i - 1] < min_sep:
            adj[i] = adj[i - 1] + min_sep
    shift = (sum(xv) - sum(adj)) / len(adj)
    adj = [a + shift for a in adj]
    return dict(zip(gs, adj))


def _pca_from_emb(emb: np.ndarray, pca_dir: str) -> np.ndarray | None:
    """Project emb (N x 512) onto stored PCA components; return (N x 3)."""
    comp_path = os.path.join(pca_dir, "pca_components.npy")
    if not os.path.isfile(comp_path):
        return None
    comps = np.load(comp_path)  # (K x 512)
    if comps.ndim != 2 or comps.shape[1] != emb.shape[1]:
        return None
    n_comp = min(3, comps.shape[0])
    return emb @ comps[:n_comp].T  # (N x n_comp)


def _pca_fallback(data: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Compute PCA directly on data matrix; return (N x n_components)."""
    pca = PCA(n_components=min(n_components, data.shape[1]), random_state=42)
    return pca.fit_transform(data)


def _subsequence_row_mask(full_seq: np.ndarray, aligned_seq: np.ndarray) -> np.ndarray | None:
    """Return a boolean mask of len(full_seq) such that full_seq[mask] == aligned_seq,
    if aligned_seq is an order-preserving subsequence of full_seq; else None.
    """
    full_seq = np.asarray(full_seq).astype(int).ravel()
    aligned_seq = np.asarray(aligned_seq).astype(int).ravel()
    n, m = len(full_seq), len(aligned_seq)
    if m > n or m == 0:
        return None
    dp = np.zeros((n + 1, m + 1), dtype=bool)
    dp[:, 0] = True
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = dp[i - 1, j]
            if full_seq[i - 1] == aligned_seq[j - 1] and dp[i - 1, j - 1]:
                dp[i, j] = True
    if not dp[n, m]:
        return None
    mask = np.zeros(n, dtype=bool)
    i, j = n, m
    while j > 0 and i > 0:
        if dp[i - 1, j]:
            i -= 1
        elif full_seq[i - 1] == aligned_seq[j - 1] and dp[i - 1, j - 1]:
            mask[i - 1] = True
            i -= 1
            j -= 1
        else:
            i -= 1
    if j != 0:
        return None
    if not np.array_equal(full_seq[mask], aligned_seq):
        return None
    return mask


def _load_dataframe_pickle_compat(pklz_path: str):
    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core", 1)
            return super().find_class(module, name)

    if not os.path.isfile(pklz_path):
        return None
    with open(pklz_path, "rb") as f:
        payload = f.read()
    try:
        return pickle.loads(payload)
    except ModuleNotFoundError:
        return _CompatUnpickler(io.BytesIO(payload)).load()


def _gendaar_td_qc_mask_from_pklz(pklz_path: str) -> np.ndarray | None:
    if not os.path.isfile(pklz_path):
        return None
    try:
        gendaar_df = _load_dataframe_pickle_compat(pklz_path)
        if gendaar_df is None:
            return None
    except Exception:
        return None

    def norm_gendaar(x):
        x = np.array(x)
        return (x - np.mean(x)) / (np.std(x) + 1e-6)

    def is_valid_gendaar(x):
        x = np.array(x)
        return x.ndim == 2 and x.shape[0] == 165 and np.isfinite(x).all()

    gendaar_df = gendaar_df.copy()
    gendaar_df["data"] = gendaar_df["data"].apply(norm_gendaar)
    original_td_indices = gendaar_df[gendaar_df["label"] == "td"].index.values
    gendaar_df = gendaar_df[gendaar_df["data"].apply(is_valid_gendaar)].copy()
    td_in_qc = gendaar_df[gendaar_df["label"] == "td"].index.values
    return np.isin(original_td_indices, td_in_qc)


def _network_fc(fc_mat: np.ndarray, roi_to_net6: np.ndarray) -> np.ndarray:
    """Compute 15 network-pair mean FC for a batch of FC matrices.

    fc_mat: (N, 246, 246)
    Returns: (N, 15) in the order of all (i,j) pairs with i < j among 6 nets.
    """
    N = fc_mat.shape[0]
    pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    out = np.zeros((N, len(pairs)), dtype=np.float32)
    for pi, (ni, nj) in enumerate(pairs):
        ri = np.where(roi_to_net6 == ni)[0]
        rj = np.where(roi_to_net6 == nj)[0]
        if len(ri) == 0 or len(rj) == 0:
            out[:, pi] = np.nan
            continue
        # mean FC between every ROI pair across networks i and j
        sub = fc_mat[:, np.ix_(ri, rj)[0]][:, :, np.ix_(ri, rj)[1]] \
            if False else fc_mat[:, ri, :][:, :, rj]
        out[:, pi] = sub.reshape(N, -1).mean(axis=1)
    return out


def load_dataset(
    ds_key: str,
    root: str,
    roi_to_net6: np.ndarray,
    gendaar_pklz: str | None = None,
    pca_dir_override: str | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Load all per-subject metrics for one dataset.

    Returns:
        dict mapping group_name -> dict mapping metric_key -> 1-D float array
    """
    cfg = _DATASET_CONFIG[ds_key]
    s7 = _abs(root, cfg["step7_dir"])
    if pca_dir_override:
        pca_dir = (
            pca_dir_override
            if os.path.isabs(pca_dir_override)
            else _abs(root, pca_dir_override)
        )
    else:
        pca_dir = _abs(root, cfg["pca_dir"])
    if pca_dir_override:
        print(f"  [{ds_key}] pca_dir : {pca_dir}")

    asd_labels = _safe_load(_abs(root, cfg["asd_labels"]))
    if asd_labels is None:
        print(f"[WARN] {ds_key}: missing ASD labels, skipping.")
        return {}
    asd_labels = asd_labels.astype(int)  # 0/1/2 -> S1/S2/S3

    n_asd = len(asd_labels)

    # ── load step7 npy ──
    alff_asd = _safe_load(os.path.join(s7, "alffs_asd_gsr.npy"))
    alff_td  = _safe_load(os.path.join(s7, "alffs_td_gsr.npy"))
    gbc_asd  = _safe_load(os.path.join(s7, "gbcs_asd_gsr.npy"))
    gbc_td   = _safe_load(os.path.join(s7, "gbcs_td_gsr.npy"))

    # FC matrices are large; only load if present
    fc_asd_path = os.path.join(s7, "fcs_asd_gsr.npy")
    fc_td_path  = os.path.join(s7, "fcs_td_gsr.npy")
    print(f"  [{ds_key}] Loading FC matrices...")
    fc_asd = _safe_load(fc_asd_path)
    fc_td  = _safe_load(fc_td_path)

    # ── check size consistency ──
    if alff_asd is not None and len(alff_asd) != n_asd:
        print(f"  [{ds_key}] ALFF ASD size mismatch ({len(alff_asd)} vs labels {n_asd})")
        alff_asd = None
    if gbc_asd is not None and len(gbc_asd) != n_asd:
        gbc_asd = None
    if fc_asd is not None and len(fc_asd) != n_asd:
        fc_asd = None

    n_td = alff_td.shape[0] if alff_td is not None else (
        gbc_td.shape[0] if gbc_td is not None else (
            fc_td.shape[0] if fc_td is not None else 0))

    # ── PCA: embeddings ──
    asd_emb_path = _abs(root, cfg.get("asd_emb", ""))
    td_emb_path  = _abs(root, cfg.get("td_emb", ""))
    asd_emb = _safe_load(asd_emb_path) if cfg.get("asd_emb") else None
    td_emb  = _safe_load(td_emb_path)  if cfg.get("td_emb")  else None

    if ds_key == "GENDAAR" and asd_emb is not None and asd_emb.shape[0] != n_asd:
        full_lab_path = cfg.get("full_asd_labels")
        if full_lab_path:
            fl = _safe_load(_abs(root, full_lab_path))
            if fl is not None and len(fl) == asd_emb.shape[0]:
                m_asd = _subsequence_row_mask(fl, asd_labels)
                if m_asd is not None and int(m_asd.sum()) == n_asd:
                    asd_emb = asd_emb[m_asd]
                    print(
                        f"  [{ds_key}] ASD embedding  step7 "
                        f"{asd_emb.shape[0]}×{asd_emb.shape[1]} gendaar_asd_labels_aligned "
                    )
                else:
                    print(f"  [{ds_key}] [WARN]  ASD embedding  embedding ")

    pklz_candidates = []
    if gendaar_pklz:
        pklz_candidates.append(gendaar_pklz if os.path.isabs(gendaar_pklz) else _abs(root, gendaar_pklz))
    pklz_candidates.extend(
        [
            _abs(root, "DATA-GENDAAR/combined_GENDAAR.pklz"),
            _abs(root, "GENDAAR-DATA/combined_asd_td_rest_run1_data.pklz"),
        ]
    )

    if ds_key == "GENDAAR" and td_emb is not None and n_td > 0 and td_emb.shape[0] != n_td:
        td_mask = None
        for pz in pklz_candidates:
            td_mask = _gendaar_td_qc_mask_from_pklz(pz)
            if td_mask is not None and len(td_mask) == td_emb.shape[0] and int(td_mask.sum()) == n_td:
                td_emb = td_emb[td_mask]
                print(
                    f"  [{ds_key}] TD embedding  pklz QC {td_emb.shape[0]}×{td_emb.shape[1]}{pz}"
                )
                break
        if td_emb is not None and td_emb.shape[0] != n_td:
            print(
                f"  [{ds_key}] TD embedding  {td_emb.shape[0]}  step7 TD {n_td} "
                f"TDC  PC  ALFF PCA step7 "
            )

    # Validate embedding vs label alignment
    pc_asd, pc_td = None, None
    if asd_emb is not None and asd_emb.shape[0] == n_asd:
        pc_asd = _pca_from_emb(asd_emb, pca_dir)
    if pc_asd is None and alff_asd is not None:
        print(f"  [{ds_key}] Embedding PCAASD ALFF-PCA")
        pc_asd = _pca_fallback(alff_asd, 3)

    if td_emb is not None and n_td > 0 and td_emb.shape[0] == n_td:
        pc_td = _pca_from_emb(td_emb, pca_dir)
    if pc_td is None and alff_td is not None:
        pc_td = _pca_fallback(alff_td, 3)

    td_guided = _td_guided_global_coop_comp(fc_asd, fc_td)
    coop_asd_full: np.ndarray | None = None
    comp_asd_full: np.ndarray | None = None
    coop_td_full: np.ndarray | None = None
    comp_td_full: np.ndarray | None = None
    if td_guided is not None:
        coop_asd_full, comp_asd_full, coop_td_full, comp_td_full = td_guided
        print(
            f"  [{ds_key}] Coop/Comp FCTDC  ASD_Brain_Measures_detailed / Fig.4K–L "
        )

    # ── helper: per-network aggregation ──
    def _net6_mean(data: np.ndarray) -> np.ndarray:
        """data (N, 246) -> (N, 6)."""
        out = np.zeros((len(data), 6), dtype=np.float32)
        for ni in range(6):
            mask = roi_to_net6 == ni
            out[:, ni] = data[:, mask].mean(axis=1) if mask.sum() > 0 else np.nan
        return out

    def _per_subject_coop_comp_fallback(fc_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        triu_r, triu_c = np.triu_indices(246, k=1)
        fv = fc_mat[:, triu_r, triu_c].astype(np.float64)
        coop = np.where(fv > 0, fv, 0.0).mean(axis=1)
        comp = np.where(fv < 0, fv, 0.0).mean(axis=1)
        return coop, comp

    # ── Build per-group metric dict ──
    def _build_group(
        idx: np.ndarray | None,
        pc: np.ndarray | None,
        alff: np.ndarray | None,
        gbc: np.ndarray | None,
        fc: np.ndarray | None,
        coop_full: np.ndarray | None = None,
        comp_full: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        metrics: dict[str, np.ndarray] = {}
        n = (len(idx) if idx is not None else
             (len(alff) if alff is not None else
              (len(gbc) if gbc is not None else 0)))
        if n == 0:
            return metrics

        # PC1/2/3
        if pc is not None:
            sel = pc[idx] if idx is not None else pc
            for i in range(min(3, sel.shape[1])):
                metrics[f"PC{i+1}"] = sel[:, i].astype(np.float64)

        # ALFF global + per network
        if alff is not None:
            sel = alff[idx] if idx is not None else alff
            metrics["ALFF_global"] = sel.mean(axis=1).astype(np.float64)
            net_alff = _net6_mean(sel)
            for ni, nn in enumerate(NET6_NAMES):
                metrics[f"ALFF_{nn}"] = net_alff[:, ni].astype(np.float64)

        # GBC all/pos/neg
        if gbc is not None:
            sel = gbc[idx] if idx is not None else gbc
            metrics["GBC_all"] = sel[:, :, 0].mean(axis=1).astype(np.float64)
            metrics["GBC_pos"] = sel[:, :, 1].mean(axis=1).astype(np.float64)
            metrics["GBC_neg"] = sel[:, :, 2].mean(axis=1).astype(np.float64)

        # Coop/Comp FC + 15 network-pair FCs
        if fc is not None:
            sel_fc = fc[idx] if idx is not None else fc
            if coop_full is not None and comp_full is not None:
                if idx is not None:
                    metrics["Coop_FC"] = coop_full[idx].astype(np.float64)
                    metrics["Comp_FC"] = comp_full[idx].astype(np.float64)
                else:
                    metrics["Coop_FC"] = coop_full.astype(np.float64)
                    metrics["Comp_FC"] = comp_full.astype(np.float64)
            else:
                c1, c2 = _per_subject_coop_comp_fallback(sel_fc)
                metrics["Coop_FC"] = c1.astype(np.float64)
                metrics["Comp_FC"] = c2.astype(np.float64)
            net_fc = _network_fc(sel_fc, roi_to_net6)
            for pi, (ni, nj) in enumerate([(i, j) for i in range(6) for j in range(i+1, 6)]):
                key = f"FC_{NET6_NAMES[ni]}_{NET6_NAMES[nj]}"
                metrics[key] = net_fc[:, pi].astype(np.float64)

        return metrics

    groups_data: dict[str, dict[str, np.ndarray]] = {}

    # ASD subtypes
    for subtype_k, group_name in enumerate(["S1", "S2", "S3"]):
        idx = np.where(asd_labels == subtype_k)[0]
        if len(idx) == 0:
            continue
        groups_data[group_name] = _build_group(
            idx,
            pc_asd,
            alff_asd,
            gbc_asd,
            fc_asd,
            coop_asd_full,
            comp_asd_full,
        )

    # TDC (no idx slicing needed, load full arrays)
    if n_td > 0:
        groups_data["TDC"] = _build_group(
            None,
            pc_td,
            alff_td,
            gbc_td,
            fc_td,
            coop_td_full,
            comp_td_full,
        )

    return groups_data


# ─────────────────────────────────────────────
# 3. STATISTICS
# ─────────────────────────────────────────────
def fdr_bh(p_values: list[float]) -> np.ndarray:
    p = np.array(p_values)
    n = len(p)
    order = np.argsort(p)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    p_adj = np.minimum(1.0, p * n / ranked)
    for i in range(n - 2, -1, -1):
        p_adj[order[i]] = min(p_adj[order[i]], p_adj[order[i + 1]])
    return p_adj


def normalized_mean_difference(a: np.ndarray, b: np.ndarray,
                               all_vals: np.ndarray) -> float:
    """Normalized mean difference between two groups: (mean_a - mean_b) / SD_all.

    The denominator is the standard deviation of the combined four-group sample
    (std(all_vals)), which keeps the metric stable when group sizes are unequal
    and when cohorts are small. Note that this is NOT Cohen's d: the denominator
    is not a pooled within-group standard deviation.

    For backward compatibility the value is still stored under the dictionary key
    'cohens_d' and exported in CSV columns named 'd_*'; those labels refer to the
    normalized mean difference defined here.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    all_vals = np.asarray(all_vals, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    all_vals = all_vals[np.isfinite(all_vals)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    if len(all_vals) < 2:
        return np.nan
    sd_all = all_vals.std(ddof=1)
    if not np.isfinite(sd_all) or sd_all < 1e-10:
        return 0.0
    return (a.mean() - b.mean()) / sd_all


def _orient_mds_coords_with_means(
    coords: dict[str, float],
    grp_vals: dict[str, np.ndarray],
) -> dict[str, float]:
    present: list[str] = []
    for g in ASD_GROUPS:
        if g not in grp_vals:
            continue
        v = np.asarray(grp_vals[g], dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        if len(v) >= 1:
            present.append(g)
    if len(present) < 2:
        return coords
    means = np.array([float(np.nanmean(grp_vals[g])) for g in present], dtype=np.float64)
    xs = np.array([coords[g] for g in present], dtype=np.float64)
    if np.nanstd(means) < 1e-12 or np.nanstd(xs) < 1e-12:
        return coords
    r = float(np.corrcoef(means, xs)[0, 1])
    if np.isfinite(r) and r < 0:
        return {g: -float(coords[g]) for g in coords}
    return coords


def compute_metric_stats(grp_vals: dict[str, np.ndarray]) -> dict:
    """Compute MDS + KW + pairwise stats for one metric in one cohort."""
    chunks = [
        np.asarray(grp_vals[g], dtype=np.float64).ravel()
        for g in GROUPS if g in grp_vals
    ]
    all_vals = (
        np.concatenate(chunks)
        if chunks
        else np.array([], dtype=np.float64)
    )

    n_grp = len(GROUPS)
    d_matrix = np.zeros((n_grp, n_grp))
    for i, gi in enumerate(GROUPS):
        for j, gj in enumerate(GROUPS):
            if i != j and gi in grp_vals and gj in grp_vals:
                d_matrix[i, j] = abs(
                    normalized_mean_difference(grp_vals[gi], grp_vals[gj], all_vals)
                )

    try:
        mds = MDS(n_components=1, dissimilarity="precomputed",
                  random_state=42, n_init=20, max_iter=1000)
        coords_raw = mds.fit_transform(d_matrix)[:, 0]
    except Exception:
        coords_raw = np.zeros(n_grp)
    coords = {g: float(coords_raw[i]) for i, g in enumerate(GROUPS)}
    coords = _orient_mds_coords_with_means(coords, grp_vals)

    asd_avail = [grp_vals[g] for g in ASD_GROUPS if g in grp_vals and len(grp_vals[g]) >= 3]
    if len(asd_avail) >= 2:
        try:
            H, p_omni = kruskal(*asd_avail)
        except Exception:
            H, p_omni = np.nan, np.nan
    else:
        H, p_omni = np.nan, np.nan

    raw_p = []
    for ga, gb in PAIRS:
        if ga in grp_vals and gb in grp_vals and len(grp_vals[ga]) >= 3 and len(grp_vals[gb]) >= 3:
            _, p = mannwhitneyu(grp_vals[ga], grp_vals[gb], alternative="two-sided")
        else:
            p = np.nan
        raw_p.append(p)
    valid = [p for p in raw_p if not np.isnan(p)]
    p_fdr = fdr_bh(valid) if valid else np.array([np.nan] * len(raw_p))
    fdr_iter = iter(p_fdr)
    pair_results = {}
    for k, (ga, gb) in enumerate(PAIRS):
        p_raw = raw_p[k]
        p_adj = float(next(fdr_iter)) if not np.isnan(p_raw) else np.nan
        d_val = normalized_mean_difference(
            grp_vals.get(ga, np.array([])),
            grp_vals.get(gb, np.array([])),
            all_vals,
        )
        pair_results[(ga, gb)] = {"p_raw": p_raw, "p_fdr": p_adj, "cohens_d": d_val}

    return {"coords": coords, "H": H, "p_omni": p_omni,
            "pair_results": pair_results, "d_matrix": d_matrix}


def p_to_stars(p: float) -> str:
    if np.isnan(p): return "n.a."
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    return "n.s."


# ─────────────────────────────────────────────
# 4. BUILD METRIC LIST
# ─────────────────────────────────────────────
def _build_metrics() -> list[dict]:
    ml = []
    ml += [{"key": f"PC{i}", "label": f"Embedding PC{i}"} for i in range(1, 4)]
    ml.append({"key": "ALFF_global", "label": "Global ALFF"})
    for nn in NET6_NAMES:
        ml.append({"key": f"ALFF_{nn}", "label": f"ALFF ({nn})"})
    ml += [
        {"key": "GBC_all", "label": "GBC (All)"},
        {"key": "GBC_pos", "label": "GBC (Positive)"},
        {"key": "GBC_neg", "label": "GBC (Negative)"},
        {"key": "Coop_FC", "label": "Cooperative FC"},
        {"key": "Comp_FC", "label": "Competitive FC"},
    ]
    pairs15 = [(NET6_NAMES[i], NET6_NAMES[j]) for i in range(6) for j in range(i+1, 6)]
    for ni, nj in pairs15:
        ml.append({"key": f"FC_{ni}_{nj}", "label": f"FC: {ni}–{nj}"})
    return ml


def _metric_figure_groups() -> list[tuple[str, str, list[dict]]]:
    full = _build_metrics()
    return [
        (
            "01_embedding_pc",
            "Embedding PCA (PC1–PC3)",
            [m for m in full if m["key"].startswith("PC")],
        ),
        (
            "02_alff",
            "ALFF (global + DevAtlas-6)",
            [m for m in full if m["key"].startswith("ALFF_")],
        ),
        (
            "03_gbc",
            "GBC (All / Positive / Negative)",
            [m for m in full if m["key"].startswith("GBC_")],
        ),
        (
            "04_coop_comp_fc",
            "Cooperative & Competitive FC (whole-brain upper triangle)",
            [m for m in full if m["key"] in ("Coop_FC", "Comp_FC")],
        ),
        (
            "05_net6_pairwise_fc",
            "Six-network pairwise mean FC (15 edges)",
            [m for m in full if m["key"].startswith("FC_")],
        ),
    ]


def _xlim_for_metrics(
    results: dict[str, dict[str, dict]],
    cohorts: list[str],
    metrics: list[dict],
    pad: float = 0.6,
) -> tuple[float, float]:
    coords: list[float] = []
    for ds in cohorts:
        for m in metrics:
            r = results[ds].get(m["key"])
            if r:
                coords += list(r["coords"].values())
    if not coords:
        return -1.0, 1.0
    return min(coords) - pad, max(coords) + pad


def plot_gradient_axis_figure(
    metrics: list[dict],
    results: dict[str, dict[str, dict]],
    cohorts: list[str],
    out_png: str,
    suptitle: str,
    x_min: float,
    x_max: float,
) -> None:
    n_m = len(metrics)
    n_c = len(cohorts)
    if n_m == 0 or n_c == 0:
        return

    fig, axes = plt.subplots(
        n_m, n_c,
        figsize=(4.2 * n_c, 1.35 * n_m),
        sharey=False,
    )
    if n_c == 1:
        axes = axes[:, np.newaxis]
    if n_m == 1:
        axes = axes[np.newaxis, :]

    for col, ds in enumerate(cohorts):
        for row, m in enumerate(metrics):
            ax = axes[row, col]
            res = results[ds].get(m["key"])

            if res is None:
                ax.axis("off")
                if col == 0:
                    ax.set_ylabel(m["label"], fontsize=8, rotation=0,
                                  ha="right", va="center", labelpad=6)
                if row == 0:
                    ax.set_title(ds, fontsize=10, fontweight="bold", pad=8)
                continue

            coords = res["coords"]
            pr = res["pair_results"]

            ax.axhline(0, color="#dddddd", lw=1.0, zorder=0)
            present = [g for g in GROUPS if g in coords]
            x_span_axis = max(x_max - x_min, 1e-6)
            plot_x = _spread_display_x(coords, present, x_span=x_span_axis)
            if len(present) >= 2:
                xs = sorted([plot_x[g] for g in present])
                ax.plot([xs[0], xs[-1]], [0, 0],
                        color="#e0e0e0", lw=2.5, zorder=1, solid_capstyle="round")

            for g in present:
                ax.plot(plot_x[g], 0,
                        marker=MARKERS[g],
                        color=COLORS[g],
                        markersize=9 if g != "TDC" else 7,
                        markeredgecolor="white",
                        markeredgewidth=0.7,
                        zorder=4, clip_on=False)

            bracket_heights = [0.30, 0.48, 0.66]
            pair_order = [("S2", "S3"), ("S1", "S3"), ("S1", "S2")]
            for bh, (ga, gb) in zip(bracket_heights, pair_order):
                key_pr = (ga, gb) if (ga, gb) in pr else (gb, ga)
                if key_pr not in pr:
                    continue
                if ga not in plot_x or gb not in plot_x:
                    continue
                p_val = pr[key_pr]["p_fdr"]
                stars = p_to_stars(p_val)
                d_val = pr[key_pr]["cohens_d"]
                x0, x1 = plot_x[ga], plot_x[gb]
                ax.annotate("", xy=(x1, bh), xytext=(x0, bh),
                            arrowprops=dict(arrowstyle="-",
                                            color="#888888", lw=0.8),
                            annotation_clip=False)
                d_str = f"{d_val:.2f}" if not np.isnan(d_val) else "n.a."
                ax.text((x0 + x1) / 2, bh + 0.03,
                        f"{stars}\nd={d_str}",
                        ha="center", va="bottom",
                        fontsize=6, color="#444444", linespacing=1.2)

            p_omni = res["p_omni"]
            omni_str = ("n.a." if np.isnan(p_omni) else
                        (f"KW p={p_omni:.3f}" if p_omni >= 0.001 else "KW p<0.001"))
            ax.text(0.99, 0.02, omni_str, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=5.5, color="#777777")

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-0.35, 0.90)
            ax.set_yticks([])
            ax.set_xticks([])
            if col == 0:
                ax.set_ylabel(m["label"], fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=6)
            if row == 0:
                ax.set_title(ds, fontsize=10, fontweight="bold", pad=8)
            for sp in ax.spines.values():
                sp.set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker="o" if g != "TDC" else "D",
               color="w", markerfacecolor=COLORS[g], markersize=8,
               label={"S1": "Subtype 1", "S2": "Subtype 2",
                      "S3": "Subtype 3", "TDC": "TDC"}[g])
        for g in GROUPS
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=4,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.text(
        0.5,
        -0.10,
        "Axis: 1D MDS on pairwise normalized mean difference (delta-mu / SD_all)  |  "
        "Brackets: FDR pairwise Mann-Whitney U  |  KW = omnibus (S1/S2/S3)",
        ha="center",
        fontsize=7,
        color="#888888",
    )
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.005)

    plt.tight_layout(h_pad=0.4, w_pad=0.5)
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()
    print(f": {out_png}")


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",   type=str, default=ROOT_DIR)
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(ROOT_DIR, "unsup_results", "step9_gradient_axis"))
    ap.add_argument("--datasets", nargs="*",
                    default=list(_DATASET_CONFIG.keys()))
    ap.add_argument("--abide-step7-dir", type=str, default=None,
                    help="Override ABIDE step7_dir (e.g. smoke_test_results/step07)")
    ap.add_argument("--abide-step3-dir", type=str, default=None,
                    help="Override ABIDE asd_labels path (directory containing abide_asd_labels.npy)")
    ap.add_argument(
        "--gendaar-pklz",
        type=str,
        default=os.path.join(ROOT_DIR, "DATA-GENDAAR", "combined_GENDAAR.pklz"),
        help="GENDAAR  pklz step7_gendaar_compute_only  TD embedding QC  TD ",
    )
    ap.add_argument(
        "--pca-dir-override",
        action="append",
        default=[],
        metavar="=",
        help=" _DATASET_CONFIG  pca_dir pca_components.npy ",
    )
    ap.add_argument(
        "--also-combined",
        action="store_true",
        help=" gradient_axis_all_metrics.png",
    )
    args = ap.parse_args()

    # Apply path overrides for smoke-test mode
    if args.abide_step7_dir:
        _DATASET_CONFIG["ABIDE"]["step7_dir"] = args.abide_step7_dir
    if args.abide_step3_dir:
        _DATASET_CONFIG["ABIDE"]["asd_labels"] = os.path.join(
            args.abide_step3_dir, "abide_asd_labels.npy"
        )
        _DATASET_CONFIG["ABIDE"]["pca_dir"] = args.abide_step3_dir
    if args.abide_step7_dir or args.abide_step3_dir:
        # Only run ABIDE in smoke-test mode
        args.datasets = ["ABIDE"]

    pca_dir_overrides: dict[str, str] = {}
    for raw in args.pca_dir_override:
        if "=" not in raw:
            print(f"[WARN]  --pca-dir-override '=': {raw}")
            continue
        name, path = raw.split("=", 1)
        name = name.strip()
        path = path.strip()
        if name:
            pca_dir_overrides[name] = path

    os.makedirs(args.outdir, exist_ok=True)

    print(" DevAtlas6 ROI  step9_*_static ...")
    roi_file6 = _build_devatlas6_roi_map(args.root)
    roi_to_net6 = _devatlas_file_index_to_static6(roi_file6)

    METRICS   = _build_metrics()
    COHORTS   = [d for d in args.datasets if d in _DATASET_CONFIG]

    # ── Load all data ──
    print("...")
    all_grp_data: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for ds in COHORTS:
        print(f"  {ds}...")
        gendaar_pklz_use = args.gendaar_pklz
        if (
            gendaar_pklz_use
            and (not os.path.isabs(gendaar_pklz_use))
            and not os.path.isfile(gendaar_pklz_use)
        ):
            gendaar_pklz_use = os.path.join(args.root, gendaar_pklz_use)
        if gendaar_pklz_use and not os.path.isfile(gendaar_pklz_use):
            print(f"[INFO]  GENDAAR pklz: {gendaar_pklz_use} TD embedding QC ")
            gendaar_pklz_use = None
        all_grp_data[ds] = load_dataset(
            ds,
            args.root,
            roi_to_net6,
            gendaar_pklz=gendaar_pklz_use,
            pca_dir_override=pca_dir_overrides.get(ds),
        )

    # ── Compute stats ──
    print("...")
    results: dict[str, dict[str, dict]] = {ds: {} for ds in COHORTS}
    for ds in COHORTS:
        grp_data = all_grp_data[ds]
        for m in METRICS:
            key = m["key"]
            grp_vals = {g: grp_data[g][key]
                        for g in GROUPS
                        if g in grp_data and key in grp_data[g]}
            if len(grp_vals) < 2:
                results[ds][key] = None
                continue
            results[ds][key] = compute_metric_stats(grp_vals)

    probe_coords: list[float] = []
    for ds in COHORTS:
        for m in METRICS:
            r = results[ds].get(m["key"])
            if r:
                probe_coords += list(r["coords"].values())
    if not probe_coords:
        print("[ERROR] ")
        return

    print("...")
    for stem, section_title, m_sub in _metric_figure_groups():
        if not m_sub:
            continue
        x_lo, x_hi = _xlim_for_metrics(results, COHORTS, m_sub, pad=0.6)
        out_png = os.path.join(args.outdir, f"gradient_axis_{stem}.png")
        plot_gradient_axis_figure(
            m_sub,
            results,
            COHORTS,
            out_png,
            f"Neurobiological Gradient Axis  ·  {section_title}",
            x_lo,
            x_hi,
        )

    if args.also_combined:
        x_min = min(probe_coords) - 0.6
        x_max = max(probe_coords) + 0.6
        out_all = os.path.join(args.outdir, "gradient_axis_all_metrics.png")
        plot_gradient_axis_figure(
            METRICS,
            results,
            COHORTS,
            out_all,
            "Neurobiological Gradient Axis  ·  All Metrics",
            x_min,
            x_max,
        )

    # ── Summary CSV ──
    rows = []
    for ds in COHORTS:
        for m in METRICS:
            res = results[ds].get(m["key"])
            if res is None:
                continue
            row = {
                "Cohort":  ds,
                "Metric":  m["label"],
                "KW_H":    round(float(res["H"]),     4) if not np.isnan(res["H"])     else np.nan,
                "KW_p":    round(float(res["p_omni"]),6) if not np.isnan(res["p_omni"]) else np.nan,
                "KW_sig":  p_to_stars(res["p_omni"]),
            }
            for g in GROUPS:
                row[f"MDS_{g}"] = round(res["coords"].get(g, np.nan), 4)
            for ga, gb in PAIRS:
                tag = f"{ga}_{gb}"
                pr  = res["pair_results"].get((ga, gb), {})
                row[f"d_{tag}"]     = round(float(pr.get("cohens_d", np.nan)), 4)
                row[f"p_fdr_{tag}"] = round(float(pr.get("p_fdr",    np.nan)), 6)
                row[f"sig_{tag}"]   = p_to_stars(pr.get("p_fdr", np.nan))
            rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.outdir, "gradient_axis_all_metrics_stats.csv")
    df.to_csv(out_csv, index=False)
    print(f": {out_csv}")
    print(f"\n: {args.outdir}")


if __name__ == "__main__":
    main()
