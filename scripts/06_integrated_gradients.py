"""
Step6: Prototype-based Integrated Gradients (IG) ROI importance maps

Key fixes vs earlier version:
1) Default IG baseline: ``zero`` (standard, reproducible; ``shuffle_time`` kept for ablation).
2) Default time aggregation: ``signed_mean_time`` so sign is preserved before per-ROI z-score.
3) ``--normalize-mode``: optional group-level normalization (none | zscore | minmax | symmetric_maxabs | dataset_max) on final [K, ROI] maps.
5) ``model.eval()`` re-asserted before the IG inner loop.
6) Warns when a subtype has very few subjects for averaging.

Outputs:
- <outdir>/proto_ig_roi_maps.npy / .csv
- <outdir>/meta.json
- Optional: subject_roi_importance.npy, network CSV/PNG if --plot-network-polar

"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List
import matplotlib.pyplot as plt
import pickle

# Keep subtype colors consistent with original plots:
# Subtype 1 -> green, Subtype 2 -> orange, Subtype 3 -> blue.
SUBTYPE_COLORS = ["#66C2A5", "#FC8D62", "#8DA0CB"]

# sklearn
try:
    from sklearn.preprocessing import OneHotEncoder
except Exception:  # compatibility across sklearn versions
    OneHotEncoder = None

# Local imports from Step1
import sys
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# stDNN model class lives in 02_train_vicreg.py; data utils in utils/
import importlib.util as _ilu, os as _os

def _load_mod(rel_path):
    fp = _os.path.join(CUR_DIR, rel_path)
    spec = _ilu.spec_from_file_location('_mod', fp)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_train_mod = _load_mod('02_train_vicreg.py')
stDNN_Embedding_WithProj = _train_mod.stDNN_Embedding_WithProj

from utils.step7_brain_metrics_abide import (
    load_and_preprocess_data,
    build_site_keys,
)


def load_model(model_path: str, input_channels: int, meta_dim: int, embedding_dim: int, dropout: float, device: torch.device) -> torch.nn.Module:
    """Instantiate model and load weights."""
    model = stDNN_Embedding_WithProj(input_channels, meta_dim, embedding_dim, dropout).to(device)
    state = torch.load(model_path, map_location=device)
    # state can be either state_dict or raw dict of tensors
    if isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values()):
        model.load_state_dict(state, strict=True)
    else:
        # If wrapped (unlikely), try common keys
        sd = state.get('state_dict', state.get('model', state))
        model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def _normalize_sex_series(s, fixed_two_class: bool):
    """Make SEX encoder-safe: no NaN, str dtype (avoids sklearn isnan errors on mixed types)."""
    import pandas as pd

    out = s.copy()
    out = out.where(pd.notna(out), other="unknown")
    out = out.astype(str).str.strip()
    out = out.replace({"nan": "unknown", "NaN": "unknown", "None": "unknown", "": "unknown"})
    if fixed_two_class:
        u = out.str.upper().str.strip()
        out = (
            u.map(
                {
                    "M": "M", "F": "F", "MALE": "M", "FEMALE": "F",
                    "1": "M", "2": "F", "0.m": "M", "0. F": "F",
                }
            )
            .fillna("M")
        )
        out = out.where(out.isin(["M", "F"]), other="M")
    return out


def fit_gender_encoder(df_fit, fixed_two_class: bool = False) -> OneHotEncoder:
    import pandas as pd

    if OneHotEncoder is None:
        raise RuntimeError("scikit-learn is required for OneHotEncoder.")
    try:
        if fixed_two_class:
            ge = OneHotEncoder(sparse_output=False, handle_unknown="ignore", categories=[["M", "F"]])
        else:
            ge = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        if fixed_two_class:
            ge = OneHotEncoder(sparse=False, handle_unknown="ignore", categories=[["M", "F"]])
        else:
            ge = OneHotEncoder(sparse=False, handle_unknown="ignore")
    if df_fit.shape[0] == 0 or "SEX" not in df_fit.columns:
        df_fit = pd.DataFrame({"SEX": ["unknown"]})
    else:
        df_fit = df_fit.copy()
        df_fit["SEX"] = _normalize_sex_series(df_fit["SEX"], fixed_two_class)
    ge.fit(df_fit[["SEX"]])
    return ge


def build_meta_for_row(row, gender_enc, age_fill_value: float, gender_two_class: bool = False) -> np.ndarray:
    import pandas as pd

    sx = row.get("SEX", "unknown")
    tok = _normalize_sex_series(pd.Series([sx]), gender_two_class).iloc[0]
    g = gender_enc.transform([[tok]])
    age_val = row.get("AGE_AT_SCAN", np.nan)
    if not (isinstance(age_val, (int, float)) and np.isfinite(age_val)):
        age_val = age_fill_value
    age = np.array([[float(age_val)]])
    return np.concatenate([g, age], axis=1)


def crop_first100(x_2d: np.ndarray) -> np.ndarray:
    """Input shape [T, ROI] -> crop to first 100 time points if available."""
    if x_2d.shape[0] >= 100:
        return x_2d[:100, :]
    return x_2d


def prototype_score(fx: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
    """s_k(x) = -||f(x) - c_k||^2 for each sample in batch.

    fx: [B, D], centroid: [D]
    returns: [B]
    """
    return -torch.sum((fx - centroid[None, :]) ** 2, dim=1)


def integrated_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    meta: torch.Tensor,
    centroid: torch.Tensor,
    centroid_mean: torch.Tensor | None,
    objective: str,
    steps: int = 32,
    baseline: str = "zero",
) -> torch.Tensor:
    """Compute Integrated Gradients of s_k(x) wrt x.

    x: [B, C, T], meta: [B, M], centroid: [D]
    returns IG wrt x with shape [B, C, T]
    """
    device = x.device
    B, C, T = x.shape
    if baseline == "zero":
        x0 = torch.zeros_like(x)
    elif baseline == "shuffle_time":
        perm = torch.randperm(T, device=device)
        x0 = x[:, :, perm]
    else:
        x0 = torch.zeros_like(x)

    total_grad = torch.zeros_like(x)
    # Use float steps to avoid dtype issues
    for alpha in torch.linspace(0.0, 1.0, steps, device=device):
        x_alpha = (x0 + alpha * (x - x0)).detach().requires_grad_(True)
        f, _ = model(x_alpha, meta)
        if objective == "prototype":
            score = prototype_score(f, centroid).sum()
        elif objective == "k_vs_mean":
            # discriminative direction: f · (c_k - meanC)
            d = centroid - centroid_mean
            score = torch.sum(f * d[None, :], dim=1).sum()
        else:
            score = prototype_score(f, centroid).sum()
        grads = torch.autograd.grad(score, x_alpha, retain_graph=False)[0]
        total_grad += grads.detach()
    ig = (x - x0) * total_grad / float(steps)
    return ig


def aggregate_ig_over_time(ig: torch.Tensor, mode: str = "signed_mean_time") -> torch.Tensor:
    """Reduce [B, C, T] -> [B, C]. Default preserves sign (mean over time)."""
    if mode == "signed_mean_time":
        return ig.mean(dim=2)
    if mode == "abs_mean_time":
        return ig.abs().mean(dim=2)
    if mode == "l2_time":
        return torch.sqrt((ig ** 2).sum(dim=2))
    return ig.mean(dim=2)


def zscore_per_subject(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Row-wise z-score normalization per subject.

    mat: [K, ROI] for one subject
    """
    mean = mat.mean(axis=1, keepdims=True)
    std = mat.std(axis=1, keepdims=True)
    return (mat - mean) / (std + eps)


def minmax01_per_subject_rows(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-subject [K, ROI] min-max normalization to [0, 1] range (row-wise)."""
    out = np.asarray(mat, dtype=np.float32).copy()
    vmin = out.min(axis=1, keepdims=True)
    vmax = out.max(axis=1, keepdims=True)
    span = vmax - vmin
    ok = span[:, 0] >= eps
    out[ok] = (out[ok] - vmin[ok]) / span[ok]
    out[~ok] = 0.0
    return out


def apply_preavg_per_subject(
    subject_importance: np.ndarray, mode: str, eps_z: float = 1e-6, eps_mm: float = 1e-8
) -> np.ndarray:
    N = subject_importance.shape[0]
    out = np.empty_like(subject_importance, dtype=np.float32)
    for i in range(N):
        if mode == "per_subject":
            out[i] = zscore_per_subject(subject_importance[i], eps=eps_z)
        elif mode == "per_subject_minmax":
            out[i] = minmax01_per_subject_rows(subject_importance[i], eps=eps_mm)
        else:
            out[i] = np.asarray(subject_importance[i], dtype=np.float32)
    return out


def group_average_roi_maps(
    subj_imp: np.ndarray, labels_used: np.ndarray, K: int, c_roi: int
) -> np.ndarray:
    roi_maps = np.zeros((K, c_roi), dtype=np.float32)
    for k in range(K):
        mask = labels_used == k
        if mask.sum() == 0:
            roi_maps[k] = 0.0
        else:
            roi_maps[k] = subj_imp[mask, k, :].mean(axis=0)
    return roi_maps


def finalize_subtype_roi_maps(roi_maps: np.ndarray, args) -> np.ndarray:
    out = roi_maps.astype(np.float32, copy=True)
    if args.subtract_global_mean:
        out = out - out.mean(axis=0, keepdims=True)
    out = normalize_group_maps(out, mode=args.normalize_mode)
    if args.abs_after_normalize:
        out = np.abs(out)
    return out


def normalize_subject_centroid_rows_by_max(
    subject_importance: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Per-subject, per-centroid row-max normalization of IG importance.

    subject_importance: [N, K, ROI]
    Rows with max <= eps are left unchanged (near-zero).
    """
    out = np.asarray(subject_importance, dtype=np.float32).copy()
    nflat = out.reshape(-1, out.shape[-1])
    mx = nflat.max(axis=1)
    ok = mx > eps
    nflat[ok] = nflat[ok] / mx[ok, np.newaxis]
    return out


def normalize_symmetric_maxabs(roi_map: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    row = np.asarray(roi_map, dtype=np.float64)
    abs_max = float(np.abs(row).max())
    if abs_max < eps:
        return row.astype(roi_map.dtype, copy=False)
    return (row / abs_max).astype(roi_map.dtype, copy=False)


def normalize_group_maps(roi_maps: np.ndarray, mode: str = "none") -> np.ndarray:
    """Optional normalization of final [K, ROI] group maps (per subtype row)."""
    if mode == "none":
        return roi_maps
    if mode == "dataset_max":
        vmax = float(np.max(roi_maps))
        if vmax < 1e-12:
            return np.zeros_like(roi_maps)
        return roi_maps / vmax
    out = roi_maps.copy()
    for k in range(roi_maps.shape[0]):
        row = roi_maps[k]
        if mode == "zscore":
            mu, sigma = row.mean(), row.std()
            out[k] = (row - mu) / (sigma + 1e-6)
        elif mode == "minmax":
            vmin, vmax = row.min(), row.max()
            span = vmax - vmin
            if span < 1e-8:
                out[k] = np.zeros_like(row)
            else:
                out[k] = 2.0 * (row - vmin) / span - 1.0
        elif mode == "symmetric_maxabs":
            out[k] = normalize_symmetric_maxabs(row)
        else:
            raise ValueError(f"unknown normalize mode: {mode}")
    return out


def load_bn246_to_devatlas6_map(mapping_csv: str) -> np.ndarray:
    """Build ROI(246) -> DevAtlas6 index mapping via BN246 Yeo_7network."""
    import pandas as pd
    # Some mapping files include a title line at row 1, with real header on row 2.
    # Try normal header first, then fallback to header=1.
    df = pd.read_csv(mapping_csv)
    if "Label" not in df.columns or "Yeo_7network" not in df.columns:
        df = pd.read_csv(mapping_csv, header=1)
    if "Label" not in df.columns or "Yeo_7network" not in df.columns:
        raise SystemExit(f" Label/Yeo_7network: {mapping_csv}")
    df = df[["Label", "Yeo_7network"]].copy()
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df["Yeo_7network"] = pd.to_numeric(df["Yeo_7network"], errors="coerce")

    roi_to_group = np.full(246, -1, dtype=int)
    yeo_to_dev6 = {1: 2, 2: 1, 3: 4, 4: 5, 6: 3, 7: 0}
    for _, row in df.dropna().iterrows():
        try:
            roi_label = int(row["Label"])
            y7 = int(row["Yeo_7network"])
        except Exception:
            continue
        if 1 <= roi_label <= 246:
            roi_to_group[roi_label - 1] = yeo_to_dev6.get(y7, -1)
    return roi_to_group


def load_bn246_to_devatlas24_map(mapping_csv: str) -> np.ndarray:
    """Build ROI(246) -> DevAtlas24 net id mapping from overlap-based table.

    Returns int array length 246 with values in {0..24}, where 0 means unmapped.
    """
    import pandas as pd
    df = pd.read_csv(mapping_csv)
    required = {"roi_id", "assigned_net_id"}
    if not required.issubset(set(df.columns)):
        raise SystemExit(f"DevAtlas24 roi_id/assigned_net_id: {mapping_csv}")
    roi_to_net = np.zeros(246, dtype=int)
    for _, row in df.iterrows():
        try:
            rid = int(row["roi_id"])
            nid = int(row["assigned_net_id"])
        except Exception:
            continue
        if 1 <= rid <= 246 and 0 <= nid <= 24:
            roi_to_net[rid - 1] = nid
    return roi_to_net


def parse_devatlas24_parent_labels(labels_txt: str) -> tuple[dict, list, list]:
    """Parse DevAtlas 24RSN parent groups file.

    Returns:
      - net_to_parent: {net_id: parent_name}
      - parent_order: [parent_name...]
      - net_order: [net_id...] ordered by parent then id
    """
    net_to_parent = {}
    parent_order = []
    with open(labels_txt, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",") if p.strip() != ""]
            if len(parts) < 2:
                continue
            parent = parts[0]
            if parent not in parent_order:
                parent_order.append(parent)
            for p in parts[1:]:
                try:
                    nid = int(p)
                except Exception:
                    continue
                if 1 <= nid <= 24:
                    net_to_parent[nid] = parent
    net_order = []
    for parent in parent_order:
        ids = sorted([nid for nid, p in net_to_parent.items() if p == parent])
        net_order.extend(ids)
    return net_to_parent, parent_order, net_order


def build_devatlas6_from_devatlas24_roi_map(
    roi_to_net24: np.ndarray, labels_txt: str
) -> tuple[np.ndarray, list]:
    """Build ROI(246) -> DevAtlas6 index mapping by collapsing DevAtlas24 RSNs into 6 parent systems.

    This guarantees DevAtlas6 is consistent with DevAtlas24 mapping + parent labels.
    Returns:
      - roi_to_group6: int array length 246 with values in {0..5}, -1 for unmapped
      - parent_order: ordered list of 6 parent system names (from labels_txt)
    """
    net_to_parent, parent_order, _ = parse_devatlas24_parent_labels(labels_txt)
    parent_to_idx = {p: i for i, p in enumerate(parent_order)}

    roi_to_group6 = np.full(int(roi_to_net24.shape[0]), -1, dtype=int)
    for i in range(int(roi_to_net24.shape[0])):
        nid = int(roi_to_net24[i])
        if nid <= 0:
            continue
        parent = net_to_parent.get(nid)
        if parent is None:
            continue
        idx = parent_to_idx.get(parent)
        if idx is None:
            continue
        roi_to_group6[i] = int(idx)
    return roi_to_group6, parent_order


def devatlas6_abbrev_names(parent_order: list[str]) -> list[str]:
    """Convert DevAtlas parent system names to the abbreviated names used in plots/CSVs."""
    m = {
        "DefaultMode": "DMN",
        "Sensorimotor": "SMN",
        "Visual": "VN",
        "Control": "Control",
        "DorsalAttention": "DAN",
        "VentralAttention": "VAN",
    }
    return [m.get(p, p) for p in parent_order]


def aggregate_roi_to_network(roi_maps: np.ndarray, roi_to_group: np.ndarray, n_networks: int = 6) -> np.ndarray:
    """Aggregate KxROI to KxNetwork by mean over ROIs within network."""
    K, C = roi_maps.shape
    out = np.zeros((K, n_networks), dtype=np.float32)
    for g in range(n_networks):
        mask = (roi_to_group[:C] == g)
        if mask.sum() == 0:
            out[:, g] = 0.0
        else:
            out[:, g] = roi_maps[:, mask].mean(axis=1)
    return out


def aggregate_roi_to_devatlas24(roi_maps: np.ndarray, roi_to_net24: np.ndarray) -> np.ndarray:
    """Aggregate KxROI to Kx24 DevAtlas subnetworks by mean over assigned ROIs."""
    K, C = roi_maps.shape
    out = np.zeros((K, 24), dtype=np.float32)
    for nid in range(1, 25):
        mask = (roi_to_net24[:C] == nid)
        if mask.sum() > 0:
            out[:, nid - 1] = roi_maps[:, mask].mean(axis=1)
        else:
            out[:, nid - 1] = 0.0
    return out


def _polar_grid_uniform(ax, alpha: float = 0.35, linewidth: float = 0.45) -> None:
    ax.grid(True, alpha=alpha, linewidth=linewidth, linestyle="-")
    try:
        ax.tick_params(
            axis="both",
            which="major",
            grid_linewidth=linewidth,
            grid_alpha=alpha,
            grid_color="0.45",
        )
    except (AttributeError, TypeError):
        pass
    try:
        for gl in ax.yaxis.get_gridlines():
            if gl is not None:
                gl.set_linewidth(linewidth)
                gl.set_alpha(alpha)
                gl.set_color((0.45, 0.45, 0.45, alpha))
                gl.set_linestyle("-")
        for gl in ax.xaxis.get_gridlines():
            if gl is not None:
                gl.set_linewidth(linewidth)
                gl.set_alpha(alpha)
                gl.set_color((0.45, 0.45, 0.45, alpha))
                gl.set_linestyle("-")
    except (AttributeError, TypeError):
        pass
    try:
        sp = ax.spines.get("polar")
        if sp is not None:
            sp.set_linewidth(linewidth)
            sp.set_edgecolor((0.45, 0.45, 0.45, alpha))
    except (AttributeError, KeyError):
        pass


def plot_network_polar_for_3_subtypes(
    network_maps: np.ndarray,
    out_png: str,
    r_min: float | None = None,
    r_max: float | None = None,
):
    """Plot one polar chart with 3 subtype curves on DevAtlas6 networks."""
    if network_maps.shape[0] < 3:
        print("[WARN]  < 3 3-subtype ")
        return
    net_names = ["DefaultMode", "Sensorimotor", "Visual", "Control", "DorsalAttention", "VentralAttention"]
    vals = network_maps[:3, :]
    n = len(net_names)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta = np.concatenate([theta, theta[:1]])

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    if r_min is None:
        r_min = float(np.min(vals))
    if r_max is None:
        r_max = float(np.max(vals))
    if r_max <= r_min:
        r_max = r_min + 1e-6
    ax.set_rlim(r_min, r_max)

    colors = SUBTYPE_COLORS
    for k in range(3):
        y = np.concatenate([vals[k], vals[k, :1]])
        ax.plot(theta, y, color=colors[k], linewidth=2.2, label=f"Subtype {k}", zorder=3)
        ax.fill(theta, y, color=colors[k], alpha=0.12, zorder=2)

    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(net_names, fontsize=10)
    _polar_grid_uniform(ax)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_network_polar_dual_preavg(
    network_maps_z: np.ndarray,
    network_maps_mm: np.ndarray,
    out_png: str,
) -> None:
    if network_maps_z.shape[0] < 3 or network_maps_mm.shape[0] < 3:
        print("[WARN]  < 3 dual ")
        return
    net_names = ["DefaultMode", "Sensorimotor", "Visual", "Control", "DorsalAttention", "VentralAttention"]
    vals_z = network_maps_z[:3, :]
    vals_mm = network_maps_mm[:3, :]
    n = len(net_names)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta = np.concatenate([theta, theta[:1]])
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 8.0), subplot_kw=dict(projection="polar"))
    titles = ("Pre-avg: z-score (246 ROI)", "Pre-avg: min-max [0,1] (246 ROI)")
    for ax, vals, tit in zip(axes, (vals_z, vals_mm), titles):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(tit, pad=14)
        r_min = float(np.min(vals))
        r_max = float(np.max(vals))
        if r_max <= r_min:
            r_max = r_min + 1e-6
        ax.set_rlim(r_min, r_max)
        colors = SUBTYPE_COLORS
        for k in range(3):
            y = np.concatenate([vals[k], vals[k, :1]])
            ax.plot(theta, y, color=colors[k], linewidth=2.2, label=f"Subtype {k}", zorder=3)
            ax.fill(theta, y, color=colors[k], alpha=0.12, zorder=2)
        ax.set_xticks(theta[:-1])
        ax.set_xticklabels(net_names, fontsize=9)
        _polar_grid_uniform(ax)
        ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_subtype_polar_dual_dev6(
    vec_z: np.ndarray,
    vec_mm: np.ndarray,
    labels: list[str],
    out_png: str,
    subtype_idx: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17.0, 7.5), subplot_kw=dict(projection="polar"))
    titles = ("z-score (pre-avg)", "min-max [0,1] (pre-avg)")
    color = SUBTYPE_COLORS[int(subtype_idx) % len(SUBTYPE_COLORS)]
    n = len(labels)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta_c = np.concatenate([theta, theta[:1]])
    for ax, vec, tit in zip(axes, (vec_z, vec_mm), titles):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(tit, pad=12)
        r_min = float(np.min(vec))
        r_max = float(np.max(vec))
        if r_max <= r_min:
            r_max = r_min + 1e-6
        ax.set_rlim(r_min, r_max)
        vals = np.concatenate([vec, vec[:1]])
        ax.plot(theta_c, vals, color=color, linewidth=2.4, zorder=3)
        ax.fill(theta_c, vals, color=color, alpha=0.18, zorder=2)
        ax.set_xticks(theta)
        ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
        _polar_grid_uniform(ax)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_subtype_polar(
    network_values: np.ndarray,
    labels: list[str],
    out_png: str,
    title: str | None = None,
    subtype_idx: int = 0,
    r_min: float | None = None,
    r_max: float | None = None,
):
    """Plot one subtype polar chart."""
    n = len(labels)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    vals = np.concatenate([network_values, network_values[:1]])
    theta_c = np.concatenate([theta, theta[:1]])
    fig = plt.figure(figsize=(9.2, 8.0))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    if r_min is None:
        r_min = float(np.min(network_values))
    if r_max is None:
        r_max = float(np.max(network_values))
    if r_max <= r_min:
        r_max = r_min + 1e-6
    ax.set_rlim(r_min, r_max)
    color = SUBTYPE_COLORS[int(subtype_idx) % len(SUBTYPE_COLORS)]
    ax.plot(theta_c, vals, color=color, linewidth=2.4, zorder=3)
    ax.fill(theta_c, vals, color=color, alpha=0.18, zorder=2)
    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
    _polar_grid_uniform(ax)
    if title:
        ax.set_title(title, pad=18)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_subtype_polar_devatlas24_grouped(
    network_values: np.ndarray,
    net_order: list[int],
    net_to_parent: dict,
    parent_order: list[str],
    out_png: str,
    title: str | None = None,
    subtype_idx: int = 0,
    r_min: float | None = None,
    r_max: float | None = None,
):
    """Plot 24-subnetwork polar chart ordered/grouped by 6 parent systems."""
    vals_ord = np.array([network_values[nid - 1] for nid in net_order], dtype=float)
    labels = [f"RSN{nid}" for nid in net_order]
    n = len(net_order)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n

    parent_colors = {
        "DefaultMode": "#FDE2E4",
        "Sensorimotor": "#DDEDEA",
        "Visual": "#E3F2FD",
        "Control": "#FFF3CD",
        "DorsalAttention": "#E8DAEF",
        "VentralAttention": "#FCE4EC",
    }

    fig = plt.figure(figsize=(11.2, 9.8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    if r_min is None:
        r_min = float(np.min(vals_ord))
    if r_max is None:
        r_max = float(np.max(vals_ord))
    if r_max <= r_min:
        r_max = r_min + 1e-6
    ax.set_rlim(r_min, r_max)

    # background wedges by parent group
    idx = 0
    for parent in parent_order:
        ids = [nid for nid in net_order if net_to_parent.get(nid, "") == parent]
        if len(ids) == 0:
            continue
        start = theta[idx] - width / 2.0
        end = theta[idx + len(ids) - 1] + width / 2.0
        center = (start + end) / 2.0
        ax.bar(
            center,
            1.0,
            width=(end - start),
            bottom=-0.02,
            color=parent_colors.get(parent, "#F0F0F0"),
            alpha=0.35,
            edgecolor="none",
            zorder=0,
            transform=ax.transData,
        )
        idx += len(ids)

    vals_c = np.concatenate([vals_ord, vals_ord[:1]])
    theta_c = np.concatenate([theta, theta[:1]])
    color = SUBTYPE_COLORS[int(subtype_idx) % len(SUBTYPE_COLORS)]
    ax.plot(theta_c, vals_c, color=color, linewidth=2.1, zorder=2)
    ax.fill(theta_c, vals_c, color=color, alpha=0.15, zorder=1)
    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    _polar_grid_uniform(ax)
    if title:
        ax.set_title(title, pad=18)

    # parent labels around outer ring
    idx = 0
    for parent in parent_order:
        ids = [nid for nid in net_order if net_to_parent.get(nid, "") == parent]
        if len(ids) == 0:
            continue
        ang = np.mean(theta[idx: idx + len(ids)])
        abbr = {
            "DefaultMode": "DMN",
            "Sensorimotor": "SMN",
            "Visual": "VN",
            "Control": "Control",
            "DorsalAttention": "DAN",
            "VentralAttention": "VAN",
        }.get(parent, parent)
        ax.text(ang, ax.get_rmax() * 1.36, abbr, ha="center", va="center", fontsize=17, fontweight="bold")
        idx += len(ids)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_subtype_polar_devatlas24_grouped_dual(
    network_values_z: np.ndarray,
    network_values_mm: np.ndarray,
    net_order: list[int],
    net_to_parent: dict,
    parent_order: list[str],
    out_png: str,
    subtype_idx: int = 0,
) -> None:
    parent_colors = {
        "DefaultMode": "#FDE2E4",
        "Sensorimotor": "#DDEDEA",
        "Visual": "#E3F2FD",
        "Control": "#FFF3CD",
        "DorsalAttention": "#E8DAEF",
        "VentralAttention": "#FCE4EC",
    }
    fig, axes = plt.subplots(1, 2, figsize=(23.0, 10.0), subplot_kw=dict(projection="polar"))
    titles = ("Pre-avg z-score (246 ROI)", "Pre-avg min-max [0,1] (246 ROI)")
    color = SUBTYPE_COLORS[int(subtype_idx) % len(SUBTYPE_COLORS)]
    for ax, network_values, tit in zip(axes, (network_values_z, network_values_mm), titles):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(tit, pad=12)
        vals_ord = np.array([network_values[nid - 1] for nid in net_order], dtype=float)
        labels = [f"RSN{nid}" for nid in net_order]
        n = len(net_order)
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        width = 2 * np.pi / n
        r_min = float(np.min(vals_ord))
        r_max = float(np.max(vals_ord))
        if r_max <= r_min:
            r_max = r_min + 1e-6
        ax.set_rlim(r_min, r_max)
        idx = 0
        for parent in parent_order:
            ids = [nid for nid in net_order if net_to_parent.get(nid, "") == parent]
            if len(ids) == 0:
                continue
            start = theta[idx] - width / 2.0
            end = theta[idx + len(ids) - 1] + width / 2.0
            center = (start + end) / 2.0
            ax.bar(
                center,
                1.0,
                width=(end - start),
                bottom=-0.02,
                color=parent_colors.get(parent, "#F0F0F0"),
                alpha=0.35,
                edgecolor="none",
                zorder=0,
                transform=ax.transData,
            )
            idx += len(ids)
        vals_c = np.concatenate([vals_ord, vals_ord[:1]])
        theta_c = np.concatenate([theta, theta[:1]])
        ax.plot(theta_c, vals_c, color=color, linewidth=2.1, zorder=2)
        ax.fill(theta_c, vals_c, color=color, alpha=0.15, zorder=1)
        ax.set_xticks(theta)
        ax.set_xticklabels(labels, fontsize=8, fontweight="bold")
        _polar_grid_uniform(ax)
        idx = 0
        for parent in parent_order:
            ids = [nid for nid in net_order if net_to_parent.get(nid, "") == parent]
            if len(ids) == 0:
                continue
            ang = np.mean(theta[idx: idx + len(ids)])
            abbr = {
                "DefaultMode": "DMN",
                "Sensorimotor": "SMN",
                "Visual": "VN",
                "Control": "Control",
                "DorsalAttention": "DAN",
                "VentralAttention": "VAN",
            }.get(parent, parent)
            ax.text(ang, ax.get_rmax() * 1.36, abbr, ha="center", va="center", fontsize=15, fontweight="bold")
            idx += len(ids)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Step6: Prototype IG ROI maps")
    ap.add_argument("--model-path", type=str, required=True, help="Path to Step1 model weights (.pth)")
    ap.add_argument("--step2-outdir", type=str, required=True, help="Directory containing final_centroids.npy and labels")
    ap.add_argument("--dataset", type=str, choices=["ABIDE", "GENDAAR", "CMI", "STANFORD"], default="GENDAAR", help="Dataset to analyze")
    ap.add_argument("--abide-pklz", type=str, default="DATA/combined_ABIDE_information_with_fMRI.pklz")
    ap.add_argument("--cmi-pklz", type=str, default=None, help="CMI pklz (optional)")
    ap.add_argument("--gendaar-pklz", type=str, default="DATA-GENDAAR/combined_GENDAAR.pklz")
    ap.add_argument("--stanford-pklz", type=str, default="DATA-Stanford/stanford_brainnetome_mean_regMov-6param_dt1_bpf008-09_246ROIs.pklz")
    ap.add_argument("--labels-file", type=str, default=None, help="Override labels filename; else auto (abide_asd_labels.npy or cmi_asd_labels.npy)")
    ap.add_argument("--outdir", type=str, default="unsup_results/step6_proto_ig")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument(
        "--baseline",
        type=str,
        choices=["zero", "shuffle_time"],
        default="zero",
        help="IG baseline: zero (default, standard IG) or shuffle_time (ablation).",
    )
    ap.add_argument(
        "--agg",
        type=str,
        choices=["signed_mean_time", "abs_mean_time", "l2_time"],
        default="signed_mean_time",
        help="Time aggregation: signed_mean_time preserves direction for z-scoring.",
    )
    ap.add_argument(
        "--normalize-mode",
        type=str,
        choices=["none", "zscore", "minmax", "symmetric_maxabs", "dataset_max"],
        default="none",
        help="ROI dataset_max= KxROI [0,1] abs_mean_time ",
    )
    ap.add_argument(
        "--abs-after-normalize",
        action="store_true",
        help=" group-level  minmax  [-1,1] ROI ",
    )
    ap.add_argument("--embedding-dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--meta-dim", type=int, default=None, help="Optional: override meta dim; else auto from encoder + age")
    ap.add_argument("--save-subject-level", action="store_true", help="Save per-subject KxROI IG maps")
    ap.add_argument(
        "--subject-row-max-norm",
        action="store_true",
        help="× [246]  max abs_mean_time per_subject z-score ",
    )
    ap.add_argument(
        "--zscore-mode",
        type=str,
        choices=["per_subject", "per_subject_minmax", "none"],
        default="per_subject",
        help="per_subject:  246 ROI z-scoreper_subject_minmax:  min-max [0,1]"
        "none: ",
    )
    ap.add_argument(
        "--plot-polar-preavg-dual",
        action="store_true",
        help=" --plot-network-polar z-score  [0,1] min-max roi_maps  CSV",
    )
    ap.add_argument("--subtract-global-mean", action="store_true")
    ap.add_argument("--gender-two-class", action="store_true", help="Force gender encoder to 2 classes {M,F} to match checkpoint meta-dim")
    ap.add_argument("--objective", type=str, choices=["prototype", "k_vs_mean"], default="prototype",
                    help="IG target: prototype distance (-||f-c_k||^2) or discriminative dot(f, c_k-meanC)")
    ap.add_argument("--network-map-csv", type=str,
                    default=os.path.join(CUR_DIR, "..", "atlas", "subregion_func_network_Yeo_updated.csv"),
                    help="BN246 mapping CSV containing Label and Yeo_7network")
    ap.add_argument("--devatlas24-map-csv", type=str,
                    default=os.path.join(CUR_DIR, "..", "atlas", "devatlas", "BN246_to_DevAtlas24_mapping.csv"),
                    help="BN246 to DevAtlas24 mapping CSV")
    ap.add_argument("--devatlas24-labels", type=str,
                    default=os.path.join(CUR_DIR, "..", "atlas", "devatlas", "DevAtlas_24RSNS_Labels.txt"),
                    help="DevAtlas 24RSN parent-group labels")
    ap.add_argument("--plot-network-polar", action="store_true",
                    help="Aggregate ROI IG to DevAtlas6 networks and plot polar chart for first 3 subtypes")
    args = ap.parse_args()

    if args.plot_polar_preavg_dual and not args.plot_network_polar:
        raise SystemExit("--plot-polar-preavg-dual  --plot-network-polar")
    if args.plot_polar_preavg_dual and args.subject_row_max_norm:
        raise SystemExit("--plot-polar-preavg-dual  --subject-row-max-norm ")
    if args.plot_polar_preavg_dual and args.zscore_mode == "per_subject_minmax":
        print("[INFO] --plot-polar-preavg-dual  min-max  CSV  z-score ")
    if args.subject_row_max_norm and args.zscore_mode == "per_subject":
        print("[INFO]  --subject-row-max-norm --zscore-mode per_subject")
    elif args.agg == "abs_mean_time" and args.zscore_mode == "per_subject":
        print(
            "[INFO] abs_mean_time + per_subject |IG|  z-score  ROI "
            "/"
        )
    if args.subject_row_max_norm and args.agg != "abs_mean_time":
        print(
            "[WARN] --subject-row-max-norm  --agg abs_mean_time max "
            " agg "
        )

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(args.device)

    # 1) Load centroids and labels
    centroids_path = os.path.join(args.step2_outdir, "final_centroids.npy")
    if not os.path.exists(centroids_path):
        raise SystemExit(f"Not found: {centroids_path}")
    centroids = np.load(centroids_path)  # [K, D]
    K = int(centroids.shape[0])

    if args.labels_file is not None:
        labels_path = os.path.join(args.step2_outdir, args.labels_file)
    elif args.dataset == "ABIDE":
        labels_path = os.path.join(args.step2_outdir, "abide_asd_labels.npy")
    elif args.dataset == "CMI":
        labels_path = os.path.join(args.step2_outdir, "cmi_asd_labels.npy")
    elif args.dataset == "GENDAAR":
        gendaar_local = os.path.join(ROOT_DIR, "unsup_results", "gendaar", "cmi_asd_labels.npy")
        alt7 = os.path.join(ROOT_DIR, "unsup_results", "step7_gendaar", "gendaar_asd_labels_aligned.npy")
        if os.path.isfile(gendaar_local):
            labels_path = gendaar_local
        elif os.path.isfile(alt7):
            labels_path = alt7
        else:
            labels_path = os.path.join(args.step2_outdir, "cmi_asd_labels.npy")
    elif args.dataset == "STANFORD":
        alt = os.path.join(ROOT_DIR, "unsup_results", "step7_stanford", "stanford_asd_labels_aligned.npy")
        labels_path = alt if os.path.isfile(alt) else os.path.join(args.step2_outdir, "cmi_asd_labels.npy")
    else:
        labels_path = os.path.join(args.step2_outdir, "cmi_asd_labels.npy")

    if not os.path.exists(labels_path):
        raise SystemExit(f"Not found: {labels_path}")
    labels = np.load(labels_path).astype(int)

    # 2) Load dataset (reusing Step1's preprocessing to ensure row order for ABIDE/CMI)
    import pandas as pd
    if args.dataset == "ABIDE":
        _cmi_arg = args.cmi_pklz if (args.cmi_pklz and __import__("os").path.exists(args.cmi_pklz)) else None
        abide_df, cmi_df = load_and_preprocess_data(args.abide_pklz, _cmi_arg)
        df_asd = abide_df[abide_df["DX_GROUP"] == 0].copy()
        df_asd.reset_index(drop=True, inplace=True)
    elif args.dataset == "CMI":
        _cmi_arg = args.cmi_pklz if (args.cmi_pklz and __import__("os").path.exists(args.cmi_pklz)) else None
        _, cmi_df = load_and_preprocess_data(args.abide_pklz, _cmi_arg)
        df_asd = cmi_df[cmi_df["DX_GROUP"] == 0].copy()
        df_asd.reset_index(drop=True, inplace=True)
    elif args.dataset == "STANFORD":
        # Align with existing Stanford handling in step7 scripts:
        # - per-subject normalization
        # - quality control on [T, 246], T>=120, finite
        # - ASD selected by label == 'asd'
        with open(args.stanford_pklz, "rb") as f:
            stanford_df = pickle.load(f)

        def _norm_stanford(ts):
            x = np.array(ts, dtype=np.float32)
            return (x - np.mean(x)) / (np.std(x) + 1e-6)

        def _valid_stanford(ts):
            x = np.array(ts)
            return (x.ndim == 2) and (x.shape[0] >= 120) and (x.shape[1] == 246) and np.isfinite(x).all()

        stanford_df = stanford_df.copy()
        stanford_df["data"] = stanford_df["data"].apply(_norm_stanford)
        original_asd_indices = stanford_df[stanford_df["label"] == "asd"].index.values
        stanford_df = stanford_df[stanford_df["data"].apply(_valid_stanford)].copy()

        df_asd = stanford_df[stanford_df["label"] == "asd"].copy().reset_index(drop=True)
        # Ensure meta columns exist for downstream encoder.
        if "SEX" not in df_asd.columns:
            if "sex" in df_asd.columns:
                df_asd["SEX"] = df_asd["sex"].astype(str)
            else:
                df_asd["SEX"] = "unknown"
        if "AGE_AT_SCAN" not in df_asd.columns:
            if "age" in df_asd.columns:
                df_asd["AGE_AT_SCAN"] = pd.to_numeric(df_asd["age"], errors="coerce")
            elif "age_years" in df_asd.columns:
                df_asd["AGE_AT_SCAN"] = pd.to_numeric(df_asd["age_years"], errors="coerce")
            else:
                df_asd["AGE_AT_SCAN"] = np.nan
        stanford_asd_indices_in_qc_df = stanford_df[stanford_df["label"] == "asd"].index.values
        keep_mask = np.isin(original_asd_indices, stanford_asd_indices_in_qc_df)
        if labels.shape[0] != keep_mask.shape[0]:
            print(
                f"[WARN] Stanford ({labels.shape[0]})ASD({keep_mask.shape[0]})"
                f""
            )
            labels = None
        else:
            labels = labels[keep_mask]
    else:  # GENDAAR
        with open(args.gendaar_pklz, "rb") as f:
            df_all = pd.read_pickle(f)
        # Ensure site keys exist (align with Step1 conventions)
        df_all = build_site_keys(df_all, dataset_tag='GENDAAR')
        # Robust DX_GROUP derivation
        if "DX_GROUP" not in df_all.columns:
            if "diagnosis" in df_all.columns:
                df_all["diagnosis"] = df_all["diagnosis"].astype(str).str.lower().str.strip()
                df_all["DX_GROUP"] = df_all["diagnosis"].map({"asd": 0, "td": 1})
            elif "label" in df_all.columns:
                df_all["label"] = df_all["label"].astype(str).str.lower().str.strip()
                df_all["DX_GROUP"] = df_all["label"].map({"asd": 0, "td": 1})
            else:
                raise SystemExit("GENDAAR pklz  DX_GROUP/diagnosis/label ASD ")
        # Ensure SEX/AGE columns exist
        if "SEX" not in df_all.columns and "sex" in df_all.columns:
            df_all["SEX"] = df_all["sex"].map({"male": "M", "female": "F"}).fillna("unknown")
        if "AGE_AT_SCAN" not in df_all.columns and "age_years" in df_all.columns:
            df_all["AGE_AT_SCAN"] = df_all["age_years"]
        df_asd = df_all[df_all["DX_GROUP"] == 0].copy()
        df_asd.reset_index(drop=True, inplace=True)

    if labels is not None and df_asd.shape[0] != labels.shape[0]:
        print(
            f"[WARN] ({labels.shape[0]}) ASD ({df_asd.shape[0]})"
            f""
        )
        labels = None

    import pandas as pd

    if "SEX" not in df_asd.columns:
        df_asd["SEX"] = "unknown"
    df_asd["SEX"] = _normalize_sex_series(df_asd["SEX"], bool(args.gender_two_class))

    # 3) Fit gender encoder & age fill
    gender_enc = fit_gender_encoder(df_asd, fixed_two_class=bool(args.gender_two_class))
    age_fill = float(df_asd["AGE_AT_SCAN"].dropna().mean()) if "AGE_AT_SCAN" in df_asd.columns else 0.0

    # 4) Build model
    # Determine meta dim dynamically: gender one-hot + age(1)
    meta_dim_actual = gender_enc.transform(df_asd[["SEX"]]).shape[1] + 1
    if args.meta_dim is not None and args.meta_dim != meta_dim_actual:
        print(f"[WARN] --meta-dim ({args.meta_dim}) != detected ({meta_dim_actual}); using detected.")
    model = load_model(
        model_path=args.model_path,
        input_channels=246,
        meta_dim=meta_dim_actual,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        device=device,
    )
    model.eval()

    # 5) Iterate in batches and compute IG per centroid
    N = df_asd.shape[0]
    C_roi = 246
    subject_importance = np.zeros((N, K, C_roi), dtype=np.float32)
    assigned_labels = np.zeros((N,), dtype=np.int64)

    def iter_batches(batch_size: int):
        total = N
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            yield start, end

    meanC = torch.tensor(centroids.mean(axis=0), device=device, dtype=torch.float32)

    for start, end in iter_batches(args.batch_size):
        batch_rows = [df_asd.iloc[i] for i in range(start, end)]
        # x: [B, C, T]
        xs = []
        metas = []
        for row in batch_rows:
            x_np = np.asarray(row["data"], dtype=np.float32)
            x_np = crop_first100(x_np)  # [T, C]
            xs.append(torch.from_numpy(x_np.T))  # [C, T]
            metas.append(build_meta_for_row(row, gender_enc, age_fill, bool(args.gender_two_class)))
        x = torch.stack(xs, dim=0).to(device)
        meta = torch.from_numpy(np.vstack(metas)).to(device).float()

        # Fallback subtype assignment by nearest centroid in embedding space.
        with torch.no_grad():
            f_batch, _ = model(x, meta)  # [B, D]
            c_mat = torch.tensor(centroids, device=device, dtype=torch.float32)  # [K, D]
            d2 = torch.cdist(f_batch, c_mat, p=2.0) ** 2  # [B, K]
            assigned_labels[start:end] = torch.argmin(d2, dim=1).detach().cpu().numpy()

        model.eval()
        # Compute IG per centroid
        with torch.inference_mode(False):
            for k in range(K):
                c_k = torch.tensor(centroids[k], device=device, dtype=torch.float32)
                ig = integrated_gradients(
                    model=model,
                    x=x,
                    meta=meta,
                    centroid=c_k,
                    centroid_mean=meanC,
                    objective=args.objective,
                    steps=args.ig_steps,
                    baseline=args.baseline,
                )  # [B, C, T]
                imp = aggregate_ig_over_time(ig, mode=args.agg)  # [B, C]
                subject_importance[start:end, k, :] = imp.detach().cpu().numpy()

    labels_used = labels if labels is not None else assigned_labels
    for k in range(K):
        n_k = int((labels_used == k).sum())
        print(f"  Subtype {k}: {n_k} subjects used for IG averaging.")
        if n_k < 5:
            print(f"  [WARN] Subtype {k} has very few subjects ({n_k}); map may be noisy.")

    roi_maps_mm: np.ndarray | None = None
    subj_imp_for_save_rowmax = None

    if args.subject_row_max_norm:
        subj_imp_work = normalize_subject_centroid_rows_by_max(subject_importance, eps=1e-8)
        subj_imp_for_save_rowmax = subj_imp_work
        roi_raw = group_average_roi_maps(subj_imp_work, labels_used, K, C_roi)
        roi_maps = finalize_subtype_roi_maps(roi_raw, args)
    elif args.plot_polar_preavg_dual:
        subj_z = apply_preavg_per_subject(subject_importance, "per_subject")
        subj_mm = apply_preavg_per_subject(subject_importance, "per_subject_minmax")
        rz = group_average_roi_maps(subj_z, labels_used, K, C_roi)
        rmm = group_average_roi_maps(subj_mm, labels_used, K, C_roi)
        roi_maps = finalize_subtype_roi_maps(rz, args)
        roi_maps_mm = finalize_subtype_roi_maps(rmm, args)
    elif args.zscore_mode == "per_subject":
        subj_imp_work = apply_preavg_per_subject(subject_importance, "per_subject")
        roi_raw = group_average_roi_maps(subj_imp_work, labels_used, K, C_roi)
        roi_maps = finalize_subtype_roi_maps(roi_raw, args)
    elif args.zscore_mode == "per_subject_minmax":
        subj_imp_work = apply_preavg_per_subject(subject_importance, "per_subject_minmax")
        roi_raw = group_average_roi_maps(subj_imp_work, labels_used, K, C_roi)
        roi_maps = finalize_subtype_roi_maps(roi_raw, args)
    else:
        roi_raw = group_average_roi_maps(subject_importance, labels_used, K, C_roi)
        roi_maps = finalize_subtype_roi_maps(roi_raw, args)

    # 7) Save outputs
    np.save(os.path.join(args.outdir, "proto_ig_roi_maps.npy"), roi_maps)
    import pandas as pd
    df_csv = pd.DataFrame(roi_maps, columns=[f"ROI_{i+1}" for i in range(C_roi)])
    df_csv.insert(0, "subtype", list(range(K)))
    df_csv.to_csv(os.path.join(args.outdir, "proto_ig_roi_maps.csv"), index=False)
    if roi_maps_mm is not None:
        np.save(os.path.join(args.outdir, "proto_ig_roi_maps_preavg_minmax01.npy"), roi_maps_mm)
        df_mm = pd.DataFrame(roi_maps_mm, columns=[f"ROI_{i+1}" for i in range(C_roi)])
        df_mm.insert(0, "subtype", list(range(K)))
        df_mm.to_csv(os.path.join(args.outdir, "proto_ig_roi_maps_preavg_minmax01.csv"), index=False)

    if args.save_subject_level:
        np.save(os.path.join(args.outdir, "subject_roi_importance.npy"), subject_importance)
        if args.subject_row_max_norm and subj_imp_for_save_rowmax is not None:
            np.save(os.path.join(args.outdir, "subject_roi_importance_rowmaxnorm.npy"), subj_imp_for_save_rowmax)

    meta_info = dict(
        K=K,
        ROI=C_roi,
        ig_steps=args.ig_steps,
        baseline=args.baseline,
        agg=args.agg,
        zscore_mode=args.zscore_mode,
        gender_two_class=bool(args.gender_two_class),
        plot_polar_preavg_dual=bool(args.plot_polar_preavg_dual),
        subject_row_max_norm=bool(args.subject_row_max_norm),
        normalize_mode=args.normalize_mode,
        subtract_global_mean=bool(args.subtract_global_mean),
        objective=args.objective,
        dataset=args.dataset,
        labels_mode=("provided" if labels is not None else "nearest_centroid"),
        labels_path=os.path.abspath(labels_path),
        centroids_path=os.path.abspath(centroids_path),
        model_path=os.path.abspath(args.model_path),
    )
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    # 8) Optional: network-level aggregation + polar plot (3 subtypes)
    if args.plot_network_polar:
        # DevAtlas6 must be derived from DevAtlas24 mapping + parent labels (NOT Yeo-based CSV).
        roi_to_net24 = load_bn246_to_devatlas24_map(args.devatlas24_map_csv)
        roi_to_group, parent_order = build_devatlas6_from_devatlas24_roi_map(
            roi_to_net24, args.devatlas24_labels
        )
        net_names = devatlas6_abbrev_names(parent_order)
        import pandas as pd

        k_plot = min(3, K)

        if args.plot_polar_preavg_dual and roi_maps_mm is not None:
            network_maps_z = aggregate_roi_to_network(roi_maps, roi_to_group, n_networks=6)
            network_maps_mm = aggregate_roi_to_network(roi_maps_mm, roi_to_group, n_networks=6)
            net_df_z = pd.DataFrame(network_maps_z, columns=net_names)
            net_df_z.insert(0, "subtype", list(range(K)))
            net_df_z.to_csv(
                os.path.join(args.outdir, "proto_ig_network_maps_devatlas6_preavg_zscore.csv"),
                index=False,
            )
            net_df_mm = pd.DataFrame(network_maps_mm, columns=net_names)
            net_df_mm.insert(0, "subtype", list(range(K)))
            net_df_mm.to_csv(
                os.path.join(args.outdir, "proto_ig_network_maps_devatlas6_preavg_minmax01.csv"),
                index=False,
            )
            net_df_z.to_csv(os.path.join(args.outdir, "proto_ig_network_maps_devatlas6.csv"), index=False)
            print(f"Saved network maps (z-score): {args.outdir}/proto_ig_network_maps_devatlas6*.csv")

            slab6 = network_maps_z[:k_plot, :]
            r6_min = float(np.min(slab6))
            r6_max = float(np.max(slab6))
            if r6_max <= r6_min:
                r6_max = r6_min + 1e-12
            plot_network_polar_for_3_subtypes(
                network_maps_z,
                os.path.join(args.outdir, "proto_ig_network_polar_3subtypes.png"),
                r_min=r6_min,
                r_max=r6_max,
            )
            plot_network_polar_dual_preavg(
                network_maps_z,
                network_maps_mm,
                os.path.join(args.outdir, "proto_ig_network_polar_3subtypes_dual.png"),
            )
            for k in range(min(3, K)):
                plot_subtype_polar_dual_dev6(
                    network_maps_z[k],
                    network_maps_mm[k],
                    net_names,
                    os.path.join(args.outdir, f"proto_ig_network6_polar_subtype{k}_dual.png"),
                    subtype_idx=k,
                )
                plot_subtype_polar(
                    network_maps_z[k],
                    labels=net_names,
                    out_png=os.path.join(args.outdir, f"proto_ig_network6_polar_subtype{k}.png"),
                    subtype_idx=k,
                    r_min=r6_min,
                    r_max=r6_max,
                )

            roi_to_net24 = load_bn246_to_devatlas24_map(args.devatlas24_map_csv)
            net24_z = aggregate_roi_to_devatlas24(roi_maps, roi_to_net24)
            net24_mm = aggregate_roi_to_devatlas24(roi_maps_mm, roi_to_net24)
            net24_labels = [f"RSN{i}" for i in range(1, 25)]
            _dfz = pd.DataFrame(net24_z, columns=net24_labels)
            _dfz.insert(0, "subtype", list(range(K)))
            _dfz.to_csv(
                os.path.join(args.outdir, "proto_ig_network_maps_devatlas24_preavg_zscore.csv"),
                index=False,
            )
            _dfm = pd.DataFrame(net24_mm, columns=net24_labels)
            _dfm.insert(0, "subtype", list(range(K)))
            _dfm.to_csv(
                os.path.join(args.outdir, "proto_ig_network_maps_devatlas24_preavg_minmax01.csv"),
                index=False,
            )
            _dfz.to_csv(os.path.join(args.outdir, "proto_ig_network_maps_devatlas24.csv"), index=False)

            slab24 = net24_z[:k_plot, :]
            r24_min = float(np.min(slab24))
            r24_max = float(np.max(slab24))
            if r24_max <= r24_min:
                r24_max = r24_min + 1e-12
            net_to_parent, parent_order, net_order = parse_devatlas24_parent_labels(args.devatlas24_labels)
            for k in range(min(3, K)):
                plot_subtype_polar_devatlas24_grouped(
                    network_values=net24_z[k],
                    net_order=net_order,
                    net_to_parent=net_to_parent,
                    parent_order=parent_order,
                    out_png=os.path.join(args.outdir, f"proto_ig_network24_polar_subtype{k}_grouped.png"),
                    subtype_idx=k,
                    r_min=r24_min,
                    r_max=r24_max,
                )
                plot_subtype_polar_devatlas24_grouped_dual(
                    net24_z[k],
                    net24_mm[k],
                    net_order,
                    net_to_parent,
                    parent_order,
                    os.path.join(args.outdir, f"proto_ig_network24_polar_subtype{k}_grouped_dual.png"),
                    subtype_idx=k,
                )
            print(f"Saved DevAtlas24 network maps under {args.outdir}/")
        else:
            network_maps = aggregate_roi_to_network(roi_maps, roi_to_group, n_networks=6)
            net_df = pd.DataFrame(network_maps, columns=net_names)
            net_df.insert(0, "subtype", list(range(K)))
            net_csv = os.path.join(args.outdir, "proto_ig_network_maps_devatlas6.csv")
            net_df.to_csv(net_csv, index=False)
            slab6 = network_maps[:k_plot, :]
            polar_01 = args.normalize_mode == "dataset_max" or bool(args.subject_row_max_norm)
            if polar_01:
                r6_min, r6_max = 0.0, 1.0
            else:
                r6_min = float(np.min(slab6))
                r6_max = float(np.max(slab6))
            if r6_max <= r6_min:
                r6_max = r6_min + 1e-12
            polar_png = os.path.join(args.outdir, "proto_ig_network_polar_3subtypes.png")
            plot_network_polar_for_3_subtypes(network_maps, polar_png, r_min=r6_min, r_max=r6_max)
            print(f"Saved network maps to: {net_csv}")
            print(f"Saved network polar plot to: {polar_png}")

            for k in range(min(3, K)):
                out6 = os.path.join(args.outdir, f"proto_ig_network6_polar_subtype{k}.png")
                plot_subtype_polar(
                    network_maps[k],
                    labels=net_names,
                    out_png=out6,
                    subtype_idx=k,
                    r_min=r6_min,
                    r_max=r6_max,
                )

            roi_to_net24 = load_bn246_to_devatlas24_map(args.devatlas24_map_csv)
            net24_maps = aggregate_roi_to_devatlas24(roi_maps, roi_to_net24)
            net24_labels = [f"RSN{i}" for i in range(1, 25)]
            net24_df = pd.DataFrame(net24_maps, columns=net24_labels)
            net24_df.insert(0, "subtype", list(range(K)))
            net24_csv = os.path.join(args.outdir, "proto_ig_network_maps_devatlas24.csv")
            net24_df.to_csv(net24_csv, index=False)
            slab24 = net24_maps[:k_plot, :]
            if polar_01:
                r24_min, r24_max = 0.0, 1.0
            else:
                r24_min = float(np.min(slab24))
                r24_max = float(np.max(slab24))
            if r24_max <= r24_min:
                r24_max = r24_min + 1e-12

            net_to_parent, parent_order, net_order = parse_devatlas24_parent_labels(args.devatlas24_labels)
            for k in range(min(3, K)):
                out24 = os.path.join(args.outdir, f"proto_ig_network24_polar_subtype{k}_grouped.png")
                plot_subtype_polar_devatlas24_grouped(
                    network_values=net24_maps[k],
                    net_order=net_order,
                    net_to_parent=net_to_parent,
                    parent_order=parent_order,
                    out_png=out24,
                    subtype_idx=k,
                    r_min=r24_min,
                    r_max=r24_max,
                )
            print(f"Saved DevAtlas24 network maps to: {net24_csv}")

    print(f"Saved ROI maps to: {os.path.join(args.outdir, 'proto_ig_roi_maps.npy')} | CSV exported.")


if __name__ == "__main__":
    main()


