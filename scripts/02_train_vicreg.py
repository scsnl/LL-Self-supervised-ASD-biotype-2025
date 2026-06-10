#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: Train & Select & Validate – VICReg-stDNN for ASD Subtyping

Scanner-invariant self-supervised training on ABIDE (1+2) ASD-only data using
Variance–Invariance–Covariance Regularization (VICReg).

Two interchangeable site-removal interfaces are provided (--site-removal):
  * orthogonal (default): a scanner-proxy branch produces a site vector and the
    biological embedding is orthogonalized against it (orthogonal projection +
    orthogonal penalty). This is the main pipeline used throughout the study.
  * adversarial: a site-prediction head attached to the embedding via a
    Gradient Reversal Layer (DANN-style); the encoder is trained to make the
    embedding non-predictive of site (SITE_KEY_BASE). Opt-in alternative.

Train & Select & Validate for ASD Subtyping (VICReg + Orthogonal site removal)

Features
- Data: ABIDE(1+2 merged) + external CMI
- Modes:
  * grouped split (by SITE_KEY_STRICT): Train/Val/Test once
  * LOSO-CV (leave-one-site-out): each site is a test fold; within remaining sites, sample a Val group
- Unsupervised training on ASD only (no labels), TD kept for later mapping/UMAP
- K sweep on Val-ASD for K in [kmin..kmax] using combined score:
    Score = z(Stability) + 0.5*z(Silhouette) - 0.5*z(CramérV) - gamma*(K-2)
- Stability metrics: Bootstrap AMI (on Train+Val), LOSO-by-site AMI (on Train+Val)
- External evaluation: ABIDE-Test-ASD & CMI-ASD (silhouette, Cramér’s V)
- Optional final fit on ALL ABIDE-ASD with best hyperparams; export embeddings for ASD/TD/CMI

Outputs (under ./asd_unsup_runs/<tag>_timestamp/)
- Per fold/config CSV summaries, best_model.pth, best_selection.json
- If --final-train-on-all: *.npy embeddings for ABIDE-ASD/TD and CMI-ASD/TD
"""

import os, argparse, warnings, random, json, math
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from collections import defaultdict

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# sklearn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score

# -------------------- Repro --------------------
def seed_all(seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def to_numpy(x: torch.Tensor):
    return x.detach().float().cpu().numpy()

def sanitize_embeddings(E: np.ndarray):
    if E is None or E.size == 0: return E
    mask = np.isfinite(E).all(axis=1)
    E = E[mask]
    if E.size == 0: return E
    n = np.linalg.norm(E, axis=1, keepdims=True); n[n==0]=1.0
    return E / n

# -------------------- Site keys --------------------
def build_site_keys(df, dataset_tag=None):
    df = df.copy()
    if dataset_tag is not None:
        dataset_series = pd.Series(dataset_tag, index=df.index)
    elif "ABIDE" in df.columns:
        dataset_series = df["ABIDE"].map({1: "ABIDE1", 2: "ABIDE2"}).fillna("ABIDE1").astype(str)
    else:
        dataset_series = pd.Series("ABIDE1", index=df.index)
    if "SITE_ID" in df.columns:
        site_id = df["SITE_ID"].astype(str)
    elif "SITE" in df.columns:
        site_id = df["SITE"].astype(str)
    else:
        site_id = pd.Series("UNKNOWN", index=df.index)
    if "SITE" in df.columns:
        site_base = df["SITE"].astype(str)
    else:
        site_base = site_id.str.replace(r"_\d+$", "", regex=True)
    df["DATASET"] = dataset_series
    df["SITE_NAME"] = site_base
    df["SITE_KEY_STRICT"] = df["DATASET"] + "/" + site_id
    df["SITE_KEY_BASE"]   = df["DATASET"] + "/" + site_base
    return df

# -------------------- Load data --------------------
def load_and_preprocess_data(ABIDE_PATH, CMI_PATH):
    print("Loading ABIDE & CMI ...")
    with open(ABIDE_PATH, "rb") as f:
        original_df = pickle.load(f)
    # QC
    original_df = original_df[(original_df["percentofvolsrepaired"] <= 10) & (original_df["mean_fd"] <= 0.5)]
    original_df["data"] = original_df["data"].apply(lambda x: np.array(x) if isinstance(x, list) else x)
    def is_valid_array(x):
        x = np.array(x); return x.ndim == 2 and x.shape[0] >= 120 and np.isfinite(x).all()
    original_df = original_df[original_df["data"].apply(is_valid_array)].copy()
    def norm_abide(x):
        x = np.array(x); x = (x - np.mean(x)) / (np.std(x) + 1e-6); return x[:120, :]
    original_df["data"] = original_df["data"].apply(norm_abide)
    original_df["DX_GROUP"] = original_df["DX_GROUP"].map({1:0, 2:1})  # 0=ASD, 1=TD
    original_df = build_site_keys(original_df)
    abide_df = original_df[(original_df["DATASET"].isin(["ABIDE1","ABIDE2"]))].copy()

    cmi_df = None
    if CMI_PATH is not None:
        with open(CMI_PATH, "rb") as f:
            cmi_df = pickle.load(f)
        def norm_cmi(x):
            x = np.array(x); return (x - np.mean(x)) / (np.std(x) + 1e-6)
        def is_valid_cmi(x):
            x=np.array(x); return x.ndim==2 and x.shape[0]==375 and np.isfinite(x).all()
        cmi_df["data"] = cmi_df["data"].apply(norm_cmi)
        cmi_df = cmi_df[cmi_df["data"].apply(is_valid_cmi)].copy()
        cmi_df["DX_GROUP"] = cmi_df["label"].map({"asd":0, "td":1})
        cmi_df["SEX"] = cmi_df["gender"]
        cmi_df["AGE_AT_SCAN"] = cmi_df["age"]
        if "SITE_ID" not in cmi_df.columns:
            cmi_df["SITE_ID"] = cmi_df.get("site", "CMI_VIRTUAL_SITE")
        if "SITE" not in cmi_df.columns:
            cmi_df["SITE"] = cmi_df["SITE_ID"].astype(str).str.replace(r"_\d+$","", regex=True)
        cmi_df = build_site_keys(cmi_df, dataset_tag='CMI')
        print(f"ABIDE total: {len(abide_df)} | CMI total: {len(cmi_df)}")
    else:
        print(f"ABIDE total: {len(abide_df)} | CMI: not provided")
    return abide_df, cmi_df

# -------------------- Model --------------------
class stDNN_Embedding_WithProj(nn.Module):
    def __init__(self, input_channels, meta_dim, embedding_dim=512, dropout_rate=0.4):
        super().__init__()
        # scanner branch (site proxy)
        self.scanner = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
        )
        # main branch
        self.conv1 = nn.Conv1d(input_channels, 128, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(128); self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(256); self.dropout2 = nn.Dropout(dropout_rate)
        self.conv3 = nn.Conv1d(256, 512, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(512); self.dropout3 = nn.Dropout(dropout_rate)
        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.fc_fmri = nn.Linear(512, 512)
        self.fc_fmri_bn = nn.BatchNorm1d(512)
        self.fc_fmri_dropout = nn.Dropout(dropout_rate)
        self.fc_meta = nn.Linear(meta_dim, 128)
        self.fc_meta_bn = nn.BatchNorm1d(128)
        self.fc_meta_dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(512 + 128, embedding_dim)
        self.fc_out_bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x, meta):
        fs = self.scanner(x)                        # [B, D]
        fs_norm2 = torch.sum(fs * fs, dim=1, keepdim=True) + 1e-6
        z = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        z = F.max_pool1d(z, 2, 1)
        z = self.dropout2(F.relu(self.bn2(self.conv2(z))))
        z = F.max_pool1d(z, 2, 1)
        z = self.dropout3(F.relu(self.bn3(self.conv3(z))))
        z = F.max_pool1d(z, 2, 1)
        z = self.global_avg(z).squeeze(-1)
        x_fmri = self.fc_fmri_dropout(F.relu(self.fc_fmri_bn(self.fc_fmri(z))))
        x_meta = self.fc_meta_dropout(F.relu(self.fc_meta_bn(self.fc_meta(meta))))
        fx = self.fc_out_bn(self.fc_out(torch.cat([x_fmri, x_meta], dim=1)))  # [B, D]
        # Orthogonal projection (remove site component)
        inner = torch.sum(fx * fs, dim=1, keepdim=True)
        proj = (inner / fs_norm2) * fs
        fx_tilde = fx - proj
        return F.normalize(fx_tilde, dim=1), F.normalize(fs, dim=1)


# -------------------- Adversarial (GRL) variant --------------------
# Optional alternative site-removal interface. The default pipeline uses the
# orthogonal-projection model above (stDNN_Embedding_WithProj). This variant
# instead removes site information via domain-adversarial training: a site
# classification head is attached to the biological embedding through a
# Gradient Reversal Layer (GRL), so the encoder is pushed to make the embedding
# *non-predictive* of site. Select it with --site-removal adversarial.

class _GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer: identity forward, negated-scaled gradient back."""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


class stDNN_Embedding_WithAdvHead(nn.Module):
    """Biological encoder identical to stDNN_Embedding_WithProj's main branch,
    but with a site-prediction head attached via a Gradient Reversal Layer
    instead of an orthogonal-projection scanner branch.

    forward(x, meta) returns (embedding, site_logits), preserving the two-value
    contract so downstream embedding extraction (`f, _ = model(...)`) is
    unchanged. `site_logits` is consumed only by the adversarial loss during
    training; at inference it can be ignored.
    """
    def __init__(self, input_channels, meta_dim, num_sites, embedding_dim=512,
                 dropout_rate=0.4, grl_lambda=1.0):
        super().__init__()
        self.grl_lambda = grl_lambda
        # main (biological) branch — same topology as the projection model
        self.conv1 = nn.Conv1d(input_channels, 128, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(128); self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(256); self.dropout2 = nn.Dropout(dropout_rate)
        self.conv3 = nn.Conv1d(256, 512, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(512); self.dropout3 = nn.Dropout(dropout_rate)
        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.fc_fmri = nn.Linear(512, 512)
        self.fc_fmri_bn = nn.BatchNorm1d(512)
        self.fc_fmri_dropout = nn.Dropout(dropout_rate)
        self.fc_meta = nn.Linear(meta_dim, 128)
        self.fc_meta_bn = nn.BatchNorm1d(128)
        self.fc_meta_dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(512 + 128, embedding_dim)
        self.fc_out_bn = nn.BatchNorm1d(embedding_dim)
        # adversarial site-prediction head (operates on GRL-reversed embedding)
        self.site_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_sites),
        )

    def forward(self, x, meta):
        z = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        z = F.max_pool1d(z, 2, 1)
        z = self.dropout2(F.relu(self.bn2(self.conv2(z))))
        z = F.max_pool1d(z, 2, 1)
        z = self.dropout3(F.relu(self.bn3(self.conv3(z))))
        z = F.max_pool1d(z, 2, 1)
        z = self.global_avg(z).squeeze(-1)
        x_fmri = self.fc_fmri_dropout(F.relu(self.fc_fmri_bn(self.fc_fmri(z))))
        x_meta = self.fc_meta_dropout(F.relu(self.fc_meta_bn(self.fc_meta(meta))))
        fx = self.fc_out_bn(self.fc_out(torch.cat([x_fmri, x_meta], dim=1)))  # [B, D]
        emb = F.normalize(fx, dim=1)
        # site head sees a gradient-reversed copy: encoder learns to fool it
        site_logits = self.site_head(grad_reverse(emb, self.grl_lambda))
        return emb, site_logits


def build_model(cfg, meta_dim, input_channels=246, embedding_dim=512, device=None):
    """Factory selecting the site-removal interface from cfg['site_removal'].

    'orthogonal' (default) -> stDNN_Embedding_WithProj (no site labels needed)
    'adversarial'          -> stDNN_Embedding_WithAdvHead (needs cfg['num_sites'])
    """
    mode = cfg.get('site_removal', 'orthogonal')
    if mode == 'adversarial':
        m = stDNN_Embedding_WithAdvHead(
            input_channels, meta_dim, int(cfg['num_sites']), embedding_dim,
            cfg['dropout_rate'], grl_lambda=float(cfg.get('grl_lambda', 1.0)))
    else:
        m = stDNN_Embedding_WithProj(
            input_channels, meta_dim, embedding_dim, cfg['dropout_rate'])
    return m.to(device) if device is not None else m


# -------------------- Dataset --------------------
class ASDUnsupervisedDataset(Dataset):
    def __init__(self, df_asd, gender_encoder, age_fill_value, augment_prob=0.6,
                 site_labels=None):
        self.df = df_asd.reset_index(drop=True)
        self.fmri = self.df["data"].values
        g1 = gender_encoder.transform(self.df[["SEX"]])
        ages = self.df["AGE_AT_SCAN"].fillna(age_fill_value).values.reshape(-1,1)
        self.meta = np.concatenate([g1, ages], axis=1)
        self.augment_prob = augment_prob
        # Optional integer site labels for adversarial training (orthogonal mode
        # leaves this None and the label is never used).
        self.site_labels = (np.asarray(site_labels).astype(np.int64)
                            if site_labels is not None else None)

    def augment_np(self, x):
        if np.random.random() < self.augment_prob:
            x = x + np.random.normal(0, 0.01, x.shape)
        return x

    def crop_first100(self, x):
        return x[:100, :] if x.shape[0] >= 100 else x

    def __getitem__(self, idx):
        a = self.crop_first100(self.augment_np(np.array(self.fmri[idx])))
        m = self.meta[idx]
        item = {
            'x': torch.FloatTensor(a.T),  # [C,T]
            'meta': torch.FloatTensor(m),
        }
        if self.site_labels is not None:
            item['site'] = torch.tensor(self.site_labels[idx], dtype=torch.long)
        return item
    def __len__(self): return len(self.df)

# -------------------- VICReg --------------------
class VICReg(nn.Module):
    def __init__(self, inv_w=25.0, var_w=25.0, cov_w=1.0, eps=1e-4):
        super().__init__()
        self.inv_w, self.var_w, self.cov_w, self.eps = inv_w, var_w, cov_w, eps
    def forward(self, z1, z2):
        inv = F.mse_loss(z1, z2)
        def variance_term(z, eps=1e-4, gamma=1.0):
            std = torch.sqrt(z.var(dim=0) + eps)
            return torch.mean(F.relu(gamma - std))
        var_loss = 0.5 * (variance_term(z1) + variance_term(z2))
        B, D = z1.shape
        def cov_offdiag(z):
            zc = z - z.mean(0, keepdim=True)
            c = (zc.T @ zc) / max(1, B - 1)
            off = c - torch.diag(torch.diag(c))
            return (off.pow(2).sum()) / D
        cov_loss = 0.5 * (cov_offdiag(z1) + cov_offdiag(z2))
        return self.inv_w*inv + self.var_w*var_loss + self.cov_w*cov_loss

# -------------------- Augment --------------------
def two_views(x, target_len=100):
    B,C,T = x.shape
    L = min(target_len, T)
    s1 = torch.randint(0, T-L+1, (1,), device=x.device).item()
    s2 = torch.randint(0, T-L+1, (1,), device=x.device).item()
    v1 = x[:,:,s1:s1+L]
    v2 = x[:,:,s2:s2+L]
    v2 = v2 + torch.randn_like(v2)*0.02
    return v1, v2

@torch.no_grad()
def four_view_mean(model, x, m, target_len=100):
    B,C,T = x.shape
    L = min(target_len, T)
    s1 = torch.randint(0, T-L+1, (1,), device=x.device).item()
    s2 = torch.randint(0, T-L+1, (1,), device=x.device).item()
    v1 = x[:,:,s1:s1+L]; v2 = x[:,:,s2:s2+L]
    v3 = v1 + torch.randn_like(v1)*0.02
    v4 = v2 + torch.randn_like(v2)*0.02
    f1,_=model(v1,m); f2,_=model(v2,m); f3,_=model(v3,m); f4,_=model(v4,m)
    return (f1+f2+f3+f4)/4.0

# -------------------- K selection metrics --------------------
def cramers_v_from_labels(cluster_labels, site_labels):
    cl = np.asarray(cluster_labels).astype(int)
    sl = np.asarray(site_labels).astype(int)
    n = len(cl)
    if n == 0: return np.nan
    cl_ids, cl_inv = np.unique(cl, return_inverse=True)
    sl_ids, sl_inv = np.unique(sl, return_inverse=True)
    r, c = len(cl_ids), len(sl_ids)
    table = np.zeros((r,c), dtype=np.float64)
    for i in range(n): table[cl_inv[i], sl_inv[i]] += 1.0
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / max(table.sum(), 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nan_to_num(((table - expected)**2 / (expected + 1e-12))).sum()
    denom = n * max(1, min(r-1, c-1))
    return float(np.sqrt(chi2 / denom)) if denom>0 else 0.0

def k_sweep_scores(E, site_labels, K_range, stability_repeats=10, parsimony_gamma=0.02, seed=123):
    rng = np.random.RandomState(seed)
    n = E.shape[0]
    tmp = {}
    for K in K_range:
        if n <= K or K < 2:
            tmp[K] = dict(stability=np.nan, sil=np.nan, cramers_v=np.nan); continue
        labels_runs = []
        for rs in range(stability_repeats):
            km = KMeans(n_clusters=K, random_state=rs+seed, n_init=1)
            labels_runs.append(km.fit_predict(E))
        # stability via AMI mean - std
        amis = []
        for i in range(stability_repeats):
            for j in range(i+1, stability_repeats):
                amis.append(adjusted_mutual_info_score(labels_runs[i], labels_runs[j]))
        stability = float(np.nanmean(amis) - np.nanstd(amis))
        # silhouette (first run)
        try:
            sil = float(silhouette_score(E, labels_runs[0])) if len(np.unique(labels_runs[0]))>1 else np.nan
        except Exception:
            sil = np.nan
        # site leakage
        try:
            v = cramers_v_from_labels(labels_runs[0], site_labels)
        except Exception:
            v = np.nan
        tmp[K] = dict(stability=stability, sil=sil, cramers_v=v)

    def zscore(arr):
        arr = np.array(arr, dtype=np.float64)
        m = np.nanmean(arr); s = np.nanstd(arr); s = s if s>1e-8 else 1.0
        return (arr - m)/s

    Ks_ok = [K for K in K_range if not np.isnan(tmp[K]['stability'])]
    if len(Ks_ok)==0: return {}
    z_stab = zscore([tmp[K]['stability'] for K in Ks_ok])
    z_sil  = zscore([tmp[K]['sil'] for K in Ks_ok])
    z_v    = zscore([tmp[K]['cramers_v'] for K in Ks_ok])
    results = {}
    for idx, K in enumerate(Ks_ok):
        score = z_stab[idx] + 0.5*z_sil[idx] - 0.5*z_v[idx] - parsimony_gamma*(K-2)
        results[K] = dict(stability=tmp[K]['stability'], sil=tmp[K]['sil'], cramers_v=tmp[K]['cramers_v'], score=float(score))
    return results

# -------------------- Stability (Bootstrap & LOSO on Train+Val) --------------------
def bootstrap_stability(E, K=2, repeats=100, frac=0.8, seed=123):
    rng = np.random.RandomState(seed)
    n = E.shape[0]
    labs = []
    for r in range(repeats):
        idx = rng.choice(n, size=max(2,int(frac*n)), replace=False)
        km = KMeans(n_clusters=K, random_state=seed+r, n_init=10).fit(E[idx])
        lab = -np.ones(n, dtype=int)
        lab[idx] = km.labels_
        # assign oob to nearest centroid
        oob = np.setdiff1d(np.arange(n), idx)
        if len(oob)>0:
            dist = ((E[oob,None,:]-km.cluster_centers_[None,:,:])**2).sum(-1)
            lab[oob] = dist.argmin(1)
        labs.append(lab)
    amis = []
    for i in range(repeats):
        for j in range(i+1, repeats):
            amis.append(adjusted_mutual_info_score(labs[i], labs[j]))
    return float(np.nanmean(amis)), float(np.nanstd(amis)), labs

def loso_site_stability(E, sites, K=2, seed=123):
    sites = np.asarray(sites).astype(int)
    uniq = np.unique(sites)
    # global clustering on all
    km_all = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(E)
    lab_all = km_all.labels_
    amis = []
    for s in uniq:
        mask_in = sites != s
        mask_out = sites == s
        if mask_in.sum() < K or mask_out.sum() < 1:
            continue
        km_in = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(E[mask_in])
        # assign left-out
        C = km_in.cluster_centers_
        dist_out = ((E[mask_out,None,:]-C[None,:,:])**2).sum(-1)
        lab_out = dist_out.argmin(1)
        # compare with lab_all on that site
        ami = adjusted_mutual_info_score(lab_all[mask_out], lab_out)
        amis.append(ami)
    return float(np.nanmean(amis)) if len(amis)>0 else np.nan, amis

# -------------------- Embedding extraction --------------------
@torch.no_grad()
def embed_dataframe(model, df, gender_enc, age_fill_value, device, views=4, target_len=100):
    model.eval()
    embs, sites = [], []
    for i in range(len(df)):
        x = torch.FloatTensor(df.iloc[i]["data"].T).unsqueeze(0).to(device)
        g = gender_enc.transform(df.iloc[[i]][["SEX"]])
        age = np.array([[df.iloc[i]["AGE_AT_SCAN"] if pd.notna(df.iloc[i]["AGE_AT_SCAN"]) else age_fill_value]])
        m = torch.FloatTensor(np.concatenate([g, age], axis=1)).to(device)
        if views >= 4:
            fmean = four_view_mean(model, x, m, target_len=target_len)
        else:
            v1,v2 = two_views(x, target_len=target_len)
            f1,_=model(v1,m); f2,_=model(v2,m)
            fmean = (f1+f2)/2.0
        embs.append(to_numpy(fmean))
        sites.append(df.iloc[i]["SITE_KEY_STRICT"])
    E = np.concatenate(embs, 0) if len(embs)>0 else np.empty((0,512))
    E = sanitize_embeddings(E)
    site_codes = pd.Series(sites).astype('category').cat.codes.values
    return E, site_codes

# -------------------- External eval via fixed centroids --------------------
def fit_centroids(E_trainval, K=2, seed=123):
    km = KMeans(n_clusters=K, random_state=seed, n_init=20).fit(E_trainval)
    return km.cluster_centers_, km.labels_

def assign_labels(E, C):
    dist = ((E[:,None,:]-C[None,:,:])**2).sum(-1)
    return dist.argmin(1), dist.min(1)

def sil_or_nan(E, lab):
    try:
        if len(np.unique(lab)) < 2: return np.nan
        return float(silhouette_score(E, lab))
    except Exception:
        return np.nan

def cramers_v_from_centroids(E, C, site_labels):
    lab,_ = assign_labels(E, C)
    return cramers_v_from_labels(lab, site_labels)

# -------------------- Split helpers --------------------
def grouped_split_by_site(df_asd, test_size=0.2, val_size=0.2, seed=42):
    rng = np.random.RandomState(seed)
    groups = df_asd["SITE_KEY_STRICT"].astype(str).values
    uniq = np.unique(groups); rng.shuffle(uniq)
    n = len(uniq)
    n_test = max(1, int(round(test_size * n)))
    n_val  = max(1, int(round(val_size * n)))
    test_sites = set(uniq[:n_test])
    val_sites  = set(uniq[n_test:n_test+n_val])
    train_sites= set(uniq[n_test+n_val:])
    train_df = df_asd[df_asd["SITE_KEY_STRICT"].astype(str).isin(train_sites)].copy()
    val_df   = df_asd[df_asd["SITE_KEY_STRICT"].astype(str).isin(val_sites)].copy()
    test_df  = df_asd[df_asd["SITE_KEY_STRICT"].astype(str).isin(test_sites)].copy()
    return train_df, val_df, test_df

def loso_folds(df_asd, val_site_frac=0.2, seed=42):
    """Yield (train_df, val_df, test_df) per test site group."""
    rng = np.random.RandomState(seed)
    all_sites = df_asd["SITE_KEY_STRICT"].astype(str)
    uniq = np.unique(all_sites.values)
    for test_site in uniq:
        test_df = df_asd[all_sites==test_site].copy()
        remain = df_asd[all_sites!=test_site].copy()
        # choose val sites from remain
        rem_sites = remain["SITE_KEY_STRICT"].astype(str).unique()
        rng.shuffle(rem_sites)
        n_val = max(1, int(round(val_site_frac * len(rem_sites))))
        val_sites = set(rem_sites[:n_val])
        train_df = remain[~remain["SITE_KEY_STRICT"].astype(str).isin(val_sites)].copy()
        val_df   = remain[ remain["SITE_KEY_STRICT"].astype(str).isin(val_sites)].copy()
        yield (train_df, val_df, test_df, test_site)

# -------------------- Train one config --------------------
def train_one(cfg, train_df, val_df, gender_enc, age_fill_value, meta_dim, device):
    site_removal = cfg.get('site_removal', 'orthogonal')

    # For adversarial mode, build an integer site vocabulary from SITE_KEY_BASE
    # (collapsing _1/_2 acquisition suffixes) over the training split.
    site_labels = None
    if site_removal == 'adversarial':
        site_vocab = cfg.get('site_vocab')
        if site_vocab is None:
            uniq = sorted(train_df["SITE_KEY_BASE"].astype(str).unique())
            site_vocab = {s: i for i, s in enumerate(uniq)}
        cfg['site_vocab'] = site_vocab
        cfg['num_sites'] = len(site_vocab)
        site_labels = train_df["SITE_KEY_BASE"].astype(str).map(
            lambda s: site_vocab.get(s, 0)).values

    model = build_model(cfg, meta_dim, device=device)
    vic = VICReg(inv_w=cfg['vic_inv'], var_w=cfg['vic_var'], cov_w=cfg['vic_cov'])
    tr_set = ASDUnsupervisedDataset(train_df, gender_enc, age_fill_value,
                                    augment_prob=cfg['augment_prob'],
                                    site_labels=site_labels)
    tr_loader = DataLoader(tr_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    opt = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['l2_weight'])
    if cfg['scheduler']=='cosine':
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2); use_plateau=False
    else:
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5); use_plateau=True

    for epoch in range(cfg['epochs']):
        model.train(); ep_loss = 0.0; batch_count = 0
        print(f"    Epoch {epoch+1}/{cfg['epochs']} - Training...")
        for batch in tr_loader:
            x = batch['x'].to(device); m = batch['meta'].to(device)
            v1,v2 = two_views(x, target_len=cfg['target_len'])
            f1,aux1 = model(v1,m); f2,aux2 = model(v2,m)
            loss_vic = vic(f1, f2)
            if site_removal == 'adversarial':
                # aux* are site logits; encoder is trained (via GRL) to make the
                # embedding non-predictive of site.
                s = batch['site'].to(device)
                loss_adv = 0.5 * (F.cross_entropy(aux1, s) + F.cross_entropy(aux2, s))
                loss = loss_vic + cfg['adv_w'] * loss_adv
            else:
                # aux* are scanner vectors; orthogonal penalty removes site dim.
                loss_ortho = torch.mean(torch.sum(f1 * aux1, dim=1)**2)
                loss = loss_vic + cfg['ortho_w']*loss_ortho
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ep_loss += float(loss.item()); batch_count += 1
            if batch_count % 10 == 0:
                print(f"      Batch {batch_count}/{len(tr_loader)}, Loss: {loss.item():.4f}")
        avg_loss = ep_loss / max(1, len(tr_loader))
        print(f"    Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}")
        if not use_plateau: sched.step(epoch+1.0)
        else: sched.step(avg_loss)
    # end training
    # Val embeddings for K sweep
    with torch.no_grad():
        E_val, site_val = embed_dataframe(model, val_df, gender_enc, age_fill_value, device, views=4, target_len=cfg['target_len'])
    return model, E_val, site_val

# -------------------- Config sampler --------------------
def sample_configs(n, base):
    rng = random.Random(42)
    LRs   = [3e-4, 1e-3]
    BSs   = [32, 64]
    OWs   = [0.3, 0.4, 0.5]
    VINV  = [15.0, 25.0, 40.0]
    VVAR  = [15.0, 25.0, 40.0]
    VCOV  = [0.5, 1.0]
    AUGP  = [0.5, 0.6, 0.7]
    DROP  = [0.3, 0.4]
    TLEN  = [80, 100, 120]
    seen=set(); cfgs=[]
    def key(c): return (c['lr'],c['bs'],c['ortho_w'],c['vinv'],c['vvar'],c['vcov'],c['augp'],c['dropout'],c['target_len'])
    while len(cfgs)<n:
        c = dict(
            epochs=base['epochs'], scheduler=base['scheduler'],
            lr=rng.choice(LRs), bs=rng.choice(BSs), ortho_w=rng.choice(OWs),
            vinv=rng.choice(VINV), vvar=rng.choice(VVAR), vcov=rng.choice(VCOV),
            augp=rng.choice(AUGP), dropout=rng.choice(DROP), target_len=rng.choice(TLEN),
            l2=base['l2'],
        )
        if key(c) not in seen:
            seen.add(key(c)); cfgs.append(c)
    return cfgs

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='vicreg_loso_or_grouped')
    parser.add_argument('--abide-path', type=str, required=True)
    parser.add_argument('--cmi-path', type=str, default=None,
                        help='Path to CMI pklz (optional; omit for ABIDE-only smoke test)')
    parser.add_argument('--stanford-path', type=str, default=None,
                        help='Stanford pklzembeddings')
    parser.add_argument('--fixed-config-json', type=str, default=None,
                        help='step0best_hyperparameters.json')
    # CV mode
    parser.add_argument('--cv-mode', type=str, default='grouped', choices=['grouped','loso'])
    parser.add_argument('--val-size', type=float, default=0.2)
    parser.add_argument('--test-size', type=float, default=0.2)
    # training base
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine','plateau'])
    parser.add_argument('--l2', type=float, default=1e-3)
    # search
    parser.add_argument('--num-configs', type=int, default=120)
    # K sweep
    parser.add_argument('--kmin', type=int, default=2)
    parser.add_argument('--kmax', type=int, default=6)
    parser.add_argument('--stab-repeats', type=int, default=10)
    parser.add_argument('--parsimony-gamma', type=float, default=0.02)
    # final all-train
    parser.add_argument('--final-train-on-all', action='store_true')
    # site-removal interface (default: orthogonal projection, the main pipeline)
    parser.add_argument('--site-removal', type=str, default='orthogonal',
                        choices=['orthogonal', 'adversarial'],
                        help="How to remove site/scanner variance. 'orthogonal' "
                             "(default): scanner proxy branch + orthogonal "
                             "projection. 'adversarial': site-prediction head "
                             "with a Gradient Reversal Layer (DANN-style).")
    parser.add_argument('--adv-w', type=float, default=1.0,
                        help='Adversarial site-loss weight (used only when '
                             '--site-removal adversarial).')
    parser.add_argument('--grl-lambda', type=float, default=1.0,
                        help='Gradient Reversal Layer strength (used only when '
                             '--site-removal adversarial).')
    args = parser.parse_known_args()[0]

    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    ROOT = os.path.join("./asd_unsup_runs", f"{args.tag}_{ts}")
    os.makedirs(ROOT, exist_ok=True)
    print("Results dir:", os.path.abspath(ROOT))

    abide_df, cmi_df = load_and_preprocess_data(args.abide_path, args.cmi_path)

    # Split ASD/TD
    abide_asd = abide_df[abide_df["DX_GROUP"]==0].copy()
    abide_td  = abide_df[abide_df["DX_GROUP"]==1].copy()
    if cmi_df is not None:
        cmi_asd = cmi_df[cmi_df["DX_GROUP"]==0].copy()
        cmi_td  = cmi_df[cmi_df["DX_GROUP"]==1].copy()
    else:
        cmi_asd = pd.DataFrame()
        cmi_td  = pd.DataFrame()

    # encoder fit scope will vary per fold to avoid leakage
    def fit_gender_encoder(df_fit):
        try:
            ge = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            ge = OneHotEncoder(sparse=False, handle_unknown="ignore")
        ge.fit(df_fit[["SEX"]])
        return ge

    # configs to try
    base = dict(epochs=args.epochs, scheduler=args.scheduler, l2=args.l2)
    if parser.parse_known_args()[0].fixed_config_json:
        with open(args.fixed_config_json, 'r') as f:
            hp = json.load(f)
        best = hp.get('best_config', hp)
        c_fixed = dict(
            epochs=int(best.get('epochs', args.epochs)),
            scheduler=str(best.get('scheduler', args.scheduler)),
            l2=args.l2,
            lr=float(best.get('learning_rate', 1e-3)),
            bs=int(best.get('batch_size', 64)),
            dropout=float(best.get('dropout_rate', 0.4)),
            augp=float(best.get('augment_prob', 0.6)),
            ortho_w=float(best.get('ortho_w', 0.4)),
            vinv=float(best.get('vic_inv', 25.0)),
            vvar=float(best.get('vic_var', 25.0)),
            vcov=float(best.get('vic_cov', 1.0)),
            target_len=int(best.get('target_len', 100)),
        )
        cfg_list = [c_fixed]
        print("Using fixed config from", args.fixed_config_json)
        print(c_fixed)
    else:
        cfg_list = sample_configs(args.num_configs, base)
        print(f"Total sampled configs: {len(cfg_list)}")

    # CV loops
    folds = []
    if args.cv_mode == 'grouped':
        tr, va, te = grouped_split_by_site(abide_asd, test_size=args.test_size, val_size=args.val_size, seed=42)
        folds.append((tr, va, te, "grouped"))
    else:
        for tr, va, te, site in loso_folds(abide_asd, val_site_frac=args.val_size, seed=42):
            folds.append((tr, va, te, site))

    all_rows = []
    best_overall = None
    best_overall_score = -np.inf
    best_overall_cfg = None

    for fi, (train_df, val_df, test_df, fold_tag) in enumerate(folds, 1):
        print("\n"+"="*90)
        print(f"FOLD {fi}/{len(folds)}  tag={fold_tag}")
        print("="*90)
        # enc/age
        gender_enc = fit_gender_encoder(train_df)
        meta_dim = gender_enc.transform(train_df[["SEX"]]).shape[1] + 1
        age_fill = float(train_df["AGE_AT_SCAN"].dropna().mean())

        FOLD_DIR = os.path.join(ROOT, f"fold_{fi}_{str(fold_tag)}"); os.makedirs(FOLD_DIR, exist_ok=True)
        fold_rows = []
        fold_best = None; fold_best_score = -np.inf; fold_best_K = None; fold_best_state = None; fold_best_cfg = None

        for ci, c in enumerate(cfg_list, 1):
            print(f"\n{'='*60}")
            print(f"Config {ci}/{len(cfg_list)} - Fold {fi}/{len(folds)}")
            print(f"{'='*60}")
            cfg = dict(
                epochs=c['epochs'], scheduler=c['scheduler'], l2_weight=args.l2,
                learning_rate=c['lr'], batch_size=c['bs'], dropout_rate=c['dropout'],
                augment_prob=c['augp'], ortho_w=c['ortho_w'],
                vic_inv=c['vinv'], vic_var=c['vvar'], vic_cov=c['vcov'],
                target_len=c['target_len'],
                site_removal=args.site_removal, adv_w=args.adv_w,
                grl_lambda=args.grl_lambda,
            )
            print(f"Training config: lr={cfg['learning_rate']}, bs={cfg['batch_size']}, epochs={cfg['epochs']}")
            print(f"VICReg weights: inv={cfg['vic_inv']}, var={cfg['vic_var']}, cov={cfg['vic_cov']}")
            print(f"Site removal: {cfg['site_removal']} "
                  f"(ortho_w={cfg['ortho_w']}, adv_w={cfg['adv_w']}), "
                  f"Dropout: {cfg['dropout_rate']}, Aug prob: {cfg['augment_prob']}")
            print(f"Target length: {cfg['target_len']}")
            print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
            # train
            model, E_val, site_val = train_one(
                dict(epochs=cfg['epochs'], scheduler=cfg['scheduler'],
                     learning_rate=cfg['learning_rate'], batch_size=cfg['batch_size'],
                     l2_weight=cfg['l2_weight'], dropout_rate=cfg['dropout_rate'],
                     augment_prob=cfg['augment_prob'], ortho_w=cfg['ortho_w'],
                     vic_inv=cfg['vic_inv'], vic_var=cfg['vic_var'], vic_cov=cfg['vic_cov'],
                     target_len=cfg['target_len'],
                     site_removal=cfg['site_removal'], adv_w=cfg['adv_w'],
                     grl_lambda=cfg['grl_lambda']),
                train_df, val_df, gender_enc, age_fill, meta_dim, device
            )

            # K sweep on Val
            K_range = list(range(args.kmin, args.kmax+1))
            results = k_sweep_scores(E_val, site_val, K_range,
                                     stability_repeats=args.stab_repeats,
                                     parsimony_gamma=args.parsimony_gamma)
            if len(results)==0:
                score = -np.inf; Kbest=None
            else:
                Kbest, stat = max(results.items(), key=lambda kv: kv[1]['score'])
                score = stat['score']

            # keep best on Val
            if score > fold_best_score:
                fold_best_score = score; fold_best = results.get(Kbest, None); fold_best_K = Kbest
                fold_best_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
                fold_best_cfg = cfg

            fold_rows.append({
                'fold': fold_tag, 'cfg_idx': ci, 'K_best': Kbest, 'val_score': score,
                **({} if results.get(Kbest) is None else results[Kbest])
            })

        # End of configs in this fold -> save fold best
        pd.DataFrame(fold_rows).to_csv(os.path.join(FOLD_DIR, "val_selection.csv"), index=False)
        if fold_best_state is None:
            print("No valid config in this fold, skipping downstream.")
            continue

        # Load best state & compute Train+Val embeddings
        # Rebuild the best model via the factory. For adversarial mode we must
        # reconstruct the same site vocabulary (SITE_KEY_BASE) used in training
        # so the site head has matching output dimension before loading weights.
        rebuild_cfg = dict(fold_best_cfg)
        if rebuild_cfg.get('site_removal') == 'adversarial':
            uniq = sorted(train_df["SITE_KEY_BASE"].astype(str).unique())
            rebuild_cfg['site_vocab'] = {s: i for i, s in enumerate(uniq)}
            rebuild_cfg['num_sites'] = len(uniq)
        model = build_model(rebuild_cfg, meta_dim, device=device)
        model.load_state_dict(fold_best_state, strict=True)

        E_tr, site_tr = embed_dataframe(model, train_df, gender_enc, age_fill, device, views=4, target_len=fold_best_cfg['target_len'])
        E_va, site_va = embed_dataframe(model, val_df,   gender_enc, age_fill, device, views=4, target_len=fold_best_cfg['target_len'])
        E_tv = sanitize_embeddings(np.vstack([E_tr, E_va]))
        site_tv = np.concatenate([site_tr, site_va])

        # Stability on Train+Val
        b_mean, b_std, _ = bootstrap_stability(E_tv, K=fold_best_K, repeats=100, frac=0.8, seed=123)
        loso_mean, loso_list = loso_site_stability(E_tv, site_tv, K=fold_best_K, seed=123)

        # Fit centroids on Train+Val, evaluate Test-ASD & CMI-ASD
        C, lab_tv = fit_centroids(E_tv, K=fold_best_K, seed=123)

        # ABIDE-Test-ASD
        E_te, site_te = embed_dataframe(model, test_df, gender_enc, age_fill, device, views=4, target_len=fold_best_cfg['target_len'])
        lab_te, _ = assign_labels(E_te, C)
        sil_te = sil_or_nan(E_te, lab_te)
        V_te   = cramers_v_from_labels(lab_te, site_te)

        # CMI-ASD external (optional)
        sil_cmi, V_cmi = np.nan, np.nan
        if cmi_df is not None and len(cmi_df) > 0:
            _cmi_asd = cmi_df[cmi_df["DX_GROUP"]==0].copy()
            try:
                E_cmi_asd, site_cmi_asd = embed_dataframe(model, _cmi_asd, gender_enc, age_fill, device, views=4, target_len=fold_best_cfg['target_len'])
                lab_cmi, _ = assign_labels(E_cmi_asd, C)
                sil_cmi = sil_or_nan(E_cmi_asd, lab_cmi)
                V_cmi   = cramers_v_from_labels(lab_cmi, site_cmi_asd)
            except Exception as e:
                print("CMI embedding failed:", e)

        # Save fold summary
        fold_summary = dict(
            fold=str(fold_tag), K_star=int(fold_best_K),
            val_score=float(fold_best_score),
            stab_boot_mean=float(b_mean), stab_boot_std=float(b_std),
            stab_loso_mean=float(loso_mean),
            test_abide_sil=float(sil_te), test_abide_V=float(V_te),
            test_cmi_sil=float(sil_cmi),  test_cmi_V=float(V_cmi),
            best_cfg=fold_best_cfg
        )
        with open(os.path.join(FOLD_DIR, "fold_best_summary.json"), "w") as f:
            json.dump(fold_summary, f, indent=2)
        torch.save(fold_best_state, os.path.join(FOLD_DIR, "best_model.pth"))
        np.save(os.path.join(FOLD_DIR, "centroids.npy"), C)

        # global table
        all_rows.append({
            'fold': str(fold_tag), 'K_star': fold_best_K,
            'val_score': fold_best_score,
            'boot_mean': b_mean, 'boot_std': b_std, 'loso_mean': loso_mean,
            'abide_test_sil': sil_te, 'abide_test_V': V_te,
            'cmi_asd_sil': sil_cmi, 'cmi_asd_V': V_cmi
        })

        # track overall best by val_score
        if fold_best_score > best_overall_score:
            best_overall_score = fold_best_score
            best_overall = dict(fold=fold_tag, K_star=fold_best_K)
            best_overall_cfg = fold_best_cfg
            # also store path
            best_overall['fold_dir'] = FOLD_DIR

    # Save global CSV
    if len(all_rows)>0:
        pd.DataFrame(all_rows).to_csv(os.path.join(ROOT, "cv_overview.csv"), index=False)
    with open(os.path.join(ROOT, "best_overall.json"), "w") as f:
        json.dump({'best': best_overall, 'best_cfg': best_overall_cfg}, f, indent=2)

    # -------- Optional: final train on ALL ABIDE-ASD with best cfg, export embeddings --------
    if args.final_train_on_all and (best_overall_cfg is not None):
        print("\n[Final training on ALL ABIDE-ASD with best hyperparams] ...")
        # prepare ALL-ASD split for encoder fitting
        ge = fit_gender_encoder(abide_asd)
        meta_dim = ge.transform(abide_asd[["SEX"]]).shape[1] + 1
        age_fill = float(abide_asd["AGE_AT_SCAN"].dropna().mean())
        cfg = best_overall_cfg
        model, E_dummy, _ = train_one(
            dict(epochs=cfg['epochs'], scheduler=cfg['scheduler'],
                 learning_rate=cfg['learning_rate'], batch_size=cfg['batch_size'],
                 l2_weight=args.l2, dropout_rate=cfg['dropout_rate'],
                 augment_prob=cfg['augment_prob'], ortho_w=cfg['ortho_w'],
                 vic_inv=cfg['vic_inv'], vic_var=cfg['vic_var'], vic_cov=cfg['vic_cov'],
                 target_len=cfg['target_len'],
                 site_removal=cfg.get('site_removal', 'orthogonal'),
                 adv_w=cfg.get('adv_w', 1.0), grl_lambda=cfg.get('grl_lambda', 1.0)),
            abide_asd, abide_asd.sample(n=min(10,len(abide_asd)), random_state=1),  # tiny dummy val
            ge, age_fill, meta_dim, device
        )
        torch.save(model.state_dict(), os.path.join(ROOT, "final_all_abide_asd_model.pth"))

        # Export embeddings for later UMAP/plotting (ABIDE/CMI + optional Stanford)
        def export_df(df, name):
            E, sites = embed_dataframe(model, df, ge, age_fill, device, views=4, target_len=cfg['target_len'])
            np.save(os.path.join(ROOT, f"{name}_emb.npy"), E)
            np.save(os.path.join(ROOT, f"{name}_sites.npy"), sites)
            return E, sites

        # ABIDE
        E_abide_asd,_ = export_df(abide_asd, "abide_asd")
        E_abide_td,_  = export_df(abide_td,  "abide_td")
        # CMI (optional)
        if cmi_df is not None and len(cmi_asd) > 0:
            E_cmi_asd,_ = export_df(cmi_asd, "cmi_asd")
            E_cmi_td,_  = export_df(cmi_td,  "cmi_td")

        # Stanford (optional)
        if args.stanford_path is not None and os.path.exists(args.stanford_path):
            import pickle, gzip, bz2, lzma
            def load_pklz(p):
                for opener in (gzip.open, bz2.open, lzma.open):
                    try:
                        with opener(p, 'rb') as f: return pickle.load(f)
                    except Exception: pass
                with open(p, 'rb') as f: return pickle.load(f)
            sta = load_pklz(args.stanford_path)
            sta = sta.copy()
            sta['data'] = sta['data'].apply(lambda x: (np.array(x)-np.array(x).mean())/(np.array(x).std()+1e-6))
            sta['DX_GROUP'] = sta['label'].map({'asd':0,'td':1})
            sta['SEX'] = sta.get('gender', sta.get('SEX','unknown'))
            sta['AGE_AT_SCAN'] = sta.get('age', sta.get('AGE_AT_SCAN', np.nan))
            if 'SITE_ID' not in sta.columns:
                sta['SITE_ID'] = sta.get('site', 'STANFORD')
            if 'SITE' not in sta.columns:
                sta['SITE'] = sta['SITE_ID'].astype(str)
            sta = build_site_keys(sta, dataset_tag='STANFORD')
            sta_asd = sta[sta['DX_GROUP']==0].copy(); sta_td = sta[sta['DX_GROUP']==1].copy()
            export_df(sta_asd, 'stanford_asd')
            export_df(sta_td,  'stanford_td')

        # Fit K* on ALL ABIDE-ASD and save centroids
        K_star = int(best_overall['K_star']) if best_overall else 2
        C_all, lab_all = fit_centroids(E_abide_asd, K=K_star, seed=123)
        np.save(os.path.join(ROOT, "final_centroids.npy"), C_all)
        np.save(os.path.join(ROOT, "abide_asd_labels.npy"), lab_all)

    print("\nDone. Artifacts ->", os.path.abspath(ROOT))

if __name__ == "__main__":
    main()
