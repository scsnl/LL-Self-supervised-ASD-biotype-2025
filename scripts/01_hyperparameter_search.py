#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 0: Large-scale Hyperparameter Search for ASD Subtyping

Purpose: Comprehensive hyperparameter optimization on server
- 300 different configurations
- VICReg + Orthogonal site removal
- K selection from 2-6
- Find optimal parameters for main training
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

# -------------------- Dataset --------------------
class ASDUnsupervisedDataset(Dataset):
    def __init__(self, df_asd, gender_encoder, age_fill_value, augment_prob=0.6):
        self.df = df_asd.reset_index(drop=True)
        self.fmri = self.df["data"].values
        g1 = gender_encoder.transform(self.df[["SEX"]])
        ages = self.df["AGE_AT_SCAN"].fillna(age_fill_value).values.reshape(-1,1)
        self.meta = np.concatenate([g1, ages], axis=1)
        self.augment_prob = augment_prob

    def augment_np(self, x):
        if np.random.random() < self.augment_prob:
            x = x + np.random.normal(0, 0.01, x.shape)
        return x

    def crop_first100(self, x):
        return x[:100, :] if x.shape[0] >= 100 else x

    def __getitem__(self, idx):
        a = self.crop_first100(self.augment_np(np.array(self.fmri[idx])))
        m = self.meta[idx]
        return {
            'x': torch.FloatTensor(a.T),  # [C,T]
            'meta': torch.FloatTensor(m),
        }
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

# -------------------- Train one config --------------------
def train_one(cfg, train_df, val_df, gender_enc, age_fill_value, meta_dim, device):
    print(f"  Training config {cfg['config_id']}: lr={cfg['learning_rate']:.1e}, bs={cfg['batch_size']}, epochs={cfg['epochs']}")
    model = stDNN_Embedding_WithProj(246, meta_dim, 512, cfg['dropout_rate']).to(device)
    vic = VICReg(inv_w=cfg['vic_inv'], var_w=cfg['vic_var'], cov_w=cfg['vic_cov'])
    tr_set = ASDUnsupervisedDataset(train_df, gender_enc, age_fill_value, augment_prob=cfg['augment_prob'])
    tr_loader = DataLoader(tr_set, batch_size=cfg['batch_size'], shuffle=True, 
                          num_workers=min(4, os.cpu_count()), pin_memory=torch.cuda.is_available(), drop_last=True)
    opt = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['l2_weight'])
    if cfg['scheduler']=='cosine':
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2); use_plateau=False
    else:
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5); use_plateau=True

    for epoch in range(cfg['epochs']):
        model.train(); ep_loss = 0.0
        for batch in tr_loader:
            x = batch['x'].to(device); m = batch['meta'].to(device)
            v1,v2 = two_views(x, target_len=cfg['target_len'])
            f1,fs1 = model(v1,m); f2,fs2 = model(v2,m)
            loss_vic = vic(f1, f2)
            loss_ortho = torch.mean(torch.sum(f1 * fs1, dim=1)**2)
            loss = loss_vic + cfg['ortho_w']*loss_ortho
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ep_loss += float(loss.item())
        avg_loss = ep_loss / max(1, len(tr_loader))
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{cfg['epochs']}: Loss = {avg_loss:.4f}")
        if not use_plateau: sched.step(epoch+1.0)
        else: sched.step(avg_loss)
    # end training
    # Val embeddings for K sweep
    with torch.no_grad():
        E_val, site_val = embed_dataframe(model, val_df, gender_enc, age_fill_value, device, views=4, target_len=cfg['target_len'])
    return model, E_val, site_val

# -------------------- Config sampler (Extended) --------------------
def sample_configs(n, base_epochs=80, seed=42):
    """Sample n different hyperparameter configurations"""
    rng = random.Random(seed)
    
    # Extended hyperparameter ranges for comprehensive search
    LRs   = [1e-4, 3e-4, 5e-4, 1e-3, 2e-3]  # 5 options
    BSs   = [16, 32, 48, 64]  # 4 options
    OWs   = [0.2, 0.3, 0.4, 0.5, 0.6]  # 5 options
    VINV  = [10.0, 15.0, 25.0, 35.0, 50.0]  # 5 options
    VVAR  = [10.0, 15.0, 25.0, 35.0, 50.0]  # 5 options
    VCOV  = [0.3, 0.5, 1.0, 1.5, 2.0]  # 5 options
    AUGP  = [0.4, 0.5, 0.6, 0.7, 0.8]  # 5 options
    DROP  = [0.2, 0.3, 0.4, 0.5]  # 4 options
    TLEN  = [80, 100, 120, 140]  # 4 options
    SCHED = ['cosine', 'plateau']  # 2 options
    L2S   = [5e-4, 1e-3, 2e-3, 5e-3]  # 4 options
    
    seen = set()
    cfgs = []
    
    def config_key(c):
        return (c['lr'], c['bs'], c['ortho_w'], c['vinv'], c['vvar'], 
                c['vcov'], c['augp'], c['dropout'], c['target_len'], 
                c['scheduler'], c['l2'])
    
    max_attempts = n * 10  # Prevent infinite loop
    attempts = 0
    
    while len(cfgs) < n and attempts < max_attempts:
        attempts += 1
        c = dict(
            config_id=len(cfgs) + 1,
            epochs=base_epochs,
            lr=rng.choice(LRs), 
            bs=rng.choice(BSs), 
            ortho_w=rng.choice(OWs),
            vinv=rng.choice(VINV), 
            vvar=rng.choice(VVAR), 
            vcov=rng.choice(VCOV),
            augp=rng.choice(AUGP), 
            dropout=rng.choice(DROP), 
            target_len=rng.choice(TLEN),
            scheduler=rng.choice(SCHED),
            l2=rng.choice(L2S),
        )
        
        key = config_key(c)
        if key not in seen:
            seen.add(key)
            cfgs.append(c)
    
    if len(cfgs) < n:
        print(f"Warning: Only generated {len(cfgs)} unique configs out of {n} requested")
    
    return cfgs

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Step 0: Hyperparameter Search for ASD Subtyping")
    parser.add_argument('--tag', type=str, default='hyperparam_search_300')
    # Data paths (provide your own; see README "Data Format")
    parser.add_argument('--abide-path', type=str,
                       default='DATA/combined_ABIDE_information_with_fMRI.pklz')
    parser.add_argument('--cmi-path', type=str,
                       default='CMI-DATA/combined_asd_td_rest_run1_data.pklz')
    # Search parameters
    parser.add_argument('--num-configs', type=int, default=300, help='Number of configurations to try')
    parser.add_argument('--epochs', type=int, default=80, help='Training epochs per config')
    parser.add_argument('--val-size', type=float, default=0.2)
    parser.add_argument('--test-size', type=float, default=0.2)
    # K sweep
    parser.add_argument('--kmin', type=int, default=2)
    parser.add_argument('--kmax', type=int, default=6)
    parser.add_argument('--stab-repeats', type=int, default=10)
    parser.add_argument('--parsimony-gamma', type=float, default=0.02)
    # Hardware
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()

    seed_all(42)
    
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    ROOT = os.path.join("./asd_unsup_runs", f"{args.tag}_{ts}")
    os.makedirs(ROOT, exist_ok=True)
    print(f"Results directory: {os.path.abspath(ROOT)}")

    # Load and split data
    abide_df, cmi_df = load_and_preprocess_data(args.abide_path, args.cmi_path)
    abide_asd = abide_df[abide_df["DX_GROUP"]==0].copy()
    train_df, val_df, test_df = grouped_split_by_site(abide_asd, test_size=args.test_size, val_size=args.val_size, seed=42)
    print(f"Data split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Prepare encoders
    try:
        ge = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        ge = OneHotEncoder(sparse=False, handle_unknown="ignore")
    ge.fit(train_df[["SEX"]])
    meta_dim = ge.transform(train_df[["SEX"]]).shape[1] + 1
    age_fill = float(train_df["AGE_AT_SCAN"].dropna().mean())
    print(f"Meta dimension: {meta_dim}, Age fill value: {age_fill:.1f}")

    # Generate configurations
    cfg_list = sample_configs(args.num_configs, base_epochs=args.epochs, seed=42)
    print(f"Generated {len(cfg_list)} unique configurations")

    # Hyperparameter search
    all_results = []
    best_score = -np.inf
    best_config = None
    best_K = None
    
    print(f"\n{'='*80}")
    print(f"STARTING HYPERPARAMETER SEARCH: {len(cfg_list)} CONFIGURATIONS")
    print(f"{'='*80}")

    for ci, c in enumerate(cfg_list, 1):
        print(f"\n[{ci}/{len(cfg_list)}] Testing configuration {c['config_id']}")
        print(f"  lr={c['lr']:.1e}, bs={c['bs']}, ortho_w={c['ortho_w']}, vic=[{c['vinv']},{c['vvar']},{c['vcov']}]")
        print(f"  dropout={c['dropout']}, augment_prob={c['augp']}, target_len={c['target_len']}")
        print(f"  scheduler={c['scheduler']}, l2={c['l2']:.1e}")
        
        try:
            # Create config dict for training
            cfg = dict(
                config_id=c['config_id'],
                epochs=c['epochs'], scheduler=c['scheduler'], l2_weight=c['l2'],
                learning_rate=c['lr'], batch_size=c['bs'], dropout_rate=c['dropout'],
                augment_prob=c['augp'], ortho_w=c['ortho_w'],
                vic_inv=c['vinv'], vic_var=c['vvar'], vic_cov=c['vcov'],
                target_len=c['target_len'],
            )
            
            # Train model
            model, E_val, site_val = train_one(cfg, train_df, val_df, ge, age_fill, meta_dim, device)
            
            # K sweep on validation set
            K_range = list(range(args.kmin, args.kmax+1))
            results = k_sweep_scores(E_val, site_val, K_range,
                                   stability_repeats=args.stab_repeats,
                                   parsimony_gamma=args.parsimony_gamma)
            
            if len(results) == 0:
                score = -np.inf
                Kbest = None
                print("  ❌ No valid K selection results")
            else:
                Kbest, stat = max(results.items(), key=lambda kv: kv[1]['score'])
                score = stat['score']
                print(f"  ✅ Best K={Kbest}, Score={score:.4f}")
                print(f"     Stability={stat['stability']:.3f}, Silhouette={stat['sil']:.3f}, Cramér's V={stat['cramers_v']:.3f}")
            
            # Track best configuration
            if score > best_score:
                best_score = score
                best_config = cfg.copy()
                best_K = Kbest
                print(f"  🏆 NEW BEST! Score={best_score:.4f}, K={best_K}")
            
            # Store results
            result_row = {
                'config_id': c['config_id'],
                'lr': c['lr'], 'batch_size': c['bs'], 'ortho_w': c['ortho_w'],
                'vic_inv': c['vinv'], 'vic_var': c['vvar'], 'vic_cov': c['vcov'],
                'dropout': c['dropout'], 'augment_prob': c['augp'], 'target_len': c['target_len'],
                'scheduler': c['scheduler'], 'l2': c['l2'],
                'best_K': Kbest, 'val_score': score
            }
            
            # Add K-specific metrics if available
            if Kbest is not None and Kbest in results:
                result_row.update({
                    'stability': results[Kbest]['stability'],
                    'silhouette': results[Kbest]['sil'],
                    'cramers_v': results[Kbest]['cramers_v']
                })
            else:
                result_row.update({
                    'stability': np.nan,
                    'silhouette': np.nan,
                    'cramers_v': np.nan
                })
            
            all_results.append(result_row)
            
        except Exception as e:
            print(f"  ❌ Configuration {c['config_id']} failed: {e}")
            # Still record the failure
            all_results.append({
                'config_id': c['config_id'],
                'lr': c['lr'], 'batch_size': c['bs'], 'ortho_w': c['ortho_w'],
                'vic_inv': c['vinv'], 'vic_var': c['vvar'], 'vic_cov': c['vcov'],
                'dropout': c['dropout'], 'augment_prob': c['augp'], 'target_len': c['target_len'],
                'scheduler': c['scheduler'], 'l2': c['l2'],
                'best_K': None, 'val_score': -np.inf,
                'stability': np.nan, 'silhouette': np.nan, 'cramers_v': np.nan
            })
        
        # Save intermediate results every 20 configs
        if ci % 20 == 0 or ci == len(cfg_list):
            df_results = pd.DataFrame(all_results)
            df_results.to_csv(os.path.join(ROOT, f"hyperparameter_search_results_intermediate_{ci}.csv"), index=False)
            print(f"  💾 Saved intermediate results ({ci} configs completed)")

    # Final results
    print(f"\n{'='*80}")
    print("HYPERPARAMETER SEARCH COMPLETED")
    print(f"{'='*80}")
    
    if best_config is not None:
        print(f"🏆 BEST CONFIGURATION (Score: {best_score:.4f}, K: {best_K}):")
        print(f"   Learning Rate: {best_config['learning_rate']:.1e}")
        print(f"   Batch Size: {best_config['batch_size']}")
        print(f"   Orthogonal Weight: {best_config['ortho_w']}")
        print(f"   VICReg Weights: inv={best_config['vic_inv']}, var={best_config['vic_var']}, cov={best_config['vic_cov']}")
        print(f"   Dropout: {best_config['dropout_rate']}")
        print(f"   Augment Prob: {best_config['augment_prob']}")
        print(f"   Target Length: {best_config['target_len']}")
        print(f"   Scheduler: {best_config['scheduler']}")
        print(f"   L2 Weight: {best_config['l2_weight']:.1e}")
    else:
        print("❌ No valid configuration found!")

    # Save final results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(ROOT, "hyperparameter_search_results_final.csv"), index=False)
    
    # Save best configuration
    if best_config is not None:
        best_summary = {
            'best_score': float(best_score),
            'best_K': int(best_K) if best_K is not None else None,
            'best_config': best_config,
            'total_configs_tested': len(all_results),
            'successful_configs': len([r for r in all_results if r['val_score'] > -np.inf])
        }
        with open(os.path.join(ROOT, "best_hyperparameters.json"), "w") as f:
            json.dump(best_summary, f, indent=2)
    
    print(f"\n📁 All results saved to: {os.path.abspath(ROOT)}")
    print(f"   - hyperparameter_search_results_final.csv")
    print(f"   - best_hyperparameters.json")

if __name__ == "__main__":
    main()
