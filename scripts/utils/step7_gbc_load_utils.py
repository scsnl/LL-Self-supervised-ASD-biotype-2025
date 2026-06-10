#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import Tuple

import numpy as np


def gbc_variants_from_fc_z(fc_z: np.ndarray) -> np.ndarray:
    z = fc_z.copy()
    np.fill_diagonal(z, np.nan)
    gbc_all = np.nanmean(z, axis=1)
    z_pos = np.where(z > 0, z, np.nan)
    gbc_pos = np.nanmean(z_pos, axis=1)
    z_neg = np.where(z < 0, z, np.nan)
    gbc_neg = np.nanmean(z_neg, axis=1)
    return np.vstack([gbc_all, gbc_pos, gbc_neg]).T


def load_gbcs(fc_dir: str, cohort: str = "", use_combat: bool | None = None) -> Tuple[np.ndarray, str]:
    """
    """
    if use_combat is None:
        use_combat = cohort.strip().lower() != "stanford"
    if use_combat:
        combat_a = os.path.join(fc_dir, "gbcs_asd_combat.npy")
        combat_td = os.path.join(fc_dir, "gbcs_td_combat.npy")
        if os.path.exists(combat_a) and os.path.exists(combat_td):
            arr = np.load(combat_a)
            if np.any(np.isfinite(arr)):
                return arr, combat_a

    for fname in ("gbcs_asd_gsr.npy", "gbcs_asd.npy"):
        p = os.path.join(fc_dir, fname)
        if os.path.exists(p):
            return np.load(p), p

    for fname in ("fcs_asd_gsr.npy", "fcs_asd.npy"):
        p = os.path.join(fc_dir, fname)
        if os.path.exists(p):
            fcs = np.load(p)
            gbcs = np.stack([gbc_variants_from_fc_z(fcs[i]) for i in range(fcs.shape[0])], axis=0)
            return gbcs, p

    raise FileNotFoundError(f" {fc_dir}  GBC/FC ")
