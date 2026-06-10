#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

from __future__ import annotations

import numpy as np


def remove_outliers_iqr(
    values: np.ndarray,
    labels: np.ndarray,
    factor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Returns:
    """
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    outlier_mask = np.zeros(len(values), dtype=bool)

    for label in np.unique(labels):
        mask = labels == label
        group_values = values[mask]
        if len(group_values) > 4:
            q1, q3 = np.percentile(group_values, [25, 75])
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            outlier_mask[mask] = (group_values < lower) | (group_values > upper)

    return values[~outlier_mask], labels[~outlier_mask], outlier_mask


def iqr_outlier_mask_1d(values: np.ndarray, factor: float = 2.0) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    mask = np.zeros(len(v), dtype=bool)
    finite = np.isfinite(v)
    if finite.sum() <= 4:
        return mask
    vf = v[finite]
    q1, q3 = np.percentile(vf, [25, 75])
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    mask[finite] = (vf < lower) | (vf > upper)
    return mask


def bivariate_outlier_mask(
    x: np.ndarray,
    y: np.ndarray,
    factor: float = 2.0,
) -> np.ndarray:
    return iqr_outlier_mask_1d(x, factor) | iqr_outlier_mask_1d(y, factor)
