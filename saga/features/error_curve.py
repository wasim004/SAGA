"""
Turns a clip's bivariate error curve (C_a2v(t), C_v2a(t)) — the output of
`saga.models.predictors.DomainMatchedPredictors` — into the 12-d per-window
feature vector and 36-d flat clip-level summary used throughout the paper
(Section 3.2).

12-d per-window features:
  [0]  C_a2v
  [1]  C_v2a
  [2]  C_a2v + C_v2a                        (summed error)
  [3]  C_a2v - C_v2a                        (signed asymmetry)
  [4]  C_v2a - clip_mean(C_v2a)             (deviation from clip baseline)
  [5]  C_a2v - clip_mean(C_a2v)
  [6]  flag = 1[(C_a2v+C_v2a) > mu + tau_k*sigma]   (adaptive per-clip threshold)
  [7]  C_v2a * flag                         (gated)
  [8]  C_a2v * flag                         (gated)
  [9]  normalized temporal position in [0, 1]
  [10] C_a2v shifted by 1 window (lag-1; 0 for the first window)
  [11] C_v2a shifted by 1 window (lag-1; 0 for the first window)

36-d flat statistics: mean, standard deviation, and maximum of each of the
12 features above, over all valid (non-padded) windows in the clip.
"""

from __future__ import annotations

import numpy as np

D_IN = 12


def build_window_features(c_a2v: np.ndarray, c_v2a: np.ndarray,
                           tau_k: float = 1.5) -> np.ndarray:
    """c_a2v, c_v2a: (T,) per-window error curves for one clip -> (T, 12) float32."""
    n = len(c_a2v)
    eps = 1e-6
    c_i = c_a2v + c_v2a

    tau = c_i.mean() + tau_k * (c_i.std() + eps)
    flag = (c_i > tau).astype(np.float32)

    mean_a2v, mean_v2a = c_a2v.mean(), c_v2a.mean()
    pos = np.linspace(0.0, 1.0, n, dtype=np.float32)
    a2v_lag1 = np.concatenate([[0.0], c_a2v[:-1]]).astype(np.float32)
    v2a_lag1 = np.concatenate([[0.0], c_v2a[:-1]]).astype(np.float32)

    return np.stack(
        [
            c_a2v, c_v2a, c_i,
            c_a2v - c_v2a,
            c_v2a - mean_v2a,
            c_a2v - mean_a2v,
            flag,
            c_v2a * flag,
            c_a2v * flag,
            pos,
            a2v_lag1,
            v2a_lag1,
        ],
        axis=1,
    ).astype(np.float32)


def pad_or_truncate(features: np.ndarray, max_windows: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """features: (T, D) -> (features padded to (max_windows, D), mask (max_windows,) bool)."""
    n_keep = min(len(features), max_windows)
    feat = features[:n_keep]
    mask = np.ones(n_keep, dtype=bool)
    if n_keep < max_windows:
        pad = max_windows - n_keep
        feat = np.concatenate([feat, np.zeros((pad, feat.shape[1]), dtype=np.float32)], axis=0)
        mask = np.concatenate([mask, np.zeros(pad, dtype=bool)])
    return feat, mask


def flatten_stats(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(T, D) features + (T,) bool mask -> (3*D,) [mean, std, max] over valid windows."""
    valid = features[mask]
    if len(valid) == 0:
        valid = features
    return np.concatenate([valid.mean(0), valid.std(0), valid.max(0)]).astype(np.float32)
