"""
Paired bootstrap significance testing (Section 6.3): for two classifiers
scored on the same test clips, resamples clip indices with replacement (the
same resample applied to both scores each draw, since they share the same
easy/hard clips) and reports the empirical 95% CI of the AUC difference.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def paired_bootstrap(bin_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
                      n_resamples: int = 5000, seed: int = 0) -> dict:
    """
    bin_true         : (N,) binary ground truth.
    score_a, score_b : (N,) real-valued scores from the two classifiers being compared.

    Returns point AUCs for each, the 95% CI of (auc_a - auc_b), and the
    fraction of resamples where a scored higher than b.
    """
    rng = np.random.default_rng(seed)
    n = len(bin_true)
    auc_a0 = roc_auc_score(bin_true, score_a)
    auc_b0 = roc_auc_score(bin_true, score_b)

    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        yb = bin_true[idx]
        if yb.min() == yb.max():  # degenerate resample, no positive/negative split
            diffs[i] = np.nan
            continue
        diffs[i] = roc_auc_score(yb, score_a[idx]) - roc_auc_score(yb, score_b[idx])
    diffs = diffs[~np.isnan(diffs)]

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "auc_a": float(auc_a0), "auc_b": float(auc_b0),
        "diff_ci_low": float(lo), "diff_ci_high": float(hi),
        "p_a_better": float(np.mean(diffs > 0)),
    }
