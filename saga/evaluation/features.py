"""Turns an `ErrorCurveDataset` (loaded via a DataLoader) into flat (X, y, uids) arrays."""

from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader

from ..features.error_curve import flatten_stats


def extract_flat_xy(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Flattens each clip's (T, D) windowed features to a (3*D,) mean/std/max vector."""
    xs, ys = [], []
    for batch in loader:
        feat = batch["features"].numpy()
        mask = batch["mask"].numpy()
        for i in range(len(feat)):
            xs.append(flatten_stats(feat[i], mask[i]))
        ys.extend(batch["label"].numpy().tolist())
    return np.array(xs), np.array(ys)


def add_sync_features(x: np.ndarray, uids: list[str], sync_features: dict) -> np.ndarray:
    """Concatenates the 2-d [confidence, min_dist] SyncNet scalars onto each row of X."""
    extra = np.array([sync_features.get(u, [0.0, 0.0]) for u in uids])
    return np.concatenate([x, extra], axis=1)


def sync_confidence_only(uids: list[str], sync_features: dict) -> np.ndarray:
    """The raw SyncNet confidence scalar alone, for the out-of-the-box threshold baseline."""
    return np.array([sync_features.get(u, [0.0, 0.0])[0] for u in uids])
