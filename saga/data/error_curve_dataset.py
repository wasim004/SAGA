"""
Loads pre-computed per-clip error curves (`saga.scripts.precompute_error_curves`
output: one `.npz` with `c_a2v`, `c_v2a` per clip) and metadata CSV, and
builds the 12-d per-window feature tensor for a given split.

Expected metadata CSV columns: `uid`, `split` (train/val/test), `generator_id`
(0 = real, 1..k = each fake generator), `source` (free-text label).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..features.error_curve import D_IN, build_window_features, pad_or_truncate

GENERATOR_NAMES = {0: "Real", 1: "FaceSwap", 2: "Wav2Lip", 3: "FSGAN"}


class ErrorCurveDataset(Dataset):
    def __init__(
        self,
        meta_csv: str,
        cache_dir: str,
        split: str = "train",  # "train" | "val" | "test" | "all"
        max_windows: int = 16,
        tau_k: float = 1.5,
        generator_ids: Optional[list[int]] = None,
        d_in: int = D_IN,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_windows = max_windows
        self.tau_k = tau_k
        self.d_in = d_in
        self.split = split

        self.samples = []
        with open(meta_csv, newline="") as f:
            for row in csv.DictReader(f):
                if split != "all" and row["split"] != split:
                    continue
                npz = self.cache_dir / f"{row['uid']}.npz"
                if not npz.exists():
                    continue
                gid = int(row["generator_id"])
                if generator_ids is not None and gid not in generator_ids:
                    continue
                self.samples.append({"uid": row["uid"], "generator_id": gid, "npz": npz})

        self._label_counts = np.bincount(
            [s["generator_id"] for s in self.samples], minlength=max(GENERATOR_NAMES) + 1
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        npz = np.load(s["npz"])
        c_a2v = npz["c_a2v"].astype(np.float32)
        c_v2a = npz["c_v2a"].astype(np.float32)

        feat = build_window_features(c_a2v, c_v2a, self.tau_k)[:, : self.d_in]
        feat, mask = pad_or_truncate(feat, self.max_windows)

        return {
            "features": torch.from_numpy(feat).float(),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(s["generator_id"], dtype=torch.long),
            "uid": s["uid"],
        }

    def class_weights(self) -> torch.Tensor:
        counts = np.where(self._label_counts == 0, 1.0, self._label_counts).astype(np.float32)
        return torch.from_numpy(counts.sum() / (len(counts) * counts)).float()


def build_loaders(meta_csv: str, cache_dir: str, batch_size: int, max_windows: int = 16,
                   tau_k: float = 1.5, d_in: int = D_IN, with_val: bool = True):
    """Returns (train_loader, [val_loader,] test_loader, class_weights)."""
    kw = dict(max_windows=max_windows, tau_k=tau_k, d_in=d_in)
    train_ds = ErrorCurveDataset(meta_csv, cache_dir, split="train", **kw)
    test_ds = ErrorCurveDataset(meta_csv, cache_dir, split="test", **kw)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                              num_workers=2, pin_memory=True)

    if not with_val:
        return train_loader, test_loader, train_ds.class_weights()

    val_ds = ErrorCurveDataset(meta_csv, cache_dir, split="val", **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds.class_weights()
