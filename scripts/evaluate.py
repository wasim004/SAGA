#!/usr/bin/env python
"""
Evaluates RF-300, Logistic Regression, and the raw-SyncNet-threshold
baseline on one protocol (Sections 4.2, 5), with and without SyncNet
features, over the paper's 3 seeds.

Usage (closed-set: random-split or identity-disjoint — pass the CSV whose
`split` column already encodes that protocol's train/test partition):
  python scripts/evaluate.py closed-set \\
      --meta-csv data/error_meta.csv --cache-dir data/cached_errors \\
      --sync-features data/sync_features.json --output results/random_split.json

Usage (open-set: `--meta-csv` train rows exclude `--held-out-label`;
`--test-meta-csv` test rows include it):
  python scripts/evaluate.py open-set \\
      --meta-csv data/error_meta_openset_train.csv \\
      --test-meta-csv data/error_meta_openset.csv \\
      --cache-dir data/cached_errors --held-out-label 3 \\
      --sync-features data/sync_features.json --output results/openset_fsgan.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saga.baselines import (
    closed_set_multiseed, logistic_regression, open_set_multiseed,
    raw_syncnet_threshold_auc, rf300,
)
from saga.data import ErrorCurveDataset
from saga.evaluation import add_sync_features, extract_flat_xy, sync_confidence_only


def load_xy(meta_csv: str, cache_dir: str, split: str, d_in: int = 12,
            max_windows: int = 16, tau_k: float = 1.5):
    ds = ErrorCurveDataset(meta_csv, cache_dir, split=split, d_in=d_in,
                            max_windows=max_windows, tau_k=tau_k)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    x, y = extract_flat_xy(loader)
    uids = [s["uid"] for s in ds.samples]
    return x, y, uids


def run_closed_set(args) -> dict:
    x_tr, y_tr, u_tr = load_xy(args.meta_csv, args.cache_dir, "train", args.d_in,
                                args.max_windows, args.tau_k)
    x_te, y_te, u_te = load_xy(args.meta_csv, args.cache_dir, "test", args.d_in,
                                args.max_windows, args.tau_k)

    out = {"RF-300 (no sync)": closed_set_multiseed(x_tr, y_tr, x_te, y_te, rf300),
           "Logistic Reg. (no sync)": closed_set_multiseed(x_tr, y_tr, x_te, y_te, logistic_regression)}

    if args.sync_features:
        sync = json.loads(Path(args.sync_features).read_text())
        x_tr_s, x_te_s = add_sync_features(x_tr, u_tr, sync), add_sync_features(x_te, u_te, sync)
        out["RF-300 (+ sync)"] = closed_set_multiseed(x_tr_s, y_tr, x_te_s, y_te, rf300)
        out["Logistic Reg. (+ sync)"] = closed_set_multiseed(x_tr_s, y_tr, x_te_s, y_te, logistic_regression)
        sync_conf = sync_confidence_only(u_te, sync)
        out["Raw SyncNet threshold"] = raw_syncnet_threshold_auc(sync_conf, y_te)

    return out


def run_open_set(args) -> dict:
    x_tr, y_tr, u_tr = load_xy(args.meta_csv, args.cache_dir, "train", args.d_in,
                                args.max_windows, args.tau_k)
    x_te, y_te, u_te = load_xy(args.test_meta_csv, args.cache_dir, "test", args.d_in,
                                args.max_windows, args.tau_k)

    out = {"RF-300 (no sync)": open_set_multiseed(x_tr, y_tr, x_te, y_te, rf300, args.held_out_label),
           "Logistic Reg. (no sync)": open_set_multiseed(x_tr, y_tr, x_te, y_te, logistic_regression,
                                                          args.held_out_label)}

    if args.sync_features:
        sync = json.loads(Path(args.sync_features).read_text())
        x_tr_s, x_te_s = add_sync_features(x_tr, u_tr, sync), add_sync_features(x_te, u_te, sync)
        out["RF-300 (+ sync)"] = open_set_multiseed(x_tr_s, y_tr, x_te_s, y_te, rf300, args.held_out_label)
        out["Logistic Reg. (+ sync)"] = open_set_multiseed(x_tr_s, y_tr, x_te_s, y_te,
                                                             logistic_regression, args.held_out_label)

    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="protocol", required=True)

    common = dict(required=True)
    for name in ("closed-set", "open-set"):
        sp = sub.add_parser(name)
        sp.add_argument("--meta-csv", **common)
        sp.add_argument("--cache-dir", **common)
        sp.add_argument("--sync-features", default=None)
        sp.add_argument("--output", **common)
        sp.add_argument("--d-in", type=int, default=12)
        sp.add_argument("--max-windows", type=int, default=16)
        sp.add_argument("--tau-k", type=float, default=1.5)
        if name == "open-set":
            sp.add_argument("--test-meta-csv", required=True)
            sp.add_argument("--held-out-label", type=int, required=True)

    args = p.parse_args()
    results = run_closed_set(args) if args.protocol == "closed-set" else run_open_set(args)

    for name, metrics in results.items():
        print(f"{name}: {metrics}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
