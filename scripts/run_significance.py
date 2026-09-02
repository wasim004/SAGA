#!/usr/bin/env python
"""
Paired bootstrap significance test (Section 6.3) between two configurations
(each: RF-300 or Logistic Regression, with or without SyncNet features) on
one closed-set protocol.

Usage:
  python scripts/run_significance.py \\
      --meta-csv data/error_meta.csv --cache-dir data/cached_errors \\
      --sync-features data/sync_features.json \\
      --clf-a logreg --sync-a --clf-b rf300 --sync-b \\
      --output results/significance_random_split.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saga.baselines import fit_predict, logistic_regression, rf300
from saga.data import ErrorCurveDataset
from saga.evaluation import add_sync_features, extract_flat_xy, paired_bootstrap

CLASSIFIERS = {"rf300": rf300, "logreg": logistic_regression}


def load_xy(meta_csv, cache_dir, split, d_in=12, max_windows=16, tau_k=1.5):
    ds = ErrorCurveDataset(meta_csv, cache_dir, split=split, d_in=d_in,
                            max_windows=max_windows, tau_k=tau_k)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    x, y = extract_flat_xy(loader)
    uids = [s["uid"] for s in ds.samples]
    return x, y, uids


def score(clf_name, use_sync, x_tr, y_tr, x_te, u_tr, u_te, sync):
    if use_sync:
        x_tr, x_te = add_sync_features(x_tr, u_tr, sync), add_sync_features(x_te, u_te, sync)
    clf = CLASSIFIERS[clf_name](random_state=42)
    _, proba = fit_predict(clf, x_tr, y_tr, x_te)
    return 1.0 - proba[:, 0]  # P(fake)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta-csv", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--sync-features", required=True)
    p.add_argument("--clf-a", choices=CLASSIFIERS, required=True)
    p.add_argument("--sync-a", action="store_true")
    p.add_argument("--clf-b", choices=CLASSIFIERS, required=True)
    p.add_argument("--sync-b", action="store_true")
    p.add_argument("--n-resamples", type=int, default=5000)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    sync = json.loads(Path(args.sync_features).read_text())
    x_tr, y_tr, u_tr = load_xy(args.meta_csv, args.cache_dir, "train")
    x_te, y_te, u_te = load_xy(args.meta_csv, args.cache_dir, "test")
    bin_true = (y_te != 0).astype(int)

    score_a = score(args.clf_a, args.sync_a, x_tr, y_tr, x_te, u_tr, u_te, sync)
    score_b = score(args.clf_b, args.sync_b, x_tr, y_tr, x_te, u_tr, u_te, sync)

    result = paired_bootstrap(bin_true, score_a, score_b, n_resamples=args.n_resamples)
    name_a = f"{args.clf_a}{'+sync' if args.sync_a else ''}"
    name_b = f"{args.clf_b}{'+sync' if args.sync_b else ''}"
    print(f"{name_a} ({result['auc_a']*100:.1f}%) vs {name_b} ({result['auc_b']*100:.1f}%): "
          f"diff 95% CI [{result['diff_ci_low']*100:+.1f}, {result['diff_ci_high']*100:+.1f}]pp, "
          f"P({name_a}>{name_b})={result['p_a_better']:.3f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
