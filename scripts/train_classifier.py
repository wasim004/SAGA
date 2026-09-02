#!/usr/bin/env python
"""
Trains the Hybrid BiLSTM classifier (Section 3.4).

Usage:
  python scripts/train_classifier.py --config configs/hybrid_bilstm.yaml \\
      --meta-csv data/error_meta.csv --cache-dir data/cached_errors \\
      --sync-features data/sync_features.json \\
      --output-dir runs/hybrid_bilstm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saga.training.train_classifier import train_classifier


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/hybrid_bilstm.yaml")
    p.add_argument("--meta-csv", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--sync-features", default=None,
                    help="Path to sync_features.json; omit for open-set (no-sync) training")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    sync_features = json.loads(Path(args.sync_features).read_text()) if args.sync_features else None

    train_classifier(
        meta_csv=args.meta_csv, cache_dir=args.cache_dir, output_dir=args.output_dir,
        sync_features=sync_features, **cfg,
    )


if __name__ == "__main__":
    main()
