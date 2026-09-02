#!/usr/bin/env python
"""
Domain-matched predictor adaptation (Section 3.1).

Usage:
  python scripts/adapt_predictor.py --config configs/predictor_adapt.yaml \\
      --meta-csv data/error_meta.csv --output-dir runs/predictor_adapt

`--meta-csv` must have `split`, `generator_id`, `video_path` columns; only
rows with split == train and generator_id == 0 (real) are used.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saga.training.adapt_predictor import train_predictors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/predictor_adapt.yaml")
    p.add_argument("--meta-csv", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))

    video_paths = [
        row["video_path"] for row in csv.DictReader(open(args.meta_csv))
        if row["split"] == "train" and row["generator_id"] == "0"
    ]
    print(f"Adapting predictors on {len(video_paths)} real training clips.")

    train_predictors(video_paths=video_paths, output_dir=args.output_dir, **cfg)


if __name__ == "__main__":
    main()
