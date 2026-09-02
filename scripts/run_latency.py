#!/usr/bin/env python
"""
Single-clip inference latency benchmark for RF-300, Logistic Regression, and
the Hybrid BiLSTM (Section 6.4), batch size 1, 200 timed calls after warmup.

Usage:
  python scripts/run_latency.py --meta-csv data/error_meta.csv \\
      --cache-dir data/cached_errors --sync-features data/sync_features.json \\
      --output results/latency_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saga.baselines import fit_predict, logistic_regression, rf300
from saga.data import ErrorCurveDataset
from saga.evaluation import add_sync_features, extract_flat_xy, time_sklearn_predict, time_torch_forward
from saga.models.error_signature_encoder import HybridBiLSTM


def load_xy(meta_csv, cache_dir, split, d_in=12, max_windows=16, tau_k=1.5):
    ds = ErrorCurveDataset(meta_csv, cache_dir, split=split, d_in=d_in,
                            max_windows=max_windows, tau_k=tau_k)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    x, y = extract_flat_xy(loader)
    uids = [s["uid"] for s in ds.samples]
    return x, y, uids, ds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta-csv", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--sync-features", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    sync = json.loads(Path(args.sync_features).read_text())
    x_tr, y_tr, u_tr, _ = load_xy(args.meta_csv, args.cache_dir, "train")
    x_te, y_te, u_te, test_ds = load_xy(args.meta_csv, args.cache_dir, "test")
    x_tr_s = add_sync_features(x_tr, u_tr, sync)
    x_te_s = add_sync_features(x_te, u_te, sync)

    results = {}

    rf = rf300(random_state=42)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_tr_s)
    rf.fit(scaler.transform(x_tr_s), y_tr)
    results["RF-300 (+sync), CPU, ms/clip"] = time_sklearn_predict(rf, scaler.transform(x_te_s))
    results["RF-300 total tree nodes"] = int(sum(t.tree_.node_count for t in rf.estimators_))

    lr = logistic_regression(random_state=42)
    lr.fit(scaler.transform(x_tr_s), y_tr)
    results["Logistic Regression (+sync), CPU, ms/clip"] = time_sklearn_predict(lr, scaler.transform(x_te_s))

    device_cpu = torch.device("cpu")
    model = HybridBiLSTM(d_in=12, d_model=128, n_layers=3, d_embed=128, n_classes=4).to(device_cpu)
    batch = next(iter(DataLoader(test_ds, batch_size=1)))
    x = batch["features"].to(device_cpu)
    mask = batch["mask"].to(device_cpu)
    sync_t = torch.tensor([sync.get(batch["uid"][0], [0.0, 0.0])], dtype=torch.float32)

    results["Hybrid BiLSTM (+sync), CPU, ms/clip"] = time_torch_forward(
        model, lambda: model(x, mask, sync=sync_t), device=device_cpu,
    )
    results["Hybrid BiLSTM param count"] = model.count_params()

    if torch.cuda.is_available():
        device_gpu = torch.device("cuda")
        model_gpu = model.to(device_gpu)
        x_g, mask_g, sync_g = x.to(device_gpu), mask.to(device_gpu), sync_t.to(device_gpu)
        results["Hybrid BiLSTM (+sync), GPU, ms/clip"] = time_torch_forward(
            model_gpu, lambda: model_gpu(x_g, mask_g, sync=sync_g), device=device_gpu,
        )

    for k, v in results.items():
        print(f"{k}: {v}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
