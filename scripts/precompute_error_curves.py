#!/usr/bin/env python
"""
Runs the domain-matched predictors over every clip in a metadata CSV and
caches its (C_a2v, C_v2a) error curve to `<cache_dir>/<uid>.npz`.

Usage:
  python scripts/precompute_error_curves.py \\
      --meta-csv data/error_meta.csv \\
      --checkpoint runs/predictor_adapt/best.pt \\
      --cache-dir data/cached_errors
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saga.data.video_io import get_frame_count, load_audio_segment, load_video_frames
from saga.data.windowing import enumerate_windows
from saga.models.encoders import DualEncoder
from saga.models.predictors import DomainMatchedPredictors


def score_clip(video_path: str, encoder: DualEncoder, predictors: DomainMatchedPredictors,
                device: torch.device, window_frames: int, stride_frames: int, max_windows: int,
                fps: int, sample_rate: int) -> tuple[np.ndarray, np.ndarray] | None:
    n_total = get_frame_count(video_path)
    starts = enumerate_windows(n_total, window_frames, stride_frames, max_windows=max_windows)

    c_a2v_wins, c_v2a_wins = [], []
    with torch.no_grad():
        for start in starts:
            frames = load_video_frames(video_path, start, window_frames).unsqueeze(0).to(device)
            waveform = load_audio_segment(
                video_path, start / fps, window_frames / fps, sample_rate
            ).unsqueeze(0).to(device)
            z_v, z_a = encoder(frames, waveform)
            out = predictors(z_v, z_a)
            c_a2v_wins.append(float(out["C_a2v"].mean().item()))
            c_v2a_wins.append(float(out["C_v2a"].mean().item()))

    if not c_a2v_wins:
        return None
    return np.array(c_a2v_wins, dtype=np.float32), np.array(c_v2a_wins, dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta-csv", required=True, help="CSV with uid, video_path columns")
    p.add_argument("--checkpoint", required=True, help="Predictor checkpoint from adapt_predictor.py")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--window-frames", type=int, default=16)
    p.add_argument("--stride-frames", type=int, default=8)
    p.add_argument("--max-windows", type=int, default=16)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    encoder = DualEncoder(proj_dim=cfg.get("proj_dim", 384), frozen=True).to(device)
    encoder.audio_proj.load_state_dict(ckpt["audio_proj_state"])
    predictors = DomainMatchedPredictors(
        d_model=cfg.get("proj_dim", 384), d_state=cfg.get("d_state", 16),
        d_conv=cfg.get("d_conv", 4), expand=cfg.get("expand", 2),
        n_layers=cfg.get("n_layers", 2), dropout=cfg.get("dropout", 0.1),
    ).to(device)
    predictors.load_state_dict(ckpt["predictors_state"])
    predictors.eval()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.meta_csv)))
    n_ok = n_fail = 0
    for i, row in enumerate(rows):
        uid = row["uid"]
        out_path = cache_dir / f"{uid}.npz"
        if out_path.exists():
            n_ok += 1
            continue
        result = score_clip(
            row["video_path"], encoder, predictors, device,
            args.window_frames, args.stride_frames, args.max_windows, args.fps, args.sample_rate,
        )
        if result is None:
            n_fail += 1
            continue
        c_a2v, c_v2a = result
        np.savez(out_path, c_a2v=c_a2v, c_v2a=c_v2a)
        n_ok += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}  (ok={n_ok}, failed={n_fail})")

    print(f"Done. {n_ok} ok, {n_fail} failed. Cache: {cache_dir}")


if __name__ == "__main__":
    main()
