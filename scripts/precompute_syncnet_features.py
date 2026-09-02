#!/usr/bin/env python
"""
Scores every clip in a metadata CSV with the frozen, off-the-shelf SyncNet
model (Section 3.3) and writes `{uid: [confidence, min_dist]}` to a JSON
file.

Usage:
  python scripts/precompute_syncnet_features.py \\
      --meta-csv data/error_meta.csv \\
      --checkpoint external/syncnet/syncnet_v2.model \\
      --output data/sync_features.json
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import python_speech_features
import torch
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saga.models.syncnet import SyncNetScorer


def extract_frames_and_audio(video_path: str, tmp_dir: Path) -> tuple[list, np.ndarray, int] | None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-nostdin", "-y", "-i", video_path,
         "-f", "image2", str(tmp_dir / "%06d.jpg")],
        check=False,
    )
    wav_path = tmp_dir / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-nostdin", "-y", "-i", video_path,
         "-async", "1", "-ac", "1", "-vn", "-acodec", "pcm_s16le", "-ar", "16000", str(wav_path)],
        check=False,
    )
    frames = []
    for fp in sorted(tmp_dir.glob("*.jpg")):
        img = cv2.imread(str(fp))
        if img is not None:
            frames.append(cv2.resize(img, (224, 224)))
    if not wav_path.exists():
        return None
    sr, audio = wavfile.read(str(wav_path))
    return frames, audio, sr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta-csv", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scorer = SyncNetScorer(args.checkpoint, device=device)

    out_path = Path(args.output)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    rows = list(csv.DictReader(open(args.meta_csv)))
    tmp_root = Path(tempfile.mkdtemp(prefix="syncnet_"))

    for i, row in enumerate(rows):
        uid = row["uid"]
        if uid in results:
            continue
        try:
            extracted = extract_frames_and_audio(row["video_path"], tmp_root / uid)
            if extracted is None:
                continue
            frames, audio, sr = extracted
            if sr != 16000:
                import librosa

                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
            mfcc = np.stack([np.array(c) for c in zip(*python_speech_features.mfcc(audio, 16000))])
            conf, min_dist = scorer.score(frames, mfcc)
            results[uid] = [conf, min_dist]
        except Exception as e:  # noqa: BLE001 — best-effort batch scoring
            print(f"  [warn] {uid}: {e}")
        finally:
            subprocess.run(["rm", "-rf", str(tmp_root / uid)], check=False)

        if (i + 1) % 100 == 0:
            out_path.write_text(json.dumps(results))
            print(f"  {i + 1}/{len(rows)} ({len(results)} scored)")

    out_path.write_text(json.dumps(results))
    print(f"Done. {len(results)}/{len(rows)} clips scored -> {out_path}")


if __name__ == "__main__":
    main()
