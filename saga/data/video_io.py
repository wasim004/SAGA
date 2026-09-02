"""Minimal video/audio frame-accurate loading, used by the sliding-window datasets."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch


def get_frame_count(video_path: str | Path) -> int:
    """Total frame count of a video file."""
    try:
        import decord

        return len(decord.VideoReader(str(video_path), ctx=decord.cpu(0)))
    except Exception:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n


def load_video_frames(video_path: str | Path, start_frame: int, n_frames: int) -> torch.Tensor:
    """Load `n_frames` frames starting at `start_frame`. Returns (n_frames, H, W, 3) uint8."""
    try:
        import decord

        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        indices = list(range(start_frame, min(start_frame + n_frames, len(vr))))
        while len(indices) < n_frames:
            indices.append(indices[-1] if indices else 0)
        return vr.get_batch(indices).byte()
    except Exception:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        while len(frames) < n_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        return torch.from_numpy(np.stack(frames)).byte()


def load_audio_segment(video_path: str | Path, start_sec: float, duration_sec: float,
                        sample_rate: int = 16000) -> torch.Tensor:
    """Extract a raw mono waveform segment via ffmpeg. Returns (L,) float32 at `sample_rate` Hz."""
    import soundfile as sf

    target_samples = int(duration_sec * sample_rate)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "quiet", "-i", str(video_path),
             "-ss", str(start_sec), "-t", str(duration_sec),
             "-ar", str(sample_rate), "-ac", "1", tmp_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        audio, _ = sf.read(tmp_path, dtype="float32")
    except Exception:
        audio = np.zeros(target_samples, dtype=np.float32)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    return torch.from_numpy(audio[:target_samples].astype(np.float32))
