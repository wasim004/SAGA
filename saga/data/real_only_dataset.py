"""
Real-only sliding-window dataset for domain-matched predictor adaptation
(Section 3.1).

Every window is drawn from a real (unmanipulated) clip in the *training*
split of the dataset being evaluated — never from validation, test, or a
different dataset — which is the precondition this paper identifies for the
zero-fake-data predictor-residual approach to carry any signal at all.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from .video_io import get_frame_count, load_audio_segment, load_video_frames
from .windowing import enumerate_windows


@dataclass
class _Window:
    video_path: str
    start_frame: int


class RealOnlyAdaptationDataset(Dataset):
    """
    Enumerates every sliding window across a list of real-clip video paths,
    then subsamples down to `n_samples` windows (default: 2,000, matching
    the paper).

    Expects clips to be single muxed audio-video files (e.g. .mp4). For
    frame-directory + separate-audio corpora (as DF-TIMIT ships), preprocess
    to muxed clips first, or subclass and override `_frame_count`/`_load`.
    """

    def __init__(
        self,
        video_paths: list[str],
        window_frames: int = 16,
        stride_frames: int = 8,
        n_samples: Optional[int] = 2000,
        fps: int = 25,
        sample_rate: int = 16000,
        seed: int = 42,
    ):
        self.window_frames = window_frames
        self.stride_frames = stride_frames
        self.fps = fps
        self.sample_rate = sample_rate

        self.windows: list[_Window] = []
        for path in video_paths:
            try:
                n_total = get_frame_count(path)
            except Exception:
                continue
            for start in enumerate_windows(n_total, window_frames, stride_frames,
                                            max_windows=None):
                self.windows.append(_Window(video_path=path, start_frame=start))

        if n_samples is not None and len(self.windows) > n_samples:
            random.Random(seed).shuffle(self.windows)
            self.windows = self.windows[:n_samples]

    @classmethod
    def from_metadata_csv(cls, csv_path: str, split_col: str = "split",
                           path_col: str = "video_path", label_col: str = "generator_id",
                           real_label: str = "0", train_value: str = "train", **kwargs):
        """
        Convenience constructor: reads a CSV with (at least) `split_col`,
        `path_col`, and `label_col` columns, and keeps only rows where
        split == train_value and label == real_label.
        """
        video_paths = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get(split_col) == train_value and row.get(label_col) == real_label:
                    video_paths.append(row[path_col])
        return cls(video_paths, **kwargs)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        w = self.windows[idx]
        frames = load_video_frames(w.video_path, w.start_frame, self.window_frames)
        waveform = load_audio_segment(
            w.video_path, w.start_frame / self.fps, self.window_frames / self.fps, self.sample_rate
        )
        return {"frames": frames, "waveform": waveform, "label": torch.tensor(0.0)}


def collate_real_only(batch: list[dict]) -> dict:
    """Pads waveforms to the shortest clip in the batch (all windows share window_frames)."""
    frames = torch.stack([b["frames"].float() for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    min_len = min(b["waveform"].shape[0] for b in batch)
    waveform = torch.stack([b["waveform"][:min_len] for b in batch])
    return {"frames": frames, "waveform": waveform, "label": labels}
