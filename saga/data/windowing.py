"""Sliding-window enumeration shared by the adaptation dataset and feature extraction."""

from __future__ import annotations


def enumerate_windows(n_total_frames: int, window_frames: int = 16,
                       stride_frames: int = 8, max_windows: int | None = None) -> list[int]:
    """
    Start indices of every window of `window_frames` that fits in a clip of
    `n_total_frames` frames, at `stride_frames` stride.

    With `max_windows=None` (the default here), this enumerates *every*
    candidate window — the un-capped pool that predictor adaptation
    subsamples from (Section 3.1: "10,578 candidate sliding windows
    subsampled to 2,000"). Pass `max_windows=16` to instead reproduce the
    capped-at-16 windowing used for downstream classifier features
    (Section 3.1: "up to a maximum of 16 windows per clip").
    """
    starts = list(range(0, max(1, n_total_frames - window_frames + 1), stride_frames))
    if not starts:
        starts = [0]
    if max_windows is not None:
        starts = starts[:max_windows]
    return starts
