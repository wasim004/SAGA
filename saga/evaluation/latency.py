"""Single-clip inference latency benchmarking (Section 6.4), batch size 1."""

from __future__ import annotations

import time

import torch


def time_sklearn_predict(model, x, n_reps: int = 200, n_warmup: int = 5) -> float:
    """Returns ms/clip for `model.predict_proba` on a single row, batch size 1."""
    for _ in range(n_warmup):
        model.predict_proba(x[:1])
    t0 = time.perf_counter()
    for i in range(n_reps):
        model.predict_proba(x[i % len(x): i % len(x) + 1])
    t1 = time.perf_counter()
    return (t1 - t0) / n_reps * 1000.0


@torch.no_grad()
def time_torch_forward(model, forward_fn, n_reps: int = 200, n_warmup: int = 5,
                        device: str | torch.device = "cpu") -> float:
    """
    Returns ms/clip for a torch model's forward pass, batch size 1.
    `forward_fn` is called with no arguments and should invoke `model(...)`
    on pre-moved-to-`device` inputs captured in its closure.
    """
    model.eval()
    for _ in range(n_warmup):
        forward_fn()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_reps):
        forward_fn()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / n_reps * 1000.0
