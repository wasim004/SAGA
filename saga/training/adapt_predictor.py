"""
Domain-matched predictor adaptation (Section 3.1): trains T_a2v / T_v2a on
real-only clips from the *target* dataset's own training split.

This is Phase 1 only — the frozen DINOv2/wav2vec2 encoders and the audio
projection head are never fine-tuned end-to-end here beyond what
`DualEncoder.audio_proj` already trains; only `DomainMatchedPredictors`'
parameters are optimized.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data.real_only_dataset import RealOnlyAdaptationDataset, collate_real_only
from ..models.encoders import DualEncoder
from ..models.predictors import DomainMatchedPredictors
from .losses import AdaptationLoss


def _warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int, steps_per_epoch: int):
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(1, total_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_predictors(
    video_paths: list[str],
    output_dir: str,
    proj_dim: int = 384,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    n_layers: int = 2,
    dropout: float = 0.1,
    window_frames: int = 16,
    stride_frames: int = 8,
    n_samples: int = 2000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 3,
    epochs: int = 30,
    batch_size: int = 16,
    clip_grad: float = 0.5,
    calibration_weight: float = 0.01,
    seed: int = 42,
    device: str | None = None,
    log_every: int = 50,
) -> Path:
    """
    Adapts T_a2v/T_v2a on `video_paths` (real clips only, training split of
    the target dataset) and saves the best checkpoint (by running average
    training loss) to `output_dir/best.pt`. Returns that path.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)

    dataset = RealOnlyAdaptationDataset(
        video_paths, window_frames=window_frames, stride_frames=stride_frames,
        n_samples=n_samples, seed=seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                         collate_fn=collate_real_only, num_workers=4, pin_memory=True)
    print(f"[adapt_predictor] {len(dataset)} candidate windows -> {len(loader)} batches/epoch")

    encoder = DualEncoder(proj_dim=proj_dim, frozen=True).to(device)
    predictors = DomainMatchedPredictors(
        d_model=proj_dim, d_state=d_state, d_conv=d_conv, expand=expand,
        n_layers=n_layers, dropout=dropout,
    ).to(device)

    trainable = list(encoder.audio_proj.parameters()) + list(predictors.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = _warmup_cosine_scheduler(optimizer, warmup_epochs, epochs, len(loader))
    loss_fn = AdaptationLoss(calibration_weight=calibration_weight)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    step = 0

    for epoch in range(1, epochs + 1):
        predictors.train()
        t0 = time.time()
        epoch_loss = 0.0
        for batch in loader:
            frames = batch["frames"].to(device)
            waveform = batch["waveform"].to(device)

            z_v, z_a = encoder(frames, waveform)
            out = predictors(z_v, z_a)
            losses = loss_fn(out, z_v, z_a)

            optimizer.zero_grad()
            losses["loss"].backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(trainable, clip_grad)
            optimizer.step()
            scheduler.step()

            epoch_loss += losses["loss"].item()
            step += 1
            if step % log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  step {step:6d} | loss {losses['loss'].item():.4f} "
                      f"| l_pred {losses['l_pred'].item():.4f} "
                      f"| l_calib {losses['l_calib'].item():.4f} | lr {lr_now:.2e}")

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch:3d}/{epochs} | avg_loss {avg_loss:.4f} | {time.time() - t0:.1f}s")

        checkpoint = {
            "epoch": epoch,
            "predictors_state": predictors.state_dict(),
            "audio_proj_state": encoder.audio_proj.state_dict(),
            "avg_loss": avg_loss,
            "config": {
                "proj_dim": proj_dim, "d_state": d_state, "d_conv": d_conv,
                "expand": expand, "n_layers": n_layers, "dropout": dropout,
            },
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, out_dir / "best.pt")
            print(f"  new best ({best_loss:.4f}) saved")

    print(f"[adapt_predictor] Done. Best avg_loss={best_loss:.4f}. Checkpoint: {out_dir / 'best.pt'}")
    return out_dir / "best.pt"
