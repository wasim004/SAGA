"""
Hybrid BiLSTM classifier (Section 3.4).

A 3-layer bidirectional LSTM processes the raw 12-d per-window error
sequence and is masked-mean-pooled into a temporal embedding. This is
concatenated with the same 36-d flat mean/std/max statistics RF-300 uses
(`saga.features.error_curve.flatten_stats`), and optionally the 2-d SyncNet
scalars, before a small feed-forward projection and a final linear
classification head.

This design is motivated by an empirical finding reported in the paper: a
temporal encoder given only the raw per-window sequence has to learn to
reconstruct aggregate statistics implicitly, and underperforms a plain
Random Forest given those statistics directly. Handing the LSTM the same
flat statistics explicitly closes most of that gap.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HybridBiLSTM(nn.Module):
    def __init__(
        self,
        d_in: int = 12,
        d_model: int = 128,
        n_layers: int = 3,
        d_embed: int = 128,
        n_classes: int = 4,
        dropout: float = 0.2,
        sync_dim: int = 2,
    ):
        super().__init__()
        self.d_in = d_in
        self.sync_dim = sync_dim
        self.d_embed = d_embed
        self.n_classes = n_classes

        self.input_proj = nn.Sequential(nn.Linear(d_in, d_model), nn.LayerNorm(d_model))
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,  # x2 directions -> d_model output
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        d_flat = d_in * 3 + sync_dim  # mean/std/max of the d_in raw features, + sync scalars
        self.flat_norm = nn.LayerNorm(d_flat)
        self.flat_proj = nn.Sequential(
            nn.Linear(d_flat, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.embed_proj = nn.Sequential(
            nn.LayerNorm(d_model + d_model // 2),
            nn.Linear(d_model + d_model // 2, d_embed),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attr_head = nn.Linear(d_embed, n_classes)

    @staticmethod
    def _masked_flat_stats(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_in), mask: (B, T) bool -> (B, 3*d_in) mean/std/max."""
        mask_f = mask.unsqueeze(-1).float()
        n = mask_f.sum(1).clamp(min=1)
        mean = (x * mask_f).sum(1) / n
        var = ((x - mean.unsqueeze(1)) ** 2 * mask_f).sum(1) / n
        std = torch.sqrt(var.clamp(min=1e-8))
        masked_for_max = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        mx, _ = masked_for_max.max(dim=1)
        mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
        return torch.cat([mean, std, mx], dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                sync: torch.Tensor | None = None) -> dict:
        """
        x    : (B, T, d_in)   raw per-window error features
        mask : (B, T)         bool, True = valid (non-padded) window
        sync : (B, sync_dim)  SyncNet scalars, or None to zero-fill (open-set mode)

        Returns {"logits": (B, n_classes), "embed": (B, d_embed)}.
        """
        h = self.input_proj(x)
        h, _ = self.lstm(h)
        mask_f = mask.unsqueeze(-1).float()
        h_pool = (h * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

        flat = self._masked_flat_stats(x, mask)
        if self.sync_dim > 0:
            if sync is None:
                sync = torch.zeros(x.size(0), self.sync_dim, device=x.device, dtype=flat.dtype)
            flat = torch.cat([flat, sync], dim=-1)
        flat = self.flat_proj(self.flat_norm(flat))

        embed = self.embed_proj(torch.cat([h_pool, flat], dim=-1))
        logits = self.attr_head(embed)
        return {"logits": logits, "embed": embed}

    @torch.no_grad()
    def predict(self, x: torch.Tensor, mask: torch.Tensor,
                sync: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward(x, mask, sync)["logits"].argmax(dim=-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
