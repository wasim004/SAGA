"""
Domain-matched cross-modal predictors T_a2v and T_v2a (Section 3.1).

Each predictor causally maps one modality's frozen feature sequence onto the
other's, using a small stack of Mamba blocks:

    v_hat_t = T_a2v(a_{<=t})
    a_hat_t = T_v2a(v_{<=t})

and reports the per-window cosine-distance error curve used everywhere else
in the pipeline:

    C_a2v(t) = 1 - cos(v_hat_t, v_t)
    C_v2a(t) = 1 - cos(a_hat_t, a_t)

`mamba_ssm` requires a CUDA GPU; on CPU-only machines this module falls back
to a single-layer GRU per block so the pipeline still runs end to end (with
degraded reconstruction quality, useful for debugging and CI only).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba

    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False


def _make_block(d_model: int, d_state: int, d_conv: int, expand: int) -> nn.Module:
    if _MAMBA_AVAILABLE:
        return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    class _GRUFallback(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(d_model, d_model, batch_first=True)

        def forward(self, x):
            out, _ = self.gru(x)
            return out

    return _GRUFallback()


class MambaStack(nn.Module):
    """`n_layers` residual Mamba blocks, each pre-normalized and dropout-regularized."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int,
                 n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Sequential(
                _make_block(d_model, d_state, d_conv, expand),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class CrossModalPredictor(nn.Module):
    """Source sequence -> Mamba stack -> linear head -> predicted target sequence."""

    def __init__(self, d_model: int = 384, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbone = MambaStack(d_model, d_state, d_conv, expand, n_layers, dropout)
        self.head = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z_src: torch.Tensor, z_tgt: torch.Tensor):
        """
        z_src, z_tgt: (B, T, d_model)

        Returns (z_hat, C) where z_hat is the raw prediction (for the
        reconstruction loss) and C(t) = 1 - cos(z_hat_t, z_tgt_t) in [0, 2].
        """
        z_hat = self.head(self.backbone(z_src))
        cos_sim = F.cosine_similarity(z_hat, z_tgt, dim=-1)
        c = 1.0 - cos_sim
        return z_hat, c


class DomainMatchedPredictors(nn.Module):
    """The paired T_a2v / T_v2a predictors used throughout the paper."""

    def __init__(self, d_model: int = 384, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        kw = dict(d_model=d_model, d_state=d_state, d_conv=d_conv,
                  expand=expand, n_layers=n_layers, dropout=dropout)
        self.a2v = CrossModalPredictor(**kw)
        self.v2a = CrossModalPredictor(**kw)

    def forward(self, z_v: torch.Tensor, z_a: torch.Tensor) -> dict:
        """
        z_v, z_a: (B, T, d_model) frozen encoder features for one clip.

        Returns dict with z_v_hat, z_a_hat, C_a2v, C_v2a (all (B, T) or
        (B, T, d_model) as appropriate) — the bivariate error curve
        (C_a2v, C_v2a) is what Section 3.2 turns into the 12-d feature
        vector per window.
        """
        z_v_hat, c_a2v = self.a2v(z_a, z_v)  # audio predicts visual
        z_a_hat, c_v2a = self.v2a(z_v, z_a)  # visual predicts audio
        return {
            "z_v_hat": z_v_hat,
            "z_a_hat": z_a_hat,
            "C_a2v": c_a2v,
            "C_v2a": c_v2a,
        }
