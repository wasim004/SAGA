"""Loss functions for predictor adaptation (Section 3.1) and classifier training (Section 3.4)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────
# Predictor adaptation: L = L_pred + calibration_weight * L_calib
# ─────────────────────────────────────────────────────────────────────────

def prediction_loss(z_hat: torch.Tensor, z_gt: torch.Tensor) -> torch.Tensor:
    """Cosine MSE between predicted and ground-truth features, magnitude-invariant."""
    return F.mse_loss(F.normalize(z_hat, dim=-1), F.normalize(z_gt.detach(), dim=-1))


def calibration_loss(c_a2v: torch.Tensor, c_v2a: torch.Tensor) -> torch.Tensor:
    """
    Temporal-smoothness regularizer on the real-only error curve: mean
    absolute first difference of C_a2v + C_v2a over time. On real, authentic
    training clips the combined error should stay small *and* smooth; this
    discourages the predictors from producing spiky, high-variance error
    curves on real data that would raise the effective detection threshold.
    """
    c_i = c_a2v + c_v2a
    return (c_i[:, 1:] - c_i[:, :-1]).abs().mean()


class AdaptationLoss(nn.Module):
    """L = L_pred(v) + L_pred(a) + calibration_weight * L_calib."""

    def __init__(self, calibration_weight: float = 0.01):
        super().__init__()
        self.calibration_weight = calibration_weight

    def forward(self, predictor_out: dict, z_v: torch.Tensor, z_a: torch.Tensor) -> dict:
        l_v = prediction_loss(predictor_out["z_v_hat"], z_v)
        l_a = prediction_loss(predictor_out["z_a_hat"], z_a)
        l_pred = l_v + l_a
        l_calib = calibration_loss(predictor_out["C_a2v"], predictor_out["C_v2a"])
        total = l_pred + self.calibration_weight * l_calib
        return {"loss": total, "l_pred": l_pred.detach(), "l_calib": l_calib.detach()}


# ─────────────────────────────────────────────────────────────────────────
# Classifier training: L = L_focal + triplet_weight * L_triplet
# ─────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Class-weighted focal loss (Lin et al., 2017)."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()
        log_pt = log_p.gather(1, labels.unsqueeze(1)).squeeze(1)
        pt = p.gather(1, labels.unsqueeze(1)).squeeze(1)
        focal = -((1 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            focal = focal * self.weight.to(logits.device)[labels]
        return focal.mean()


class BatchHardTripletLoss(nn.Module):
    """Batch-hard triplet loss (Hermans et al., 2017) on L2-normalized embeddings."""

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embed: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embed = F.normalize(embed, dim=-1)
        dists = torch.cdist(embed, embed, p=2)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)

        pos_dists = dists.clone()
        pos_dists[~same] = 0.0
        hard_pos = pos_dists.max(dim=1).values

        neg_dists = dists.clone()
        neg_dists[same] = 1e9
        hard_neg = neg_dists.min(dim=1).values

        return F.relu(hard_pos - hard_neg + self.margin).mean()
