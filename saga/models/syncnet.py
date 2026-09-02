"""
Off-the-shelf SyncNet synchrony scorer (Section 3.3).

Wraps the original, frozen SyncNet model (`external/syncnet/syncnet_model.py`)
to produce two scalars per clip:

  min_dist : the minimum mean audio-visual embedding distance across all
             tested temporal offsets — how well-aligned the best offset is.
  conf     : median mean distance minus min_dist — the "synchrony
             confidence": how much better the best offset is than a typical
             one. A well-defined, unambiguous sync point gives high
             confidence; a clip with no clear alignment gives low
             confidence, the signature of desynchronized content.

No fine-tuning is performed anywhere in this repository — SyncNet is used
purely for inference, exactly as released.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_EXTERNAL_SYNCNET = Path(__file__).resolve().parents[2] / "external" / "syncnet"
if str(_EXTERNAL_SYNCNET) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_SYNCNET))


class SyncNetScorer:
    """Loads the frozen SyncNet checkpoint and scores (video frames, audio) pairs."""

    VSHIFT = 15  # +/- 15-frame search window, matches the original evaluation protocol

    def __init__(self, checkpoint_path: str | Path, device: str | torch.device = "cpu"):
        from syncnet_model import S

        self.device = torch.device(device)
        self.net = S(num_layers_in_fc_layers=1024).to(self.device)
        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        self.net.load_state_dict(state)
        self.net.eval()

    @staticmethod
    def _pairwise_dists(lip_feat: torch.Tensor, aud_feat: torch.Tensor,
                         vshift: int) -> list[torch.Tensor]:
        win = vshift * 2 + 1
        aud_padded = F.pad(aud_feat, (0, 0, vshift, vshift))
        return [
            F.pairwise_distance(lip_feat[[i], :].repeat(win, 1), aud_padded[i:i + win, :])
            for i in range(len(lip_feat))
        ]

    @torch.no_grad()
    def score(self, lip_frames: np.ndarray, mfcc: np.ndarray) -> tuple[float, float]:
        """
        lip_frames : (T, 224, 224, 3) BGR uint8 video frames, >= 6 frames.
        mfcc       : (13, 4*T) MFCC audio features (see `saga.features.audio`).

        Returns (conf, min_dist).
        """
        im = np.transpose(np.expand_dims(np.stack(lip_frames, axis=3), 0), (0, 3, 4, 1, 2))
        im_t = torch.from_numpy(im.astype(np.float64)).float()

        cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
        cc_t = torch.from_numpy(cc.astype(np.float64)).float()

        last_frame = min(len(lip_frames), mfcc.shape[1] // 4) - 5
        if last_frame < 1:
            raise ValueError("clip too short to score (need >= 6 usable frames)")

        lip_feats, aud_feats = [], []
        batch_size = 20
        for i in range(0, last_frame, batch_size):
            j = min(last_frame, i + batch_size)
            lip_in = torch.cat([im_t[:, :, k:k + 5, :, :] for k in range(i, j)], 0)
            lip_feats.append(self.net.forward_lip(lip_in.to(self.device)).cpu())
            aud_in = torch.cat([cc_t[:, :, :, 4 * k:4 * k + 20] for k in range(i, j)], 0)
            aud_feats.append(self.net.forward_aud(aud_in.to(self.device)).cpu())

        lip_feats = torch.cat(lip_feats, 0)
        aud_feats = torch.cat(aud_feats, 0)

        dists = self._pairwise_dists(lip_feats, aud_feats, self.VSHIFT)
        mean_dists = torch.mean(torch.stack(dists, 1), 1)
        min_dist, _ = torch.min(mean_dists, 0)
        conf = torch.median(mean_dists) - min_dist
        return float(conf.item()), float(min_dist.item())
