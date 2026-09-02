"""
Frozen audio-visual encoders.

Visual : DINOv2 ViT-S/14 (384-d CLS token per frame), via timm.
Audio  : wav2vec2-base (768-d per frame, after temporal alignment), via
         HuggingFace `transformers`.

Both encoders are pretrained and frozen for the entire pipeline: only the
audio-to-visual-dimension projection head is ever trained (Section 3.1 of
the paper). Weights are downloaded automatically on first use.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualEncoder(nn.Module):
    """DINOv2 ViT-S/14, frozen. Returns the 384-d [CLS] token per frame."""

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    EMBED_DIM = 384

    def __init__(self, frozen: bool = True):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            img_size=224,
            dynamic_img_size=True,
        )
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) normalized frames -> (B, 384)."""
        out = self.backbone.forward_features(x)
        if isinstance(out, dict):
            return out["x_norm_clstoken"]
        return out[:, 0] if out.dim() == 3 else out

    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, H, W, 3) uint8 [0,255] or float [0,1] -> (B*T, 3, 224, 224)."""
        b, t = frames.shape[:2]
        x = frames.reshape(b * t, *frames.shape[2:])
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        if x.shape[-1] != 224:
            x = F.interpolate(x, (224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor(self.MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.STD, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std


class AudioEncoder(nn.Module):
    """wav2vec2-base, frozen. Returns a (T, 768) feature sequence per clip."""

    EMBED_DIM = 768

    def __init__(self, frozen: bool = True):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        if frozen:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (B, L) float32 @ 16kHz -> (B, T_audio, 768)."""
        with torch.no_grad():
            return self.model(waveform).last_hidden_state

    @staticmethod
    def align_to_video(audio_feat: torch.Tensor, n_video_frames: int) -> torch.Tensor:
        """Linearly resample the audio feature sequence to the video frame rate."""
        if audio_feat.shape[1] == n_video_frames:
            return audio_feat
        x = audio_feat.permute(0, 2, 1)
        x = F.interpolate(x, size=n_video_frames, mode="linear", align_corners=False)
        return x.permute(0, 2, 1)


class DualEncoder(nn.Module):
    """VisualEncoder + AudioEncoder, with the trainable audio -> proj_dim head."""

    def __init__(self, proj_dim: int = 384, frozen: bool = True):
        super().__init__()
        self.visual = VisualEncoder(frozen=frozen)
        self.audio = AudioEncoder(frozen=frozen)
        self.audio_proj = nn.Sequential(
            nn.Linear(AudioEncoder.EMBED_DIM, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.proj_dim = proj_dim

    def encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        b, t = frames.shape[:2]
        x = self.visual.preprocess(frames)
        z = self.visual(x)
        return z.view(b, t, -1)

    def encode_audio(self, waveform: torch.Tensor, n_video_frames: int) -> torch.Tensor:
        feat = self.audio(waveform)
        feat = self.audio.align_to_video(feat, n_video_frames)
        return self.audio_proj(feat)

    def forward(self, frames: torch.Tensor, waveform: torch.Tensor):
        """Returns (z_v, z_a), each (B, T, proj_dim)."""
        t = frames.shape[1]
        z_v = self.encode_video(frames)
        z_a = self.encode_audio(waveform, t)
        return z_v, z_a
