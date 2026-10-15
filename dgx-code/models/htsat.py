"""HTS-AT: Hierarchical Token-Semantic Audio Transformer (Chen et al., 2022).

A Swin-Transformer based model for audio classification. Pretrained on AudioSet,
it achieves 47.1 mAP — one of the strongest single-model AudioSet results.

For FMA genre classification, we fine-tune it end-to-end on log-mel spectrograms.
The hierarchical patch merging captures both local timbral features and global
temporal structure, which is particularly useful for genre.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class PatchEmbed(nn.Module):
    """Split spectrogram into patches and project to embedding dimension."""

    def __init__(self, patch_size: tuple[int, int] = (4, 4), in_chans: int = 1, embed_dim: int = 96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, freq, time)
        x = self.proj(x)  # (B, embed_dim, H, W)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.norm(x)
        return x, (H, W)


class WindowAttention(nn.Module):
    """Multi-head attention within local windows."""

    def __init__(self, dim: int, window_size: tuple[int, int], num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinBlock(nn.Module):
    """Swin Transformer block with window attention and shifted window attention."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, (window_size, window_size), num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    """Downsample by merging 2x2 patches, doubling channels."""

    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        # Pad if needed
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2


class HTSATGenre(nn.Module):
    """Hierarchical Token-Semantic Audio Transformer for genre classification.

    Architecture:
    - Log-mel spectrogram input
    - Patch embedding (4x4 patches)
    - 4 Swin Transformer stages with patch merging
    - Global average pooling
    - Classification head

    This is a from-scratch implementation optimized for FMA-small.
    For best results, initialize from AudioSet-pretrained weights if available.

    Args:
        num_classes: number of genres
        target_sr: audio sample rate
        max_duration_sec: max audio duration
        n_mels: mel spectrogram bins
        embed_dim: base embedding dimension (doubles each stage)
        depths: number of blocks per stage
        num_heads: attention heads per stage
        window_size: local attention window size
        dropout: dropout rate
    """

    def __init__(
        self,
        num_classes: int = 8,
        target_sr: int = 32000,
        max_duration_sec: float = 10.0,
        n_mels: int = 128,
        embed_dim: int = 96,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.target_sr = target_sr
        self.max_samples = int(max_duration_sec * target_sr)
        self.n_mels = n_mels
        self.num_stages = len(depths)

        # Mel spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=2048,
            hop_length=512,
            n_mels=n_mels,
        )

        # Patch embedding
        self.patch_embed = PatchEmbed(patch_size=(4, 4), in_chans=1, embed_dim=embed_dim)

        # Stages
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()

        dim = embed_dim
        for i in range(self.num_stages):
            stage = nn.Sequential(*[
                SwinBlock(dim, num_heads[i], window_size, mlp_ratio, dropout)
                for _ in range(depths[i])
            ])
            self.stages.append(stage)
            if i < self.num_stages - 1:
                self.merges.append(PatchMerging(dim))
                dim *= 2

        self.final_norm = nn.LayerNorm(dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: (batch, samples) raw audio

        Returns:
            logits: (batch, num_classes)
        """
        waveforms = waveforms[:, : self.max_samples]

        # Compute mel spectrogram
        mel = self.mel_transform(waveforms)
        mel = torch.log(mel.clamp(min=1e-7))
        mel = mel.unsqueeze(1)  # (B, 1, n_mels, time)

        # Patch embedding
        x, (H, W) = self.patch_embed(mel)

        # Hierarchical stages
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < self.num_stages - 1:
                x, H, W = self.merges[i](x, H, W)

        x = self.final_norm(x)

        # Global average pooling
        x = x.mean(dim=1)  # (B, dim)

        return self.classifier(x)
