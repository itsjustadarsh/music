"""ResNet-style CNN classifier over log-mel spectrograms.

Deeper than GenreCNN (4 blocks, 1.2M params) with residual connections,
squeeze-and-excitation, and more capacity (~3-5M params). Designed to
extract stronger features from the same log-mel spectrograms.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.common import NUM_CLASSES


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ResBlock(nn.Module):
    """Pre-activation residual block with optional downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.conv2(torch.relu(self.bn2(out)))
        out = self.se(out)
        return out + self.shortcut(x)


class GenreResNet(nn.Module):
    """Input: (B, 1, n_mels, time). Output: (B, num_classes) logits.

    6-block ResNet with SE attention, ~4M params.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4, embed_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.layer1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        embedding = self.embedder(x)
        logits = self.head(embedding)
        if return_embedding:
            return logits, embedding
        return logits


if __name__ == "__main__":
    model = GenreResNet()
    dummy = torch.randn(4, 1, 128, 130)
    out = model(dummy)
    print(f"Output: {out.shape}")  # (4, 8)
    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n:,}")
