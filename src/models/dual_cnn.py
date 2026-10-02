"""Dual-path CNN with residual connections for genre classification.

Reproduces Ashraf et al. (2025) who used a Dual CNN with parallel max+avg
pooling paths and modified residual learning to achieve 68.5% on FMA-small.
Each conv block splits into two pooling branches (max and avg) that are
concatenated, with residual connections between blocks.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.common import NUM_CLASSES


class DualPoolBlock(nn.Module):
    """Conv -> BN -> ReLU -> parallel [MaxPool, AvgPool] -> concat along channels."""

    def __init__(self, in_ch: int, out_ch: int, pool_size: tuple[int, int] = (2, 2)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(pool_size)
        self.avgpool = nn.AvgPool2d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return torch.cat([self.maxpool(x), self.avgpool(x)], dim=1)  # 2*out_ch


class DualCNN(nn.Module):
    """Input: (B, 1, 128, 130). Output: (B, num_classes) logits.

    4 dual-pool blocks with residual connections. Each block doubles the
    channel count via dual pooling (max+avg concat), then a 1x1 residual
    conv matches dimensions for the skip connection.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        # Channel progression: 1 -> 32*2=64 -> 64*2=128 -> 128*2=256 -> 256*2=512
        channels = [1, 32, 64, 128, 256]
        self.blocks = nn.ModuleList()
        self.residuals = nn.ModuleList()

        for i in range(4):
            in_ch = channels[i] * 2 if i > 0 else channels[i]  # after dual pool, channels double
            out_ch = channels[i + 1]
            self.blocks.append(DualPoolBlock(in_ch, out_ch))
            # Residual: match input dims to output dims (out_ch * 2 after dual pool)
            self.residuals.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch * 2, kernel_size=1),
                nn.AvgPool2d(2),
            ))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        for block, res in zip(self.blocks, self.residuals):
            x = block(x) + res(x)
        x = self.pool(x)
        embedding = self.embedder(x)
        logits = self.head(embedding)
        if return_embedding:
            return logits, embedding
        return logits


if __name__ == "__main__":
    model = DualCNN()
    dummy = torch.randn(4, 1, 128, 130)
    out = model(dummy)
    print(out.shape)  # (4, 8)
    print(sum(p.numel() for p in model.parameters()), "params")
