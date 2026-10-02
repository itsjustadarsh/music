"""CNN + Bidirectional GRU hybrid for genre classification.

Reproduces Kostrzewa et al. (2021) who used CNN feature extraction followed
by RNN temporal modeling to achieve 54.9% F1 on FMA-small. CNN blocks extract
frame-level features from log-mel spectrograms, then a BiGRU processes the
temporal sequence for classification.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.common import NUM_CLASSES


class CNNBiGRU(nn.Module):
    """Input: (B, 1, 128, 130). Output: (B, num_classes) logits.

    CNN extracts frequency features while preserving time dimension,
    then BiGRU models temporal dependencies.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3,
                 gru_hidden: int = 128, gru_layers: int = 2):
        super().__init__()
        # CNN: pool only in frequency, keep time dimension
        self.cnn = nn.Sequential(
            # Block 1: (B, 1, 128, T) -> (B, 32, 64, T)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # pool freq only
            # Block 2: (B, 32, 64, T) -> (B, 64, 32, T)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            # Block 3: (B, 64, 32, T) -> (B, 128, 16, T)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        # After CNN: (B, 128, 16, T) -> reshape to (B, T, 128*16) -> project
        self.freq_proj = nn.Linear(128 * 16, gru_hidden * 2)

        self.gru = nn.GRU(
            input_size=gru_hidden * 2,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        embed_dim = gru_hidden * 2  # bidirectional
        self.embedder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # CNN feature extraction
        x = self.cnn(x)  # (B, 128, F', T)
        B, C, F, T = x.shape
        # Reshape: merge channel and freq dims, transpose to (B, T, C*F)
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)
        x = self.freq_proj(x)  # (B, T, gru_hidden*2)
        # BiGRU
        x, _ = self.gru(x)  # (B, T, gru_hidden*2)
        # Use mean pooling over time
        x = x.mean(dim=1)  # (B, gru_hidden*2)
        embedding = self.embedder(x)
        logits = self.head(embedding)
        if return_embedding:
            return logits, embedding
        return logits


if __name__ == "__main__":
    model = CNNBiGRU()
    dummy = torch.randn(4, 1, 128, 130)
    out = model(dummy)
    print(out.shape)  # (4, 8)
    print(sum(p.numel() for p in model.parameters()), "params")
