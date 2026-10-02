"""CNN + Transformer hybrid for genre classification.

Reproduces CT-GateNet (Ma et al. 2026, PLoS ONE) which used a 5-stage CNN +
2-layer Transformer encoder with attention fusion to achieve 89.42% on
FMA-small. Our simplified version: CNN backbone extracts spatial features,
reshape to sequence, Transformer encoder models global dependencies.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.common import NUM_CLASSES


class CNNTransformer(nn.Module):
    """Input: (B, 1, 128, 130). Output: (B, num_classes) logits.

    CNN backbone -> reshape to sequence -> Transformer encoder -> classification.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3,
                 d_model: int = 256, nhead: int = 4, num_transformer_layers: int = 2):
        super().__init__()
        # CNN backbone: 4 conv blocks with progressive pooling
        self.cnn = nn.Sequential(
            # Block 1: (B, 1, 128, 130) -> (B, 32, 64, 65)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: -> (B, 64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: -> (B, 128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4: -> (B, 256, 8, 8)
            nn.Conv2d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Positional encoding for the flattened spatial sequence
        max_seq_len = 8 * 8  # 64 positions
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers,
        )

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.embedder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        B = x.size(0)

        # CNN feature extraction
        x = self.cnn(x)  # (B, d_model, H, W)
        _, C, H, W = x.shape
        # Reshape to sequence: (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, d_model)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1 + H*W, d_model)

        # Transformer
        x = self.transformer(x)  # (B, 1 + H*W, d_model)
        cls_out = x[:, 0]  # CLS token output

        embedding = self.embedder(cls_out)
        logits = self.head(embedding)
        if return_embedding:
            return logits, embedding
        return logits


if __name__ == "__main__":
    model = CNNTransformer()
    dummy = torch.randn(4, 1, 128, 130)
    out = model(dummy)
    print(out.shape)  # (4, 8)
    print(sum(p.numel() for p in model.parameters()), "params")
