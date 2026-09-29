"""Flat MERT probe - concatenates all layers instead of weighted sum.

Uses all 13 layers × 768 hidden = 9,984 dimensional feature vector.
More information than the weighted-sum probe, at the cost of a larger
classifier head. Requires more regularization to avoid overfitting.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.common import NUM_CLASSES
from src.models.mert_embed import MERT_HIDDEN_SIZE, MERT_NUM_LAYERS


class MertFlatProbe(nn.Module):
    """Input: (B, 13, 768) cached MERT embeddings. Output: (B, num_classes) logits."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        num_layers: int = MERT_NUM_LAYERS,
        hidden_size: int = MERT_HIDDEN_SIZE,
        proj_size: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()
        flat_dim = num_layers * hidden_size  # 13*768 = 9984

        self.norm = nn.LayerNorm(flat_dim)
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, proj_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(proj_size * 2),
            nn.Linear(proj_size * 2, proj_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(proj_size, num_classes)

        # Compatibility attributes
        self.learn_layer_weights = False
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))

    def layer_weights(self):
        return torch.softmax(self.layer_logits, dim=0)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        x = x.float()
        # Flatten all layers: (B, 13, 768) -> (B, 9984)
        flat = x.reshape(x.size(0), -1)
        flat = self.norm(flat)
        embedding = self.encoder(flat)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits


if __name__ == "__main__":
    probe = MertFlatProbe()
    dummy = torch.randn(4, MERT_NUM_LAYERS, MERT_HIDDEN_SIZE)
    logits, emb = probe(dummy, return_embedding=True)
    print(f"Logits: {logits.shape}, Embed: {emb.shape}")
    n = sum(p.numel() for p in probe.parameters())
    print(f"Parameters: {n:,}")
