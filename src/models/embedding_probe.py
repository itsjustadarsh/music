"""Generic DNN probe over a frozen single-vector audio embedding.

Shared by the BYOL-A (Rai & Bhattacharjee 2026) and PANNs/CNN14 (Liu et al. 2022)
reproductions: both backbones emit one flat vector per clip, unlike MERT's
per-layer stack, so one classifier head serves both and only `embed_dim` changes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.common import NUM_CLASSES


class EmbeddingProbe(nn.Module):
    """Input: (B, embed_dim) frozen embeddings. Output: (B, num_classes) logits."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        embed_dim: int = 2048,
        hidden_sizes: tuple[int, ...] = (512, 256),
        dropout: float = 0.4,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

        layers: list[nn.Module] = []
        in_dim = embed_dim
        for size in hidden_sizes:
            layers += [nn.Linear(in_dim, size), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = size
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        embedding = self.encoder(self.norm(x.float()))
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits


if __name__ == "__main__":
    probe = EmbeddingProbe()
    logits, emb = probe(torch.randn(4, 2048), return_embedding=True)
    print(f"Logits: {logits.shape}, Embed: {emb.shape}")
    print(f"Parameters: {sum(p.numel() for p in probe.parameters()):,}")
