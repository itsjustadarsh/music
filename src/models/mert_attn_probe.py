"""Attention-based classifier head over frozen MERT embeddings.

Instead of a simple weighted sum over 13 MERT layers, this probe uses
multi-head self-attention to let layers interact, then pools the result
with a learnable query. The classifier is a deeper 3-layer MLP with
residual connections and layer normalization.

This is designed to extract significantly more information from the same
cached (13, 768) embeddings that the simple probe uses.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.common import NUM_CLASSES
from src.models.mert_embed import MERT_HIDDEN_SIZE, MERT_NUM_LAYERS


class MertAttentionProbe(nn.Module):
    """Input: (B, 13, 768) cached MERT embeddings. Output: (B, num_classes) logits.

    Architecture:
    1. LayerNorm + positional encoding over the 13 layers
    2. Multi-head self-attention (layers attend to each other)
    3. Learnable [CLS]-style query token pools the attended layers
    4. 3-layer MLP with residual connections for classification
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        num_layers: int = MERT_NUM_LAYERS,
        hidden_size: int = MERT_HIDDEN_SIZE,
        n_heads: int = 8,
        attn_layers: int = 2,
        mlp_size: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Learnable positional embeddings for the 13 MERT layers
        self.layer_pos = nn.Parameter(torch.randn(1, num_layers, hidden_size) * 0.02)

        # Pre-norm
        self.input_norm = nn.LayerNorm(hidden_size)

        # Self-attention over layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=attn_layers)

        # Learnable query for pooling
        self.cls_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, n_heads, dropout=dropout, batch_first=True
        )
        self.pool_norm = nn.LayerNorm(hidden_size)

        # Deep classifier with residual
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(mlp_size),
            nn.Linear(mlp_size, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(mlp_size, num_classes)

        # For compatibility with cache_outputs / fusion
        self.learn_layer_weights = True
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))  # dummy for compat

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        x = x.float()  # (B, 13, 768)

        # Add positional encoding
        x = self.input_norm(x) + self.layer_pos

        # Self-attention over layers
        x = self.attention(x)  # (B, 13, 768)

        # Pool with learnable query via cross-attention
        B = x.size(0)
        query = self.cls_query.expand(B, -1, -1)  # (B, 1, 768)
        pooled, _ = self.cross_attn(query, x, x)  # (B, 1, 768)
        pooled = self.pool_norm(pooled.squeeze(1))  # (B, 768)

        # Deep MLP classifier
        embedding = self.mlp(pooled)  # (B, mlp_size)
        logits = self.classifier(embedding)

        if return_embedding:
            return logits, embedding
        return logits


if __name__ == "__main__":
    probe = MertAttentionProbe()
    dummy = torch.randn(4, MERT_NUM_LAYERS, MERT_HIDDEN_SIZE)
    logits, emb = probe(dummy, return_embedding=True)
    print(f"Logits: {logits.shape}, Embedding: {emb.shape}")
    n_params = sum(p.numel() for p in probe.parameters())
    print(f"Parameters: {n_params:,}")
