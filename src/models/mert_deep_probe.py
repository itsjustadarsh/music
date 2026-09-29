"""Deep MERT probe with residual blocks and multi-scale layer aggregation.

Instead of a single weighted sum over layers, this probe:
1. Groups MERT layers into early/mid/late blocks
2. Takes weighted sums within each group
3. Concatenates the group representations
4. Passes through a deeper classifier with residual connections

This gives the model access to more nuanced layer combinations while
keeping parameter count reasonable.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.common import NUM_CLASSES
from src.models.mert_embed import MERT_HIDDEN_SIZE, MERT_NUM_LAYERS


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class MertDeepProbe(nn.Module):
    """Multi-scale layer aggregation with deeper classifier.

    Input: (B, num_layers, hidden_size) cached MERT embeddings.
    Output: (B, num_classes) logits.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        num_layers: int = MERT_NUM_LAYERS,
        hidden_size: int = MERT_HIDDEN_SIZE,
        proj_size: int = 512,
        n_groups: int = 3,
        n_residual_blocks: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.n_groups = n_groups

        # Split layers into groups and learn weights within each
        layers_per_group = num_layers // n_groups
        self.group_ranges = []
        for g in range(n_groups):
            start = g * layers_per_group
            end = start + layers_per_group if g < n_groups - 1 else num_layers
            self.group_ranges.append((start, end))

        # Learned weights per group
        self.group_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(end - start))
            for start, end in self.group_ranges
        ])

        # Also learn a global weighted sum as an additional signal
        self.global_logits = nn.Parameter(torch.zeros(num_layers))

        # Norms for each group + global
        self.group_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(n_groups)])
        self.global_norm = nn.LayerNorm(hidden_size)

        # Project concatenated groups + global into shared space
        input_dim = hidden_size * (n_groups + 1)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual classifier blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(proj_size, dropout) for _ in range(n_residual_blocks)]
        )

        self.classifier = nn.Linear(proj_size, num_classes)

        # Compatibility
        self.learn_layer_weights = True
        self.layer_logits = self.global_logits

    def layer_weights(self):
        return torch.softmax(self.global_logits, dim=0)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        x = x.float()  # (B, L, H)

        parts = []
        # Group-level weighted sums
        for g, ((start, end), logits, norm) in enumerate(
            zip(self.group_ranges, self.group_logits, self.group_norms)
        ):
            w = torch.softmax(logits, dim=0)
            pooled = (x[:, start:end, :] * w[None, :, None]).sum(dim=1)
            parts.append(norm(pooled))

        # Global weighted sum
        gw = torch.softmax(self.global_logits, dim=0)
        global_pooled = (x * gw[None, :, None]).sum(dim=1)
        parts.append(self.global_norm(global_pooled))

        # Concatenate and project
        concat = torch.cat(parts, dim=1)  # (B, H*(n_groups+1))
        projected = self.projection(concat)
        embedding = self.residual_blocks(projected)
        logits = self.classifier(embedding)

        if return_embedding:
            return logits, embedding
        return logits


if __name__ == "__main__":
    probe = MertDeepProbe()
    dummy = torch.randn(4, MERT_NUM_LAYERS, MERT_HIDDEN_SIZE)
    logits, emb = probe(dummy, return_embedding=True)
    print(f"Logits: {logits.shape}, Embed: {emb.shape}")
    n = sum(p.numel() for p in probe.parameters())
    print(f"Parameters: {n:,}")
    print(f"Group ranges: {probe.group_ranges}")
