"""Baseline CNN classifier over log-mel spectrograms.

A modest 4-block conv stack (~1.2M params) -- standard "CNN baseline" architecture
for genre classification on spectrograms (c.f. Choi et al. 2016 style models),
intentionally simple enough to train in well under an hour on a laptop.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.common import NUM_CLASSES


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: tuple[int, int] = (2, 2)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GenreCNN(nn.Module):
    """Input: (B, 1, n_mels, time). Output: (B, num_classes) logits."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Split at the penultimate activation so the fusion gate in src/fusion/ can
        # condition on this backbone's representation, not just its logits.
        self.embedder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self._register_load_state_dict_pre_hook(self._remap_legacy_classifier)

    @staticmethod
    def _remap_legacy_classifier(state_dict, prefix, *args, **kwargs):
        """Load checkpoints saved before `classifier` was split into embedder/head.

        The original head was Sequential(Flatten, Dropout, Linear, ReLU, Dropout,
        Linear), so `classifier.2` is now `embedder.2` and `classifier.5` is now
        `head.1`. Remapping in place keeps the already-trained CNN runs loadable
        instead of silently failing or, worse, loading partially.
        """
        mapping = {"classifier.2": "embedder.2", "classifier.5": "head.1"}
        for old_key, new_key in mapping.items():
            for suffix in ("weight", "bias"):
                src = f"{prefix}{old_key}.{suffix}"
                if src in state_dict:
                    state_dict[f"{prefix}{new_key}.{suffix}"] = state_dict.pop(src)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        if x.dim() == 3:  # (B, n_mels, time) -> add channel dim
            x = x.unsqueeze(1)
        x = self.blocks(x)
        x = self.pool(x)
        embedding = self.embedder(x)
        logits = self.head(embedding)
        if return_embedding:
            return logits, embedding
        return logits


if __name__ == "__main__":
    model = GenreCNN()
    dummy = torch.randn(4, 1, 128, 130)
    out = model(dummy)
    print(out.shape)  # (4, 8)
    print(sum(p.numel() for p in model.parameters()), "params")
