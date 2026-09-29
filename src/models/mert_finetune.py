"""MERT-v1-95M with selectively unfrozen layers for fine-tuning.

Instead of just probing frozen embeddings, we unfreeze the last N transformer
layers and fine-tune them jointly with the classification head. This lets the
model adapt its representations to genre classification.

To keep this tractable on a laptop:
- Only unfreeze last N layers (default 2)
- Process raw audio episodes (3s @ 24kHz)
- Use gradient checkpointing to reduce memory
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

from src.models.common import NUM_CLASSES
from src.models.mert_embed import MERT_HIDDEN_SIZE, MERT_SAMPLE_RATE, load_mert

log = logging.getLogger(__name__)


class MertFinetune(nn.Module):
    """End-to-end MERT with unfrozen top layers + classification head.

    Input: (B, n_samples) raw audio at 24kHz
    Output: (B, num_classes) logits
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        hidden_size: int = MERT_HIDDEN_SIZE,
        unfreeze_layers: int = 2,
        proj_size: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.unfreeze_layers = unfreeze_layers

        # Load MERT backbone (starts fully frozen)
        self.backbone, self.extractor = load_mert()

        # Unfreeze last N transformer layers
        n_total_layers = len(self.backbone.encoder.layers)
        for i in range(n_total_layers - unfreeze_layers, n_total_layers):
            for param in self.backbone.encoder.layers[i].parameters():
                param.requires_grad = True
            log.info("Unfroze transformer layer %d/%d", i, n_total_layers)

        # Also unfreeze the final layer norm
        if hasattr(self.backbone.encoder, 'layer_norm'):
            for param in self.backbone.encoder.layer_norm.parameters():
                param.requires_grad = True

        # Classification head
        self.norm = nn.LayerNorm(hidden_size)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, proj_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(proj_size, num_classes)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        log.info("Trainable: %d / %d params (%.1f%%)", n_trainable, n_total, 100 * n_trainable / n_total)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """x: (B, n_samples) raw audio waveform at 24kHz."""
        # Feature extraction (normalization)
        inputs = self.extractor(
            list(x.cpu().numpy()),
            sampling_rate=MERT_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(x.device)

        # Forward through backbone
        outputs = self.backbone(input_values)
        # Mean pool over time dimension
        hidden = outputs.last_hidden_state  # (B, T, hidden)
        pooled = hidden.mean(dim=1)  # (B, hidden)

        # Classification
        embedding = self.encoder(self.norm(pooled))
        logits = self.classifier(embedding)

        if return_embedding:
            return logits, embedding
        return logits


if __name__ == "__main__":
    import numpy as np
    model = MertFinetune(unfreeze_layers=2)
    dummy = torch.randn(2, 72000)  # 2 clips of 3s @ 24kHz
    logits = model(dummy)
    print(f"Logits: {logits.shape}")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n:,}")
