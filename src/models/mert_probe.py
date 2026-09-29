"""Light classifier head over frozen MERT embeddings.

The backbone is never fine-tuned, so this probe is the only thing that trains: it
sees cached `(13, 768)` per-episode embeddings and predicts a genre. Two design
points matter for the paper:

1. **Learned layer weighting.** Genre information in a self-supervised encoder is
   not concentrated in the final layer -- middle layers usually carry more of it.
   Rather than picking a layer by hand, the probe learns a softmax distribution over
   all 13 and takes the weighted sum. `mert_probe_lastlayer.yaml` ablates this away
   so the choice is justified by a number rather than by assertion.
2. **Penultimate embedding exposed.** `forward(..., return_embedding=True)` returns
   the hidden representation alongside the logits, because the fusion gate in
   `src/fusion/` conditions on both backbones' embeddings, not just their logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.common import NUM_CLASSES
from src.models.mert_embed import MERT_HIDDEN_SIZE, MERT_NUM_LAYERS


class MertProbe(nn.Module):
    """Input: (B, 13, 768) cached MERT embeddings. Output: (B, num_classes) logits."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        num_layers: int = MERT_NUM_LAYERS,
        hidden_size: int = MERT_HIDDEN_SIZE,
        proj_size: int = 512,
        dropout: float = 0.3,
        learn_layer_weights: bool = True,
        layer_index: int | None = None,
    ):
        super().__init__()
        self.learn_layer_weights = learn_layer_weights
        self.layer_index = layer_index
        if learn_layer_weights:
            self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        elif layer_index is None:
            raise ValueError("layer_index is required when learn_layer_weights is False")

        self.norm = nn.LayerNorm(hidden_size)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, proj_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(proj_size, num_classes)

    def layer_weights(self) -> torch.Tensor:
        """The learned distribution over MERT layers -- reported in the paper as
        evidence for which depths carry genre information."""
        if not self.learn_layer_weights:
            weights = torch.zeros(MERT_NUM_LAYERS, device=self.norm.weight.device)
            weights[self.layer_index] = 1.0
            return weights
        return torch.softmax(self.layer_logits, dim=0)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        x = x.float()
        if self.learn_layer_weights:
            weights = torch.softmax(self.layer_logits, dim=0)
            pooled = (x * weights[None, :, None]).sum(dim=1)
        else:
            pooled = x[:, self.layer_index, :]

        embedding = self.encoder(self.norm(pooled))
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits


if __name__ == "__main__":
    probe = MertProbe()
    dummy = torch.randn(4, MERT_NUM_LAYERS, MERT_HIDDEN_SIZE)
    logits, emb = probe(dummy, return_embedding=True)
    print(logits.shape, emb.shape)
    print(sum(p.numel() for p in probe.parameters()), "params")
    print("layer weights sum:", probe.layer_weights().sum().item())
