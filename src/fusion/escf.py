"""Epoch-Scheduled Conditional Fusion (ESCF).

Two backbones describe the same 3-second episode: a from-scratch log-mel CNN and a
frozen music-SSL (MERT) probe. The claim this module tests is that the right way to
combine them is *not* a single global weight, because the two fail on different parts
of a track -- a percussive passage and a sparse harmonic one are not equally legible
to a spectrogram CNN and a self-supervised encoder. So the trust ratio is predicted
per episode, from both backbones' representations.

    h_i   = [h_A_i ; h_B_i ; confidence(z_A_i) ; confidence(z_B_i)]
    alpha_i = sigmoid(g(h_i))          # which backbone to trust, per episode
    beta_i  = softmax_i(a(h_i))        # which episodes matter, per track
    z_i   = alpha_i * z_A_i + (1 - alpha_i) * z_B_i
    z_T   = sum_i beta_i * z_i

The epoch curriculum interpolates the learned gate against a fixed prior:

    alpha_i(t) = lambda_t * alpha_i + (1 - lambda_t) * alpha_prior

Related work this must be measured against, not just cited: MATT (2022) weights
segments for a single model; the IISE Transactions (2026) optimisation ensemble
learns *global* segment weights. Neither conditions the weighting on the input, and
neither weights across heterogeneous backbones. `variant=` below collapses this class
down to each of those baselines so the comparison runs on identical data and splits.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.fusion.schedules import get_schedule
from src.models.common import NUM_CLASSES

NEG_INF = -1e9


def confidence_features(logits: torch.Tensor) -> torch.Tensor:
    """Cheap, scale-free summaries of how sure a backbone is about an episode:
    max probability, negative entropy, and top-2 margin."""
    probs = torch.softmax(logits, dim=-1)
    top2 = probs.topk(2, dim=-1).values
    max_prob = top2[..., 0:1]
    margin = (top2[..., 0] - top2[..., 1]).unsqueeze(-1)
    neg_entropy = (probs * torch.log(probs.clamp_min(1e-9))).sum(-1, keepdim=True)
    return torch.cat([max_prob, margin, neg_entropy], dim=-1)


class ESCF(nn.Module):
    """Configurable fusion head. `variant` selects an ablation row:

    - ``"cnn_only"`` / ``"mert_only"``: single backbone (alpha pinned to 1 / 0)
    - ``"fixed"``: one global alpha, not learned (grid-searched outside)
    - ``"global"``: one global alpha, learned -- no input conditioning
    - ``"conditional"``: per-episode alpha (the proposed gate)

    `use_attention` toggles beta between learned MIL attention and uniform mean.
    """

    VARIANTS = ("cnn_only", "mert_only", "fixed", "global", "conditional")

    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        num_classes: int = NUM_CLASSES,
        variant: str = "conditional",
        use_attention: bool = True,
        hidden: int = 128,
        dropout: float = 0.2,
        alpha_prior: float = 0.5,
        fixed_alpha: float = 0.5,
        schedule: str = "cosine",
    ):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; expected one of {self.VARIANTS}")
        self.variant = variant
        self.use_attention = use_attention
        self.alpha_prior = alpha_prior
        self.fixed_alpha = fixed_alpha
        self.schedule = get_schedule(schedule)
        self.schedule_name = schedule

        in_dim = dim_a + dim_b + 2 * 3  # embeddings + confidence features per backbone

        if variant == "global":
            self.global_alpha_logit = nn.Parameter(torch.zeros(1))
        elif variant == "conditional":
            self.gate = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        if use_attention:
            self.attention = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )

    def _episode_features(self, batch) -> torch.Tensor:
        return torch.cat(
            [
                batch.emb_a,
                batch.emb_b,
                confidence_features(batch.logits_a),
                confidence_features(batch.logits_b),
            ],
            dim=-1,
        )

    def episode_alpha(self, batch, features: torch.Tensor, lam: float) -> torch.Tensor:
        """Per-episode weight on backbone A, after applying the epoch curriculum."""
        shape = batch.mask.shape
        if self.variant == "cnn_only":
            return torch.ones(shape, device=features.device)
        if self.variant == "mert_only":
            return torch.zeros(shape, device=features.device)
        if self.variant == "fixed":
            return torch.full(shape, self.fixed_alpha, device=features.device)
        if self.variant == "global":
            return torch.sigmoid(self.global_alpha_logit).expand(shape)

        learned = torch.sigmoid(self.gate(features)).squeeze(-1)
        # lambda_t hands control from the fixed prior to the learned gate.
        return lam * learned + (1.0 - lam) * self.alpha_prior

    def episode_beta(self, batch, features: torch.Tensor) -> torch.Tensor:
        """Per-episode relevance, normalised within a track over real episodes only."""
        mask = batch.mask
        if not self.use_attention:
            counts = mask.sum(dim=1, keepdim=True).clamp_min(1)
            return mask.float() / counts

        scores = self.attention(features).squeeze(-1)
        scores = scores.masked_fill(~mask, NEG_INF)
        return torch.softmax(scores, dim=1)

    def forward(self, batch, epoch: int = 1, total_epochs: int = 1, return_weights: bool = False):
        features = self._episode_features(batch)
        lam = self.schedule(epoch, total_epochs)

        alpha = self.episode_alpha(batch, features, lam)
        beta = self.episode_beta(batch, features)

        episode_logits = alpha.unsqueeze(-1) * batch.logits_a + (1 - alpha).unsqueeze(-1) * batch.logits_b
        episode_logits = episode_logits * batch.mask.unsqueeze(-1)
        track_logits = (beta.unsqueeze(-1) * episode_logits).sum(dim=1)

        if return_weights:
            return track_logits, {"alpha": alpha, "beta": beta, "lambda": lam}
        return track_logits
