"""Capsule Network for genre classification.

Reproduces CapsNet+UIGMM (2024, Scientific Reports) which used capsule
networks on spectrograms to achieve 68.87% on FMA-small. Implements the
standard CapsNet architecture (Sabour et al. 2017) with dynamic routing:
conv stem -> primary capsules -> genre capsules.

Uses margin loss as in the original CapsNet paper, but also supports
cross-entropy by taking the L2 norm of output capsules as class scores.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common import NUM_CLASSES


def squash(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Squashing non-linearity: short vectors get shrunk to ~0,
    long vectors get shrunk to just under 1."""
    sq_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = sq_norm / (1 + sq_norm)
    return scale * x / (sq_norm.sqrt() + 1e-8)


class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels: int = 256, caps_channels: int = 32,
                 caps_dim: int = 8, kernel_size: int = 9, stride: int = 2):
        super().__init__()
        self.caps_dim = caps_dim
        self.conv = nn.Conv2d(in_channels, caps_channels * caps_dim,
                              kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)  # (B, caps_channels * caps_dim, H, W)
        B = x.size(0)
        # Reshape to (B, num_capsules, caps_dim)
        x = x.view(B, -1, self.caps_dim)
        return squash(x)


class GenreCapsules(nn.Module):
    """Digit/genre capsules with dynamic routing."""

    def __init__(self, num_capsules: int, in_capsules: int, in_dim: int = 8,
                 out_dim: int = 16, num_routing: int = 3):
        super().__init__()
        self.num_capsules = num_capsules
        self.num_routing = num_routing
        # W: (num_capsules, in_capsules, out_dim, in_dim)
        self.W = nn.Parameter(
            torch.randn(num_capsules, in_capsules, out_dim, in_dim) * 0.01
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_capsules, in_dim)
        B, N, D = x.shape
        # u_hat: prediction vectors
        # (B, 1, N, 1, D) @ (1, num_caps, N, D, out_dim) -> (B, num_caps, N, 1, out_dim)
        x = x.unsqueeze(1).unsqueeze(3)  # (B, 1, N, 1, D)
        W = self.W.unsqueeze(0)  # (1, num_caps, N, out_dim, D)
        u_hat = torch.matmul(x, W.transpose(-1, -2)).squeeze(3)  # (B, num_caps, N, out_dim)

        # Dynamic routing
        b = torch.zeros(B, self.num_capsules, N, 1, device=x.device)
        for i in range(self.num_routing):
            c = F.softmax(b, dim=1)  # (B, num_caps, N, 1)
            s = (c * u_hat).sum(dim=2)  # (B, num_caps, out_dim)
            v = squash(s)  # (B, num_caps, out_dim)
            if i < self.num_routing - 1:
                # (B, num_caps, N, out_dim) @ (B, num_caps, out_dim, 1)
                agreement = torch.matmul(u_hat, v.unsqueeze(-1))  # (B, num_caps, N, 1)
                b = b + agreement.detach()

        return v  # (B, num_capsules, out_dim)


class CapsNet(nn.Module):
    """Input: (B, 1, 128, 130). Output: (B, num_classes) logits (L2 norms)."""

    def __init__(self, num_classes: int = NUM_CLASSES, primary_caps_dim: int = 8,
                 genre_caps_dim: int = 16, num_routing: int = 3):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.primary_caps = PrimaryCapsules(
            in_channels=256, caps_channels=32,
            caps_dim=primary_caps_dim, kernel_size=9, stride=2,
        )
        # Compute number of primary capsules: depends on spatial dims after convs
        # Input (128, 130) -> conv_stem keeps size (padding=4) -> primary_caps reduces
        # (128, 130) -> conv(k=9, s=2) -> (60, 61), so 32 * 60 * 61 = 117,120 capsules
        # That's too many for the routing matrix. Add pooling to reduce.
        # Deeper conv stem to reduce spatial dims before capsules
        self.conv_stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (64, 64, 65)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (128, 32, 32)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (256, 16, 16)
        )
        self.primary_caps = PrimaryCapsules(
            in_channels=256, caps_channels=32,
            caps_dim=primary_caps_dim, kernel_size=3, stride=2,
        )
        # Compute n_primary dynamically from a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 130)
            feat = self.conv_stem(dummy)
            caps = self.primary_caps(feat)
            n_primary = caps.size(1)
        self.genre_caps = GenreCapsules(
            num_capsules=num_classes,
            in_capsules=n_primary,
            in_dim=primary_caps_dim,
            out_dim=genre_caps_dim,
            num_routing=num_routing,
        )
        self.genre_caps_dim = genre_caps_dim

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv_stem(x)
        x = self.primary_caps(x)  # (B, num_primary, caps_dim)
        caps = self.genre_caps(x)  # (B, num_classes, genre_caps_dim)
        # L2 norm of each capsule = class probability
        logits = caps.norm(dim=-1)  # (B, num_classes)
        if return_embedding:
            embedding = caps.reshape(caps.size(0), -1)  # flatten all capsules
            return logits, embedding
        return logits


class MarginLoss(nn.Module):
    """CapsNet margin loss from Sabour et al. 2017."""

    def __init__(self, m_pos: float = 0.9, m_neg: float = 0.1, lam: float = 0.5):
        super().__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lam = lam

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, num_classes) — L2 norms of capsule vectors
        one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pos = F.relu(self.m_pos - logits) ** 2
        neg = F.relu(logits - self.m_neg) ** 2
        loss = one_hot * pos + self.lam * (1 - one_hot) * neg
        return loss.sum(dim=-1).mean()


if __name__ == "__main__":
    model = CapsNet()
    dummy = torch.randn(4, 1, 128, 130)
    out = model(dummy)
    print(out.shape)  # (4, 8)
    print(sum(p.numel() for p in model.parameters()), "params")
