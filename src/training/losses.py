"""Custom loss functions for genre classification training."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017) — down-weights well-classified examples
    so training concentrates on hard/misclassified ones.

    With gamma=0 this reduces to standard cross-entropy.
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


def build_criterion(cfg: dict) -> nn.Module:
    """Build a loss function from config.

    Supported values for cfg["loss"]:
        - "cross_entropy" (default): nn.CrossEntropyLoss with optional label_smoothing
        - "focal": FocalLoss with cfg["focal_gamma"] (default 2.0) and optional label_smoothing
    """
    loss_name = cfg.get("loss", "cross_entropy")
    label_smoothing = cfg.get("label_smoothing", 0.0)

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif loss_name == "focal":
        gamma = cfg.get("focal_gamma", 2.0)
        return FocalLoss(gamma=gamma, label_smoothing=label_smoothing)
    elif loss_name == "margin":
        from src.models.capsnet import MarginLoss
        return MarginLoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name!r}")
