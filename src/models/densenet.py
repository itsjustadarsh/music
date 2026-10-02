"""DenseNet121 adapted for log-mel spectrogram genre classification.

Reproduces Bian, Wang, Zhuang et al. (2019) who used DenseNet with
time-overlap/pitch-shift augmentation + SVM stacking to achieve 68.9% on
FMA-small. We use torchvision's DenseNet121 (ImageNet-pretrained) adapted
for single-channel input, without the SVM stacking layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

from src.models.common import NUM_CLASSES


class GenreDenseNet(nn.Module):
    """DenseNet121 for (B, 1, 128, 130) log-mel spectrograms."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3,
                 pretrained: bool = True):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        base = densenet121(weights=weights)

        # Replace first conv: 3-channel RGB -> 1-channel spectrogram
        old_conv = base.features.conv0
        self.features = base.features
        self.features.conv0 = nn.Conv2d(
            1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
            stride=old_conv.stride, padding=old_conv.padding, bias=False,
        )
        if pretrained:
            # Average the 3-channel weights into 1 channel
            self.features.conv0.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)

        self.pool = nn.AdaptiveAvgPool2d(1)
        embed_dim = base.classifier.in_features  # 1024
        self.embedder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = torch.relu(x)
        x = self.pool(x)
        embedding = self.embedder(x)
        logits = self.head(embedding)
        if return_embedding:
            return logits, embedding
        return logits


if __name__ == "__main__":
    model = GenreDenseNet()
    dummy = torch.randn(4, 1, 128, 130)
    out = model(dummy)
    print(out.shape)  # (4, 8)
    print(sum(p.numel() for p in model.parameters()), "params")
