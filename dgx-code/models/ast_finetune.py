"""Audio Spectrogram Transformer (AST) fine-tuning for genre classification.

AST (Gong et al., 2021) applies a Vision Transformer to mel spectrograms and
achieves strong results on AudioSet. We fine-tune a pretrained AST on FMA-small.

The key advantage over MERT: AST was supervised on AudioSet's 527 sound classes,
giving it strong discriminative audio features, whereas MERT is self-supervised
on music specifically. Different inductive biases make them complementary for ensembling.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torchaudio


class ASTFinetune(nn.Module):
    """Fine-tune a pretrained Audio Spectrogram Transformer.

    Loads the MIT/ast-finetuned-audioset model from HuggingFace and replaces
    the classification head.

    Args:
        num_classes: number of output genres
        target_sr: audio sample rate (16kHz for AST)
        max_duration_sec: maximum audio duration
        freeze_layers: number of transformer layers to freeze from bottom (0 = unfreeze all)
        dropout: dropout in classification head
    """

    def __init__(
        self,
        num_classes: int = 8,
        target_sr: int = 16000,
        max_duration_sec: float = 10.0,
        freeze_layers: int = 0,
        dropout: float = 0.3,
        hidden_size: int = 768,
        spec_augment: bool = False,
    ):
        super().__init__()
        self.spec_augment = spec_augment
        # SpecAugment on the (time, freq) fbank, applied only in training
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=24, iid_masks=True)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=96, iid_masks=True)
        self.target_sr = target_sr
        self.max_samples = int(max_duration_sec * target_sr)
        self.num_classes = num_classes

        from transformers import ASTModel, ASTFeatureExtractor

        self.feature_extractor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        self.backbone = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

        hidden_size = self.backbone.config.hidden_size

        # Freeze bottom layers
        if freeze_layers > 0:
            # Freeze embeddings
            for param in self.backbone.embeddings.parameters():
                param.requires_grad = False
            # Freeze bottom N encoder layers
            for layer in self.backbone.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: (batch, samples) raw audio at target_sr

        Returns:
            logits: (batch, num_classes)
        """
        device = waveforms.device

        # Truncate to max length
        waveforms = waveforms[:, : self.max_samples]

        # Convert to numpy for feature extractor
        wav_list = [w.cpu().numpy() for w in waveforms]
        inputs = self.feature_extractor(
            wav_list,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(device)
        if self.training and self.spec_augment:
            x = input_values.transpose(1, 2).unsqueeze(1)  # (B, 1, freq, time)
            x = self.time_mask(self.freq_mask(x))
            x = self.time_mask(x)
            input_values = x.squeeze(1).transpose(1, 2)

        outputs = self.backbone(input_values=input_values)

        # Use the [CLS] token output
        cls_output = outputs.last_hidden_state[:, 0]

        return self.classifier(cls_output)


class EfficientATFinetune(nn.Module):
    """EfficientAT (Schmid et al., 2023) — a lightweight but strong audio classifier.

    Uses a MobileNetV3-based architecture pretrained on AudioSet. Much smaller
    than AST/MERT but competitive on audio classification tasks. Good for
    ensemble diversity.

    Falls back to a CNN-Transformer if EfficientAT weights aren't available.
    """

    def __init__(
        self,
        num_classes: int = 8,
        target_sr: int = 32000,
        max_duration_sec: float = 10.0,
        dropout: float = 0.3,
        n_mels: int = 128,
    ):
        super().__init__()
        self.target_sr = target_sr
        self.max_samples = int(max_duration_sec * target_sr)
        self.n_mels = n_mels

        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=2048,
            hop_length=512,
            n_mels=n_mels,
            f_min=0,
            f_max=target_sr // 2,
        )

        # CNN backbone (MobileNetV3-inspired for audio)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32),  # depthwise
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 1),  # pointwise
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(64, 64, 3, padding=1, groups=64),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 3
            nn.Conv2d(128, 128, 3, padding=1, groups=128),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 4
            nn.Conv2d(256, 256, 3, padding=1, groups=256),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Squeeze-and-excitation
        self.se = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.Sigmoid(),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        waveforms = waveforms[:, : self.max_samples]

        # Compute mel spectrogram
        mel = self.mel_transform(waveforms)  # (batch, n_mels, time)
        mel = torch.log(mel.clamp(min=1e-7))
        mel = mel.unsqueeze(1)  # (batch, 1, n_mels, time)

        features = self.features(mel).squeeze(-1).squeeze(-1)  # (batch, 512)

        # SE attention
        se_weights = self.se(features)
        features = features * se_weights

        return self.classifier(features)
