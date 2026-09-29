"""Modern architecture: fine-tuned Audio Spectrogram Transformer (AST).

Uses a pretrained AST checkpoint (trained on AudioSet) from HuggingFace
`transformers` and replaces its classification head for our 8 FMA genres.
Transfer learning from AudioSet is what makes it feasible to fine-tune a
transformer to a competitive accuracy within a laptop-scale time budget,
unlike training a transformer from scratch on only 8,000 tracks.

Reference: Gong, Chung & Glass, "AST: Audio Spectrogram Transformer" (2021).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification

from src.data.preprocess import AST_SAMPLE_RATE
from src.models.common import IDX_TO_GENRE, NUM_CLASSES

AST_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"


def load_feature_extractor() -> ASTFeatureExtractor:
    return ASTFeatureExtractor.from_pretrained(AST_CHECKPOINT)


class ASTGenreClassifier(nn.Module):
    """Wraps a pretrained AST with a fresh classification head sized to NUM_CLASSES.

    Forward pass accepts already-extracted `input_values` (B, time_frames, n_mels)
    as produced by ASTFeatureExtractor -- feature extraction is done once at
    dataset-load time (see src/training/dataset.py), not inside the model, so it
    can run efficiently in a DataLoader worker.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, freeze_encoder_layers: int = 0):
        super().__init__()
        config = ASTConfig.from_pretrained(AST_CHECKPOINT)
        config.num_labels = num_classes
        self.ast = ASTForAudioClassification.from_pretrained(
            AST_CHECKPOINT, config=config, ignore_mismatched_sizes=True
        )

        if freeze_encoder_layers > 0:
            for i, layer in enumerate(self.ast.audio_spectrogram_transformer.layers):
                if i < freeze_encoder_layers:
                    for p in layer.parameters():
                        p.requires_grad_(False)

    def forward(self, input_values: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        if return_embedding:
            outputs = self.ast.audio_spectrogram_transformer(input_values)
            hidden = outputs.last_hidden_state[:, 0]  # CLS token
            logits = self.ast.classifier(hidden)
            return logits, hidden
        return self.ast(input_values=input_values).logits


def extract_features(feature_extractor: ASTFeatureExtractor, waveform_16k) -> torch.Tensor:
    """waveform_16k: 1D numpy array or list of them, sampled at AST_SAMPLE_RATE (16kHz)."""
    out = feature_extractor(waveform_16k, sampling_rate=AST_SAMPLE_RATE, return_tensors="pt")
    return out["input_values"]


if __name__ == "__main__":
    import numpy as np

    fe = load_feature_extractor()
    dummy_wave = np.random.randn(AST_SAMPLE_RATE * 3).astype(np.float32)  # 3s @ 16kHz
    feats = extract_features(fe, dummy_wave)
    print("feature shape:", feats.shape)

    model = ASTGenreClassifier()
    logits = model(feats)
    print("logits shape:", logits.shape)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable params")
