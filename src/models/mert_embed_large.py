"""Frozen MERT-v1-330M backbone used as a per-episode feature extractor.

Same approach as mert_embed.py but for the larger 330M model:
- 24 transformer layers + 1 conv = 25 layers
- Hidden size 1024 (vs 768)
- Same 24kHz sample rate

We load into stock HubertModel (same trick as the 95M version).
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import AutoConfig, HubertConfig, HubertModel, Wav2Vec2FeatureExtractor

from src.models.common import get_device

log = logging.getLogger(__name__)

MERT_LARGE_CHECKPOINT = "m-a-p/MERT-v1-330M"
MERT_LARGE_SAMPLE_RATE = 24000
MERT_LARGE_NUM_LAYERS = 25  # 24 transformer layers + conv feature projection
MERT_LARGE_HIDDEN_SIZE = 1024


def load_mert_large(device: torch.device | None = None) -> tuple[HubertModel, Wav2Vec2FeatureExtractor]:
    """Return the frozen MERT-330M encoder and its feature extractor, in eval mode."""
    src_config = AutoConfig.from_pretrained(MERT_LARGE_CHECKPOINT, trust_remote_code=True)
    src_config.conv_pos_batch_norm = False

    allowed = set(HubertConfig().to_dict())
    config = HubertConfig(**{k: v for k, v in src_config.to_dict().items() if k in allowed})

    model = HubertModel.from_pretrained(MERT_LARGE_CHECKPOINT, config=config)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    device = device or get_device()
    model.to(device)

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_LARGE_CHECKPOINT, trust_remote_code=True)
    if extractor.sampling_rate != MERT_LARGE_SAMPLE_RATE:
        raise ValueError(
            f"Expected MERT feature extractor at {MERT_LARGE_SAMPLE_RATE}Hz, got {extractor.sampling_rate}"
        )
    return model, extractor


@torch.no_grad()
def embed_episodes_large(
    episodes: np.ndarray,
    model: HubertModel,
    extractor: Wav2Vec2FeatureExtractor,
    device: torch.device,
) -> np.ndarray:
    """Embed a batch of equal-length 24kHz episodes.

    `episodes` is (n_episodes, n_samples). Returns (n_episodes, 25, 1024) float16:
    every layer's activations mean-pooled over time.
    """
    inputs = extractor(
        list(episodes), sampling_rate=MERT_LARGE_SAMPLE_RATE, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)

    # (layers, batch, time, hidden) -> mean over time -> (batch, layers, hidden)
    stacked = torch.stack(outputs.hidden_states)
    pooled = stacked.mean(dim=2).permute(1, 0, 2)
    return pooled.cpu().numpy().astype(np.float16)
