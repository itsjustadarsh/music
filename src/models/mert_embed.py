"""Frozen MERT-v1-95M backbone used as a per-episode feature extractor.

MERT is a music-specific self-supervised model (masked language modelling over
20k hours of music). We never fine-tune it: we cache mean-pooled hidden states per
episode once, then train light probes on top. That is the standard probing protocol
the MERT paper itself reports under, and it is what makes a multi-seed ablation grid
affordable on a laptop -- the expensive forward pass happens exactly once.

Loading note: MERT ships custom modeling code that (a) needs a `conv_pos_batch_norm`
attribute that `transformers >= 4.44` expects but its config lacks, and (b) silently
returns `hidden_states=None`, ignoring `output_hidden_states`. Since MERT-v1-95M is
architecturally a stock HuBERT (768 hidden, 12 layers, 7-layer conv feature encoder,
`feature_extractor_cqt` disabled), we load the weights into `transformers.HubertModel`
instead. This was verified to be *bit-identical* to the remote-code model's
`last_hidden_state` (max abs diff 0.0) while correctly exposing all 13 layers, and it
drops the `trust_remote_code` dependency -- which matters for reproducibility.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import AutoConfig, HubertConfig, HubertModel, Wav2Vec2FeatureExtractor

from src.models.common import get_device

log = logging.getLogger(__name__)

MERT_CHECKPOINT = "m-a-p/MERT-v1-95M"
MERT_SAMPLE_RATE = 24000
MERT_NUM_LAYERS = 13  # 12 transformer layers + the conv feature projection output
MERT_HIDDEN_SIZE = 768


def load_mert(device: torch.device | None = None) -> tuple[HubertModel, Wav2Vec2FeatureExtractor]:
    """Return the frozen MERT encoder and its feature extractor, in eval mode."""
    src_config = AutoConfig.from_pretrained(MERT_CHECKPOINT, trust_remote_code=True)
    src_config.conv_pos_batch_norm = False

    allowed = set(HubertConfig().to_dict())
    config = HubertConfig(**{k: v for k, v in src_config.to_dict().items() if k in allowed})

    model = HubertModel.from_pretrained(MERT_CHECKPOINT, config=config)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    device = device or get_device()
    model.to(device)

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_CHECKPOINT, trust_remote_code=True)
    if extractor.sampling_rate != MERT_SAMPLE_RATE:
        raise ValueError(
            f"Expected MERT feature extractor at {MERT_SAMPLE_RATE}Hz, got {extractor.sampling_rate}"
        )
    return model, extractor


@torch.no_grad()
def embed_episodes(
    episodes: np.ndarray,
    model: HubertModel,
    extractor: Wav2Vec2FeatureExtractor,
    device: torch.device,
) -> np.ndarray:
    """Embed a batch of equal-length 24kHz episodes.

    `episodes` is (n_episodes, n_samples). Returns (n_episodes, 13, 768) float16:
    every layer's activations mean-pooled over time.

    All 13 layers are kept rather than just the last one because genre information is
    not concentrated in the final layer of a self-supervised encoder -- the probe
    learns its own layer weighting (see `src/models/mert_probe.py`), and collapsing
    here would throw that choice away.
    """
    inputs = extractor(
        list(episodes), sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)

    # (layers, batch, time, hidden) -> mean over time -> (batch, layers, hidden)
    stacked = torch.stack(outputs.hidden_states)
    pooled = stacked.mean(dim=2).permute(1, 0, 2)
    return pooled.cpu().numpy().astype(np.float16)
