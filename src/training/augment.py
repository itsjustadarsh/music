"""Data augmentation, training split only.

SpecAugment (frequency/time masking, Park et al. 2019) over the cached log-mel
spectrograms, used by every from-scratch spectrogram model. Pure numpy, so it is
easy to unit test and safe to call from a DataLoader worker without extra state.
"""
from __future__ import annotations

import numpy as np


def spec_augment(
    logmel: np.ndarray,
    freq_mask_param: int = 16,
    time_mask_param: int = 20,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """logmel: (n_mels, time). Returns a masked copy; input is not mutated."""
    rng = rng or np.random.default_rng()
    out = logmel.copy()
    n_mels, n_frames = out.shape
    fill_value = out.min()

    for _ in range(num_freq_masks):
        width = rng.integers(0, freq_mask_param + 1)
        if width == 0 or width >= n_mels:
            continue
        start = rng.integers(0, n_mels - width)
        out[start : start + width, :] = fill_value

    for _ in range(num_time_masks):
        width = rng.integers(0, time_mask_param + 1)
        if width == 0 or width >= n_frames:
            continue
        start = rng.integers(0, n_frames - width)
        out[:, start : start + width] = fill_value

    return out
