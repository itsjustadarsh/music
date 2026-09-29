"""Data augmentation for the training-split only.

- SpecAugment (frequency/time masking) is applied to cached log-mel spectrograms
  -- used by GenreCNN. Cheap, standard, and well established for audio CNNs
  (Park et al. 2019, "SpecAugment").
- Pitch-shift / time-stretch are applied to the raw 16kHz waveform before AST
  feature extraction -- used by ASTGenreClassifier.

Both are pure numpy/librosa functions so they're easy to unit test and to call
from a DataLoader worker without extra state.
"""
from __future__ import annotations

import numpy as np
import librosa


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


def pitch_shift_waveform(wave: np.ndarray, sr: int, n_steps_range: tuple[float, float] = (-2.0, 2.0),
                          rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    n_steps = rng.uniform(*n_steps_range)
    return librosa.effects.pitch_shift(wave, sr=sr, n_steps=n_steps).astype(np.float32)


def time_stretch_waveform(wave: np.ndarray, rate_range: tuple[float, float] = (0.9, 1.1),
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """Time-stretches then crops/pads back to the original length so tensor shapes stay fixed."""
    rng = rng or np.random.default_rng()
    rate = rng.uniform(*rate_range)
    stretched = librosa.effects.time_stretch(wave, rate=rate).astype(np.float32)
    target_len = len(wave)
    if len(stretched) >= target_len:
        return stretched[:target_len]
    pad = target_len - len(stretched)
    return np.pad(stretched, (0, pad))
