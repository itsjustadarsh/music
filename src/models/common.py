"""Shared model utilities: device auto-detection and label encoding."""
from __future__ import annotations

import os

import torch

from src.config import FMA_GENRES

# Must be set before any MPS op runs so unsupported ops silently fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

GENRE_TO_IDX = {g: i for i, g in enumerate(FMA_GENRES)}
IDX_TO_GENRE = {i: g for g, i in GENRE_TO_IDX.items()}
NUM_CLASSES = len(FMA_GENRES)


def get_device(prefer: str | None = None) -> torch.device:
    """Auto-detect the best available device: MPS (Apple Silicon) > CUDA > CPU.

    Pass `prefer` to force a specific backend (useful for CPU-only smoke tests).
    """
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
