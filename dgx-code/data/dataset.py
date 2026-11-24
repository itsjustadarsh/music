"""FMA-small dataset for DGX training.

Two modes:
1. Raw audio (for fine-tuning MERT/AST end-to-end) — loads full 30s or variable-length crops
2. Pre-extracted spectrograms (for AST/PaSST fine-tuning)

Always uses the official artist-filtered split. Supports both episode-level (3s clips)
and track-level (full 30s) loading.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

DATA_DIR = Path(os.getenv("FMA_DATA_DIR", "fma_data"))
RAW_DIR = DATA_DIR / "raw"
AUDIO_DIR = RAW_DIR / "fma_small"
METADATA_DIR = RAW_DIR / "fma_metadata"

FMA_GENRES = [
    "Electronic", "Experimental", "Folk", "Hip-Hop",
    "Instrumental", "International", "Pop", "Rock",
]
GENRE_TO_IDX = {g: i for i, g in enumerate(FMA_GENRES)}
IDX_TO_GENRE = {i: g for g, i in GENRE_TO_IDX.items()}
NUM_CLASSES = len(FMA_GENRES)


def load_fma_metadata() -> pd.DataFrame:
    """Load FMA metadata and return a DataFrame with track_id, genre, split."""
    from data import memory_cache
    if memory_cache.is_loaded():
        return memory_cache.METADATA.copy()

    tracks_path = METADATA_DIR / "tracks.csv"
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])

    # Extract the fields we need
    subset = tracks[("set", "subset")]
    split = tracks[("set", "split")]
    genre = tracks[("track", "genre_top")]

    df = pd.DataFrame({
        "track_id": subset.index,
        "genre": genre.values,
        "split": split.values,
        "subset": subset.values,
    })

    # Keep only fma_small tracks with a valid genre
    df = df[df["subset"] == "small"].copy()
    df = df[df["genre"].isin(FMA_GENRES)].copy()
    df = df.dropna(subset=["split"])

    # Map split names
    split_map = {"training": "train", "validation": "val", "test": "test"}
    df["split"] = df["split"].map(split_map)
    df = df.dropna(subset=["split"])

    return df.reset_index(drop=True)


def load_waveform(track_id: int, target_sr: int) -> torch.Tensor:
    """Mono waveform at target_sr. Uses the in-memory cache when loaded, else reads the MP3."""
    from data import memory_cache
    if memory_cache.is_loaded() and target_sr in memory_cache.AUDIO:
        return torch.from_numpy(memory_cache.AUDIO[target_sr][memory_cache.ROW_OF[track_id]])

    audio_path = track_id_to_path(track_id)
    try:
        waveform, sr = torchaudio.load(str(audio_path))
    except Exception:
        import librosa
        wav, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
        waveform = torch.from_numpy(wav).unsqueeze(0)
        sr = target_sr
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


def track_id_to_path(track_id: int) -> Path:
    """Convert a track ID to its audio file path."""
    tid_str = f"{track_id:06d}"
    return AUDIO_DIR / tid_str[:3] / f"{tid_str}.mp3"


class FMAFullTrackDataset(Dataset):
    """Loads full 30-second tracks as raw waveforms.

    For MERT fine-tuning: returns audio at target_sr (24kHz for MERT).
    For AST/PaSST: returns audio at 16kHz.

    Each track is padded/truncated to exactly `max_duration_sec` seconds.
    """

    def __init__(
        self,
        split: str,
        target_sr: int = 24000,
        max_duration_sec: float = 30.0,
        augment: bool = False,
    ):
        self.metadata = load_fma_metadata()
        self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        self.target_sr = target_sr
        self.max_samples = int(max_duration_sec * target_sr)
        self.augment = augment and split == "train"

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        track_id = int(row["track_id"])
        genre = row["genre"]
        label = GENRE_TO_IDX[genre]

        waveform = load_waveform(track_id, self.target_sr)

        # Pad or truncate to max_samples
        if waveform.shape[0] < self.max_samples:
            pad = torch.zeros(self.max_samples - waveform.shape[0])
            waveform = torch.cat([waveform, pad])
        else:
            waveform = waveform[: self.max_samples]

        if self.augment:
            waveform = self._augment(waveform)

        return waveform, label, track_id

    def _augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Simple waveform augmentations for training."""
        # Random gain (-6dB to +6dB)
        gain_db = torch.empty(1).uniform_(-6, 6).item()
        waveform = waveform * (10 ** (gain_db / 20))

        # Random time shift (up to 0.5s)
        max_shift = int(0.5 * self.target_sr)
        shift = torch.randint(-max_shift, max_shift, (1,)).item()
        if shift > 0:
            waveform = torch.cat([torch.zeros(shift), waveform[:-shift]])
        elif shift < 0:
            waveform = torch.cat([waveform[-shift:], torch.zeros(-shift)])

        # Additive Gaussian noise
        if torch.rand(1).item() < 0.5:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise

        return waveform


class FMAEpisodeDataset(Dataset):
    """Loads 3-second episodes from FMA tracks.

    Splits each 30s track into 10 non-overlapping 3s episodes.
    Compatible with the laptop pipeline's episode protocol.
    """

    def __init__(
        self,
        split: str,
        target_sr: int = 24000,
        episode_sec: float = 3.0,
        augment: bool = False,
    ):
        self.metadata = load_fma_metadata()
        self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        self.target_sr = target_sr
        self.episode_samples = int(episode_sec * target_sr)
        self.episodes_per_track = int(30.0 / episode_sec)
        self.augment = augment and split == "train"

        # Build episode index: (track_row_idx, episode_idx)
        self.episodes = []
        for i in range(len(self.metadata)):
            for ep in range(self.episodes_per_track):
                self.episodes.append((i, ep))

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int):
        track_idx, ep_idx = self.episodes[idx]
        row = self.metadata.iloc[track_idx]
        track_id = int(row["track_id"])
        label = GENRE_TO_IDX[row["genre"]]

        waveform = load_waveform(track_id, self.target_sr)

        # Extract the episode
        start = ep_idx * self.episode_samples
        end = start + self.episode_samples
        if end > waveform.shape[0]:
            episode = torch.zeros(self.episode_samples)
            available = waveform.shape[0] - start
            if available > 0:
                episode[:available] = waveform[start : start + available]
        else:
            episode = waveform[start:end]

        if self.augment:
            # Lighter augmentation for short episodes
            if torch.rand(1).item() < 0.5:
                gain_db = torch.empty(1).uniform_(-3, 3).item()
                episode = episode * (10 ** (gain_db / 20))
            if torch.rand(1).item() < 0.3:
                noise = torch.randn_like(episode) * 0.003
                episode = episode + noise

        return episode, label, track_id


class FMARandomCropDataset(Dataset):
    """Training-only: each item is a random `crop_sec` window from a full track.

    `repeats` = how many random crops per track per epoch. Gives far more
    variety than fixed episodes, which reduces overfitting of large models.
    """

    def __init__(self, split: str, target_sr: int = 16000, crop_sec: float = 10.0,
                 repeats: int = 3, augment: bool = True):
        self.metadata = load_fma_metadata()
        self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        self.track_ids = self.metadata["track_id"].astype(int).to_numpy()
        self.labels = self.metadata["genre"].map(GENRE_TO_IDX).to_numpy()
        self.target_sr = target_sr
        self.crop = int(crop_sec * target_sr)
        self.repeats = repeats
        self.augment = augment

    def __len__(self) -> int:
        return len(self.metadata) * self.repeats

    def __getitem__(self, idx: int):
        i = idx % len(self.metadata)
        tid = int(self.track_ids[i])
        wav = load_waveform(tid, self.target_sr)
        max_start = max(0, wav.shape[0] - self.crop)
        s = int(torch.randint(0, max_start + 1, (1,)))
        x = wav[s : s + self.crop]
        if x.shape[0] < self.crop:
            x = torch.cat([x, torch.zeros(self.crop - x.shape[0])])
        if self.augment:
            x = x * (10 ** (torch.empty(1).uniform_(-6, 6).item() / 20))
            if torch.rand(1).item() < 0.3:
                x = x + torch.randn_like(x) * 0.003
        return x, int(self.labels[i]), tid
