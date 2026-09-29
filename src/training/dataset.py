"""PyTorch Datasets built on top of the preprocessed manifest.csv.

Two datasets, one per model, since the cached representations differ:
- LogMelDataset  -> GenreCNN            (reads logmel/*.npy)
- WaveformDataset -> ASTGenreClassifier (reads waveform16k/*.npy, feature-extracted
                                          lazily so augmentation happens on raw audio first)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.models.common import GENRE_TO_IDX
from src.training.augment import pitch_shift_waveform, spec_augment, time_stretch_waveform


SPLIT_MODES = ("official", "random_clip")
RANDOM_CLIP_SEED = 42
RANDOM_CLIP_FRACTIONS = (0.8, 0.1, 0.1)


def assign_random_clip_splits(
    df: pd.DataFrame, seed: int = RANDOM_CLIP_SEED
) -> pd.DataFrame:
    """Re-split at the *clip* level, 8:1:1, ignoring track and artist boundaries.

    This deliberately reproduces the protocol used by several published FMA-small
    results (e.g. Ashraf et al. 2025, which splits 3-second clips 8:1:1), under which
    clips of the same track land in both train and test. FMA's own paper warns that
    letting the same artist -- let alone the same track -- span splits "leads to
    over-optimistic accuracy", so this mode exists only to quantify that premium by
    running an identical model under both protocols. It must never be used for a
    headline result; see docs/METHODOLOGY.md.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    n_train = int(RANDOM_CLIP_FRACTIONS[0] * len(df))
    n_val = int(RANDOM_CLIP_FRACTIONS[1] * len(df))

    assigned = np.empty(len(df), dtype=object)
    assigned[order[:n_train]] = "train"
    assigned[order[n_train : n_train + n_val]] = "val"
    assigned[order[n_train + n_val :]] = "test"

    out = df.copy()
    out["split"] = assigned
    return out


def load_manifest(
    manifest_path: Path,
    split: str,
    max_subclips_per_track: int | None = None,
    split_mode: str = "official",
) -> pd.DataFrame:
    """Rows belonging to one split.

    `max_subclips_per_track` keeps only the first N sub-clips of each track. Every
    30s track yields ~10 sub-clips, which is more than AST fine-tuning can chew
    through in a laptop time budget; subsampling here rather than at preprocessing
    time means both models share one cache and one manifest.

    `split_mode` selects the evaluation protocol: "official" uses FMA's own
    artist-filtered split (the honest one, and the default); "random_clip" re-splits
    clips 8:1:1 to replicate the leaky protocol used by parts of the literature.
    """
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"Unknown split_mode {split_mode!r}; expected one of {SPLIT_MODES}")

    df = pd.read_csv(manifest_path)
    if split_mode == "random_clip":
        df = assign_random_clip_splits(df)
    df = df[df["split"] == split]
    if max_subclips_per_track is not None:
        df = df[df["subclip_idx"] < max_subclips_per_track]
    return df.reset_index(drop=True)


class LogMelDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        processed_dir: Path,
        split: str,
        augment: bool = False,
        return_track_ids: bool = False,
        split_mode: str = "official",
    ):
        self.df = load_manifest(manifest_path, split, split_mode=split_mode)
        self.processed_dir = Path(processed_dir)
        self.augment = augment and split == "train"
        self.return_track_ids = return_track_ids

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        logmel = np.load(self.processed_dir / row["logmel_path"])
        if self.augment:
            logmel = spec_augment(logmel)
        label = GENRE_TO_IDX[row["genre"]]
        x = torch.from_numpy(logmel).float().unsqueeze(0)
        if self.return_track_ids:
            return x, label, int(row["track_id"])
        return x, label


class WaveformDataset(Dataset):
    """Returns raw (possibly augmented) 16kHz waveforms; AST feature extraction
    happens in the collate_fn so it can be batched efficiently."""

    def __init__(
        self,
        manifest_path: Path,
        processed_dir: Path,
        split: str,
        sample_rate: int = 16000,
        pitch_shift: bool = False,
        time_stretch: bool = False,
        max_subclips_per_track: int | None = None,
        return_track_ids: bool = False,
        split_mode: str = "official",
    ):
        self.df = load_manifest(manifest_path, split, max_subclips_per_track, split_mode=split_mode)
        self.processed_dir = Path(processed_dir)
        self.sample_rate = sample_rate
        self.pitch_shift = pitch_shift and split == "train"
        self.time_stretch = time_stretch and split == "train"
        self.return_track_ids = return_track_ids

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wave = np.load(self.processed_dir / row["waveform_path"])
        if self.pitch_shift:
            wave = pitch_shift_waveform(wave, self.sample_rate)
        if self.time_stretch:
            wave = time_stretch_waveform(wave)
        label = GENRE_TO_IDX[row["genre"]]
        if self.return_track_ids:
            return wave, label, int(row["track_id"])
        return wave, label


def make_ast_collate_fn(feature_extractor):
    def collate(batch):
        # Datasets yield (wave, label) normally and (wave, label, track_id) when
        # return_track_ids is set for track-level evaluation.
        if len(batch[0]) == 3:
            waves, labels, track_ids = zip(*batch)
        else:
            waves, labels = zip(*batch)
            track_ids = None
        out = feature_extractor(list(waves), sampling_rate=16000, return_tensors="pt")
        labels = torch.tensor(labels, dtype=torch.long)
        if track_ids is None:
            return out["input_values"], labels
        return out["input_values"], labels, torch.tensor(track_ids, dtype=torch.long)

    return collate


class MertEmbeddingDataset(Dataset):
    """Cached frozen-MERT episode embeddings, shape (13, 768) per episode.

    Embeddings are stored one file per track, so this caches the most recently
    opened track's array: the manifest is written in track order and never shuffled
    at read time, which turns ~75k file opens into ~8k.
    """

    def __init__(
        self,
        manifest_path: Path,
        embeddings_dir: Path,
        split: str,
        return_track_ids: bool = False,
        split_mode: str = "official",
    ):
        self.df = load_manifest(manifest_path, split, split_mode=split_mode)
        self.embeddings_dir = Path(embeddings_dir)
        self.return_track_ids = return_track_ids
        self._cache_path: str | None = None
        self._cache_array: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, rel_path: str) -> np.ndarray:
        if rel_path != self._cache_path:
            self._cache_array = np.load(self.embeddings_dir / rel_path)
            self._cache_path = rel_path
        return self._cache_array

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        track_array = self._load(row["embedding_path"])
        embedding = torch.from_numpy(track_array[int(row["episode_idx"])].astype(np.float32))
        label = GENRE_TO_IDX[row["genre"]]
        if self.return_track_ids:
            return embedding, label, int(row["track_id"])
        return embedding, label
