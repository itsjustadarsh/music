"""PyTorch Datasets built on top of the preprocessed manifest.csv.

Two datasets, because the cached representations differ:
- LogMelDataset       -> the from-scratch spectrogram models (reads logmel/*.npy)
- MertEmbeddingDataset -> the frozen-backbone probes (reads cached layer stacks)

Both take `split_mode`, which is what puts a run on a rung of the protocol ladder.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.models.common import GENRE_TO_IDX
from src.training.augment import spec_augment


SPLIT_MODES = ("official", "random_clip", "random_track")
RANDOM_CLIP_SEED = 42
RANDOM_CLIP_FRACTIONS = (0.8, 0.1, 0.1)


def assign_random_track_splits(
    df: pd.DataFrame, seed: int = RANDOM_CLIP_SEED
) -> pd.DataFrame:
    """Re-split 8:1:1 at the *track* level: no clip leakage, but no artist filter either.

    The middle rung of the protocol ladder. Several published FMA-small results state a
    random 8:1:1 ratio over songs without mentioning artists (Zhang 2025 is explicit about
    splitting songs; Ma et al. 2026 give the ratio but never say what unit it applies to).
    Scoring those against either "official" or "random_clip" would misattribute the gap, so
    this mode isolates the artist/album effect on its own, with same-track leakage removed.
    """
    rng = np.random.default_rng(seed)
    track_ids = np.sort(df["track_id"].unique())
    order = rng.permutation(len(track_ids))
    n_train = int(RANDOM_CLIP_FRACTIONS[0] * len(track_ids))
    n_val = int(RANDOM_CLIP_FRACTIONS[1] * len(track_ids))

    assigned = np.empty(len(track_ids), dtype=object)
    assigned[order[:n_train]] = "train"
    assigned[order[n_train : n_train + n_val]] = "val"
    assigned[order[n_train + n_val :]] = "test"

    out = df.copy()
    out["split"] = pd.Series(df["track_id"]).map(dict(zip(track_ids, assigned)))
    return out


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

    `split_mode` selects the evaluation protocol, in increasing order of leakage:
    "official" uses FMA's own artist-filtered split (the honest one, and the default);
    "random_track" re-splits whole tracks 8:1:1, so clips never span splits but artists
    do; "random_clip" re-splits individual clips 8:1:1, so the same track appears in both
    train and test. Only "official" may produce a headline number.
    """
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"Unknown split_mode {split_mode!r}; expected one of {SPLIT_MODES}")

    df = pd.read_csv(manifest_path)
    if split_mode == "random_clip":
        df = assign_random_clip_splits(df)
    elif split_mode == "random_track":
        df = assign_random_track_splits(df)
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
