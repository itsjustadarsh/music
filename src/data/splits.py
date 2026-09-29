"""Load the FMA-small subset and its official train/val/test split from tracks.csv.

FMA's tracks.csv has a 3-row multi-index header. The columns we care about are
('set', 'subset') -- which size tier a track belongs to (small/medium/large) --
and ('set', 'split') -- the official training/validation/test assignment used
by the FMA paper, plus ('track', 'genre_top') for the top-level genre label.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import FMA_GENRES, FMA_METADATA_DIR

TRACKS_CSV = FMA_METADATA_DIR / "tracks.csv"

SPLIT_NAME_MAP = {
    "training": "train",
    "validation": "val",
    "test": "test",
}


def load_fma_small_tracks(tracks_csv: Path = TRACKS_CSV) -> pd.DataFrame:
    """Return a DataFrame indexed by track_id, restricted to the fma_small subset,
    with flat columns: genre_top, split ('train'/'val'/'test')."""
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])

    small = tracks[tracks[("set", "subset")] == "small"]
    df = pd.DataFrame(
        {
            "genre_top": small[("track", "genre_top")],
            "split": small[("set", "split")].map(SPLIT_NAME_MAP),
        },
        index=small.index,
    )
    df.index.name = "track_id"
    df = df.dropna(subset=["genre_top", "split"])
    return df


def track_id_to_path(track_id: int, audio_dir: Path) -> Path:
    """FMA stores audio as data/raw/fma_small/<first 3 digits>/<zero-padded id>.mp3"""
    tid_str = f"{track_id:06d}"
    return audio_dir / tid_str[:3] / f"{tid_str}.mp3"


def genre_to_label(genre: str) -> int:
    return FMA_GENRES.index(genre)


if __name__ == "__main__":
    df = load_fma_small_tracks()
    print(df["split"].value_counts())
    print(df["genre_top"].value_counts())
