"""Cross-check mood tagging against FMA's echonest.csv subset.

`echonest.csv` (part of fma_metadata.zip) has Spotify-style audio features
(valence, energy, etc.) for a subset of FMA tracks that also have a genre_top
label in tracks.csv. We tag each of those tracks with our mood taxonomy and
build a genre x mood crosstab -- not a hard validation (mood and genre are
legitimately different axes), but a sanity check that the mapping produces
musically sensible results (e.g. Hip-Hop skewing energetic, Instrumental
skewing calm) rather than noise.

Usage:
    python -m src.mood.echonest_crosscheck
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config import DOCS_DIR, FMA_METADATA_DIR
from src.data.splits import load_fma_small_tracks
from src.mood.mood_tagger import tag_mood

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ECHONEST_CSV = FMA_METADATA_DIR / "echonest.csv"


def load_echonest_features() -> pd.DataFrame:
    """echonest.csv has a 3-row multi-index header like tracks.csv. We only need
    the ('echonest', 'audio_features', {valence, energy}) columns."""
    df = pd.read_csv(ECHONEST_CSV, index_col=0, header=[0, 1, 2])
    features = df["echonest"]["audio_features"][["valence", "energy"]].copy()
    features.index.name = "track_id"
    return features.dropna()


def build_genre_mood_crosstab() -> pd.DataFrame:
    features = load_echonest_features()
    genres = load_fma_small_tracks()["genre_top"]

    joined = features.join(genres, how="inner")
    joined["mood"] = joined.apply(lambda r: tag_mood(r["valence"], r["energy"]).mood.value, axis=1)

    crosstab = pd.crosstab(joined["genre_top"], joined["mood"], normalize="index")
    return crosstab.round(3)


def main() -> None:
    crosstab = build_genre_mood_crosstab()
    log.info("Genre x mood distribution (row-normalized):\n%s", crosstab)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_DIR / "genre_mood_crosstab.csv"
    crosstab.to_csv(out_path)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
