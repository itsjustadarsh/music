"""Sanity-check both datasets after download and append observed stats to docs/DATA_CARD.md.

Usage:
    python -m src.data.validate
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from src.config import (
    DOCS_DIR,
    FMA_SMALL_EXPECTED_TRACKS,
    SPOTIFY_TRACKS_CSV,
    SPOTIFY_TRACKS_EXPECTED_ROWS,
    SPOTIFY_TRACKS_ROW_TOLERANCE,
)
from src.data.splits import load_fma_small_tracks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def validate_fma_small() -> dict:
    df = load_fma_small_tracks()
    n = len(df)
    ok = n == FMA_SMALL_EXPECTED_TRACKS
    if not ok:
        log.warning("fma_small track count = %d, expected %d", n, FMA_SMALL_EXPECTED_TRACKS)
    else:
        log.info("fma_small track count OK: %d", n)
    return {
        "dataset": "fma_small",
        "n_tracks": n,
        "expected": FMA_SMALL_EXPECTED_TRACKS,
        "ok": ok,
        "split_counts": df["split"].value_counts().to_dict(),
        "genre_counts": df["genre_top"].value_counts().to_dict(),
    }


def validate_spotify_tracks() -> dict:
    df = pd.read_csv(SPOTIFY_TRACKS_CSV)
    n = len(df)
    lo = SPOTIFY_TRACKS_EXPECTED_ROWS - SPOTIFY_TRACKS_ROW_TOLERANCE
    hi = SPOTIFY_TRACKS_EXPECTED_ROWS + SPOTIFY_TRACKS_ROW_TOLERANCE
    ok = lo <= n <= hi
    if not ok:
        log.warning("spotify_tracks row count = %d, expected ~%d", n, SPOTIFY_TRACKS_EXPECTED_ROWS)
    else:
        log.info("spotify_tracks row count OK: %d", n)
    return {
        "dataset": "spotify_tracks",
        "n_rows": n,
        "expected_approx": SPOTIFY_TRACKS_EXPECTED_ROWS,
        "ok": ok,
        "n_unique_genres": int(df["track_genre"].nunique()) if "track_genre" in df.columns else None,
    }


def append_data_card(fma_stats: dict, spotify_stats: dict) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / "DATA_CARD.md"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    block = [
        f"\n## Validation run: {stamp}\n",
        f"- **fma_small**: {fma_stats['n_tracks']} tracks (expected {fma_stats['expected']}) "
        f"-- {'OK' if fma_stats['ok'] else 'MISMATCH'}",
        f"  - Split counts: {fma_stats['split_counts']}",
        f"  - Genre counts: {fma_stats['genre_counts']}",
        f"- **spotify_tracks**: {spotify_stats['n_rows']} rows "
        f"(expected ~{spotify_stats['expected_approx']}) -- {'OK' if spotify_stats['ok'] else 'MISMATCH'}",
        f"  - Unique genres: {spotify_stats['n_unique_genres']}",
        "",
    ]
    with open(path, "a") as f:
        f.write("\n".join(block))
    log.info("Appended validation results to %s", path)


def main() -> None:
    fma_stats = validate_fma_small()
    spotify_stats = validate_spotify_tracks()
    append_data_card(fma_stats, spotify_stats)
    if not (fma_stats["ok"] and spotify_stats["ok"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
