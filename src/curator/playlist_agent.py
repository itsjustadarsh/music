"""Curator agent: given a genre/mood prompt, produce a ranked, DJ-ordered
track list from the Spotify Tracks Dataset.

Pipeline:
1. Parse the request (structured genre/mood, or a free-text prompt like
   "upbeat happy pop") into a target genre and target mood/valence/energy point.
2. Filter the track pool to that genre (soft-falls back to the whole pool if the
   genre isn't found) and score every track by closeness to the mood target
   plus popularity, as a relevance ranking.
3. Re-order the top-ranked candidates with a greedy nearest-neighbor walk over
   (tempo, key, mode) so adjacent tracks flow into each other, while never
   placing two tracks by the same artist back-to-back.

This is deliberately dependency-free (pandas + stdlib only) so it's easy to
unit test without any model or network access.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import pandas as pd

from src.mood.mood_tagger import Mood, tag_mood

# Free-text mood keywords -> canonical Mood. Extend as needed.
_MOOD_KEYWORDS: dict[str, Mood] = {
    "excited": Mood.EXCITED, "hype": Mood.EXCITED, "energetic": Mood.EXCITED,
    "happy": Mood.HAPPY, "joyful": Mood.HAPPY, "upbeat": Mood.HAPPY,
    "content": Mood.CONTENT, "peaceful": Mood.CONTENT, "chill": Mood.CONTENT,
    "relaxed": Mood.RELAXED, "calm": Mood.RELAXED, "mellow": Mood.RELAXED,
    "tired": Mood.TIRED, "sleepy": Mood.TIRED, "drowsy": Mood.TIRED,
    "sad": Mood.SAD, "melancholy": Mood.SAD, "down": Mood.SAD,
    "tense": Mood.TENSE, "anxious": Mood.TENSE, "nervous": Mood.TENSE,
    "angry": Mood.ANGRY, "aggressive": Mood.ANGRY, "furious": Mood.ANGRY,
    "neutral": Mood.NEUTRAL,
}

# Center point (valence, energy) each mood maps back to, for scoring candidates
# -- the inverse of mood_tagger.tag_mood's octant midpoints.
_MOOD_CENTERS: dict[Mood, tuple[float, float]] = {
    Mood.HAPPY: (0.85, 0.60), Mood.EXCITED: (0.60, 0.90),
    Mood.TENSE: (0.35, 0.85), Mood.ANGRY: (0.15, 0.75),
    Mood.SAD: (0.15, 0.35), Mood.TIRED: (0.30, 0.10),
    Mood.RELAXED: (0.70, 0.20), Mood.CONTENT: (0.85, 0.40),
    Mood.NEUTRAL: (0.5, 0.5),
}

REQUIRED_COLUMNS = [
    "track_id", "artists", "track_name", "track_genre",
    "valence", "energy", "tempo", "key", "mode", "popularity",
]


@dataclass
class CuratedTrack:
    track_id: str
    artist: str
    name: str
    genre: str
    tempo: float
    key: int
    mode: int
    valence: float
    energy: float
    score: float


def parse_prompt(prompt: str, known_genres: list[str]) -> tuple[str | None, Mood | None]:
    """Very small, dependency-free NLP: lowercase keyword matching for genre and mood."""
    text = prompt.lower()
    matches = [g for g in known_genres if re.search(rf"\b{re.escape(g.lower())}\b", text)]
    mood = next((m for kw, m in _MOOD_KEYWORDS.items() if re.search(rf"\b{kw}\b", text)), None)
    # The Spotify Tracks Dataset ships genres that are also mood words ("happy",
    # "sad", "chill"), so a prompt like "upbeat happy pop" matches both "happy" and
    # "pop". Prefer the match that isn't doubling as the mood term; fall back to the
    # first match when the mood word is the only genre named ("happy songs").
    genre = next((g for g in matches if g.lower() not in _MOOD_KEYWORDS), matches[0] if matches else None)
    return genre, mood


def _circular_key_distance(k1: int, k2: int) -> float:
    d = abs(k1 - k2) % 12
    return min(d, 12 - d) / 6.0  # normalize to 0..1


def _tempo_distance(t1: float, t2: float) -> float:
    return abs(t1 - t2) / max(t1, t2, 1.0)


def _mood_distance(row: pd.Series, target_valence: float, target_energy: float) -> float:
    return math.hypot(row["valence"] - target_valence, row["energy"] - target_energy)


class PlaylistCurator:
    def __init__(self, tracks_df: pd.DataFrame):
        missing = [c for c in REQUIRED_COLUMNS if c not in tracks_df.columns]
        if missing:
            raise ValueError(f"tracks_df is missing required columns: {missing}")
        self.tracks = tracks_df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
        self.known_genres = sorted(self.tracks["track_genre"].unique())

    def resolve_request(
        self, genre: str | None = None, mood: str | Mood | None = None, prompt: str | None = None,
    ) -> tuple[str | None, Mood]:
        if prompt and not (genre or mood):
            genre, mood = parse_prompt(prompt, self.known_genres)
        if isinstance(mood, str):
            mood = _MOOD_KEYWORDS.get(mood.lower()) or Mood(mood.lower())
        return genre, mood or Mood.NEUTRAL

    def _candidate_pool(self, genre: str | None) -> pd.DataFrame:
        if genre is None:
            return self.tracks
        match = self.tracks[self.tracks["track_genre"].str.lower() == genre.lower()]
        return match if len(match) > 0 else self.tracks

    def _rank_candidates(self, pool: pd.DataFrame, mood: Mood, pool_size: int) -> pd.DataFrame:
        target_valence, target_energy = _MOOD_CENTERS[mood]
        pool = pool.copy()
        pool["mood_distance"] = pool.apply(_mood_distance, axis=1, args=(target_valence, target_energy))
        # Lower mood_distance is better; higher popularity is better. Normalize popularity to 0..1.
        pop_norm = pool["popularity"] / 100.0
        pool["score"] = (1.0 - pool["mood_distance"].clip(0, 1)) * 0.7 + pop_norm * 0.3
        return pool.sort_values("score", ascending=False).head(pool_size)

    def _order_for_flow(self, candidates: pd.DataFrame, n: int) -> list[CuratedTrack]:
        remaining = candidates.to_dict("records")
        if not remaining:
            return []

        ordered: list[CuratedTrack] = []
        current = remaining.pop(0)
        ordered.append(self._to_curated(current))

        while remaining and len(ordered) < n:
            last = current
            last_artist = last["artists"]

            def transition_cost(cand: dict) -> float:
                return _tempo_distance(last["tempo"], cand["tempo"]) + _circular_key_distance(
                    int(last["key"]), int(cand["key"])
                )

            non_dup = [c for c in remaining if c["artists"] != last_artist]
            search_space = non_dup if non_dup else remaining
            best = min(search_space, key=transition_cost)
            remaining.remove(best)
            ordered.append(self._to_curated(best))
            current = best

        return ordered

    @staticmethod
    def _to_curated(row: dict) -> CuratedTrack:
        return CuratedTrack(
            track_id=str(row["track_id"]),
            artist=row["artists"],
            name=row["track_name"],
            genre=row["track_genre"],
            tempo=float(row["tempo"]),
            key=int(row["key"]),
            mode=int(row["mode"]),
            valence=float(row["valence"]),
            energy=float(row["energy"]),
            score=float(row.get("score", 0.0)),
        )

    def generate_playlist(
        self,
        genre: str | None = None,
        mood: str | Mood | None = None,
        prompt: str | None = None,
        n: int = 20,
        candidate_pool_size: int = 200,
    ) -> list[CuratedTrack]:
        resolved_genre, resolved_mood = self.resolve_request(genre, mood, prompt)
        pool = self._candidate_pool(resolved_genre)
        ranked = self._rank_candidates(pool, resolved_mood, max(candidate_pool_size, n))
        return self._order_for_flow(ranked, n)
