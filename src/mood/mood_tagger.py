"""Map Spotify-style audio features to a mood label using Russell's Circumplex
Model of Affect.

Reference: Russell, J. A. (1980). "A circumplex model of affect."
Journal of Personality and Social Psychology, 39(6), 1161-1178.

The circumplex places affective states on a 2D plane: valence (pleasant <->
unpleasant, x-axis) and arousal (activated <-> deactivated, y-axis). We treat
Spotify's `valence` feature directly as the x-axis, and `energy` as a proxy for
arousal (the y-axis) -- a common substitution in music-emotion-recognition
literature since Spotify does not expose a separate "arousal" feature.

Both valence and energy are already 0..1 in the Spotify Tracks Dataset, so we
recenter them to -1..1 and bucket the resulting point into one of 8 octants,
with a small "neutral" radius at the center for tracks that aren't strongly
characterized either way.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Mood(str, Enum):
    EXCITED = "excited"        # high arousal, positive valence
    HAPPY = "happy"             # high-ish arousal, high valence
    CONTENT = "content"         # low-ish arousal, high valence
    RELAXED = "relaxed"         # low arousal, positive valence
    TIRED = "tired"              # low arousal, negative valence
    SAD = "sad"                  # low-ish arousal, low valence
    TENSE = "tense"              # high-ish arousal, low valence
    ANGRY = "angry"              # high arousal, negative valence
    NEUTRAL = "neutral"          # near the center of the plane


# Octant boundaries in degrees, measured counterclockwise from the positive
# valence axis (0 degrees = pure positive valence, 90 = pure high arousal).
_OCTANT_ORDER = [
    Mood.HAPPY,     # 0-45
    Mood.EXCITED,   # 45-90
    Mood.TENSE,     # 90-135
    Mood.ANGRY,     # 135-180
    Mood.SAD,       # 180-225
    Mood.TIRED,     # 225-270
    Mood.RELAXED,   # 270-315
    Mood.CONTENT,   # 315-360
]

NEUTRAL_RADIUS = 0.15  # points within this distance of the origin are "neutral"


@dataclass(frozen=True)
class MoodResult:
    mood: Mood
    valence: float
    arousal: float
    radius: float
    angle_degrees: float


def _recenter(x: float) -> float:
    """Map a 0..1 Spotify feature to -1..1."""
    return max(0.0, min(1.0, x)) * 2 - 1


def tag_mood(valence: float, energy: float) -> MoodResult:
    """valence, energy: Spotify audio features in 0..1."""
    v = _recenter(valence)
    a = _recenter(energy)
    radius = math.hypot(v, a)

    if radius < NEUTRAL_RADIUS:
        return MoodResult(Mood.NEUTRAL, v, a, radius, 0.0)

    angle = math.degrees(math.atan2(a, v)) % 360
    octant_idx = int(angle // 45) % 8
    mood = _OCTANT_ORDER[octant_idx]
    return MoodResult(mood, v, a, radius, angle)


def tag_mood_batch(records: list[dict]) -> list[MoodResult]:
    """records: list of {'valence': float, 'energy': float, ...}"""
    return [tag_mood(r["valence"], r["energy"]) for r in records]
