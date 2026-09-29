"""Pydantic request/response models for the FastAPI service."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ClassifyResponse(BaseModel):
    genre: str
    confidence: float
    all_scores: dict[str, float]
    model_used: str


class MoodRequest(BaseModel):
    valence: float = Field(..., ge=0.0, le=1.0)
    energy: float = Field(..., ge=0.0, le=1.0)


class MoodResponse(BaseModel):
    mood: str
    valence: float
    energy: float
    radius: float


class PlaylistRequest(BaseModel):
    prompt: str | None = Field(None, description="Free-text prompt, e.g. 'upbeat happy pop'")
    genre: str | None = None
    mood: str | None = None
    n: int = Field(20, ge=1, le=100)


class TrackOut(BaseModel):
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


class PlaylistResponse(BaseModel):
    resolved_genre: str | None
    resolved_mood: str
    tracks: list[TrackOut]


class PushPlaylistRequest(BaseModel):
    playlist_name: str
    prompt: str | None = None
    genre: str | None = None
    mood: str | None = None
    n: int = Field(20, ge=1, le=100)
    public: bool = False


class PushPlaylistResponse(BaseModel):
    playlist_id: str
    playlist_url: str
    added_count: int
    skipped_tracks: list[str]
