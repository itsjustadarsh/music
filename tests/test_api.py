import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.api.main as main


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def synthetic_curator(monkeypatch):
    from src.curator.playlist_agent import PlaylistCurator

    rows = []
    for i in range(20):
        rows.append(
            {
                "track_id": f"t{i}",
                "artists": f"Artist {i % 4}",
                "track_name": f"Track {i}",
                "track_genre": "pop" if i % 2 == 0 else "rock",
                "valence": 0.1 + (i % 10) * 0.09,
                "energy": 0.1 + (i % 7) * 0.13,
                "tempo": 100 + (i % 5) * 10,
                "key": i % 12,
                "mode": i % 2,
                "popularity": (i * 3) % 100,
            }
        )
    curator = PlaylistCurator(pd.DataFrame(rows))
    monkeypatch.setattr(main, "get_curator", lambda: curator)
    return curator


def test_health_reports_missing_model_and_curator(client, monkeypatch):
    monkeypatch.setattr(main, "get_classifier", lambda: None)
    monkeypatch.setattr(main, "get_curator", lambda: None)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert body["curator_loaded"] is False


def test_mood_endpoint_returns_expected_shape(client):
    resp = client.post("/mood", json={"valence": 0.9, "energy": 0.9})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mood"] in {"happy", "excited"}
    assert "radius" in body


def test_mood_endpoint_rejects_out_of_range(client):
    resp = client.post("/mood", json={"valence": 1.5, "energy": 0.5})
    assert resp.status_code == 422


def test_classify_without_model_returns_503(client, monkeypatch):
    monkeypatch.setattr(main, "get_classifier", lambda: None)
    fake_wav = io.BytesIO(b"not a real wav, but request should fail before decoding matters if model missing")
    resp = client.post("/classify", files={"file": ("clip.wav", fake_wav, "audio/wav")})
    assert resp.status_code in (400, 503)


def test_generate_playlist_without_curator_returns_503(client, monkeypatch):
    monkeypatch.setattr(main, "get_curator", lambda: None)
    resp = client.post("/generate-playlist", json={"prompt": "happy pop"})
    assert resp.status_code == 503


def test_generate_playlist_with_curator_returns_tracks(client, synthetic_curator):
    resp = client.post("/generate-playlist", json={"genre": "pop", "mood": "happy", "n": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["tracks"]) == 5
    assert body["resolved_genre"] == "pop"


def test_generate_playlist_parses_free_text_prompt(client, synthetic_curator):
    resp = client.post("/generate-playlist", json={"prompt": "rock songs that feel sad", "n": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["resolved_genre"] == "rock"
    assert body["resolved_mood"] == "sad"


def test_push_playlist_without_spotify_credentials_returns_503(client, synthetic_curator, monkeypatch):
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("SPOTIFY_REDIRECT_URI", raising=False)
    resp = client.post("/push-playlist", json={"playlist_name": "Test", "genre": "pop", "n": 3})
    assert resp.status_code == 503
