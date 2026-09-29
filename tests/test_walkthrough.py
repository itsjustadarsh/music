"""Integration walkthrough tests — run with ``pytest tests/test_walkthrough.py -v``
to see a feature-by-feature demo of the Music Agent system.

These tests use real data and trained models when available, and skip
gracefully (with a reason) when they aren't.
"""
from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import CLIP_LENGTH_SAMPLES, OUTPUT_DIR, SAMPLE_RATE, SPOTIFY_TRACKS_CSV
from src.models.common import IDX_TO_GENRE, NUM_CLASSES

_has_spotify_csv = SPOTIFY_TRACKS_CSV.exists()
_best_run = OUTPUT_DIR / "cnn_tuned_lr"
_has_trained_model = (_best_run / "best.pt").exists() and (_best_run / "config.json").exists()

skip_no_csv = pytest.mark.skipif(not _has_spotify_csv, reason="Spotify CSV not available")
skip_no_model = pytest.mark.skipif(not _has_trained_model, reason="Trained model not available")


# ---------------------------------------------------------------------------
# 1. Mood tagging — Russell's Circumplex covers all 8 octants + neutral
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "valence, energy, expected_mood",
    [
        (0.90, 0.70, "happy"),
        (0.70, 0.95, "excited"),
        (0.20, 0.85, "tense"),
        (0.05, 0.80, "angry"),
        (0.10, 0.20, "sad"),
        (0.25, 0.05, "tired"),
        (0.80, 0.15, "relaxed"),
        (0.90, 0.35, "content"),
        (0.50, 0.50, "neutral"),
    ],
    ids=["happy", "excited", "tense", "angry", "sad", "tired", "relaxed", "content", "neutral"],
)
def test_mood_circumplex_maps_all_eight_octants(valence, energy, expected_mood):
    """Each (valence, energy) pair lands in the expected mood octant."""
    from src.mood.mood_tagger import tag_mood

    result = tag_mood(valence, energy)
    assert result.mood.value == expected_mood


# ---------------------------------------------------------------------------
# 2. Mood batch processing — tag multiple Spotify-style feature dicts at once
# ---------------------------------------------------------------------------

def test_mood_batch_tags_spotify_style_features():
    from src.mood.mood_tagger import MoodResult, tag_mood_batch

    records = [
        {"valence": 0.9, "energy": 0.8},
        {"valence": 0.1, "energy": 0.1},
        {"valence": 0.5, "energy": 0.5},
    ]
    results = tag_mood_batch(records)
    assert len(results) == 3
    assert all(isinstance(r, MoodResult) for r in results)
    assert results[0].mood.value in ("happy", "excited")
    assert results[1].mood.value in ("sad", "tired")
    assert results[2].mood.value == "neutral"


# ---------------------------------------------------------------------------
# 3. Curator loads the real Spotify Tracks Dataset
# ---------------------------------------------------------------------------

@skip_no_csv
def test_playlist_curator_loads_real_spotify_dataset():
    from src.curator.playlist_agent import PlaylistCurator

    df = pd.read_csv(SPOTIFY_TRACKS_CSV)
    curator = PlaylistCurator(df)
    genres = curator.tracks["track_genre"].nunique()
    assert genres > 100, f"Expected 100+ genres, got {genres}"
    assert len(curator.tracks) > 100_000


# ---------------------------------------------------------------------------
# 4. Natural language prompt -> playlist with genre + mood resolution
# ---------------------------------------------------------------------------

@skip_no_csv
def test_playlist_from_natural_language_prompt():
    from src.curator.playlist_agent import PlaylistCurator
    from src.mood.mood_tagger import Mood

    curator = PlaylistCurator(pd.read_csv(SPOTIFY_TRACKS_CSV))
    genre, mood = curator.resolve_request(genre=None, mood=None, prompt="calm relaxed jazz")
    assert genre == "jazz"
    assert mood == Mood.RELAXED

    tracks = curator.generate_playlist(prompt="calm relaxed jazz", n=10)
    assert len(tracks) == 10
    assert all(t.score > 0 for t in tracks)

    # No back-to-back same artist
    artists = [t.artist for t in tracks]
    for a, b in zip(artists, artists[1:]):
        assert a != b, f"Back-to-back artist: {a}"


# ---------------------------------------------------------------------------
# 5. Mood filtering produces coherent results — happy > sad in valence
# ---------------------------------------------------------------------------

@skip_no_csv
def test_mood_filtering_produces_coherent_results():
    from src.curator.playlist_agent import PlaylistCurator

    curator = PlaylistCurator(pd.read_csv(SPOTIFY_TRACKS_CSV))
    happy = curator.generate_playlist(genre="pop", mood="happy", n=20)
    sad = curator.generate_playlist(genre="pop", mood="sad", n=20)

    avg_v_happy = sum(t.valence for t in happy) / len(happy)
    avg_v_sad = sum(t.valence for t in sad) / len(sad)
    assert avg_v_happy > avg_v_sad, (
        f"Happy avg valence ({avg_v_happy:.2f}) should exceed sad ({avg_v_sad:.2f})"
    )


# ---------------------------------------------------------------------------
# 6. Flow ordering minimizes tempo jumps vs random shuffle
# ---------------------------------------------------------------------------

@skip_no_csv
def test_flow_ordering_minimizes_tempo_jumps():
    from src.curator.playlist_agent import PlaylistCurator

    curator = PlaylistCurator(pd.read_csv(SPOTIFY_TRACKS_CSV))
    tracks = curator.generate_playlist(genre="pop", mood="happy", n=20)
    tempos = [t.tempo for t in tracks]

    ordered_jumps = sum(abs(a - b) for a, b in zip(tempos, tempos[1:])) / (len(tempos) - 1)

    # Compare against 50 random shuffles of the same tracks
    rng = random.Random(42)
    shuffle_avgs = []
    for _ in range(50):
        shuffled = tempos[:]
        rng.shuffle(shuffled)
        avg = sum(abs(a - b) for a, b in zip(shuffled, shuffled[1:])) / (len(shuffled) - 1)
        shuffle_avgs.append(avg)

    random_avg = sum(shuffle_avgs) / len(shuffle_avgs)
    assert ordered_jumps < random_avg, (
        f"Ordered tempo jumps ({ordered_jumps:.1f}) should be less than random ({random_avg:.1f})"
    )


# ---------------------------------------------------------------------------
# 7. API /health reports system status
# ---------------------------------------------------------------------------

def test_api_health_endpoint_reports_system_status():
    import src.api.main as api_main

    client = TestClient(api_main.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)
    assert isinstance(body["curator_loaded"], bool)


# ---------------------------------------------------------------------------
# 8. API /mood returns circumplex result
# ---------------------------------------------------------------------------

def test_api_mood_endpoint_returns_circumplex_result():
    import src.api.main as api_main

    client = TestClient(api_main.app)
    resp = client.post("/mood", json={"valence": 0.9, "energy": 0.9})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mood"] in ("happy", "excited")
    assert "radius" in body
    assert body["radius"] > 0


# ---------------------------------------------------------------------------
# 9. API /generate-playlist end-to-end
# ---------------------------------------------------------------------------

@skip_no_csv
def test_api_generate_playlist_end_to_end():
    import src.api.main as api_main

    client = TestClient(api_main.app)
    resp = client.post("/generate-playlist", json={"prompt": "energetic rock", "n": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["resolved_genre"] == "rock"
    assert len(body["tracks"]) == 10
    assert all("name" in t and "artist" in t for t in body["tracks"])


# ---------------------------------------------------------------------------
# 10. Genre classifier loads and predicts on synthetic audio
# ---------------------------------------------------------------------------

@skip_no_model
def test_genre_classifier_loads_and_predicts():
    import torch

    from src.data.preprocess import audio_to_logmel
    from src.models.cnn import GenreCNN

    # Synthetic 3-second sine wave (440 Hz)
    t = np.linspace(0, 3.0, CLIP_LENGTH_SAMPLES, dtype=np.float32)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)

    logmel = audio_to_logmel(sine)
    assert logmel.shape[0] == 128, f"Expected 128 mel bins, got {logmel.shape[0]}"

    model = GenreCNN(num_classes=NUM_CLASSES)
    ckpt = torch.load(_best_run / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    x = torch.from_numpy(logmel).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

    assert probs.shape == (NUM_CLASSES,)
    assert abs(probs.sum() - 1.0) < 1e-4, "Probabilities should sum to ~1"
    predicted = IDX_TO_GENRE[int(probs.argmax())]
    assert predicted in IDX_TO_GENRE.values(), f"Unknown genre: {predicted}"


# ---------------------------------------------------------------------------
# 11. Evaluation metrics exist and are reasonable
# ---------------------------------------------------------------------------

@skip_no_model
def test_evaluation_metrics_are_available():
    metrics = json.loads((_best_run / "eval_metrics.json").read_text())
    track = metrics["track_level"]

    assert 0.40 < track["accuracy"] < 1.0, f"Accuracy {track['accuracy']} out of expected range"
    assert 0.40 < track["macro_f1"] < 1.0

    from src.config import FMA_GENRES

    for genre in FMA_GENRES:
        assert genre in track["per_genre"], f"Missing metrics for {genre}"
        g = track["per_genre"][genre]
        assert 0 < g["f1"] < 1.0
        assert g["support"] > 0


# ---------------------------------------------------------------------------
# 12. Baseline comparison table can be generated
# ---------------------------------------------------------------------------

@skip_no_model
def test_baseline_comparison_table_can_be_generated():
    from src.evaluation.compare_baselines import build_comparison_table, load_our_results

    our = load_our_results(["cnn_tuned_lr"])
    assert len(our) == 1
    table = build_comparison_table(our)
    assert "cnn_tuned_lr" in table
    assert "Ours" in table
