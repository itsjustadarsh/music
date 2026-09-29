#!/usr/bin/env python3
"""Interactive demo walkthrough of the Music Agent system.

Run:
    python scripts/demo.py

Each section prints a clear header and result. Sections that depend on
data/models skip gracefully so the script always finishes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def divider(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def section_skip(reason: str) -> None:
    print(f"  [SKIP] {reason}\n")


# -------------------------------------------------------------------------
# 1. Mood Tagging — Russell's Circumplex Model
# -------------------------------------------------------------------------

def demo_mood_tagging():
    divider("1. Mood Tagging  (Russell's Circumplex Model)")
    from src.mood.mood_tagger import tag_mood

    cases = [
        (0.90, 0.70, "high valence, high energy"),
        (0.70, 0.95, "mid-high valence, very high energy"),
        (0.20, 0.85, "low valence, high energy"),
        (0.05, 0.80, "very low valence, high energy"),
        (0.10, 0.20, "low valence, low energy"),
        (0.25, 0.05, "low valence, very low energy"),
        (0.80, 0.15, "high valence, low energy"),
        (0.90, 0.35, "high valence, mid-low energy"),
        (0.50, 0.50, "center point"),
    ]

    print(f"  {'Valence':>8}  {'Energy':>8}  {'Mood':<10}  {'Radius':>6}  Description")
    print(f"  {'-------':>8}  {'------':>8}  {'----':<10}  {'------':>6}  -----------")
    for v, e, desc in cases:
        r = tag_mood(v, e)
        print(f"  {v:8.2f}  {e:8.2f}  {r.mood.value:<10}  {r.radius:6.3f}  {desc}")
    print()


# -------------------------------------------------------------------------
# 2. Playlist Curation — generate from natural-language prompt
# -------------------------------------------------------------------------

def demo_playlist_curation():
    divider("2. Playlist Curation  (Spotify Tracks Dataset)")
    from src.config import SPOTIFY_TRACKS_CSV

    if not SPOTIFY_TRACKS_CSV.exists():
        section_skip("Spotify CSV not found")
        return False

    import pandas as pd
    from src.curator.playlist_agent import PlaylistCurator

    df = pd.read_csv(SPOTIFY_TRACKS_CSV)
    curator = PlaylistCurator(df)
    print(f"  Loaded {len(df):,} tracks across {df['track_genre'].nunique()} genres\n")

    prompt = "upbeat happy pop"
    print(f'  Prompt: "{prompt}"')
    genre, mood = curator.resolve_request(genre=None, mood=None, prompt=prompt)
    print(f"  Resolved -> genre: {genre}, mood: {mood.value}\n")

    tracks = curator.generate_playlist(prompt=prompt, n=10)
    print(f"  {'#':>3}  {'Track':<35}  {'Artist':<20}  {'Tempo':>5}  {'Key':>3}  {'Score':>5}")
    print(f"  {'---':>3}  {'-----':<35}  {'------':<20}  {'-----':>5}  {'---':>3}  {'-----':>5}")
    for i, t in enumerate(tracks, 1):
        name = t.name[:33] + ".." if len(t.name) > 35 else t.name
        artist = t.artist[:18] + ".." if len(t.artist) > 20 else t.artist
        print(f"  {i:3d}  {name:<35}  {artist:<20}  {t.tempo:5.0f}  {t.key:3d}  {t.score:5.2f}")
    print()
    return True


# -------------------------------------------------------------------------
# 3. Mood Coherence Proof — happy vs sad playlist comparison
# -------------------------------------------------------------------------

def demo_mood_coherence():
    divider("3. Mood Coherence  (happy vs sad pop)")
    from src.config import SPOTIFY_TRACKS_CSV

    if not SPOTIFY_TRACKS_CSV.exists():
        section_skip("Spotify CSV not found")
        return

    import pandas as pd
    from src.curator.playlist_agent import PlaylistCurator

    curator = PlaylistCurator(pd.read_csv(SPOTIFY_TRACKS_CSV))
    happy = curator.generate_playlist(genre="pop", mood="happy", n=20)
    sad = curator.generate_playlist(genre="pop", mood="sad", n=20)

    avg_v_happy = sum(t.valence for t in happy) / len(happy)
    avg_e_happy = sum(t.energy for t in happy) / len(happy)
    avg_v_sad = sum(t.valence for t in sad) / len(sad)
    avg_e_sad = sum(t.energy for t in sad) / len(sad)

    print(f"  Happy pop playlist (20 tracks):")
    print(f"    avg valence = {avg_v_happy:.3f}   avg energy = {avg_e_happy:.3f}")
    print(f"  Sad pop playlist (20 tracks):")
    print(f"    avg valence = {avg_v_sad:.3f}   avg energy = {avg_e_sad:.3f}")
    print(f"\n  Valence delta: {avg_v_happy - avg_v_sad:+.3f}  (positive = happy is brighter)")
    print()


# -------------------------------------------------------------------------
# 4. Genre Classification — CNN inference on synthetic audio
# -------------------------------------------------------------------------

def demo_genre_classification():
    divider("4. Genre Classification  (GenreCNN on synthetic audio)")
    from src.config import OUTPUT_DIR

    best_run = OUTPUT_DIR / "cnn_tuned_lr"
    if not (best_run / "best.pt").exists():
        section_skip("No trained model found at outputs/cnn_tuned_lr/")
        return

    import numpy as np
    import torch

    from src.config import CLIP_LENGTH_SAMPLES, SAMPLE_RATE
    from src.data.preprocess import audio_to_logmel
    from src.models.cnn import GenreCNN
    from src.models.common import IDX_TO_GENRE, NUM_CLASSES, get_device

    t = np.linspace(0, 3.0, CLIP_LENGTH_SAMPLES, dtype=np.float32)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)

    logmel = audio_to_logmel(sine)
    device = get_device()
    model = GenreCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(best_run / "best.pt", map_location=device, weights_only=True))
    model.to(device).eval()

    x = torch.from_numpy(logmel).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().numpy()

    print(f"  Input: 440 Hz sine wave (3 seconds, synthetic)")
    print(f"  Model: GenreCNN from cnn_tuned_lr checkpoint\n")
    print(f"  {'Genre':<15}  {'Confidence':>10}")
    print(f"  {'-----':<15}  {'----------':>10}")
    for i in sorted(range(NUM_CLASSES), key=lambda i: -probs[i]):
        bar = "#" * int(probs[i] * 40)
        marker = " <--" if i == probs.argmax() else ""
        print(f"  {IDX_TO_GENRE[i]:<15}  {probs[i]:10.1%}  {bar}{marker}")
    print()


# -------------------------------------------------------------------------
# 5. Evaluation Results — accuracy + per-genre F1
# -------------------------------------------------------------------------

def demo_evaluation_results():
    divider("5. Evaluation Results  (cnn_tuned_lr on FMA-small test set)")
    from src.config import OUTPUT_DIR

    eval_path = OUTPUT_DIR / "cnn_tuned_lr" / "eval_metrics.json"
    if not eval_path.exists():
        section_skip("eval_metrics.json not found")
        return

    metrics = json.loads(eval_path.read_text())
    track = metrics["track_level"]

    print(f"  Track-level accuracy:  {track['accuracy']:.1%}")
    print(f"  Track-level macro-F1:  {track['macro_f1']:.1%}")
    print(f"  Test tracks: {metrics['n_tracks']}   Test clips: {metrics['n_clips']}\n")

    print(f"  {'Genre':<15}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}  {'Support':>7}")
    print(f"  {'-----':<15}  {'----':>6}  {'------':>6}  {'--':>6}  {'-------':>7}")
    for genre, g in track["per_genre"].items():
        print(f"  {genre:<15}  {g['precision']:6.1%}  {g['recall']:6.1%}  {g['f1']:6.1%}  {g['support']:7d}")
    print()


# -------------------------------------------------------------------------
# 6. API Health Check
# -------------------------------------------------------------------------

def demo_api_health():
    divider("6. API Health Check  (localhost:8000)")
    try:
        import httpx

        resp = httpx.get("http://127.0.0.1:8000/health", timeout=3)
        body = resp.json()
        print(f"  Status:         {body['status']}")
        print(f"  Model loaded:   {body['model_loaded']}")
        print(f"  Curator loaded: {body['curator_loaded']}")
    except Exception:
        print("  [SKIP] API server not running (start with: uvicorn src.api.main:app --port 8000)")
    print()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  MUSIC AGENT — System Walkthrough Demo")
    print("=" * 60)

    demo_mood_tagging()
    csv_ok = demo_playlist_curation()
    if csv_ok:
        demo_mood_coherence()
    demo_genre_classification()
    demo_evaluation_results()
    demo_api_health()

    divider("Done")
    print("  All sections complete. To start the full web demo:\n")
    print("    uvicorn src.api.main:app --port 8000")
    print("    open frontend/index.html\n")


if __name__ == "__main__":
    main()
