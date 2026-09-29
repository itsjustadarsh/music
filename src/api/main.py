"""FastAPI service exposing classification, mood tagging, and playlist
generation/creation.

Run with:
    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

import io
import json
import logging
import tempfile
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    ClassifyResponse,
    MoodRequest,
    MoodResponse,
    PlaylistRequest,
    PlaylistResponse,
    PushPlaylistRequest,
    PushPlaylistResponse,
    TrackOut,
)
from src.config import CLIP_LENGTH_SAMPLES, OUTPUT_DIR, SAMPLE_RATE, SPOTIFY_TRACKS_CSV
from src.curator.playlist_agent import CuratedTrack, PlaylistCurator
from src.models.cnn import GenreCNN
from src.models.common import IDX_TO_GENRE, NUM_CLASSES, get_device
from src.mood.mood_tagger import tag_mood

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Music Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Lazy singletons -- trained model / dataset may not exist yet at API startup
# (e.g. right after cloning the repo, before running the training pipeline).
# --------------------------------------------------------------------------

def _find_trained_run() -> Path | None:
    """Picks the best outputs/<run>/ directory that has both config.json and best.pt.

    Only considers models that the API can serve (cnn, ast). MERT probes require
    the full MERT backbone for live inference, which is too heavy for the API.
    Prefers runs whose eval_metrics.json reports the highest macro_f1."""
    if not OUTPUT_DIR.exists():
        return None
    candidates = []
    for d in OUTPUT_DIR.iterdir():
        if not d.is_dir() or not (d / "config.json").exists() or not (d / "best.pt").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        if cfg.get("model") in ("cnn", "ast"):
            candidates.append(d)
    if not candidates:
        return None

    def score(d: Path) -> float:
        eval_path = d / "eval_metrics.json"
        if eval_path.exists():
            data = json.loads(eval_path.read_text())
            track = data.get("track_level", {})
            return track.get("macro_f1", data.get("macro_f1", 0.0))
        return -1.0

    return max(candidates, key=score)


@lru_cache(maxsize=1)
def get_classifier():
    run_dir = _find_trained_run()
    if run_dir is None:
        return None
    cfg = json.loads((run_dir / "config.json").read_text())
    device = get_device()

    if cfg["model"] == "cnn":
        model = GenreCNN(num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
        model.to(device).eval()
        return {"model": model, "type": "cnn", "device": device, "run_name": run_dir.name}

    from src.models.modern_model import ASTGenreClassifier, load_feature_extractor

    model = ASTGenreClassifier(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    model.to(device).eval()
    fe = load_feature_extractor()
    return {"model": model, "type": "ast", "device": device, "run_name": run_dir.name, "feature_extractor": fe}


@lru_cache(maxsize=1)
def get_curator() -> PlaylistCurator | None:
    if not SPOTIFY_TRACKS_CSV.exists():
        return None
    df = pd.read_csv(SPOTIFY_TRACKS_CSV)
    return PlaylistCurator(df)


# --------------------------------------------------------------------------
# Inference helpers
# --------------------------------------------------------------------------

def _logmel_from_waveform(y: np.ndarray) -> np.ndarray:
    from src.data.preprocess import audio_to_logmel

    return audio_to_logmel(y)


def _classify_waveform(y_22k: np.ndarray, y_16k: np.ndarray) -> ClassifyResponse:
    classifier = get_classifier()
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model checkpoint found under outputs/. Train a model first "
            "(see FINALSTEPS.md: 'Running it' step 3) before calling /classify.",
        )

    device = classifier["device"]
    if classifier["type"] == "cnn":
        logmel = _logmel_from_waveform(y_22k)
        x = torch.from_numpy(logmel).float().unsqueeze(0).unsqueeze(0).to(device)
    else:
        fe = classifier["feature_extractor"]
        out = fe([y_16k], sampling_rate=16000, return_tensors="pt")
        x = out["input_values"].to(device)

    with torch.no_grad():
        logits = classifier["model"](x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_idx = int(probs.argmax())
    return ClassifyResponse(
        genre=IDX_TO_GENRE[top_idx],
        confidence=float(probs[top_idx]),
        all_scores={IDX_TO_GENRE[i]: float(probs[i]) for i in range(NUM_CLASSES)},
        model_used=classifier["run_name"],
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": get_classifier() is not None,
        "curator_loaded": get_curator() is not None,
    }


@app.get("/models")
def list_models():
    """List all trained model runs with their evaluation metrics."""
    if not OUTPUT_DIR.exists():
        return {"models": []}
    models = []
    for d in sorted(OUTPUT_DIR.iterdir()):
        if not d.is_dir() or not (d / "config.json").exists() or not (d / "best.pt").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        info = {"run_name": d.name, "model_type": cfg.get("model", "unknown")}
        eval_path = d / "eval_metrics.json"
        if eval_path.exists():
            em = json.loads(eval_path.read_text())
            track = em.get("track_level", {})
            info["accuracy"] = track.get("accuracy")
            info["macro_f1"] = track.get("macro_f1")
            info["per_genre"] = track.get("per_genre", {})
            info["n_tracks"] = em.get("n_tracks")
            info["n_clips"] = em.get("n_clips")
        models.append(info)
    return {"models": models}


@app.post("/classify-all")
async def classify_all(file: UploadFile = File(...)):
    """Classify with ALL trained models and return each model's predictions."""
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "clip.wav").suffix) as tmp:
        tmp.write(contents)
        tmp.flush()
        try:
            y_22k, _ = librosa.load(tmp.name, sr=SAMPLE_RATE, mono=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not decode audio file: {exc}")

    if len(y_22k) < CLIP_LENGTH_SAMPLES:
        y_22k = np.pad(y_22k, (0, CLIP_LENGTH_SAMPLES - len(y_22k)))
    else:
        y_22k = y_22k[:CLIP_LENGTH_SAMPLES]

    results = []
    device = get_device()
    for d in sorted(OUTPUT_DIR.iterdir()):
        if not d.is_dir() or not (d / "config.json").exists() or not (d / "best.pt").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        if cfg.get("model") != "cnn":
            continue
        try:
            model = GenreCNN(num_classes=NUM_CLASSES)
            model.load_state_dict(torch.load(d / "best.pt", map_location=device))
            model.to(device).eval()
            logmel = _logmel_from_waveform(y_22k)
            x = torch.from_numpy(logmel).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            top_idx = int(probs.argmax())

            # Load eval metrics if available
            eval_info = {}
            eval_path = d / "eval_metrics.json"
            if eval_path.exists():
                em = json.loads(eval_path.read_text())
                track = em.get("track_level", {})
                eval_info = {"accuracy": track.get("accuracy"), "macro_f1": track.get("macro_f1")}

            results.append({
                "run_name": d.name,
                "genre": IDX_TO_GENRE[top_idx],
                "confidence": float(probs[top_idx]),
                "all_scores": {IDX_TO_GENRE[i]: float(probs[i]) for i in range(NUM_CLASSES)},
                **eval_info,
            })
        except Exception as exc:
            log.warning("Failed to classify with %s: %s", d.name, exc)

    if not results:
        raise HTTPException(status_code=503, detail="No trained CNN models found.")
    return {"results": results}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "clip.wav").suffix) as tmp:
        tmp.write(contents)
        tmp.flush()
        try:
            y_22k, _ = librosa.load(tmp.name, sr=SAMPLE_RATE, mono=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not decode audio file: {exc}")

    if len(y_22k) < CLIP_LENGTH_SAMPLES:
        y_22k = np.pad(y_22k, (0, CLIP_LENGTH_SAMPLES - len(y_22k)))
    else:
        y_22k = y_22k[:CLIP_LENGTH_SAMPLES]
    y_16k = librosa.resample(y_22k, orig_sr=SAMPLE_RATE, target_sr=16000).astype(np.float32)

    return _classify_waveform(y_22k, y_16k)


@app.post("/mood", response_model=MoodResponse)
def mood(req: MoodRequest):
    result = tag_mood(req.valence, req.energy)
    return MoodResponse(mood=result.mood.value, valence=result.valence, energy=result.arousal, radius=result.radius)


@app.post("/generate-playlist", response_model=PlaylistResponse)
def generate_playlist(req: PlaylistRequest):
    curator = get_curator()
    if curator is None:
        raise HTTPException(
            status_code=503,
            detail="Spotify Tracks Dataset not found. Run `python -m src.data.fetch_spotify` first.",
        )
    resolved_genre, resolved_mood = curator.resolve_request(req.genre, req.mood, req.prompt)
    tracks = curator.generate_playlist(genre=req.genre, mood=req.mood, prompt=req.prompt, n=req.n)
    return PlaylistResponse(
        resolved_genre=resolved_genre,
        resolved_mood=resolved_mood.value,
        tracks=[TrackOut(**t.__dict__) for t in tracks],
    )


@app.post("/push-playlist", response_model=PushPlaylistResponse)
def push_playlist(req: PushPlaylistRequest):
    curator = get_curator()
    if curator is None:
        raise HTTPException(
            status_code=503,
            detail="Spotify Tracks Dataset not found. Run `python -m src.data.fetch_spotify` first.",
        )
    tracks = curator.generate_playlist(genre=req.genre, mood=req.mood, prompt=req.prompt, n=req.n)
    if not tracks:
        raise HTTPException(status_code=404, detail="No tracks matched that genre/mood combination.")

    from src.spotify.client import SpotifyCredentialsError, create_playlist_from_tracks, get_spotify_client

    try:
        sp = get_spotify_client()
    except SpotifyCredentialsError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    result = create_playlist_from_tracks(sp, req.playlist_name, tracks, public=req.public)
    return PushPlaylistResponse(
        playlist_id=result.playlist_id,
        playlist_url=result.playlist_url,
        added_count=len(result.added_track_ids),
        skipped_tracks=result.skipped_tracks,
    )


# --------------------------------------------------------------------------
# Static files -- serve the frontend and demo audio files so everything
# works from a single ``uvicorn`` process.
# --------------------------------------------------------------------------

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

_frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
_demo_dir = Path(__file__).resolve().parent.parent.parent / "demo-files"

# Genre label mapping for demo files
_GENRE_LABELS = {
    "electronic": "Electronic", "experimental": "Experimental", "folk": "Folk",
    "hip_hop": "Hip-Hop", "instrumental": "Instrumental", "international": "International",
    "pop": "Pop", "rock": "Rock", "hiphop": "Hip-Hop",
}


@app.get("/demo-files-list")
def list_demo_files():
    """List available demo audio files, preferring real FMA tracks over synthetic."""
    if not _demo_dir.exists():
        return {"files": []}

    files = sorted(f.name for f in _demo_dir.iterdir() if f.suffix == ".wav")

    # Group by genre, prefer FMA over synthetic
    genre_files: dict[str, dict] = {}
    for fname in files:
        name = fname.replace(".wav", "")
        is_fma = "_fma_" in name
        # Extract genre from filename
        genre_key = name.split("_fma_")[0] if is_fma else name.replace("_synth_demo", "").replace("_beat_demo", "").replace("_powerchord_demo", "").replace("_arpeggio_demo", "").replace("_melody_demo", "").replace("_glitch_demo", "").replace("_piano_demo", "").replace("_drone_demo", "")

        label = _GENRE_LABELS.get(genre_key, genre_key.replace("_", " ").title())
        entry = {
            "file": fname,
            "label": f"{label} (Real)" if is_fma else label,
            "type": "fma" if is_fma else "synthetic",
        }

        # Prefer FMA over synthetic for the same genre
        if genre_key not in genre_files or is_fma:
            genre_files[genre_key] = entry

    return {"files": list(genre_files.values())}

if _demo_dir.exists():
    app.mount("/demo-files", StaticFiles(directory=str(_demo_dir)), name="demo-files")

if _frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


@app.get("/")
def serve_frontend():
    index = _frontend_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Frontend not found. Open frontend/index.html directly."}
