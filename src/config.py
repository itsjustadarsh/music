"""Central paths and constants shared across the data, training, and serving code."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("MUSIC_AGENT_DATA_DIR", PROJECT_ROOT / "data")).resolve()
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = Path(os.getenv("MUSIC_AGENT_OUTPUT_DIR", PROJECT_ROOT / "outputs")).resolve()
DOCS_DIR = PROJECT_ROOT / "docs"

FMA_SMALL_AUDIO_DIR = RAW_DIR / "fma_small"
FMA_METADATA_DIR = RAW_DIR / "fma_metadata"
SPOTIFY_TRACKS_CSV = RAW_DIR / "spotify_tracks" / "dataset.csv"

for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, CACHE_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- FMA-small dataset ---------------------------------------------------
FMA_SMALL_ZIP_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_METADATA_ZIP_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
# Published in the "Usage" section of https://github.com/mdeff/fma/blob/master/README.md
# (verified 2026-09-11). These are SHA1, not MD5, despite the filename convention elsewhere.
FMA_SMALL_ZIP_SHA1 = "ade154f733639d52e35e32f5593efe5be76c6d70"
FMA_METADATA_ZIP_SHA1 = "f0df49ffe5f2a6008d7dc83c6915b31835dfe733"
FMA_SMALL_EXPECTED_TRACKS = 8000

FMA_GENRES = [
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Pop",
    "Rock",
]

# --- Audio / spectrogram settings ----------------------------------------
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
CLIP_DURATION_SEC = 3.0
CLIP_LENGTH_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_SEC)

# --- Kaggle dataset --------------------------------------------------------
SPOTIFY_KAGGLE_HANDLE = "maharshipandya/-spotify-tracks-dataset"
SPOTIFY_TRACKS_EXPECTED_ROWS = 114000
SPOTIFY_TRACKS_ROW_TOLERANCE = 2000  # allow for minor dataset revisions
