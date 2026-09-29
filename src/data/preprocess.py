"""Preprocess FMA-small audio into cached tensors for both models.

Each 30s track is split into non-overlapping 3s sub-clips (standard practice for
genre classification on FMA -- see docs/METHODOLOGY.md). For each sub-clip we
cache two representations, since the two models consume audio differently:

- ``logmel/``: a log-mel spectrogram at SAMPLE_RATE (22050Hz), for GenreCNN.
- ``waveform16k/``: the raw waveform resampled to 16kHz, for the AST model
  (HuggingFace's ASTFeatureExtractor computes its own fbank features from
  16kHz audio internally, so we cache the resampled waveform rather than
  pre-computing AST-specific features).

A manifest CSV records the mapping from cache files -> (track_id, genre, split,
subclip_idx) so training can build a Dataset without re-decoding audio every epoch.

Usage:
    python -m src.data.preprocess [--workers 4] [--limit N]
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    CLIP_LENGTH_SAMPLES,
    FMA_SMALL_AUDIO_DIR,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    PROCESSED_DIR,
    SAMPLE_RATE,
)
from src.data.splits import load_fma_small_tracks, track_id_to_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"
AST_SAMPLE_RATE = 16000


def audio_to_logmel(clip: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=clip, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def process_track(args: tuple) -> list[dict]:
    """Load one track, cut into 3s sub-clips, cache both representations. Returns manifest rows."""
    track_id, genre, split, audio_dir, out_dir = args
    rows: list[dict] = []
    path = track_id_to_path(track_id, audio_dir)
    if not path.exists():
        return rows
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:  # FMA-small has a small number of known-corrupt mp3s
        log.warning("Skipping unreadable track %s (%s): %s", track_id, path, exc)
        return rows

    n_clips = len(y) // CLIP_LENGTH_SAMPLES
    if n_clips == 0:
        log.warning("Track %s shorter than one sub-clip window, skipping", track_id)
        return rows

    logmel_dir = out_dir / "logmel" / split / genre
    wav_dir = out_dir / "waveform16k" / split / genre
    logmel_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_clips):
        start = i * CLIP_LENGTH_SAMPLES
        clip = y[start : start + CLIP_LENGTH_SAMPLES]

        logmel = audio_to_logmel(clip)
        logmel_path = logmel_dir / f"{track_id:06d}_{i}.npy"
        np.save(logmel_path, logmel)

        clip_16k = librosa.resample(clip, orig_sr=SAMPLE_RATE, target_sr=AST_SAMPLE_RATE).astype(np.float32)
        wav_path = wav_dir / f"{track_id:06d}_{i}.npy"
        np.save(wav_path, clip_16k)

        rows.append(
            {
                "logmel_path": str(logmel_path.relative_to(out_dir)),
                "waveform_path": str(wav_path.relative_to(out_dir)),
                "track_id": track_id,
                "genre": genre,
                "split": split,
                "subclip_idx": i,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N tracks (debugging)")
    parser.add_argument("--audio-dir", type=Path, default=FMA_SMALL_AUDIO_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    tracks = load_fma_small_tracks()
    if args.limit:
        tracks = tracks.iloc[: args.limit]

    jobs = [
        (track_id, row["genre_top"], row["split"], args.audio_dir, args.out_dir)
        for track_id, row in tracks.iterrows()
    ]

    all_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_track, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            all_rows.extend(fut.result())

    manifest = pd.DataFrame(all_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # The manifest lives alongside the cache it indexes -- its paths are relative to
    # out_dir, so writing it to a fixed location would leave a manifest pointing at
    # files that aren't there whenever --out-dir is overridden.
    manifest_path = args.out_dir / MANIFEST_PATH.name
    manifest.to_csv(manifest_path, index=False)
    log.info("Wrote manifest with %d sub-clips to %s", len(manifest), manifest_path)
    log.info("Per-split sub-clip counts:\n%s", manifest["split"].value_counts())


if __name__ == "__main__":
    main()
