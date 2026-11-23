"""In-memory FMA-small loader — nothing is written to disk.

Streams fma_small.zip and fma_metadata.zip over HTTP straight into RAM,
verifies SHA1, decodes every MP3 (soundfile/libsndfile) in parallel and
resamples to each requested sample rate. Decoded audio is kept in shared
anonymous memory as one (n_tracks, 30s * sr) float32 array per sample rate.

Usage (from a driver process, before building datasets):
    from data import memory_cache
    memory_cache.load(sample_rates=[24000, 16000])
"""
from __future__ import annotations

import hashlib
import io
import logging
import mmap
import multiprocessing as mp
import os
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from data.download import (
    FMA_METADATA_SHA1, FMA_METADATA_URL, FMA_SMALL_SHA1, FMA_SMALL_URL,
)

log = logging.getLogger(__name__)

CLIP_SEC = 30.0

# Populated by load()
METADATA: pd.DataFrame | None = None          # track_id, genre, split (decodable tracks only)
AUDIO: dict[int, np.ndarray] = {}             # sr -> (n_tracks, CLIP_SEC*sr) float32
ROW_OF: dict[int, int] = {}                   # track_id -> row in AUDIO arrays

# Worker globals (inherited via fork)
_ZIP_BYTES: bytes | None = None
_SHARED: dict[int, np.ndarray] = {}


def is_loaded() -> bool:
    return METADATA is not None


def _fetch(url: str, sha1: str) -> bytes:
    log.info("Streaming %s into memory...", url)
    t0 = time.time()
    h = hashlib.sha1()
    buf = io.BytesIO()
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=1 << 22):
            buf.write(chunk)
            h.update(chunk)
    data = buf.getvalue()
    if h.hexdigest() != sha1:
        raise RuntimeError(f"SHA1 mismatch for {url}: {h.hexdigest()} != {sha1}")
    log.info("  %.2f GB in %.0fs, SHA1 OK", len(data) / 1e9, time.time() - t0)
    return data


def _read_metadata(meta_zip: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(meta_zip)) as zf:
        with zf.open("fma_metadata/tracks.csv") as f:
            tracks = pd.read_csv(f, index_col=0, header=[0, 1])
    from data.dataset import FMA_GENRES
    df = pd.DataFrame({
        "track_id": tracks.index,
        "genre": tracks[("track", "genre_top")].values,
        "split": tracks[("set", "split")].values,
        "subset": tracks[("set", "subset")].values,
    })
    df = df[(df["subset"] == "small") & df["genre"].isin(FMA_GENRES)].copy()
    df["split"] = df["split"].map({"training": "train", "validation": "val", "test": "test"})
    df = df.dropna(subset=["split"])
    return df.reset_index(drop=True)


def _decode_one(args) -> tuple[int, bool]:
    import soundfile as sf
    import soxr

    row, track_id = args
    tid = f"{track_id:06d}"
    try:
        with zipfile.ZipFile(io.BytesIO(_ZIP_BYTES)) as zf:
            raw = zf.read(f"fma_small/{tid[:3]}/{tid}.mp3")
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if wav.size < sr:  # < 1s of audio: treat as corrupt
            return row, False
        for target_sr, arr in _SHARED.items():
            y = soxr.resample(wav, sr, target_sr) if sr != target_sr else wav
            n = min(y.size, arr.shape[1])
            arr[row, :n] = y[:n]
            arr[row, n:] = 0.0
        return row, True
    except Exception:
        return row, False


def _shared_array(n_rows: int, n_cols: int) -> np.ndarray:
    buf = mmap.mmap(-1, n_rows * n_cols * 4)  # anonymous, MAP_SHARED: visible across fork
    return np.frombuffer(buf, dtype=np.float32).reshape(n_rows, n_cols)


def load(sample_rates: list[int], n_workers: int | None = None) -> None:
    """Download + decode FMA-small into RAM for the given sample rates."""
    global METADATA, AUDIO, ROW_OF, _ZIP_BYTES, _SHARED
    if is_loaded() and all(sr in AUDIO for sr in sample_rates):
        return

    meta = _read_metadata(_fetch(FMA_METADATA_URL, FMA_METADATA_SHA1))
    _ZIP_BYTES = _fetch(FMA_SMALL_URL, FMA_SMALL_SHA1)

    n = len(meta)
    _SHARED = {sr: _shared_array(n, int(CLIP_SEC * sr)) for sr in sorted(set(sample_rates))}
    n_workers = n_workers or min(64, os.cpu_count() or 8)
    log.info("Decoding %d tracks at %s Hz with %d workers...", n, list(_SHARED), n_workers)
    t0 = time.time()
    ok = np.zeros(n, dtype=bool)
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        jobs = list(enumerate(meta["track_id"].astype(int)))
        for i, (row, good) in enumerate(pool.imap_unordered(_decode_one, jobs, chunksize=16)):
            ok[row] = good
            if (i + 1) % 1000 == 0:
                log.info("  decoded %d/%d", i + 1, n)
    _ZIP_BYTES = None  # free the 7.7 GB zip

    bad = meta.loc[~ok, "track_id"].tolist()
    log.info("Decoded %d/%d tracks in %.0fs; dropped %d undecodable: %s",
             ok.sum(), n, time.time() - t0, len(bad), bad)

    keep = np.flatnonzero(ok)
    AUDIO = {sr: np.ascontiguousarray(arr[keep]) for sr, arr in _SHARED.items()}
    _SHARED = {}
    METADATA = meta.iloc[keep].reset_index(drop=True)
    ROW_OF = {int(t): i for i, t in enumerate(METADATA["track_id"])}
    log.info("Split sizes: %s", METADATA["split"].value_counts().to_dict())
