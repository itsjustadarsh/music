"""Cache frozen MERT-v1-330M embeddings for every episode in the manifest.

Same structure as extract_embeddings.py but for the 330M model.
Writes to data/embeddings/mert_large/ with (n_episodes, 25, 1024) float16 per track.

Usage:
    python -m src.data.extract_embeddings_large [--workers 4] [--limit N]
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import CLIP_DURATION_SEC, DATA_DIR, FMA_SMALL_AUDIO_DIR, PROCESSED_DIR
from src.data.splits import track_id_to_path
from src.models.common import get_device
from src.models.mert_embed_large import (
    MERT_LARGE_HIDDEN_SIZE,
    MERT_LARGE_NUM_LAYERS,
    MERT_LARGE_SAMPLE_RATE,
    embed_episodes_large,
    load_mert_large,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SOURCE_MANIFEST = PROCESSED_DIR / "manifest.csv"
EMBEDDINGS_DIR = DATA_DIR / "embeddings" / "mert_large"
EMBEDDING_MANIFEST = EMBEDDINGS_DIR / "manifest.csv"
EPISODE_SAMPLES = int(MERT_LARGE_SAMPLE_RATE * CLIP_DURATION_SEC)
DECODE_CHUNK = 16  # smaller chunks for larger model


def decode_track(args: tuple) -> tuple[int, np.ndarray | None]:
    track_id, subclip_indices, audio_dir = args
    path = track_id_to_path(track_id, audio_dir)
    try:
        y, _ = librosa.load(path, sr=MERT_LARGE_SAMPLE_RATE, mono=True)
    except Exception as exc:
        log.warning("Skipping unreadable track %s: %s", track_id, exc)
        return track_id, None

    episodes = []
    for idx in subclip_indices:
        start = idx * EPISODE_SAMPLES
        clip = y[start : start + EPISODE_SAMPLES]
        if len(clip) < EPISODE_SAMPLES:
            clip = np.pad(clip, (0, EPISODE_SAMPLES - len(clip)))
        episodes.append(clip)
    return track_id, np.stack(episodes).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--audio-dir", type=Path, default=FMA_SMALL_AUDIO_DIR)
    parser.add_argument("--out-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8, help="Episodes per GPU batch (330M is ~3.5x larger)")
    args = parser.parse_args()

    manifest = pd.read_csv(SOURCE_MANIFEST)
    grouped = manifest.groupby("track_id")
    track_ids = sorted(grouped.groups)
    if args.limit:
        track_ids = track_ids[: args.limit]
    log.info("Embedding %d tracks with MERT-v1-330M", len(track_ids))

    device = get_device()
    model, extractor = load_mert_large(device)
    log.info("Using device: %s", device)

    meta = {tid: grouped.get_group(tid).iloc[0] for tid in track_ids}

    def output_path(track_id: int) -> Path:
        row = meta[track_id]
        return args.out_dir / row["split"] / row["genre"] / f"{track_id:06d}.npy"

    def already_done(track_id: int, n_episodes: int) -> bool:
        path = output_path(track_id)
        if not path.exists():
            return False
        try:
            cached = np.load(path, mmap_mode="r")
        except Exception:
            log.warning("Discarding unreadable cache for track %s", track_id)
            return False
        return cached.shape == (n_episodes, MERT_LARGE_NUM_LAYERS, MERT_LARGE_HIDDEN_SIZE)

    rows: list[dict] = []
    jobs = []
    resumed = 0
    for track_id in track_ids:
        group = grouped.get_group(track_id).sort_values("subclip_idx")
        subclip_indices = list(group["subclip_idx"])
        if not args.force and already_done(track_id, len(subclip_indices)):
            resumed += 1
            continue
        jobs.append((track_id, subclip_indices, args.audio_dir))

    if resumed:
        log.info("Resuming: %d tracks already embedded, %d to go", resumed, len(jobs))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        with tqdm(total=len(jobs), desc="Embedding (330M)") as pbar:
            for start in range(0, len(jobs), DECODE_CHUNK):
                chunk = jobs[start : start + DECODE_CHUNK]
                for track_id, episodes in pool.map(decode_track, chunk):
                    pbar.update(1)
                    if episodes is None:
                        continue
                    row0 = meta[track_id]
                    split, genre = row0["split"], row0["genre"]
                    out_dir = args.out_dir / split / genre
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{track_id:06d}.npy"

                    # Process in smaller batches for the larger model
                    all_embeddings = []
                    for i in range(0, len(episodes), args.batch_size):
                        batch = episodes[i : i + args.batch_size]
                        emb = embed_episodes_large(batch, model, extractor, device)
                        all_embeddings.append(emb)
                    embeddings = np.concatenate(all_embeddings, axis=0)
                    np.save(out_path, embeddings)

                    group = grouped.get_group(track_id).sort_values("subclip_idx")
                    for episode_idx, subclip_idx in enumerate(group["subclip_idx"]):
                        rows.append(
                            {
                                "embedding_path": str(out_path.relative_to(args.out_dir)),
                                "episode_idx": episode_idx,
                                "track_id": track_id,
                                "genre": genre,
                                "split": split,
                                "subclip_idx": int(subclip_idx),
                            }
                        )

    # Include previously-embedded tracks in the manifest
    embedded = {r["track_id"] for r in rows}
    for track_id in track_ids:
        if track_id in embedded:
            continue
        group = grouped.get_group(track_id).sort_values("subclip_idx")
        if not already_done(track_id, len(group)):
            continue
        row0 = meta[track_id]
        rel = output_path(track_id).relative_to(args.out_dir)
        for episode_idx, subclip_idx in enumerate(group["subclip_idx"]):
            rows.append(
                {
                    "embedding_path": str(rel),
                    "episode_idx": episode_idx,
                    "track_id": track_id,
                    "genre": row0["genre"],
                    "split": row0["split"],
                    "subclip_idx": int(subclip_idx),
                }
            )

    rows.sort(key=lambda r: (r["track_id"], r["subclip_idx"]))
    out_manifest = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / EMBEDDING_MANIFEST.name
    out_manifest.to_csv(manifest_path, index=False)
    log.info("Wrote manifest with %d episodes to %s", len(out_manifest), manifest_path)
    log.info("Per-split episode counts:\n%s", out_manifest["split"].value_counts())


if __name__ == "__main__":
    main()
