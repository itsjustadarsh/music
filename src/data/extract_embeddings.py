"""Cache frozen MERT embeddings for every episode in the preprocessing manifest.

By default episodes are taken from `data/processed/manifest.csv` rather than
recomputed, so the MERT episodes align exactly -- same track, same 3-second span --
with the log-mel episodes the spectrogram models consume. That alignment is what
makes per-episode comparison across backbones well-defined.

`--episode-seconds` re-cuts the grid to a longer window instead. This exists to test
the one assumption every result in this project inherits: every embedding we have is
computed from 3 seconds of audio, and genre is partly a long-timescale property
(form, instrumentation turnover, section changes), so 3s may be a ceiling rather than
a neutral choice -- see `docs/METHODOLOGY.md` §5. A longer grid necessarily breaks
log-mel alignment and yields fewer, longer episodes per track, so it writes to its
own directory and manifest and is a separate feature set, not a drop-in replacement.

One file per track (`(n_episodes, 13, 768)` float16) rather than one per episode,
which keeps this to ~8k files instead of ~75k.

Usage:
    python -m src.data.extract_embeddings [--workers 4] [--limit N]
    python -m src.data.extract_embeddings --episode-seconds 10
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
from src.models.mert_embed import (
    MERT_HIDDEN_SIZE,
    MERT_NUM_LAYERS,
    MERT_SAMPLE_RATE,
    embed_episodes,
    load_mert,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SOURCE_MANIFEST = PROCESSED_DIR / "manifest.csv"
EMBEDDINGS_DIR = DATA_DIR / "embeddings" / "mert"
EMBEDDING_MANIFEST = EMBEDDINGS_DIR / "manifest.csv"
DECODE_CHUNK = 32


def decode_track(args: tuple) -> tuple[int, np.ndarray | None]:
    """Decode one mp3 at MERT's 24kHz and cut it into the planned episodes."""
    track_id, subclip_indices, audio_dir, episode_samples = args
    path = track_id_to_path(track_id, audio_dir)
    try:
        y, _ = librosa.load(path, sr=MERT_SAMPLE_RATE, mono=True)
    except Exception as exc:  # same handful of corrupt FMA mp3s that preprocess.py skips
        log.warning("Skipping unreadable track %s: %s", track_id, exc)
        return track_id, None

    episodes = []
    for idx in subclip_indices:
        start = idx * episode_samples
        clip = y[start : start + episode_samples]
        if len(clip) < episode_samples:  # trailing partial episode
            clip = np.pad(clip, (0, episode_samples - len(clip)))
        episodes.append(clip)
    return track_id, np.stack(episodes).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="mp3 decoding processes")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N tracks (debugging)")
    parser.add_argument("--audio-dir", type=Path, default=FMA_SMALL_AUDIO_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--episode-seconds", type=float, default=CLIP_DURATION_SEC,
                        help="Episode length. Anything other than the preprocessing "
                             "default re-cuts the grid and writes to its own directory.")
    parser.add_argument("--force", action="store_true", help="Re-embed tracks that are already cached")
    args = parser.parse_args()

    episode_samples = int(MERT_SAMPLE_RATE * args.episode_seconds)
    aligned = args.episode_seconds == CLIP_DURATION_SEC
    if args.out_dir is None:
        suffix = "" if aligned else f"_{args.episode_seconds:g}s"
        args.out_dir = EMBEDDINGS_DIR.with_name(EMBEDDINGS_DIR.name + suffix)

    manifest = pd.read_csv(SOURCE_MANIFEST)
    grouped = manifest.groupby("track_id")
    track_ids = sorted(grouped.groups)
    if args.limit:
        track_ids = track_ids[: args.limit]

    def episode_plan(track_id: int) -> list[int]:
        """Episode indices to cut for one track.

        Aligned mode reuses the preprocessing manifest's own indices, so a MERT
        episode and a log-mel episode name the same span of audio. In re-cut mode the
        grid is derived from the audio the preprocessing pass already accounted for
        (n 3-second episodes = 3n seconds), which keeps the plan deterministic and
        avoids decoding every mp3 twice just to learn its length.
        """
        group = grouped.get_group(track_id)
        if aligned:
            return sorted(int(i) for i in group["subclip_idx"])
        covered_sec = len(group) * CLIP_DURATION_SEC
        return list(range(int(covered_sec // args.episode_seconds)))

    log.info("Embedding %d tracks with MERT at %.3gs per episode -> %s",
             len(track_ids), args.episode_seconds, args.out_dir)

    device = get_device()
    model, extractor = load_mert(device)
    log.info("Using device: %s", device)

    meta = {tid: grouped.get_group(tid).iloc[0] for tid in track_ids}

    def output_path(track_id: int) -> Path:
        row = meta[track_id]
        return args.out_dir / row["split"] / row["genre"] / f"{track_id:06d}.npy"

    def already_done(track_id: int, n_episodes: int) -> bool:
        """A cached file counts only if it loads and has the expected shape.

        np.save is not atomic, so a run interrupted mid-write can leave a truncated
        file. Validating the header here means resuming never silently inherits a
        corrupt embedding -- the worst case is re-embedding one track.
        """
        path = output_path(track_id)
        if not path.exists():
            return False
        try:
            cached = np.load(path, mmap_mode="r")
        except Exception:
            log.warning("Discarding unreadable cache for track %s", track_id)
            return False
        return cached.shape == (n_episodes, MERT_NUM_LAYERS, MERT_HIDDEN_SIZE)

    rows: list[dict] = []
    jobs = []
    resumed = 0
    plans = {tid: episode_plan(tid) for tid in track_ids}
    for track_id in track_ids:
        plan = plans[track_id]
        if not plan:  # too short to yield even one episode at this length
            continue
        if not args.force and already_done(track_id, len(plan)):
            resumed += 1
            continue
        jobs.append((track_id, plan, args.audio_dir, episode_samples))

    if resumed:
        log.info("Resuming: %d tracks already embedded, %d to go", resumed, len(jobs))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        with tqdm(total=len(jobs), desc="Embedding") as pbar:
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

                    embeddings = embed_episodes(episodes, model, extractor, device)
                    np.save(out_path, embeddings)

                    for episode_idx, subclip_idx in enumerate(plans[track_id]):
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

    # Rows were only appended for tracks embedded in this run; the manifest must
    # describe every cached track, including ones carried over from an earlier run.
    embedded = {r["track_id"] for r in rows}
    for track_id in track_ids:
        if track_id in embedded:
            continue
        if not already_done(track_id, len(plans[track_id])):
            continue  # genuinely missing (e.g. the corrupt FMA files)
        row0 = meta[track_id]
        rel = output_path(track_id).relative_to(args.out_dir)
        for episode_idx, subclip_idx in enumerate(plans[track_id]):
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
    if args.limit and manifest_path.exists():
        # The manifest is rebuilt from `track_ids`, which --limit truncates. Writing it
        # here would silently shrink a complete manifest to the debugging subset, making
        # every cached embedding outside that subset unreachable to every dataset.
        manifest_path = manifest_path.with_suffix(f".limit{args.limit}.csv")
        log.warning("--limit is set; writing %s rather than overwriting the full manifest",
                    manifest_path.name)
    out_manifest.to_csv(manifest_path, index=False)
    log.info("Wrote manifest with %d episodes to %s", len(out_manifest), manifest_path)
    log.info("Per-split episode counts:\n%s", out_manifest["split"].value_counts())


if __name__ == "__main__":
    main()
