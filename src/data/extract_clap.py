"""Cache frozen CLAP audio embeddings for every episode in the preprocessing manifest.

CLAP (Contrastive Language-Audio Pretraining) uses a different pretraining paradigm
than MERT/BYOL-A/PANNs — it's trained with audio-text pairs, making its representations
potentially complementary. We use the `laion/larger_clap_music_and_speech` checkpoint
which was trained on music data.

One file per track (`(n_episodes, 512)` float16), mirroring extract_byola.py.

Usage:
    python -m src.data.extract_clap [--workers 4] [--limit N]
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import CLIP_DURATION_SEC, DATA_DIR, FMA_SMALL_AUDIO_DIR, PROCESSED_DIR
from src.data.splits import track_id_to_path
from src.models.common import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SOURCE_MANIFEST = PROCESSED_DIR / "manifest.csv"
EMBEDDINGS_DIR = DATA_DIR / "embeddings" / "clap"

CLAP_SAMPLE_RATE = 48000
CLAP_EMBED_DIM = 512
EPISODE_SAMPLES = int(CLAP_SAMPLE_RATE * CLIP_DURATION_SEC)
DECODE_CHUNK = 32
BATCH_SIZE = 16  # CLAP is heavier than BYOL-A


def load_clap(device: torch.device):
    from transformers import ClapModel, ClapProcessor

    model_id = "laion/larger_clap_music_and_speech"
    processor = ClapProcessor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id).to(device)
    model.eval()
    log.info("Loaded CLAP: %s (%.1fM params)",
             model_id, sum(p.numel() for p in model.parameters()) / 1e6)
    return model, processor


def decode_track(args: tuple) -> tuple[int, np.ndarray | None]:
    track_id, subclip_indices, audio_dir = args
    path = track_id_to_path(track_id, audio_dir)
    try:
        y, _ = librosa.load(path, sr=CLAP_SAMPLE_RATE, mono=True)
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


@torch.no_grad()
def embed_episodes(episodes: np.ndarray, model, processor, device) -> np.ndarray:
    """(n_episodes, samples) waveforms -> (n_episodes, 512) float16 embeddings."""
    out = []
    for start in range(0, len(episodes), BATCH_SIZE):
        batch = list(episodes[start : start + BATCH_SIZE])
        inputs = processor(audio=batch, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        features = model.get_audio_features(**inputs)
        out.append(features.pooler_output.cpu().numpy())
    return np.concatenate(out).astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="mp3 decoding processes")
    parser.add_argument("--limit", type=int, default=None, help="Only first N tracks (debug)")
    parser.add_argument("--audio-dir", type=Path, default=FMA_SMALL_AUDIO_DIR)
    parser.add_argument("--out-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--force", action="store_true", help="Re-embed cached tracks")
    args = parser.parse_args()

    manifest = pd.read_csv(SOURCE_MANIFEST)
    grouped = manifest.groupby("track_id")
    track_ids = sorted(grouped.groups)
    if args.limit:
        track_ids = track_ids[: args.limit]
    log.info("Embedding %d tracks with CLAP", len(track_ids))

    device = get_device()
    model, processor = load_clap(device)

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
            return False
        return cached.shape == (n_episodes, CLAP_EMBED_DIM)

    rows: list[dict] = []
    jobs = []
    resumed = 0
    for track_id in track_ids:
        group = grouped.get_group(track_id).sort_values("subclip_idx")
        job = (track_id, list(group["subclip_idx"]), args.audio_dir)
        if not args.force and already_done(track_id, len(group)):
            resumed += 1
        else:
            jobs.append(job)

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
                    out_path = args.out_dir / split / genre / f"{track_id:06d}.npy"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(out_path, embed_episodes(episodes, model, processor, device))

                    group = grouped.get_group(track_id).sort_values("subclip_idx")
                    for episode_idx, subclip_idx in enumerate(group["subclip_idx"]):
                        rows.append({
                            "embedding_path": str(out_path.relative_to(args.out_dir)),
                            "episode_idx": episode_idx,
                            "track_id": track_id,
                            "genre": genre,
                            "split": split,
                            "subclip_idx": int(subclip_idx),
                        })

    # Include already-cached tracks in manifest
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
            rows.append({
                "embedding_path": str(rel),
                "episode_idx": episode_idx,
                "track_id": track_id,
                "genre": row0["genre"],
                "split": row0["split"],
                "subclip_idx": int(subclip_idx),
            })

    rows.sort(key=lambda r: (r["track_id"], r["subclip_idx"]))
    out_manifest = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.csv"
    out_manifest.to_csv(manifest_path, index=False)
    log.info("Wrote manifest with %d episodes to %s", len(out_manifest), manifest_path)
    log.info("Per-split episode counts:\n%s", out_manifest["split"].value_counts())


if __name__ == "__main__":
    main()
