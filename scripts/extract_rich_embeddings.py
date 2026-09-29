"""Re-extract MERT-95M embeddings with temporal statistics, not just mean pooling.

Instead of (13, 768) per episode with only mean-pooled time steps, this
extracts (13, 768*4) = (13, 3072) per episode with mean, std, max, min
over the time dimension. This preserves temporal dynamics that mean-only
pooling discards.

Usage:
    python scripts/extract_rich_embeddings.py [--workers 4]
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
from src.models.mert_embed import MERT_HIDDEN_SIZE, MERT_NUM_LAYERS, MERT_SAMPLE_RATE, load_mert

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SOURCE_MANIFEST = PROCESSED_DIR / "manifest.csv"
EMBEDDINGS_DIR = DATA_DIR / "embeddings" / "mert_rich"
EPISODE_SAMPLES = int(MERT_SAMPLE_RATE * CLIP_DURATION_SEC)
RICH_FEATURE_DIM = MERT_HIDDEN_SIZE * 4  # mean, std, max, min


def decode_track(args):
    track_id, subclip_indices, audio_dir = args
    path = track_id_to_path(track_id, audio_dir)
    try:
        y, _ = librosa.load(path, sr=MERT_SAMPLE_RATE, mono=True)
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
def embed_episodes_rich(episodes, model, extractor, device):
    """Embed with temporal statistics: mean, std, max, min per layer.

    Returns (n_episodes, 13, 3072) float16.
    """
    from transformers import Wav2Vec2FeatureExtractor
    inputs = extractor(list(episodes), sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)

    stacked = torch.stack(outputs.hidden_states)  # (layers, batch, time, hidden)

    mean = stacked.mean(dim=2)
    std = stacked.std(dim=2)
    max_val = stacked.max(dim=2).values
    min_val = stacked.min(dim=2).values

    # Concatenate stats: (layers, batch, hidden*4)
    rich = torch.cat([mean, std, max_val, min_val], dim=-1)
    # (batch, layers, hidden*4)
    return rich.permute(1, 0, 2).cpu().numpy().astype(np.float16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest = pd.read_csv(SOURCE_MANIFEST)
    grouped = manifest.groupby("track_id")
    track_ids = sorted(grouped.groups)
    if args.limit:
        track_ids = track_ids[:args.limit]

    device = get_device()
    model, extractor = load_mert(device)
    log.info("Extracting rich MERT embeddings for %d tracks", len(track_ids))

    meta = {tid: grouped.get_group(tid).iloc[0] for tid in track_ids}

    def output_path(tid):
        r = meta[tid]
        return EMBEDDINGS_DIR / r["split"] / r["genre"] / f"{tid:06d}.npy"

    def already_done(tid, n_ep):
        p = output_path(tid)
        if not p.exists():
            return False
        try:
            c = np.load(p, mmap_mode="r")
        except Exception:
            return False
        return c.shape == (n_ep, MERT_NUM_LAYERS, RICH_FEATURE_DIM)

    jobs = []
    resumed = 0
    for tid in track_ids:
        g = grouped.get_group(tid).sort_values("subclip_idx")
        if already_done(tid, len(g)):
            resumed += 1
            continue
        jobs.append((tid, list(g["subclip_idx"]), FMA_SMALL_AUDIO_DIR))

    if resumed:
        log.info("Resuming: %d done, %d to go", resumed, len(jobs))

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        with tqdm(total=len(jobs), desc="Rich MERT") as pbar:
            for start in range(0, len(jobs), 32):
                chunk = jobs[start:start+32]
                for tid, episodes in pool.map(decode_track, chunk):
                    pbar.update(1)
                    if episodes is None:
                        continue
                    r0 = meta[tid]
                    out_dir = EMBEDDINGS_DIR / r0["split"] / r0["genre"]
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out = out_dir / f"{tid:06d}.npy"

                    # Process in batches
                    all_emb = []
                    for i in range(0, len(episodes), 16):
                        emb = embed_episodes_rich(episodes[i:i+16], model, extractor, device)
                        all_emb.append(emb)
                    np.save(out, np.concatenate(all_emb, axis=0))

                    g = grouped.get_group(tid).sort_values("subclip_idx")
                    for ei, si in enumerate(g["subclip_idx"]):
                        rows.append({
                            "embedding_path": str(out.relative_to(EMBEDDINGS_DIR)),
                            "episode_idx": ei, "track_id": tid,
                            "genre": r0["genre"], "split": r0["split"],
                            "subclip_idx": int(si),
                        })

    # Include resumed tracks
    embedded = {r["track_id"] for r in rows}
    for tid in track_ids:
        if tid in embedded:
            continue
        g = grouped.get_group(tid).sort_values("subclip_idx")
        if not already_done(tid, len(g)):
            continue
        r0 = meta[tid]
        rel = output_path(tid).relative_to(EMBEDDINGS_DIR)
        for ei, si in enumerate(g["subclip_idx"]):
            rows.append({
                "embedding_path": str(rel), "episode_idx": ei,
                "track_id": tid, "genre": r0["genre"],
                "split": r0["split"], "subclip_idx": int(si),
            })

    rows.sort(key=lambda r: (r["track_id"], r["subclip_idx"]))
    df = pd.DataFrame(rows)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(EMBEDDINGS_DIR / "manifest.csv", index=False)
    log.info("Wrote %d episodes to %s", len(df), EMBEDDINGS_DIR / "manifest.csv")


if __name__ == "__main__":
    main()
