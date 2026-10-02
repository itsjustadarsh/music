"""Cache frozen PANNs/CNN14 embeddings for every episode in the preprocessing manifest.

Reproduces the feature side of Liu et al. 2022 (AudioSet-pretrained CNN14 + classifier,
60.9% on FMA-small). Episodes are read from `data/processed/manifest.csv` so the PANNs
episodes line up exactly with the log-mel and MERT episodes for the same track.

One file per track (`(n_episodes, 2048)` float16), mirroring `extract_embeddings.py`.

Usage:
    python -m src.data.extract_panns [--workers 4] [--limit N]
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
EMBEDDINGS_DIR = DATA_DIR / "embeddings" / "panns"

PANNS_SAMPLE_RATE = 32000
PANNS_EMBED_DIM = 2048
EPISODE_SAMPLES = int(PANNS_SAMPLE_RATE * CLIP_DURATION_SEC)
DECODE_CHUNK = 32
BATCH_SIZE = 16


def load_cnn14(device: torch.device):
    """CNN14 with the AudioSet-pretrained weights panns_inference downloads to ~/panns_data."""
    from panns_inference.models import Cnn14

    checkpoint_path = Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing PANNs checkpoint at {checkpoint_path}. Download it with:\n"
            "  curl -L -o ~/panns_data/'Cnn14_mAP=0.431.pth' "
            "'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'"
        )
    model = Cnn14(
        sample_rate=PANNS_SAMPLE_RATE,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    return model


@torch.no_grad()
def embed_episodes(episodes: np.ndarray, model, device: torch.device) -> np.ndarray:
    """(n_episodes, samples) waveforms -> (n_episodes, 2048) float16 embeddings."""
    out = []
    for start in range(0, len(episodes), BATCH_SIZE):
        batch = torch.from_numpy(episodes[start : start + BATCH_SIZE]).to(device)
        out.append(model(batch)["embedding"].cpu().numpy())
    return np.concatenate(out).astype(np.float16)


def decode_track(args: tuple) -> tuple[int, np.ndarray | None]:
    """Decode one mp3 at PANNs' 32kHz and cut it into the manifest's episodes."""
    track_id, subclip_indices, audio_dir = args
    path = track_id_to_path(track_id, audio_dir)
    try:
        y, _ = librosa.load(path, sr=PANNS_SAMPLE_RATE, mono=True)
    except Exception as exc:  # same handful of corrupt FMA mp3s that preprocess.py skips
        log.warning("Skipping unreadable track %s: %s", track_id, exc)
        return track_id, None

    episodes = []
    for idx in subclip_indices:
        start = idx * EPISODE_SAMPLES
        clip = y[start : start + EPISODE_SAMPLES]
        if len(clip) < EPISODE_SAMPLES:  # trailing partial episode
            clip = np.pad(clip, (0, EPISODE_SAMPLES - len(clip)))
        episodes.append(clip)
    return track_id, np.stack(episodes).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="mp3 decoding processes")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N tracks (debugging)")
    parser.add_argument("--audio-dir", type=Path, default=FMA_SMALL_AUDIO_DIR)
    parser.add_argument("--out-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--force", action="store_true", help="Re-embed tracks that are already cached")
    args = parser.parse_args()

    manifest = pd.read_csv(SOURCE_MANIFEST)
    grouped = manifest.groupby("track_id")
    track_ids = sorted(grouped.groups)
    if args.limit:
        track_ids = track_ids[: args.limit]
    log.info("Embedding %d tracks with PANNs CNN14", len(track_ids))

    device = get_device()
    model = load_cnn14(device)
    log.info("Using device: %s", device)

    meta = {tid: grouped.get_group(tid).iloc[0] for tid in track_ids}

    def output_path(track_id: int) -> Path:
        row = meta[track_id]
        return args.out_dir / row["split"] / row["genre"] / f"{track_id:06d}.npy"

    def already_done(track_id: int, n_episodes: int) -> bool:
        """A cached file counts only if it loads and has the expected shape.

        np.save is not atomic, so an interrupted run can leave a truncated file;
        validating the header here keeps a resume from inheriting a corrupt one.
        """
        path = output_path(track_id)
        if not path.exists():
            return False
        try:
            cached = np.load(path, mmap_mode="r")
        except Exception:
            log.warning("Discarding unreadable cache for track %s", track_id)
            return False
        return cached.shape == (n_episodes, PANNS_EMBED_DIM)

    rows: list[dict] = []
    jobs = []
    resumed = 0
    for track_id in track_ids:
        group = grouped.get_group(track_id).sort_values("subclip_idx")
        if not args.force and already_done(track_id, len(group)):
            resumed += 1
            continue
        jobs.append((track_id, list(group["subclip_idx"]), args.audio_dir))

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
                    np.save(out_path, embed_episodes(episodes, model, device))

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

    # Rows were only appended for tracks embedded in this run; the manifest must
    # describe every cached track, including ones carried over from an earlier run.
    embedded = {r["track_id"] for r in rows}
    for track_id in track_ids:
        if track_id in embedded:
            continue
        group = grouped.get_group(track_id).sort_values("subclip_idx")
        if not already_done(track_id, len(group)):
            continue  # genuinely missing (e.g. the corrupt FMA files)
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
    manifest_path = args.out_dir / "manifest.csv"
    out_manifest.to_csv(manifest_path, index=False)
    log.info("Wrote manifest with %d episodes to %s", len(out_manifest), manifest_path)
    log.info("Per-split episode counts:\n%s", out_manifest["split"].value_counts())


if __name__ == "__main__":
    main()
