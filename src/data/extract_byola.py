"""Cache frozen BYOL-A embeddings for every episode in the preprocessing manifest.

Reproduces the feature side of Rai & Bhattacharjee 2026 (BYOL-A self-supervised
embeddings + tuned DNN, 64.3% on FMA-small). Episodes come from
`data/processed/manifest.csv` so they line up with the log-mel, MERT and PANNs
episodes for the same track.

The BYOL-A encoder lives in the upstream repo (not on PyPI); clone it first:

    git clone --depth 1 https://github.com/nttcslab/byol-a.git third_party/byol_a_repo

Only `byol_a.models` is imported. `byol_a.augmentations` is not, because it pulls in
`byol_a.common`, which needs pytorch_lightning and calls the long-removed
`torchaudio.set_audio_backend`; its normalizer is one subtraction and divide, inlined below.

One file per track (`(n_episodes, 2048)` float16), mirroring `extract_embeddings.py`.

Usage:
    python -m src.data.extract_byola [--workers 4] [--limit N]
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from src.config import CLIP_DURATION_SEC, DATA_DIR, FMA_SMALL_AUDIO_DIR, PROCESSED_DIR
from src.data.splits import track_id_to_path
from src.models.common import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SOURCE_MANIFEST = PROCESSED_DIR / "manifest.csv"
EMBEDDINGS_DIR = DATA_DIR / "embeddings" / "byola"
BYOLA_REPO = Path(__file__).resolve().parents[2] / "third_party" / "byol_a_repo"

# From the upstream config.yaml; the pretrained weights only match these settings.
BYOLA_SAMPLE_RATE = 16000
BYOLA_EMBED_DIM = 2048
MELSPEC_KWARGS = dict(
    sample_rate=BYOLA_SAMPLE_RATE,
    n_fft=1024,
    win_length=1024,
    hop_length=160,
    n_mels=64,
    f_min=60,
    f_max=7800,
)

EPISODE_SAMPLES = int(BYOLA_SAMPLE_RATE * CLIP_DURATION_SEC)
DECODE_CHUNK = 32
BATCH_SIZE = 32
N_STATS_EPISODES = 10000


def load_byola(device: torch.device):
    weights = BYOLA_REPO / "pretrained_weights" / f"AudioNTT2020-BYOLA-64x96d{BYOLA_EMBED_DIM}.pth"
    if not weights.exists():
        raise FileNotFoundError(
            f"Missing BYOL-A weights at {weights}. Clone the upstream repo:\n"
            "  git clone --depth 1 https://github.com/nttcslab/byol-a.git third_party/byol_a_repo"
        )
    sys.path.insert(0, str(BYOLA_REPO))
    from byol_a.models import AudioNTT2020

    model = AudioNTT2020(d=BYOLA_EMBED_DIM)
    model.load_weight(str(weights), device)
    model.eval().to(device)
    return model


def log_melspec(episodes: np.ndarray, to_melspec) -> torch.Tensor:
    """(n_episodes, samples) waveforms -> (n_episodes, 1, 64, T) log-mel spectrograms."""
    wav = torch.from_numpy(episodes)
    return (to_melspec(wav) + torch.finfo(torch.float32).eps).log().unsqueeze(1)


@torch.no_grad()
def embed_episodes(episodes: np.ndarray, model, to_melspec, stats, device) -> np.ndarray:
    """(n_episodes, samples) waveforms -> (n_episodes, 2048) float16 embeddings."""
    mean, std = stats
    lms = (log_melspec(episodes, to_melspec) - mean) / std
    out = []
    for start in range(0, len(lms), BATCH_SIZE):
        out.append(model(lms[start : start + BATCH_SIZE].to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float16)


def decode_track(args: tuple) -> tuple[int, np.ndarray | None]:
    """Decode one mp3 at BYOL-A's 16kHz and cut it into the manifest's episodes."""
    track_id, subclip_indices, audio_dir = args
    path = track_id_to_path(track_id, audio_dir)
    try:
        y, _ = librosa.load(path, sr=BYOLA_SAMPLE_RATE, mono=True)
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


def calc_norm_stats(jobs: list[tuple], to_melspec, workers: int, rng: np.random.Generator):
    """Mean/std of the log-mel features, as BYOL-A's evaluate.py does before probing.

    Upstream samples training files only; we sample tracks from the jobs list, which is
    train/val/test mixed. That is deliberate: these are two scalars describing the audio
    corpus, not label information, and using an identical normalizer across splits keeps
    the artist-filtered and clip-random runs comparable.
    """
    n_tracks = min(len(jobs), max(1, N_STATS_EPISODES // 10))
    sample = [jobs[i] for i in rng.choice(len(jobs), size=n_tracks, replace=False)]
    log.info("Calculating log-mel mean/std from %d tracks", n_tracks)

    total = count = 0.0
    total_sq = 0.0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _, episodes in tqdm(pool.map(decode_track, sample), total=len(sample), desc="Norm stats"):
            if episodes is None:
                continue
            lms = log_melspec(episodes, to_melspec)
            total += lms.sum().item()
            total_sq += (lms**2).sum().item()
            count += lms.numel()

    mean = total / count
    std = (total_sq / count - mean**2) ** 0.5
    log.info("Normalization stats: mean=%.6f std=%.6f", mean, std)
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="mp3 decoding processes")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N tracks (debugging)")
    parser.add_argument("--audio-dir", type=Path, default=FMA_SMALL_AUDIO_DIR)
    parser.add_argument("--out-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--force", action="store_true", help="Re-embed tracks that are already cached")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = pd.read_csv(SOURCE_MANIFEST)
    grouped = manifest.groupby("track_id")
    track_ids = sorted(grouped.groups)
    if args.limit:
        track_ids = track_ids[: args.limit]
    log.info("Embedding %d tracks with BYOL-A", len(track_ids))

    device = get_device()
    model = load_byola(device)
    to_melspec = torchaudio.transforms.MelSpectrogram(**MELSPEC_KWARGS)
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
        return cached.shape == (n_episodes, BYOLA_EMBED_DIM)

    rows: list[dict] = []
    jobs = []
    all_jobs = []
    resumed = 0
    for track_id in track_ids:
        group = grouped.get_group(track_id).sort_values("subclip_idx")
        job = (track_id, list(group["subclip_idx"]), args.audio_dir)
        all_jobs.append(job)
        if not args.force and already_done(track_id, len(group)):
            resumed += 1
            continue
        jobs.append(job)

    if resumed:
        log.info("Resuming: %d tracks already embedded, %d to go", resumed, len(jobs))

    # Stats come from all_jobs, not jobs, so a resumed run normalizes identically to
    # the run that produced the already-cached embeddings.
    rng = np.random.default_rng(args.seed)
    stats = calc_norm_stats(all_jobs, to_melspec, args.workers, rng)

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
                    np.save(out_path, embed_episodes(episodes, model, to_melspec, stats, device))

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
