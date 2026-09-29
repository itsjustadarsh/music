"""Join two backbones' per-episode caches into track-batched tensors.

Fusion is defined per episode but supervised per track, so the natural unit here is
a track: a variable-length bag of episodes with one genre label. Tracks are padded to
the longest bag in the split and carry a boolean mask, so attention never spends
weight on padding.

The join is on (track_id, subclip_idx) -- the reason `extract_embeddings.py` derives
its episodes from the preprocessing manifest instead of recomputing them. When the
two caches have different episode sets (e.g. MERT extraction skipped a few corrupt
tracks), we fall back to the intersection rather than failing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.config import OUTPUT_DIR


@dataclass
class TrackBatch:
    """One split, padded to (n_tracks, max_episodes, ...)."""

    emb_a: torch.Tensor
    emb_b: torch.Tensor
    logits_a: torch.Tensor
    logits_b: torch.Tensor
    mask: torch.Tensor  # (n_tracks, max_episodes) bool
    labels: torch.Tensor  # (n_tracks,)
    track_ids: np.ndarray

    def to(self, device) -> "TrackBatch":
        return TrackBatch(
            self.emb_a.to(device),
            self.emb_b.to(device),
            self.logits_a.to(device),
            self.logits_b.to(device),
            self.mask.to(device),
            self.labels.to(device),
            self.track_ids,
        )

    def __len__(self) -> int:
        return len(self.labels)


def _load(run_name: str) -> dict:
    path = OUTPUT_DIR / run_name / "episode_outputs.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m src.evaluation.cache_outputs --run {run_name}` first"
        )
    return dict(np.load(path))


def build_track_batch(run_a: str, run_b: str, split: str) -> TrackBatch:
    """Join runs `run_a` and `run_b` on (track_id, subclip_idx) for one split."""
    a, b = _load(run_a), _load(run_b)

    key_a = list(zip(a[f"{split}_track_ids"].tolist(), a[f"{split}_subclip_idx"].tolist()))
    key_b = list(zip(b[f"{split}_track_ids"].tolist(), b[f"{split}_subclip_idx"].tolist()))

    set_a, set_b = set(key_a), set(key_b)
    common = set_a & set_b
    if not common:
        raise ValueError(f"No common episodes between {run_a} and {run_b} on split {split!r}")
    if set_a != set_b:
        import logging
        _log = logging.getLogger(__name__)
        _log.warning(
            "Episode sets differ on %s: %d in %s only, %d in %s only, %d in common — using intersection",
            split, len(set_a - set_b), run_a, len(set_b - set_a), run_b, len(common),
        )

    # Filter both to intersection, maintaining order from A
    keep_a = [i for i, k in enumerate(key_a) if k in common]
    index_b = {key: i for i, key in enumerate(key_b)}
    keep_b = [index_b[key_a[i]] for i in keep_a]

    labels_a = a[f"{split}_labels"][keep_a]
    labels_b = b[f"{split}_labels"][keep_b]
    if not np.array_equal(labels_a, labels_b):
        raise ValueError(f"Labels disagree between {run_a} and {run_b} on split {split!r}")

    emb_a = a[f"{split}_embeddings"][keep_a]
    emb_b = b[f"{split}_embeddings"][keep_b]
    log_a = a[f"{split}_logits"][keep_a]
    log_b = b[f"{split}_logits"][keep_b]
    track_ids = a[f"{split}_track_ids"][keep_a]

    # group episodes by track, preserving first-seen track order
    order: dict[int, list[int]] = {}
    for position, tid in enumerate(track_ids.tolist()):
        order.setdefault(tid, []).append(position)

    unique_tracks = sorted(order)
    max_eps = max(len(v) for v in order.values())
    n = len(unique_tracks)

    out_emb_a = np.zeros((n, max_eps, emb_a.shape[1]), dtype=np.float32)
    out_emb_b = np.zeros((n, max_eps, emb_b.shape[1]), dtype=np.float32)
    out_log_a = np.zeros((n, max_eps, log_a.shape[1]), dtype=np.float32)
    out_log_b = np.zeros((n, max_eps, log_b.shape[1]), dtype=np.float32)
    mask = np.zeros((n, max_eps), dtype=bool)
    labels = np.zeros(n, dtype=np.int64)

    for row, tid in enumerate(unique_tracks):
        positions = order[tid]
        k = len(positions)
        out_emb_a[row, :k] = emb_a[positions]
        out_emb_b[row, :k] = emb_b[positions]
        out_log_a[row, :k] = log_a[positions]
        out_log_b[row, :k] = log_b[positions]
        mask[row, :k] = True
        labels[row] = labels_a[positions[0]]

    return TrackBatch(
        emb_a=torch.from_numpy(out_emb_a),
        emb_b=torch.from_numpy(out_emb_b),
        logits_a=torch.from_numpy(out_log_a),
        logits_b=torch.from_numpy(out_log_b),
        mask=torch.from_numpy(mask),
        labels=torch.from_numpy(labels),
        track_ids=np.array(unique_tracks),
    )
