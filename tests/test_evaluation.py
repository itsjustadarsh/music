"""Tests for track-level aggregation of per-episode predictions."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import aggregate_to_tracks


def test_mean_pooling_overrules_clip_majority():
    """The case that motivates averaging probabilities instead of voting.

    Two low-confidence episodes say class 0 and one certain episode says class 1.
    A majority vote returns 0; mean-pooling returns 1, because the vote throws away
    exactly the confidence that should break the tie.
    """
    probs = np.array([[0.51, 0.49], [0.51, 0.49], [0.0, 1.0]])
    labels = [1, 1, 1]
    track_ids = [7, 7, 7]

    clip_majority = np.bincount(probs.argmax(axis=1)).argmax()
    assert clip_majority == 0

    track_labels, track_preds = aggregate_to_tracks(probs, labels, track_ids)
    assert track_labels == [1]
    assert track_preds == [1]


def test_one_prediction_per_track_ordered_by_track_id():
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9], [0.7, 0.3]])
    labels = [0, 0, 1, 1, 0]
    track_ids = [30, 30, 11, 11, 22]

    track_labels, track_preds = aggregate_to_tracks(probs, labels, track_ids)

    assert len(track_preds) == 3  # tracks, not episodes
    assert track_labels == [1, 0, 0]  # ordered by track_id: 11, 22, 30
    assert track_preds == [1, 0, 0]


def test_uneven_episode_counts_are_averaged_not_summed():
    """A track with more episodes must not outvote itself -- we divide by count."""
    probs = np.array([[0.6, 0.4], [0.6, 0.4], [0.6, 0.4], [0.3, 0.7]])
    labels = [0, 0, 0, 1]
    track_ids = [1, 1, 1, 2]

    track_labels, track_preds = aggregate_to_tracks(probs, labels, track_ids)
    assert track_preds == [0, 1]
    assert track_labels == [0, 1]


def test_single_episode_track_passes_through():
    probs = np.array([[0.2, 0.8]])
    track_labels, track_preds = aggregate_to_tracks(probs, [1], [5])
    assert track_labels == [1]
    assert track_preds == [1]


def test_conflicting_labels_for_one_track_raise():
    """A track has exactly one genre; disagreement means the join went wrong."""
    probs = np.array([[0.9, 0.1], [0.9, 0.1]])
    with pytest.raises(ValueError, match="conflicting labels"):
        aggregate_to_tracks(probs, [0, 1], [3, 3])


# --- evaluation protocol / split modes ------------------------------------

import pandas as pd

from src.training.dataset import assign_random_clip_splits, load_manifest


def _fake_manifest(tmp_path, n_tracks=50, clips_per_track=10):
    rows = []
    for t in range(n_tracks):
        split = "train" if t < n_tracks * 0.8 else ("val" if t < n_tracks * 0.9 else "test")
        for i in range(clips_per_track):
            rows.append(
                {
                    "logmel_path": f"logmel/{split}/Rock/{t:06d}_{i}.npy",
                    "waveform_path": f"waveform16k/{split}/Rock/{t:06d}_{i}.npy",
                    "track_id": t,
                    "genre": "Rock",
                    "split": split,
                    "subclip_idx": i,
                }
            )
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _tracks_in(path, split, mode):
    return set(load_manifest(path, split, split_mode=mode)["track_id"])


def test_official_split_never_lets_a_track_straddle_splits(tmp_path):
    path = _fake_manifest(tmp_path)
    train = _tracks_in(path, "train", "official")
    val = _tracks_in(path, "val", "official")
    test = _tracks_in(path, "test", "official")

    assert train and val and test
    assert not (train & test)
    assert not (train & val)
    assert not (val & test)


def test_random_clip_split_does_let_tracks_straddle_splits(tmp_path):
    """The leakage this mode exists to measure -- if it stopped happening, the
    protocol-audit result would silently become meaningless."""
    path = _fake_manifest(tmp_path)
    train = _tracks_in(path, "train", "random_clip")
    test = _tracks_in(path, "test", "random_clip")

    assert train & test, "random_clip must place clips of one track in both train and test"


def test_random_clip_split_is_deterministic_and_covers_every_clip(tmp_path):
    path = _fake_manifest(tmp_path, n_tracks=20, clips_per_track=5)
    counts = {s: len(load_manifest(path, s, split_mode="random_clip")) for s in ("train", "val", "test")}
    assert sum(counts.values()) == 100
    assert counts["train"] == 80 and counts["val"] == 10 and counts["test"] == 10
    # same seed -> same assignment
    again = load_manifest(path, "test", split_mode="random_clip")
    assert list(again["track_id"]) == list(load_manifest(path, "test", split_mode="random_clip")["track_id"])


def test_unknown_split_mode_rejected(tmp_path):
    path = _fake_manifest(tmp_path, n_tracks=5, clips_per_track=2)
    with pytest.raises(ValueError, match="Unknown split_mode"):
        load_manifest(path, "train", split_mode="whatever")
