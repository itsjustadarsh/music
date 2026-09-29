#!/usr/bin/env python3
"""Evaluate a multi-model ensemble by averaging track-level probabilities.

This script loads cached episode_outputs.npz from multiple runs, averages their
per-track softmax probabilities, and reports the combined accuracy and macro-F1.

Usage:
    python scripts/ensemble_evaluate.py --runs cnn_tuned_lr mert_probe_wide ast_finetune
    python scripts/ensemble_evaluate.py --runs cnn_tuned_lr mert_probe_wide --weights 0.3 0.7
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUTPUT_DIR
from src.models.common import IDX_TO_GENRE, NUM_CLASSES


def load_track_probs(run_name: str, split: str = "test") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load cached outputs and compute per-track averaged softmax probabilities.

    Returns (track_ids, track_probs, track_labels) sorted by track_id.
    """
    path = OUTPUT_DIR / run_name / "episode_outputs.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.evaluation.cache_outputs --run {run_name}"
        )
    data = dict(np.load(path))

    logits = data[f"{split}_logits"]
    track_ids = data[f"{split}_track_ids"]
    labels = data[f"{split}_labels"]

    # softmax
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)

    # average per track
    track_groups: dict[int, list[int]] = defaultdict(list)
    for i, tid in enumerate(track_ids):
        track_groups[int(tid)].append(i)

    unique_tracks = sorted(track_groups)
    n = len(unique_tracks)
    track_probs = np.zeros((n, NUM_CLASSES), dtype=np.float64)
    track_labels = np.zeros(n, dtype=np.int64)

    for row, tid in enumerate(unique_tracks):
        indices = track_groups[tid]
        track_probs[row] = probs[indices].mean(axis=0)
        track_labels[row] = labels[indices[0]]

    return np.array(unique_tracks), track_probs, track_labels


def ensemble_evaluate(
    run_names: list[str],
    weights: list[float] | None = None,
    split: str = "test",
) -> dict:
    """Average track-level probabilities across runs and evaluate."""
    if weights is None:
        weights = [1.0 / len(run_names)] * len(run_names)
    assert len(weights) == len(run_names), "weights must match number of runs"
    weights = np.array(weights) / sum(weights)  # normalize

    # Load all runs
    all_data = []
    for name in run_names:
        tids, probs, labels = load_track_probs(name, split)
        all_data.append((name, tids, probs, labels))

    # Find common tracks (intersection)
    common_tids = set(all_data[0][1].tolist())
    for _, tids, _, _ in all_data[1:]:
        common_tids &= set(tids.tolist())
    common_tids = sorted(common_tids)

    if not common_tids:
        raise ValueError("No common tracks across all runs")

    n = len(common_tids)
    tid_to_idx = {tid: i for i, tid in enumerate(common_tids)}

    # Weighted average of probabilities
    ensemble_probs = np.zeros((n, NUM_CLASSES), dtype=np.float64)
    labels = np.zeros(n, dtype=np.int64)

    for (name, tids, probs, lbls), w in zip(all_data, weights):
        for orig_idx, tid in enumerate(tids):
            tid = int(tid)
            if tid in tid_to_idx:
                row = tid_to_idx[tid]
                ensemble_probs[row] += w * probs[orig_idx]
                labels[row] = lbls[orig_idx]

    preds = ensemble_probs.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    # Per-genre
    per_genre = {}
    for g_idx in range(NUM_CLASSES):
        mask = labels == g_idx
        if mask.sum() > 0:
            g_acc = (preds[mask] == g_idx).mean()
            per_genre[IDX_TO_GENRE[g_idx]] = {"accuracy": float(g_acc), "count": int(mask.sum())}

    # Individual model scores for comparison
    individual = {}
    for name, tids, probs, lbls in all_data:
        mask = np.isin(tids, common_tids)
        p = probs[mask].argmax(axis=1)
        l = lbls[mask]
        individual[name] = {
            "accuracy": float(accuracy_score(l, p)),
            "macro_f1": float(f1_score(l, p, average="macro")),
        }

    return {
        "ensemble_accuracy": float(acc),
        "ensemble_macro_f1": float(f1),
        "n_tracks": n,
        "weights": {name: float(w) for name, w in zip(run_names, weights)},
        "individual_models": individual,
        "per_genre": per_genre,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="Run names to ensemble")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="Weights per run (default: equal)")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    result = ensemble_evaluate(args.runs, args.weights, args.split)

    print(f"\n{'='*60}")
    print(f"  ENSEMBLE EVALUATION ({args.split} split)")
    print(f"{'='*60}")
    print(f"\n  Models: {', '.join(args.runs)}")
    print(f"  Weights: {', '.join(f'{w:.2f}' for w in result['weights'].values())}")
    print(f"  Tracks: {result['n_tracks']}")
    print(f"\n  {'Model':<30} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    for name, scores in result["individual_models"].items():
        print(f"  {name:<30} {scores['accuracy']:>10.2%} {scores['macro_f1']:>10.2%}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    print(f"  {'ENSEMBLE':<30} {result['ensemble_accuracy']:>10.2%} {result['ensemble_macro_f1']:>10.2%}")

    print(f"\n  Per-genre accuracy:")
    for genre, info in sorted(result["per_genre"].items()):
        bar = "█" * int(info["accuracy"] * 30)
        print(f"    {genre:<15} {info['accuracy']:>6.1%} {bar}")

    print()

    # Save results
    out_path = OUTPUT_DIR / "ensemble_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to {out_path}\n")


if __name__ == "__main__":
    main()
