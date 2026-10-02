#!/usr/bin/env python3
"""Train XGBoost on MERT embeddings for genre classification.

Uses the learned layer weighting from the best MERT probe to create a single
embedding vector per episode, then trains XGBoost on these features.

Usage:
    python scripts/train_xgboost.py [--mert-run mert_probe_wide]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, OUTPUT_DIR
from src.models.common import IDX_TO_GENRE, NUM_CLASSES
from src.training.dataset import MertEmbeddingDataset


MERT_EMB_DIR = DATA_DIR / "embeddings" / "mert"
MERT_MANIFEST = MERT_EMB_DIR / "manifest.csv"


def load_embeddings(split: str, layer_weights: np.ndarray | None = None, split_mode: str = "official"):
    """Load MERT embeddings for a split, optionally using learned layer weights.

    Returns (features, labels, track_ids) where features is (n_episodes, hidden_dim).
    """
    ds = MertEmbeddingDataset(
        MERT_MANIFEST, MERT_EMB_DIR, split, return_track_ids=True, split_mode=split_mode
    )
    features, labels, track_ids = [], [], []

    for i in range(len(ds)):
        emb, label, tid = ds[i]  # emb: (13, 768) tensor
        emb = emb.numpy()

        if layer_weights is not None:
            # Both: weighted sum (768) + concatenated all layers (9984)
            weighted = (layer_weights[:, None] * emb).sum(axis=0)  # (768,)
            concat = emb.reshape(-1)  # (9984,)
            feat = np.concatenate([weighted, concat])  # (10752,)
        else:
            feat = emb.reshape(-1)  # (9984,)

        features.append(feat)
        labels.append(label)
        track_ids.append(tid)

    return np.array(features), np.array(labels), np.array(track_ids)


def track_level_evaluate(probs: np.ndarray, labels: np.ndarray, track_ids: np.ndarray):
    """Average per-episode probabilities by track, then argmax."""
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

    preds = track_probs.argmax(axis=1)
    acc = accuracy_score(track_labels, preds)
    f1 = f1_score(track_labels, preds, average="macro")

    per_genre = {}
    for g_idx in range(NUM_CLASSES):
        mask = track_labels == g_idx
        if mask.sum() > 0:
            g_acc = float((preds[mask] == g_idx).mean())
            per_genre[IDX_TO_GENRE[g_idx]] = g_acc

    return acc, f1, per_genre, track_probs, track_labels, np.array(unique_tracks)


def main(mert_run: str = "mert_probe_wide", split_mode: str = "official", run_name: str | None = None):
    # Load layer weights from the MERT probe
    layer_weights = None
    probe_dir = OUTPUT_DIR / mert_run
    if probe_dir.exists():
        cfg = json.loads((probe_dir / "config.json").read_text())
        if cfg.get("learn_layer_weights", False):
            state = torch.load(probe_dir / "best.pt", map_location="cpu", weights_only=True)
            logits = state.get("layer_logits")
            if logits is not None:
                layer_weights = torch.softmax(logits, dim=0).numpy()
                print(f"Using learned layer weights from {mert_run}:")
                for i, w in enumerate(layer_weights):
                    bar = "█" * int(w * 80)
                    print(f"  Layer {i:2d}: {w:.3f} {bar}")
                print()

    print(f"Loading embeddings (split_mode={split_mode})...")
    X_train, y_train, tid_train = load_embeddings("train", layer_weights, split_mode)
    X_val, y_val, tid_val = load_embeddings("val", layer_weights, split_mode)
    X_test, y_test, tid_test = load_embeddings("test", layer_weights, split_mode)

    # Subsample training to 1 episode per track to reduce memory
    unique_train_tids = np.unique(tid_train)
    first_idx = []
    seen = set()
    for i, tid in enumerate(tid_train):
        if tid not in seen:
            first_idx.append(i)
            seen.add(tid)
    X_train = X_train[first_idx]
    y_train = y_train[first_idx]
    tid_train = tid_train[first_idx]
    print(f"  Subsampled train to {len(X_train)} (1 per track)")
    print(f"  Train: {len(X_train)} episodes, Val: {len(X_val)}, Test: {len(X_test)}")

    # Hyperparameter search
    best_acc = 0
    best_params = None
    best_model = None

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

    classifiers = [
        ("RandomForest_500", RandomForestClassifier(n_estimators=500, max_depth=None, n_jobs=-1, random_state=42)),
        ("RandomForest_1000", RandomForestClassifier(n_estimators=1000, max_depth=None, n_jobs=-1, random_state=42)),
        ("ExtraTrees_500", ExtraTreesClassifier(n_estimators=500, max_depth=None, n_jobs=-1, random_state=42)),
        ("ExtraTrees_1000", ExtraTreesClassifier(n_estimators=1000, max_depth=None, n_jobs=-1, random_state=42)),
    ]

    print(f"\nTraining {len(classifiers)} classifiers...")
    for name, clf in classifiers:
        print(f"  Training {name}...", flush=True)
        clf.fit(X_train, y_train)

        # Track-level evaluation on val
        val_probs = clf.predict_proba(X_val)
        val_acc, val_f1, _, _, _, _ = track_level_evaluate(val_probs, y_val, tid_val)

        print(f"    {name}: val track-acc={val_acc:.4f}, val track-F1={val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_params = {"name": name}
            best_model = clf

    # Evaluate best model on test
    print(f"\nBest classifier: {best_params}")
    print(f"Best val track-acc: {best_acc:.4f}")

    test_probs = best_model.predict_proba(X_test)
    test_acc, test_f1, per_genre, track_probs, track_labels, unique_tids = track_level_evaluate(
        test_probs, y_test, tid_test
    )

    print(f"\n{'='*60}")
    print(f"  XGBoost TEST RESULTS (track-level)")
    print(f"{'='*60}")
    print(f"  Accuracy: {test_acc:.2%}")
    print(f"  Macro F1: {test_f1:.2%}")
    print(f"\n  Per-genre accuracy:")
    for genre, acc in sorted(per_genre.items()):
        bar = "█" * int(acc * 30)
        print(f"    {genre:<15} {acc:>6.1%} {bar}")

    # Save results
    run_dir = OUTPUT_DIR / (run_name or "xgboost_mert")
    run_dir.mkdir(exist_ok=True)

    # Save config
    config = {
        "model": "xgboost_mert",
        "mert_run": mert_run,
        "split_mode": split_mode,
        "best_params": best_params,
        "batch_size": 256,  # for compatibility with cache_outputs
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save eval metrics in same format as neural models
    eval_metrics = {
        "track_level": {
            "accuracy": float(test_acc),
            "macro_f1": float(test_f1),
            "per_genre": {g: {"accuracy": float(a)} for g, a in per_genre.items()},
            "n_tracks": len(unique_tids),
        },
    }
    with open(run_dir / "eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    # Save model
    import pickle
    with open(run_dir / "best.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Save track-level probabilities for ensembling
    np.savez_compressed(
        run_dir / "track_probs.npz",
        track_ids=unique_tids,
        probs=track_probs.astype(np.float32),
        labels=track_labels,
    )

    print(f"\n  Results saved to {run_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mert-run", default="mert_probe_wide")
    parser.add_argument("--split-mode", default="official", choices=["official", "random_track", "random_clip"])
    parser.add_argument("--run-name", default=None, help="output dir name; defaults to xgboost_mert")
    args = parser.parse_args()
    main(args.mert_run, args.split_mode, args.run_name)
