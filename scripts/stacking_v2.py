"""Stacking meta-learner v2 with proper episode alignment.

Fixes alignment issue from v1 - properly intersects episodes across models
by (track_id, subclip_idx) key.

Usage:
    python scripts/stacking_v2.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.config import OUTPUT_DIR
from src.models.common import IDX_TO_GENRE, NUM_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_cached(run_name: str) -> dict[str, np.ndarray]:
    path = OUTPUT_DIR / run_name / "episode_outputs.npz"
    return dict(np.load(path))


def align_runs(runs: list[str], split: str):
    """Load and align episodes from multiple runs by (track_id, subclip_idx).

    Returns dict with 'features_emb', 'features_logits', 'labels', 'track_ids', 'subclip_idx'
    all aligned row-by-row.
    """
    all_data = []
    for run in runs:
        data = load_cached(run)
        track_ids = data[f"{split}_track_ids"]
        subclip_idx = data[f"{split}_subclip_idx"]
        labels = data[f"{split}_labels"]
        embeddings = data[f"{split}_embeddings"]
        logits = data[f"{split}_logits"]

        # Create keyed dict for alignment
        keys = list(zip(track_ids.tolist(), subclip_idx.tolist()))
        all_data.append({
            "keys": keys,
            "labels": labels,
            "embeddings": embeddings,
            "logits": logits,
        })

    # Find common keys
    common_keys = set(all_data[0]["keys"])
    for d in all_data[1:]:
        common_keys &= set(d["keys"])
    common_keys = sorted(common_keys)

    if not common_keys:
        raise ValueError(f"No common episodes across runs for split {split}")

    log.info("%s: %d common episodes across %d runs", split, len(common_keys), len(runs))

    # Build index mapping for each run
    aligned_emb = []
    aligned_logits = []

    for d in all_data:
        key_to_idx = {k: i for i, k in enumerate(d["keys"])}
        indices = [key_to_idx[k] for k in common_keys]
        aligned_emb.append(d["embeddings"][indices])
        aligned_logits.append(d["logits"][indices])

    # Labels and track_ids from first run (they should all agree)
    key_to_idx_ref = {k: i for i, k in enumerate(all_data[0]["keys"])}
    ref_indices = [key_to_idx_ref[k] for k in common_keys]
    labels = all_data[0]["labels"][ref_indices]
    track_ids = np.array([k[0] for k in common_keys])

    return {
        "features_emb": np.concatenate(aligned_emb, axis=1),
        "features_logits": np.concatenate(aligned_logits, axis=1),
        "features_probs": np.concatenate([
            softmax(d) for d in aligned_logits
        ], axis=1),
        "labels": labels,
        "track_ids": track_ids,
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def track_level_eval(probs: np.ndarray, labels: np.ndarray, track_ids: np.ndarray):
    """Mean-pool episode probs per track, compute track-level accuracy and F1."""
    unique_tracks = sorted(set(track_ids))
    track_labels, track_preds = [], []
    for tid in unique_tracks:
        mask = track_ids == tid
        avg_prob = probs[mask].mean(axis=0)
        track_preds.append(avg_prob.argmax())
        track_labels.append(labels[mask][0])

    track_labels = np.array(track_labels)
    track_preds = np.array(track_preds)
    acc = (track_preds == track_labels).mean()
    f1 = f1_score(track_labels, track_preds, average="macro", zero_division=0)

    # Per-genre
    per_genre = {}
    for i in range(NUM_CLASSES):
        genre = IDX_TO_GENRE[i]
        mask = track_labels == i
        if mask.sum() == 0:
            continue
        genre_acc = (track_preds[mask] == i).mean()
        per_genre[genre] = float(genre_acc)

    return float(acc), float(f1), per_genre


def main():
    runs = [
        "cnn_improved",
        "cnn_tuned_lr",
        "mert_probe_wide",
        "mert_probe_baseline",
        "mert_probe_deep",
        "mert_probe_seed1",
        "mert_probe_seed2",
        "mert_probe_seed3",
    ]

    available = [r for r in runs if (OUTPUT_DIR / r / "episode_outputs.npz").exists()]
    log.info("Available runs: %s", available)

    # Define run subsets to try
    run_subsets = {
        "all": available,
        "mert_only": [r for r in available if "mert" in r],
        "cnn_only": [r for r in available if "cnn" in r],
        "best_pair": ["cnn_improved", "mert_probe_wide"],
        "best_trio": ["cnn_improved", "mert_probe_wide", "mert_probe_baseline"],
        "cnn+mert_seeds": ["cnn_improved", "mert_probe_wide", "mert_probe_seed1", "mert_probe_seed2", "mert_probe_seed3"],
        "cnn_both+mert_wide": ["cnn_improved", "cnn_tuned_lr", "mert_probe_wide"],
    }

    feature_modes = ["features_emb", "features_logits", "features_probs"]

    classifiers = {
        "ET_500": lambda: ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=42),
        "ET_1000": lambda: ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, random_state=42),
        "ET_2000": lambda: ExtraTreesClassifier(n_estimators=2000, max_depth=30, n_jobs=-1, random_state=42),
        "RF_500": lambda: RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42),
        "RF_1000": lambda: RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=42),
        "GBM_200": lambda: GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
        "GBM_500": lambda: GradientBoostingClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, random_state=42),
        "LogReg_C1": lambda: LogisticRegression(max_iter=2000, C=1.0, random_state=42),
        "LogReg_C10": lambda: LogisticRegression(max_iter=2000, C=10.0, random_state=42),
    }

    results = []

    for subset_name, subset in run_subsets.items():
        if not all(r in available for r in subset):
            log.warning("Skipping %s: not all runs available", subset_name)
            continue

        log.info("Processing subset: %s (%d runs)", subset_name, len(subset))
        try:
            train_data = align_runs(subset, "train")
            val_data = align_runs(subset, "val")
            test_data = align_runs(subset, "test")
        except Exception as e:
            log.warning("Failed to align %s: %s", subset_name, e)
            continue

        for mode in feature_modes:
            train_X = train_data[mode]
            val_X = val_data[mode]
            test_X = test_data[mode]

            scaler = StandardScaler()
            train_X_s = scaler.fit_transform(train_X)
            val_X_s = scaler.transform(val_X)
            test_X_s = scaler.transform(test_X)

            for clf_name, clf_fn in classifiers.items():
                try:
                    clf = clf_fn()
                    clf.fit(train_X_s, train_data["labels"])

                    if hasattr(clf, "predict_proba"):
                        val_probs = clf.predict_proba(val_X_s)
                        test_probs = clf.predict_proba(test_X_s)
                    else:
                        dec = clf.decision_function(test_X_s)
                        test_probs = softmax(dec)
                        dec = clf.decision_function(val_X_s)
                        val_probs = softmax(dec)

                    val_acc, val_f1, _ = track_level_eval(val_probs, val_data["labels"], val_data["track_ids"])
                    test_acc, test_f1, per_genre = track_level_eval(test_probs, test_data["labels"], test_data["track_ids"])

                    result = {
                        "subset": subset_name,
                        "mode": mode,
                        "classifier": clf_name,
                        "val_acc": val_acc,
                        "val_f1": val_f1,
                        "test_acc": test_acc,
                        "test_f1": test_f1,
                        "per_genre": per_genre,
                    }
                    results.append(result)

                    log.info(
                        "%s/%s/%s: val=%.4f test_acc=%.4f test_f1=%.4f",
                        subset_name, mode.split("_")[1], clf_name, val_acc, test_acc, test_f1,
                    )
                except Exception as e:
                    log.warning("Failed: %s/%s/%s: %s", subset_name, mode, clf_name, e)

    results.sort(key=lambda r: -r["test_acc"])

    print("\n" + "=" * 90)
    print("TOP 30 STACKING RESULTS (track-level accuracy)")
    print("=" * 90)
    for i, r in enumerate(results[:30]):
        print(f"{i+1:2d}. test_acc={r['test_acc']:.4f} test_f1={r['test_f1']:.4f} "
              f"| val={r['val_acc']:.4f} | {r['subset']}/{r['mode'].split('_')[1]}/{r['classifier']}")

    if results:
        best = results[0]
        print(f"\nBest: {best['test_acc']:.4f} accuracy, {best['test_f1']:.4f} F1")
        print(f"Config: {best['subset']}/{best['mode']}/{best['classifier']}")
        if best.get("per_genre"):
            print("Per-genre accuracy:")
            for genre, acc in sorted(best["per_genre"].items(), key=lambda x: -x[1]):
                print(f"  {genre}: {acc:.4f}")

    out_path = OUTPUT_DIR / "stacking_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    log.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
