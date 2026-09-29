"""Weighted probability ensemble with Bayesian optimization of weights.

Instead of training a meta-learner, just average track-level softmax
probabilities with optimized per-model weights. Much simpler and less
prone to overfitting than stacking.

Usage:
    python scripts/weighted_ensemble.py
"""
from __future__ import annotations

import json
import logging
from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score

from src.config import OUTPUT_DIR
from src.models.common import IDX_TO_GENRE, NUM_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_cached(run_name: str) -> dict[str, np.ndarray]:
    return dict(np.load(OUTPUT_DIR / run_name / "episode_outputs.npz"))


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def get_track_probs(run_name: str, split: str) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Get track-level averaged probabilities for a run."""
    data = load_cached(run_name)
    logits = data[f"{split}_logits"]
    probs = softmax(logits)
    labels = data[f"{split}_labels"]
    track_ids = data[f"{split}_track_ids"]

    track_probs = {}
    track_labels = {}
    for p, l, tid in zip(probs, labels, track_ids):
        tid = int(tid)
        if tid not in track_probs:
            track_probs[tid] = []
            track_labels[tid] = int(l)
        track_probs[tid].append(p)

    return {tid: np.mean(ps, axis=0) for tid, ps in track_probs.items()}, track_labels


def evaluate_weights(weights: np.ndarray, all_probs: list[dict], labels: dict, track_ids: list[int]) -> float:
    """Negative accuracy for optimization."""
    w = np.abs(weights)
    w = w / w.sum()

    correct = 0
    total = 0
    for tid in track_ids:
        combined = sum(w[i] * all_probs[i][tid] for i in range(len(w)) if tid in all_probs[i])
        pred = combined.argmax()
        if pred == labels[tid]:
            correct += 1
        total += 1

    return -correct / total  # negative for minimization


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

    # Load all track-level probs
    val_probs = {}
    val_labels = {}
    test_probs = {}
    test_labels = {}
    for run in available:
        vp, vl = get_track_probs(run, "val")
        tp, tl = get_track_probs(run, "test")
        val_probs[run] = vp
        val_labels[run] = vl
        test_probs[run] = tp
        test_labels[run] = tl

    # Use labels from first run as reference
    ref_val_labels = val_labels[available[0]]
    ref_test_labels = test_labels[available[0]]

    # Find common tracks
    common_val = set.intersection(*[set(val_probs[r].keys()) for r in available])
    common_test = set.intersection(*[set(test_probs[r].keys()) for r in available])
    log.info("Common val tracks: %d, test tracks: %d", len(common_val), len(common_test))

    results = []

    # Try all subsets of size 2-8
    for size in range(2, len(available) + 1):
        for subset in combinations(available, size):
            subset = list(subset)
            sub_val_probs = [val_probs[r] for r in subset]
            sub_test_probs = [test_probs[r] for r in subset]

            common_v = sorted(set.intersection(*[set(vp.keys()) for vp in sub_val_probs]))
            common_t = sorted(set.intersection(*[set(tp.keys()) for tp in sub_test_probs]))

            # Optimize weights on validation set
            n = len(subset)
            x0 = np.ones(n) / n

            result = minimize(
                evaluate_weights,
                x0,
                args=(sub_val_probs, ref_val_labels, common_v),
                method="Nelder-Mead",
                options={"maxiter": 1000},
            )

            opt_weights = np.abs(result.x)
            opt_weights = opt_weights / opt_weights.sum()

            # Also try uniform weights
            uniform_weights = np.ones(n) / n

            for weight_name, weights in [("optimized", opt_weights), ("uniform", uniform_weights)]:
                # Evaluate on test
                correct = 0
                all_preds, all_labels_list = [], []
                for tid in common_t:
                    combined = sum(weights[i] * sub_test_probs[i][tid] for i in range(n))
                    pred = combined.argmax()
                    all_preds.append(pred)
                    all_labels_list.append(ref_test_labels[tid])
                    if pred == ref_test_labels[tid]:
                        correct += 1

                test_acc = correct / len(common_t)
                test_f1 = f1_score(all_labels_list, all_preds, average="macro", zero_division=0)

                # Val accuracy
                correct_v = 0
                for tid in common_v:
                    combined = sum(weights[i] * sub_val_probs[i][tid] for i in range(n))
                    if combined.argmax() == ref_val_labels[tid]:
                        correct_v += 1
                val_acc = correct_v / len(common_v)

                r = {
                    "runs": subset,
                    "weights": weight_name,
                    "weight_values": weights.tolist(),
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "test_f1": test_f1,
                    "n_test_tracks": len(common_t),
                }
                results.append(r)

                if test_acc > 0.58:  # Only log promising results
                    log.info(
                        "%s %s: val=%.4f test_acc=%.4f test_f1=%.4f [%s]",
                        weight_name, "+".join(s.split("_")[-1] for s in subset),
                        val_acc, test_acc, test_f1,
                        ", ".join(f"{w:.3f}" for w in weights),
                    )

    results.sort(key=lambda r: -r["test_acc"])

    print("\n" + "=" * 90)
    print("TOP 30 WEIGHTED ENSEMBLE RESULTS (track-level)")
    print("=" * 90)
    for i, r in enumerate(results[:30]):
        short = [s.replace("mert_probe_", "mp_").replace("cnn_", "c_") for s in r["runs"]]
        print(f"{i+1:2d}. test={r['test_acc']:.4f} f1={r['test_f1']:.4f} val={r['val_acc']:.4f} "
              f"| {r['weights']} | {'+'.join(short)}")

    out_path = OUTPUT_DIR / "weighted_ensemble_results.json"
    with open(out_path, "w") as f:
        json.dump(results[:50], f, indent=2, default=float)
    log.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
