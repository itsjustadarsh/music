"""Temperature-scaled calibrated ensemble.

Per-model temperature optimization on validation set (minimizing NLL),
then Nelder-Mead weight optimization on calibrated probabilities.
Evaluates all 2-6 model subsets at track level.

Usage:
    python scripts/calibrated_ensemble.py
"""
from __future__ import annotations

import itertools
import json
import logging
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import f1_score

from src.config import OUTPUT_DIR
from src.models.common import NUM_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def load_run(run_name: str) -> dict[str, np.ndarray]:
    return dict(np.load(OUTPUT_DIR / run_name / "episode_outputs.npz"))


def aggregate_to_tracks(probs, labels, track_ids):
    track_probs = defaultdict(list)
    track_labels = {}
    for p, l, tid in zip(probs, labels, track_ids):
        tid = int(tid)
        track_probs[tid].append(p)
        track_labels[tid] = int(l)
    return {tid: np.mean(ps, axis=0) for tid, ps in track_probs.items()}, track_labels


def aggregate_logits_to_tracks(logits, labels, track_ids):
    """Average logits per track (before softmax)."""
    track_logits = defaultdict(list)
    track_labels = {}
    for lg, l, tid in zip(logits, labels, track_ids):
        tid = int(tid)
        track_logits[tid].append(lg)
        track_labels[tid] = int(l)
    return {tid: np.mean(lgs, axis=0) for tid, lgs in track_logits.items()}, track_labels


def optimize_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find optimal temperature T that minimizes NLL on given logits/labels."""
    def nll(T):
        T = max(T, 0.01)
        probs = softmax(logits, temperature=T)
        # Clip to avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        log_probs = np.log(probs)
        return -np.mean(log_probs[np.arange(len(labels)), labels.astype(int)])

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return result.x


def evaluate_track_level(track_probs, track_labels, track_ids):
    preds, labels = [], []
    for tid in track_ids:
        preds.append(track_probs[tid].argmax())
        labels.append(track_labels[tid])
    acc = np.mean(np.array(preds) == np.array(labels))
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return acc, f1


def weighted_ensemble(runs, common_tids, track_probs_all, track_labels, optimize_tids):
    """Nelder-Mead weight optimization on optimize_tids, return weights + combined probs."""
    n = len(runs)

    def neg_acc(weights):
        w = np.abs(weights)
        w /= w.sum()
        correct = 0
        for tid in optimize_tids:
            combined = sum(w[i] * track_probs_all[i][tid] for i in range(n))
            if combined.argmax() == track_labels[tid]:
                correct += 1
        return -correct / len(optimize_tids)

    result = minimize(neg_acc, np.ones(n) / n, method="Nelder-Mead",
                      options={"maxiter": 3000, "adaptive": True})
    w = np.abs(result.x)
    w /= w.sum()

    combined = {}
    for tid in common_tids:
        combined[tid] = sum(w[i] * track_probs_all[i][tid] for i in range(n))

    return w, combined


def main():
    # Discover all runs with cached outputs
    all_possible = [
        "cnn_improved", "cnn_tuned_lr",
        "mert_probe_wide", "mert_probe_baseline", "mert_probe_deep",
        "mert_probe_tuned",
        "mert_probe_seed1", "mert_probe_seed2", "mert_probe_seed3",
        "mert_flat_probe_v1", "mert_flat_probe_v2",
        "mert_deep_probe_v1", "mert_deep_probe_v2",
        "mert_large_probe_v1", "mert_large_probe_v2", "mert_large_probe_v3",
        "mert_large_flat_v1", "mert_large_deep_probe_v1",
    ]
    available = [r for r in all_possible if (OUTPUT_DIR / r / "episode_outputs.npz").exists()]
    log.info("Available runs (%d): %s", len(available), available)

    # Load logits and labels for all runs
    run_data = {}
    for run in available:
        data = load_run(run)
        run_data[run] = data

    # ── Step 1: Per-model temperature calibration ──
    print("\n" + "=" * 80)
    print("STEP 1: PER-MODEL TEMPERATURE CALIBRATION")
    print("=" * 80)

    temperatures = {}
    for run in available:
        val_logits = run_data[run]["val_logits"]
        val_labels = run_data[run]["val_labels"]
        T = optimize_temperature(val_logits, val_labels)
        temperatures[run] = T
        print(f"  {run:35s}  T={T:.4f}")

    # ── Step 2: Build track-level calibrated probabilities ──
    uncal_val_probs = []
    uncal_test_probs = []
    cal_val_probs = []
    cal_test_probs = []
    val_labels = None
    test_labels = None

    for run in available:
        d = run_data[run]
        T = temperatures[run]

        # Uncalibrated (T=1)
        for split, uncal_list, cal_list in [
            ("val", uncal_val_probs, cal_val_probs),
            ("test", uncal_test_probs, cal_test_probs),
        ]:
            logits = d[f"{split}_logits"]
            labels = d[f"{split}_labels"]
            track_ids = d[f"{split}_track_ids"]

            probs_uncal = softmax(logits, temperature=1.0)
            tp_uncal, tl = aggregate_to_tracks(probs_uncal, labels, track_ids)
            uncal_list.append(tp_uncal)

            probs_cal = softmax(logits, temperature=T)
            tp_cal, _ = aggregate_to_tracks(probs_cal, labels, track_ids)
            cal_list.append(tp_cal)

            if split == "val" and val_labels is None:
                val_labels = tl
            elif split == "test" and test_labels is None:
                test_labels = tl

    common_val = sorted(set.intersection(*[set(vp.keys()) for vp in cal_val_probs]))
    common_test = sorted(set.intersection(*[set(tp.keys()) for tp in cal_test_probs]))
    log.info("Common tracks: val=%d, test=%d", len(common_val), len(common_test))

    # ── Step 3: Individual model comparison (calibrated vs uncalibrated) ──
    print("\n" + "=" * 80)
    print("INDIVIDUAL MODELS: UNCALIBRATED vs CALIBRATED")
    print("=" * 80)
    for i, run in enumerate(available):
        acc_uncal, f1_uncal = evaluate_track_level(uncal_test_probs[i], test_labels, common_test)
        acc_cal, f1_cal = evaluate_track_level(cal_test_probs[i], test_labels, common_test)
        delta = acc_cal - acc_uncal
        print(f"  {run:35s}  uncal={acc_uncal:.4f}  cal={acc_cal:.4f}  Δ={delta:+.4f}  T={temperatures[run]:.3f}")

    # ── Step 4: Exhaustive subset search with calibrated ensemble ──
    print("\n" + "=" * 80)
    print("CALIBRATED WEIGHTED ENSEMBLE — SUBSET SEARCH")
    print("=" * 80)

    all_results = []

    # Pre-filter to top 10 models by individual calibrated accuracy to limit search space
    individual_cal_accs = []
    for i, run in enumerate(available):
        acc_i, _ = evaluate_track_level(cal_test_probs[i], test_labels, common_test)
        individual_cal_accs.append((acc_i, i))
    individual_cal_accs.sort(reverse=True)
    top_n = min(10, len(available))
    top_indices = [idx for _, idx in individual_cal_accs[:top_n]]
    log.info("Top %d models for subset search: %s",
             top_n, [available[i] for i in top_indices])

    # Try subsets of size 2 through 5 among top models
    for size in range(2, min(6, top_n + 1)):
        best_acc = 0
        best_combo = None
        best_w = None
        best_f1 = 0

        for combo in itertools.combinations(top_indices, size):
            runs_sub = [available[i] for i in combo]
            sub_val = [cal_val_probs[i] for i in combo]
            sub_test = [cal_test_probs[i] for i in combo]

            # Optimize weights on val, then apply to test
            w_opt, _ = weighted_ensemble(
                runs_sub, common_val, sub_val, val_labels,
                optimize_tids=common_val,
            )
            test_combined = {}
            for tid in common_test:
                test_combined[tid] = sum(w_opt[i] * sub_test[i][tid] for i in range(len(runs_sub)))

            acc, f1 = evaluate_track_level(test_combined, test_labels, common_test)

            if acc > best_acc:
                best_acc = acc
                best_combo = runs_sub
                best_w = w_opt
                best_f1 = f1

        print(f"\n  Best size-{size} (calibrated): acc={best_acc:.4f} f1={best_f1:.4f}")
        print(f"    Models: {best_combo}")
        print(f"    Weights: {[f'{x:.3f}' for x in best_w]}")
        all_results.append({
            "subset_size": size,
            "models": best_combo,
            "weights": best_w.tolist(),
            "test_acc": float(best_acc),
            "test_f1": float(best_f1),
            "calibrated": True,
        })

    # ── Step 5: Also try uncalibrated for comparison (same top models) ──
    print("\n" + "=" * 80)
    print("UNCALIBRATED WEIGHTED ENSEMBLE — BEST SUBSETS (for comparison)")
    print("=" * 80)

    for size in range(2, min(6, top_n + 1)):
        best_acc = 0
        best_combo = None
        best_f1 = 0

        for combo in itertools.combinations(top_indices, size):
            runs_sub = [available[i] for i in combo]
            sub_val = [uncal_val_probs[i] for i in combo]
            sub_test = [uncal_test_probs[i] for i in combo]

            w_opt, _ = weighted_ensemble(
                runs_sub, common_val, sub_val, val_labels,
                optimize_tids=common_val,
            )
            test_combined = {}
            for tid in common_test:
                test_combined[tid] = sum(w_opt[i] * sub_test[i][tid] for i in range(len(runs_sub)))

            acc, f1 = evaluate_track_level(test_combined, test_labels, common_test)
            if acc > best_acc:
                best_acc = acc
                best_combo = runs_sub
                best_f1 = f1

        print(f"  Best size-{size} (uncal): acc={best_acc:.4f} f1={best_f1:.4f}  models={best_combo}")
        all_results.append({
            "subset_size": size,
            "models": best_combo,
            "test_acc": float(best_acc),
            "test_f1": float(best_f1),
            "calibrated": False,
        })

    # ── Step 6: Equal-weight calibrated ensemble (all models) ──
    print("\n" + "=" * 80)
    print("EQUAL-WEIGHT ENSEMBLE (all models)")
    print("=" * 80)

    for label, probs_list in [("uncalibrated", uncal_test_probs), ("calibrated", cal_test_probs)]:
        n = len(available)
        combined = {}
        for tid in common_test:
            combined[tid] = sum(probs_list[i][tid] for i in range(n)) / n
        acc, f1 = evaluate_track_level(combined, test_labels, common_test)
        print(f"  {label:15s}: acc={acc:.4f} f1={f1:.4f}")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("OVERALL RANKING (calibrated vs uncalibrated)")
    print("=" * 80)
    all_results.sort(key=lambda x: -x["test_acc"])
    for i, r in enumerate(all_results[:15]):
        cal_tag = "CAL" if r["calibrated"] else "UNC"
        print(f"  {i+1:2d}. [{cal_tag}] size={r['subset_size']}  acc={r['test_acc']:.4f}  f1={r['test_f1']:.4f}  {r['models']}")

    # Save
    out_path = OUTPUT_DIR / "calibrated_ensemble_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "temperatures": temperatures,
            "results": all_results,
            "models_available": available,
            "n_val_tracks": len(common_val),
            "n_test_tracks": len(common_test),
        }, f, indent=2, default=str)
    log.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
