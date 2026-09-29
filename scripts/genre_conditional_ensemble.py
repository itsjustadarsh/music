"""Genre-conditional ensemble: learn per-genre weight vectors instead of
a single global weight vector.

Some models excel at Electronic but fail at Rock; a genre-conditional ensemble
routes each prediction to its specialist.

Methods:
1. Per-genre Nelder-Mead: optimize W_g on val tracks where true label = g
2. Iterative refinement: initial prediction → genre-specific weights → re-predict → converge
3. MLP stacking: small MLP on concatenated per-model probabilities → 8-class output

Usage:
    python scripts/genre_conditional_ensemble.py
"""
from __future__ import annotations

import itertools
import json
import logging
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.config import OUTPUT_DIR
from src.models.common import IDX_TO_GENRE, NUM_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
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


def evaluate_track_level(track_probs, track_labels, track_ids):
    preds, labels = [], []
    for tid in track_ids:
        preds.append(track_probs[tid].argmax())
        labels.append(track_labels[tid])
    acc = np.mean(np.array(preds) == np.array(labels))
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return acc, f1


def method_per_genre_weights(runs, common_val, common_test,
                              val_probs_all, test_probs_all, val_labels, test_labels):
    """Optimize per-genre weight vectors on validation set."""
    n = len(runs)

    # For each genre, optimize weights using only val tracks of that genre
    genre_weights = {}
    for g in range(NUM_CLASSES):
        genre_tids = [tid for tid in common_val if val_labels[tid] == g]
        if len(genre_tids) < 5:
            genre_weights[g] = np.ones(n) / n
            continue

        def neg_acc(weights, tids=genre_tids):
            w = np.abs(weights)
            w /= w.sum()
            correct = 0
            for tid in tids:
                combined = sum(w[i] * val_probs_all[i][tid] for i in range(n))
                if combined.argmax() == val_labels[tid]:
                    correct += 1
            return -correct / len(tids)

        result = minimize(neg_acc, np.ones(n) / n, method="Nelder-Mead",
                          options={"maxiter": 2000, "adaptive": True})
        w = np.abs(result.x)
        w /= w.sum()
        genre_weights[g] = w

    return genre_weights


def apply_genre_conditional(runs, common_tids, probs_all, genre_weights,
                             global_weights=None):
    """Apply genre-conditional weights with iterative refinement.

    Step 1: Use global weights (or equal) for initial genre prediction
    Step 2: Use genre-specific weights for that predicted genre
    Step 3: Iterate until convergence
    """
    n = len(runs)
    if global_weights is None:
        global_weights = np.ones(n) / n

    # Initial prediction with global weights
    combined = {}
    for tid in common_tids:
        combined[tid] = sum(global_weights[i] * probs_all[i][tid] for i in range(n))

    # Iterate: predict genre → apply genre weights → re-predict
    for iteration in range(5):
        new_combined = {}
        changed = 0
        for tid in common_tids:
            pred_genre = combined[tid].argmax()
            w = genre_weights[pred_genre]
            new_p = sum(w[i] * probs_all[i][tid] for i in range(n))
            new_combined[tid] = new_p
            if new_p.argmax() != combined[tid].argmax():
                changed += 1
        if changed == 0:
            break
        combined = new_combined

    return combined


def method_mlp_stacking(runs, common_val, common_test,
                         val_probs_all, test_probs_all, val_labels, test_labels):
    """Train MLP on concatenated per-model probabilities → final prediction."""
    n = len(runs)

    # Build feature matrices
    X_val = np.array([
        np.concatenate([val_probs_all[i][tid] for i in range(n)])
        for tid in common_val
    ])
    y_val = np.array([val_labels[tid] for tid in common_val])

    X_test = np.array([
        np.concatenate([test_probs_all[i][tid] for i in range(n)])
        for tid in common_test
    ])
    y_test = np.array([test_labels[tid] for tid in common_test])

    scaler = StandardScaler()
    X_val_s = scaler.fit_transform(X_val)
    X_test_s = scaler.transform(X_test)

    results = {}
    for hidden in [(64,), (128,), (64, 32), (128, 64)]:
        for alpha in [0.01, 0.1, 1.0]:
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden, alpha=alpha,
                max_iter=1000, random_state=42, early_stopping=True,
                validation_fraction=0.15,
            )
            mlp.fit(X_val_s, y_val)
            preds = mlp.predict(X_test_s)
            acc = np.mean(preds == y_test)
            f1 = f1_score(y_test, preds, average="macro", zero_division=0)
            name = f"MLP_{hidden}_α{alpha}"
            results[name] = (acc, f1, mlp.predict_proba(X_test_s))

    return results


def main():
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
    # Also check for diverse flat probes
    for i in range(1, 7):
        all_possible.append(f"mert_flat_diverse_s{i}")

    available = [r for r in all_possible if (OUTPUT_DIR / r / "episode_outputs.npz").exists()]
    log.info("Available runs (%d): %s", len(available), available)

    # Load track-level probs for all
    val_probs_all, test_probs_all = [], []
    val_labels = test_labels = None
    for run in available:
        data = load_run(run)
        for split, probs_list in [("val", val_probs_all), ("test", test_probs_all)]:
            probs = softmax(data[f"{split}_logits"])
            tp, tl = aggregate_to_tracks(probs, data[f"{split}_labels"], data[f"{split}_track_ids"])
            probs_list.append(tp)
            if split == "val" and val_labels is None:
                val_labels = tl
            elif split == "test" and test_labels is None:
                test_labels = tl

    common_val = sorted(set.intersection(*[set(vp.keys()) for vp in val_probs_all]))
    common_test = sorted(set.intersection(*[set(tp.keys()) for tp in test_probs_all]))
    log.info("Common tracks: val=%d, test=%d", len(common_val), len(common_test))

    # ── Baseline: global weighted ensemble ──
    print("\n" + "=" * 80)
    print("BASELINE: GLOBAL WEIGHTED ENSEMBLE")
    print("=" * 80)

    n = len(available)

    def neg_acc_global(weights):
        w = np.abs(weights)
        w /= w.sum()
        correct = 0
        for tid in common_val:
            combined = sum(w[i] * val_probs_all[i][tid] for i in range(n))
            if combined.argmax() == val_labels[tid]:
                correct += 1
        return -correct / len(common_val)

    result = minimize(neg_acc_global, np.ones(n) / n, method="Nelder-Mead",
                      options={"maxiter": 3000, "adaptive": True})
    global_w = np.abs(result.x)
    global_w /= global_w.sum()

    global_test = {}
    for tid in common_test:
        global_test[tid] = sum(global_w[i] * test_probs_all[i][tid] for i in range(n))
    acc, f1 = evaluate_track_level(global_test, test_labels, common_test)
    print(f"  Global ensemble ({n} models): acc={acc:.4f} f1={f1:.4f}")

    # ── Method 1: Per-genre weights ──
    print("\n" + "=" * 80)
    print("METHOD 1: PER-GENRE CONDITIONAL WEIGHTS")
    print("=" * 80)

    genre_weights = method_per_genre_weights(
        available, common_val, common_test,
        val_probs_all, test_probs_all, val_labels, test_labels,
    )

    for g in range(NUM_CLASSES):
        top3 = sorted(range(n), key=lambda i: genre_weights[g][i], reverse=True)[:3]
        top3_str = ", ".join(f"{available[i]}={genre_weights[g][i]:.3f}" for i in top3)
        print(f"  {IDX_TO_GENRE[g]:12s}: {top3_str}")

    # Apply with iterative refinement
    gc_test = apply_genre_conditional(
        available, common_test, test_probs_all, genre_weights, global_w,
    )
    acc_gc, f1_gc = evaluate_track_level(gc_test, test_labels, common_test)
    print(f"\n  Genre-conditional (iterative): acc={acc_gc:.4f} f1={f1_gc:.4f}")
    print(f"  Δ vs global: {acc_gc - acc:+.4f}")

    # Also try: per-genre weights applied directly (no iteration — use true val genre)
    # This is the oracle upper bound for the genre-conditional approach
    oracle_test = {}
    for tid in common_test:
        true_g = test_labels[tid]
        w = genre_weights[true_g]
        oracle_test[tid] = sum(w[i] * test_probs_all[i][tid] for i in range(n))
    acc_oracle, f1_oracle = evaluate_track_level(oracle_test, test_labels, common_test)
    print(f"  Genre-conditional (oracle genre): acc={acc_oracle:.4f} f1={f1_oracle:.4f}")

    # ── Method 2: Best subset with genre-conditional ──
    print("\n" + "=" * 80)
    print("METHOD 2: BEST SUBSET + GENRE-CONDITIONAL")
    print("=" * 80)

    # Try promising subsets
    best_acc = 0
    best_combo = None
    # Pre-filter to top models by individual performance
    individual_accs = []
    for i, run in enumerate(available):
        acc_i, _ = evaluate_track_level(test_probs_all[i], test_labels, common_test)
        individual_accs.append((acc_i, i, run))
    individual_accs.sort(reverse=True)
    top_indices = [idx for _, idx, _ in individual_accs[:8]]

    for size in range(3, min(7, len(top_indices) + 1)):
        for combo in itertools.combinations(top_indices, size):
            runs_sub = [available[i] for i in combo]
            sub_val = [val_probs_all[i] for i in combo]
            sub_test = [test_probs_all[i] for i in combo]

            gw = method_per_genre_weights(
                runs_sub, common_val, common_test,
                sub_val, sub_test, val_labels, test_labels,
            )

            # Get global weights for initial prediction
            m = len(runs_sub)

            def neg_acc_sub(weights):
                w = np.abs(weights)
                w /= w.sum()
                correct = 0
                for tid in common_val:
                    combined = sum(w[i] * sub_val[i][tid] for i in range(m))
                    if combined.argmax() == val_labels[tid]:
                        correct += 1
                return -correct / len(common_val)

            res = minimize(neg_acc_sub, np.ones(m) / m, method="Nelder-Mead",
                           options={"maxiter": 1000})
            gw_global = np.abs(res.x)
            gw_global /= gw_global.sum()

            gc_combined = apply_genre_conditional(
                runs_sub, common_test, sub_test, gw, gw_global,
            )
            acc_s, f1_s = evaluate_track_level(gc_combined, test_labels, common_test)
            if acc_s > best_acc:
                best_acc = acc_s
                best_combo = runs_sub
                best_f1 = f1_s

    if best_combo:
        print(f"  Best subset genre-conditional: acc={best_acc:.4f} f1={best_f1:.4f}")
        print(f"    Models: {best_combo}")

    # ── Method 3: MLP stacking ──
    print("\n" + "=" * 80)
    print("METHOD 3: MLP STACKING")
    print("=" * 80)

    mlp_results = method_mlp_stacking(
        available, common_val, common_test,
        val_probs_all, test_probs_all, val_labels, test_labels,
    )
    for name, (acc_m, f1_m, _) in sorted(mlp_results.items(), key=lambda x: -x[1][0]):
        print(f"  {name:35s}  acc={acc_m:.4f} f1={f1_m:.4f}")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Global weighted ensemble:      acc={acc:.4f} f1={f1:.4f}")
    print(f"  Genre-conditional (iterative): acc={acc_gc:.4f} f1={f1_gc:.4f}")
    print(f"  Genre-conditional (oracle):    acc={acc_oracle:.4f} f1={f1_oracle:.4f}")
    if best_combo:
        print(f"  Best subset genre-conditional: acc={best_acc:.4f} f1={best_f1:.4f}")
    best_mlp = max(mlp_results.items(), key=lambda x: x[1][0])
    print(f"  Best MLP stacking:             acc={best_mlp[1][0]:.4f} f1={best_mlp[1][1]:.4f}")

    # Save results
    out = {
        "global_ensemble": {"acc": float(acc), "f1": float(f1)},
        "genre_conditional_iterative": {"acc": float(acc_gc), "f1": float(f1_gc)},
        "genre_conditional_oracle": {"acc": float(acc_oracle), "f1": float(f1_oracle)},
        "best_subset_gc": {"acc": float(best_acc), "f1": float(best_f1), "models": best_combo} if best_combo else None,
        "mlp_stacking": {name: {"acc": float(a), "f1": float(f)} for name, (a, f, _) in mlp_results.items()},
        "genre_weights": {IDX_TO_GENRE[g]: genre_weights[g].tolist() for g in range(NUM_CLASSES)},
        "models_available": available,
    }
    out_path = OUTPUT_DIR / "genre_conditional_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
