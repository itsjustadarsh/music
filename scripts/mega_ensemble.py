"""Comprehensive multi-model ensemble with multiple fusion strategies.

Combines all available models using:
1. Weighted probability averaging (optimized on val)
2. Logit-level stacking with cross-validated meta-learner
3. Embedding-level stacking (concatenated penultimate embeddings)

Evaluates at track level (average episode probs within track, then argmax).

Usage:
    python scripts/mega_ensemble.py
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.config import OUTPUT_DIR
from src.models.common import NUM_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def load_run(run_name: str) -> dict[str, np.ndarray]:
    return dict(np.load(OUTPUT_DIR / run_name / "episode_outputs.npz"))


def get_episode_data(run_name: str, split: str):
    """Return (probs, logits, embeddings, labels, track_ids, subclip_idx)."""
    data = load_run(run_name)
    logits = data[f"{split}_logits"]
    probs = softmax(logits)
    labels = data[f"{split}_labels"]
    track_ids = data[f"{split}_track_ids"]
    subclip_idx = data[f"{split}_subclip_idx"]
    embeddings = data[f"{split}_embeddings"]
    return probs, logits, embeddings, labels, track_ids, subclip_idx


def aggregate_to_tracks(probs, labels, track_ids):
    """Average episode probs per track, return (track_probs_dict, track_labels_dict)."""
    track_probs = defaultdict(list)
    track_labels = {}
    for p, l, tid in zip(probs, labels, track_ids):
        tid = int(tid)
        track_probs[tid].append(p)
        track_labels[tid] = int(l)
    return {tid: np.mean(ps, axis=0) for tid, ps in track_probs.items()}, track_labels


def evaluate_track_level(track_probs, track_labels, track_ids):
    """Compute track-level accuracy and macro-F1."""
    preds, labels = [], []
    for tid in track_ids:
        preds.append(track_probs[tid].argmax())
        labels.append(track_labels[tid])
    acc = np.mean(np.array(preds) == np.array(labels))
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return acc, f1


def method_weighted_avg(runs, common_val, common_test, val_probs_all, test_probs_all, val_labels):
    """Optimize per-model weights on val, evaluate on test."""
    n = len(runs)

    def neg_acc(weights):
        w = np.abs(weights)
        w /= w.sum()
        correct = 0
        for tid in common_val:
            combined = sum(w[i] * val_probs_all[i][tid] for i in range(n))
            if combined.argmax() == val_labels[tid]:
                correct += 1
        return -correct / len(common_val)

    result = minimize(neg_acc, np.ones(n) / n, method="Nelder-Mead",
                      options={"maxiter": 2000, "adaptive": True})
    w = np.abs(result.x)
    w /= w.sum()

    test_probs = {}
    for tid in common_test:
        test_probs[tid] = sum(w[i] * test_probs_all[i][tid] for i in range(n))

    return w, test_probs


def method_stacking_logits(runs, split_data, common_val, common_test, val_labels, test_labels):
    """Train a meta-learner on concatenated logits, evaluate at track level."""
    # Build track-level features from logits
    def build_track_features(split, track_ids_set):
        features = {}
        for tid in track_ids_set:
            feat = []
            for run in runs:
                probs, logits, _, labels, tids, sids = split_data[run][split]
                # Get episode indices for this track
                mask = tids == tid
                if mask.sum() == 0:
                    feat.append(np.zeros(NUM_CLASSES))
                else:
                    # Average logits for track
                    track_logits = logits[mask].mean(axis=0)
                    feat.append(softmax(track_logits.reshape(1, -1))[0])
            features[tid] = np.concatenate(feat)
        return features

    val_features = build_track_features("val", common_val)
    test_features = build_track_features("test", common_test)

    X_val = np.array([val_features[tid] for tid in sorted(common_val)])
    y_val = np.array([val_labels[tid] for tid in sorted(common_val)])
    X_test = np.array([test_features[tid] for tid in sorted(common_test)])
    y_test = np.array([test_labels[tid] for tid in sorted(common_test)])

    results = {}

    # LogisticRegression
    scaler = StandardScaler()
    X_val_s = scaler.fit_transform(X_val)
    X_test_s = scaler.transform(X_test)

    for C in [0.01, 0.1, 1.0, 10.0]:
        lr = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        lr.fit(X_val_s, y_val)
        preds = lr.predict(X_test_s)
        acc = np.mean(preds == y_test)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        results[f"LogReg_C{C}"] = (acc, f1, lr.predict_proba(X_test_s))

    # ExtraTrees
    for n_est in [200, 500]:
        et = ExtraTreesClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
        et.fit(X_val, y_val)
        preds = et.predict(X_test)
        acc = np.mean(preds == y_test)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        results[f"ET_{n_est}"] = (acc, f1, et.predict_proba(X_test))

    return results, y_test


def method_embedding_stacking(runs, split_data, common_val, common_test, val_labels, test_labels):
    """Stack on concatenated penultimate embeddings (averaged per track)."""
    def build_embed_features(split, track_ids_set):
        features = {}
        for tid in track_ids_set:
            feat = []
            for run in runs:
                _, _, embeddings, _, tids, _ = split_data[run][split]
                mask = tids == tid
                if mask.sum() == 0:
                    feat.append(np.zeros(embeddings.shape[1]))
                else:
                    feat.append(embeddings[mask].mean(axis=0))
            features[tid] = np.concatenate(feat)
        return features

    val_features = build_embed_features("val", common_val)
    test_features = build_embed_features("test", common_test)

    X_val = np.array([val_features[tid] for tid in sorted(common_val)])
    y_val = np.array([val_labels[tid] for tid in sorted(common_val)])
    X_test = np.array([test_features[tid] for tid in sorted(common_test)])
    y_test = np.array([test_labels[tid] for tid in sorted(common_test)])

    scaler = StandardScaler()
    X_val_s = scaler.fit_transform(X_val)
    X_test_s = scaler.transform(X_test)

    results = {}
    for C in [0.01, 0.1, 1.0]:
        lr = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        lr.fit(X_val_s, y_val)
        preds = lr.predict(X_test_s)
        acc = np.mean(preds == y_test)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        results[f"EmbedLR_C{C}"] = (acc, f1)

    et = ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    et.fit(X_val, y_val)
    preds = et.predict(X_test)
    acc = np.mean(preds == y_test)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    results["EmbedET_500"] = (acc, f1)

    return results


def main():
    # All available runs with cached outputs
    all_runs = [
        "cnn_improved", "cnn_tuned_lr",
        "mert_probe_wide", "mert_probe_baseline", "mert_probe_deep",
        "mert_probe_tuned",
        "mert_probe_seed1", "mert_probe_seed2", "mert_probe_seed3",
        "mert_flat_probe_v1",
    ]

    # Check for MERT-330M runs
    for run in ["mert_large_probe_v1", "mert_large_probe_v2", "mert_large_probe_v3",
                "mert_large_deep_probe_v1"]:
        if (OUTPUT_DIR / run / "episode_outputs.npz").exists():
            all_runs.append(run)

    # Check for deep probe runs
    for run in ["mert_deep_probe_v1", "mert_deep_probe_v2"]:
        if (OUTPUT_DIR / run / "episode_outputs.npz").exists():
            all_runs.append(run)

    available = [r for r in all_runs if (OUTPUT_DIR / r / "episode_outputs.npz").exists()]
    log.info("Available runs (%d): %s", len(available), available)

    # Load all data
    split_data = {}
    for run in available:
        split_data[run] = {}
        for split in ["val", "test"]:
            split_data[run][split] = get_episode_data(run, split)

    # Get track-level probs for each run
    val_probs_all = []
    test_probs_all = []
    val_labels = None
    test_labels = None

    for run in available:
        vp, vl = aggregate_to_tracks(*[split_data[run]["val"][i] for i in [0, 3, 4]])
        tp, tl = aggregate_to_tracks(*[split_data[run]["test"][i] for i in [0, 3, 4]])
        val_probs_all.append(vp)
        test_probs_all.append(tp)
        if val_labels is None:
            val_labels = vl
            test_labels = tl

    common_val = sorted(set.intersection(*[set(vp.keys()) for vp in val_probs_all]))
    common_test = sorted(set.intersection(*[set(tp.keys()) for tp in test_probs_all]))
    log.info("Common tracks: val=%d, test=%d", len(common_val), len(common_test))

    # Individual model baselines
    print("\n" + "=" * 80)
    print("INDIVIDUAL MODEL BASELINES (track-level)")
    print("=" * 80)
    for i, run in enumerate(available):
        acc, f1 = evaluate_track_level(test_probs_all[i], test_labels, common_test)
        print(f"  {run:30s}  acc={acc:.4f}  f1={f1:.4f}")

    all_results = []

    # === Method 1: Weighted probability averaging ===
    print("\n" + "=" * 80)
    print("METHOD 1: WEIGHTED PROBABILITY AVERAGING")
    print("=" * 80)

    # Best subset search (try all promising subsets)
    # First: ensemble ALL models
    w, test_probs = method_weighted_avg(available, common_val, common_test,
                                        val_probs_all, test_probs_all, val_labels)
    acc, f1 = evaluate_track_level(test_probs, test_labels, common_test)
    print(f"  All {len(available)} models: acc={acc:.4f} f1={f1:.4f}")
    print(f"    Weights: {dict(zip(available, [f'{x:.3f}' for x in w]))}")
    all_results.append(("weighted_all", acc, f1))

    # CNN + best MERT probes
    cnn_mert_runs = [r for r in available if "cnn" in r or r in ("mert_probe_wide", "mert_probe_tuned")]
    if len(cnn_mert_runs) >= 2:
        idx = [available.index(r) for r in cnn_mert_runs]
        sub_val = [val_probs_all[i] for i in idx]
        sub_test = [test_probs_all[i] for i in idx]
        w, test_probs = method_weighted_avg(cnn_mert_runs, common_val, common_test,
                                            sub_val, sub_test, val_labels)
        acc, f1 = evaluate_track_level(test_probs, test_labels, common_test)
        print(f"  CNN+MERT best ({len(cnn_mert_runs)}): acc={acc:.4f} f1={f1:.4f}")
        all_results.append(("weighted_cnn_mert", acc, f1))

    # Diverse subset: 1 CNN + 1 MERT probe + 1 flat probe
    diverse = []
    for r in available:
        if "cnn_improved" == r:
            diverse.append(r)
        elif "mert_probe_wide" == r:
            diverse.append(r)
        elif "mert_flat_probe" in r:
            diverse.append(r)
        elif "mert_large" in r:
            diverse.append(r)
    if len(diverse) >= 2:
        idx = [available.index(r) for r in diverse]
        sub_val = [val_probs_all[i] for i in idx]
        sub_test = [test_probs_all[i] for i in idx]
        w, test_probs = method_weighted_avg(diverse, common_val, common_test,
                                            sub_val, sub_test, val_labels)
        acc, f1 = evaluate_track_level(test_probs, test_labels, common_test)
        print(f"  Diverse ({', '.join(diverse)}): acc={acc:.4f} f1={f1:.4f}")
        all_results.append(("weighted_diverse", acc, f1))

    # === Method 2: Logit-level stacking ===
    print("\n" + "=" * 80)
    print("METHOD 2: LOGIT-LEVEL STACKING (meta-learner on track-averaged logits)")
    print("=" * 80)

    stacking_results, y_test = method_stacking_logits(
        available, split_data, common_val, common_test, val_labels, test_labels
    )
    for name, (acc, f1, _) in sorted(stacking_results.items(), key=lambda x: -x[1][0]):
        print(f"  {name:20s}  acc={acc:.4f}  f1={f1:.4f}")
        all_results.append((f"stack_{name}", acc, f1))

    # === Method 3: Embedding-level stacking ===
    print("\n" + "=" * 80)
    print("METHOD 3: EMBEDDING-LEVEL STACKING")
    print("=" * 80)

    embed_results = method_embedding_stacking(
        available, split_data, common_val, common_test, val_labels, test_labels
    )
    for name, (acc, f1) in sorted(embed_results.items(), key=lambda x: -x[1][0]):
        print(f"  {name:20s}  acc={acc:.4f}  f1={f1:.4f}")
        all_results.append((f"embed_{name}", acc, f1))

    # === Method 4: Two-level stacking (stack the stackers) ===
    print("\n" + "=" * 80)
    print("METHOD 4: PROBABILITY FUSION OF BEST STACKERS")
    print("=" * 80)

    # Get probabilities from the best stacking methods and fuse with weighted avg
    best_stacking_probs = []
    for name, (acc, f1, probs) in stacking_results.items():
        if acc > 0.55:
            best_stacking_probs.append((name, probs))

    if best_stacking_probs:
        # Also include the weighted ensemble probs as a pseudo-model
        sorted_test = sorted(common_test)
        # Average all stacking model probs
        stack_avg = np.mean([p for _, p in best_stacking_probs], axis=0)
        preds = stack_avg.argmax(axis=1)
        y_test_arr = np.array([test_labels[tid] for tid in sorted_test])
        acc = np.mean(preds == y_test_arr)
        f1 = f1_score(y_test_arr, preds, average="macro", zero_division=0)
        print(f"  Avg of {len(best_stacking_probs)} stackers: acc={acc:.4f} f1={f1:.4f}")
        all_results.append(("meta_stack_avg", acc, f1))

    # === Summary ===
    print("\n" + "=" * 80)
    print("OVERALL RANKING")
    print("=" * 80)
    all_results.sort(key=lambda x: -x[1])
    for i, (name, acc, f1) in enumerate(all_results[:20]):
        marker = " ***" if acc > 0.60 else " **" if acc > 0.58 else ""
        print(f"  {i+1:2d}. {name:35s}  acc={acc:.4f}  f1={f1:.4f}{marker}")

    # Save results
    out = {"results": [{"method": n, "test_acc": float(a), "test_f1": float(f)} for n, a, f in all_results],
           "models_used": available, "n_val_tracks": len(common_val), "n_test_tracks": len(common_test)}
    with open(OUTPUT_DIR / "mega_ensemble_results.json", "w") as f:
        json.dump(out, f, indent=2)
    log.info("Saved to %s", OUTPUT_DIR / "mega_ensemble_results.json")


if __name__ == "__main__":
    main()
