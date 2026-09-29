"""Best ensemble approach: train K hybrid fusion MLPs with different seeds,
then combine with individual model predictions using Nelder-Mead weights.

This produces our best result: ~60.0-60.3% track-level accuracy on
FMA-small with the official artist-filtered split.

Architecture:
    1. Concatenate CNN + MERT-95M embeddings (640 dim)
       plus softmax probabilities from 8 diverse models (64 dim)
       → 704-dim input
    2. Train MLP (768→384) with mixup + label smoothing, K seeds
    3. Average fusion model predictions across seeds
    4. Combine with individual model predictions via Nelder-Mead weights

Usage:
    python scripts/best_ensemble.py [--seeds 10]
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.config import OUTPUT_DIR
from src.models.common import NUM_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def softmax_np(logits):
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def load_run(run_name):
    return dict(np.load(OUTPUT_DIR / run_name / "episode_outputs.npz"))


def to_track_probs(probs, labels, track_ids):
    tp = defaultdict(list)
    tl = {}
    for p, l, tid in zip(probs, labels, track_ids):
        tid = int(tid)
        tp[tid].append(p)
        tl[tid] = int(l)
    return {t: np.mean(ps, axis=0) for t, ps in tp.items()}, tl


class FusionMLP(nn.Module):
    def __init__(self, in_dim, hidden=(768, 384), drop=0.5):
        super().__init__()
        layers = [nn.LayerNorm(in_dim)]
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(drop)]
            prev = h
        self.enc = nn.Sequential(*layers)
        self.cls = nn.Linear(prev, NUM_CLASSES)

    def forward(self, x):
        return self.cls(self.enc(x))


def build_features(embed_runs, prob_runs, split):
    """Concatenate embeddings + softmax probabilities."""
    parts = []
    labels = track_ids = None
    for run in embed_runs:
        d = load_run(run)
        parts.append(d[f"{split}_embeddings"])
        if labels is None:
            labels = d[f"{split}_labels"]
            track_ids = d[f"{split}_track_ids"]
    for run in prob_runs:
        d = load_run(run)
        parts.append(softmax_np(d[f"{split}_logits"]))
    return np.concatenate(parts, axis=1), labels, track_ids


def train_one_fusion(features, y, tids, device, seed):
    """Train one fusion MLP and return track-level probs for val and test."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train = features["train"], y["train"]
    model = FusionMLP(X_train.shape[1], (768, 384), 0.5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 80)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    best_val_acc = 0
    best_state = None
    patience = 0

    for epoch in range(80):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            lam = float(torch.distributions.Beta(0.3, 0.3).sample())
            idx = torch.randperm(xb.size(0), device=device)
            xb = lam * xb + (1 - lam) * xb[idx]
            logits = model(xb)
            loss = lam * criterion(logits, yb) + (1 - lam) * criterion(logits, yb[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vl = model(torch.FloatTensor(features["val"]).to(device)).cpu().numpy()
        vtp, _ = to_track_probs(softmax_np(vl), y["val"], tids["val"])
        ordered = sorted(vtp)
        preds = [vtp[t].argmax() for t in ordered]
        true = [_ for _ in range(len(preds))]  # placeholder
        # Proper eval
        vtl = {}
        for p, l, tid in zip(softmax_np(vl), y["val"], tids["val"]):
            vtl[int(tid)] = int(l)
        true = [vtl[t] for t in ordered]
        preds_arr = np.array(preds)
        true_arr = np.array(true)
        val_acc = np.mean(preds_arr == true_arr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 15:
                break

    model.load_state_dict(best_state)
    model.eval()
    results = {}
    for split in ["val", "test"]:
        with torch.no_grad():
            logits = model(torch.FloatTensor(features[split]).to(device)).cpu().numpy()
        tp, tl = to_track_probs(softmax_np(logits), y[split], tids[split])
        results[split] = (tp, tl)

    test_tp, test_tl = results["test"]
    ordered = sorted(test_tp)
    preds = [test_tp[t].argmax() for t in ordered]
    true = [test_tl[t] for t in ordered]
    test_acc = np.mean(np.array(preds) == np.array(true))
    log.info("Fusion seed %d: test_acc=%.4f", seed, test_acc)

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10, help="Number of fusion seeds")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Backbone embeddings
    embed_runs = ["cnn_improved", "mert_flat_probe_v1"]
    # Probability features from diverse models
    prob_runs = [
        "cnn_improved", "cnn_tuned_lr",
        "mert_probe_wide", "mert_probe_tuned",
        "mert_flat_probe_v1", "mert_flat_probe_v2",
        "mert_large_probe_v1", "mert_large_probe_v3",
    ]
    # Individual models for meta-ensemble
    individual_runs = ["cnn_improved", "mert_probe_wide", "mert_flat_probe_v1", "mert_large_probe_v1"]

    # Check availability
    all_needed = set(embed_runs + prob_runs + individual_runs)
    for run in all_needed:
        if not (OUTPUT_DIR / run / "episode_outputs.npz").exists():
            raise FileNotFoundError(f"Missing cached outputs for {run}")

    # Build feature matrices
    features = {}
    y = {}
    tids = {}
    for split in ["train", "val", "test"]:
        features[split], y[split], tids[split] = build_features(embed_runs, prob_runs, split)
    log.info("Feature dim: %d", features["train"].shape[1])

    # Train K fusion models
    fusion_results = []
    for seed in range(args.seeds):
        res = train_one_fusion(features, y, tids, device, seed)
        fusion_results.append(res)

    # Collect all track-level probs: K fusion + individual models
    all_val_probs = []
    all_test_probs = []
    labels = {"val": None, "test": None}

    for res in fusion_results:
        for split in ["val", "test"]:
            tp, tl = res[split]
            if split == "val":
                all_val_probs.append(tp)
            else:
                all_test_probs.append(tp)
            if labels[split] is None:
                labels[split] = tl

    for run in individual_runs:
        d = load_run(run)
        for split, plist in [("val", all_val_probs), ("test", all_test_probs)]:
            tp, tl = to_track_probs(
                softmax_np(d[f"{split}_logits"]), d[f"{split}_labels"], d[f"{split}_track_ids"]
            )
            plist.append(tp)

    common_val = sorted(set.intersection(*[set(p.keys()) for p in all_val_probs]))
    common_test = sorted(set.intersection(*[set(p.keys()) for p in all_test_probs]))
    n = len(all_val_probs)

    # Method 1: Simple average of fusion models
    fusion_avg_test = {
        tid: np.mean([fusion_results[i]["test"][0][tid] for i in range(args.seeds)], axis=0)
        for tid in common_test
    }
    preds = [fusion_avg_test[t].argmax() for t in common_test]
    true = [labels["test"][t] for t in common_test]
    acc_avg = np.mean(np.array(preds) == np.array(true))
    f1_avg = f1_score(true, preds, average="macro", zero_division=0)
    print(f"\nFusion seed-ensemble ({args.seeds} seeds): acc={acc_avg:.4f} f1={f1_avg:.4f}")

    # Method 2: Nelder-Mead weighted ensemble of all
    def neg_acc(weights):
        w = np.abs(weights)
        w /= w.sum()
        correct = 0
        for tid in common_val:
            combined = sum(w[i] * all_val_probs[i][tid] for i in range(n))
            if combined.argmax() == labels["val"][tid]:
                correct += 1
        return -correct / len(common_val)

    result = minimize(neg_acc, np.ones(n) / n, method="Nelder-Mead",
                      options={"maxiter": 5000, "adaptive": True})
    w = np.abs(result.x)
    w /= w.sum()

    test_combined = {
        tid: sum(w[i] * all_test_probs[i][tid] for i in range(n))
        for tid in common_test
    }
    preds = [test_combined[t].argmax() for t in common_test]
    true = [labels["test"][t] for t in common_test]
    acc_meta = np.mean(np.array(preds) == np.array(true))
    f1_meta = f1_score(true, preds, average="macro", zero_division=0)
    print(f"Meta-ensemble ({n} models): acc={acc_meta:.4f} f1={f1_meta:.4f}")

    model_labels = [f"fusion_s{i}" for i in range(args.seeds)] + individual_runs
    top_w = sorted(zip(model_labels, w), key=lambda x: -x[1])[:8]
    print(f"Top weights: {[(n, f'{v:.3f}') for n, v in top_w]}")

    # Save results
    out = {
        "fusion_seed_ensemble": {"acc": float(acc_avg), "f1": float(f1_avg)},
        "meta_ensemble": {"acc": float(acc_meta), "f1": float(f1_meta)},
        "n_fusion_seeds": args.seeds,
        "embed_runs": embed_runs,
        "prob_runs": prob_runs,
        "individual_runs": individual_runs,
        "n_val_tracks": len(common_val),
        "n_test_tracks": len(common_test),
    }
    out_path = OUTPUT_DIR / "best_ensemble_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
