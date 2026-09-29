"""Hybrid feature fusion: concatenate cached embeddings from multiple backbones
and train a joint MLP classifier.

This is early/mid fusion at the embedding level — captures cross-modal
interactions that probability-level ensembles cannot.

Trains on cached episode_outputs.npz arrays, so no backbone forward pass needed.
Each configuration trains in seconds.

Usage:
    python scripts/train_hybrid_fusion.py
"""
from __future__ import annotations

import itertools
import json
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.config import OUTPUT_DIR
from src.models.common import NUM_CLASSES, IDX_TO_GENRE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class HybridFusionMLP(nn.Module):
    """MLP that classifies from concatenated multi-backbone embeddings."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (512, 256),
                 dropout: float = 0.5, num_classes: int = NUM_CLASSES):
        super().__init__()
        layers = []
        layers.append(nn.LayerNorm(input_dim))
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, num_classes)

    def forward(self, x):
        h = self.encoder(x)
        return self.classifier(h)


def softmax_np(logits):
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def load_run(run_name):
    return dict(np.load(OUTPUT_DIR / run_name / "episode_outputs.npz"))


def aggregate_to_tracks(probs, labels, track_ids):
    track_probs = defaultdict(list)
    track_labels = {}
    for p, l, tid in zip(probs, labels, track_ids):
        tid = int(tid)
        track_probs[tid].append(p)
        track_labels[tid] = int(l)
    return {tid: np.mean(ps, axis=0) for tid, ps in track_probs.items()}, track_labels


def evaluate_track_level_from_logits(logits, labels, track_ids):
    """Aggregate episode logits to track level and evaluate."""
    probs = softmax_np(logits)
    track_probs, track_labels = aggregate_to_tracks(probs, labels, track_ids)
    ordered = sorted(track_probs.keys())
    preds = [track_probs[tid].argmax() for tid in ordered]
    true = [track_labels[tid] for tid in ordered]
    acc = np.mean(np.array(preds) == np.array(true))
    f1 = f1_score(true, preds, average="macro", zero_division=0)
    return acc, f1


def build_joined_features(runs, split):
    """Load and concatenate embeddings from multiple runs, joined by episode order."""
    all_embeddings = []
    labels = None
    track_ids = None

    for run in runs:
        data = load_run(run)
        emb = data[f"{split}_embeddings"]
        all_embeddings.append(emb)
        if labels is None:
            labels = data[f"{split}_labels"]
            track_ids = data[f"{split}_track_ids"]

    return np.concatenate(all_embeddings, axis=1), labels, track_ids


def train_fusion(runs, config, device="cpu"):
    """Train one hybrid fusion model and return test metrics."""
    # Build datasets
    X_train, y_train, tids_train = build_joined_features(runs, "train")
    X_val, y_val, tids_val = build_joined_features(runs, "val")
    X_test, y_test, tids_test = build_joined_features(runs, "test")

    input_dim = X_train.shape[1]
    log.info("Fusion input dim: %d (from %s)", input_dim, [r for r in runs])

    # Create model
    model = HybridFusionMLP(
        input_dim=input_dim,
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
    ).to(device)

    # Data loaders
    def make_loader(X, y, shuffle=False):
        ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        return DataLoader(ds, batch_size=config["batch_size"], shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                   weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.1))

    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        # Train
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Mixup
            if config.get("mixup_alpha", 0) > 0:
                lam = float(torch.distributions.Beta(
                    config["mixup_alpha"], config["mixup_alpha"]).sample())
                idx = torch.randperm(X_batch.size(0), device=device)
                X_batch = lam * X_batch + (1 - lam) * X_batch[idx]
                logits = model(X_batch)
                loss = lam * criterion(logits, y_batch) + (1 - lam) * criterion(logits, y_batch[idx])
            else:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validate
        model.eval()
        val_logits_all = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch.to(device))
                val_logits_all.append(logits.cpu().numpy())
        val_logits_np = np.concatenate(val_logits_all)
        val_acc, val_f1 = evaluate_track_level_from_logits(val_logits_np, y_val, tids_val)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.get("patience", 15):
                break

    # Evaluate on test
    model.load_state_dict(best_state)
    model.eval()
    test_logits_all = []
    with torch.no_grad():
        test_tensor = torch.FloatTensor(X_test).to(device)
        # Process in batches to avoid OOM
        for i in range(0, len(X_test), config["batch_size"]):
            batch = test_tensor[i:i + config["batch_size"]]
            test_logits_all.append(model(batch).cpu().numpy())
    test_logits_np = np.concatenate(test_logits_all)
    test_acc, test_f1 = evaluate_track_level_from_logits(test_logits_np, y_test, tids_test)

    return test_acc, test_f1, best_val_acc


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Available backbone combinations
    all_possible = [
        "cnn_improved", "cnn_tuned_lr",
        "mert_probe_wide", "mert_probe_baseline", "mert_probe_tuned",
        "mert_flat_probe_v1", "mert_flat_probe_v2",
        "mert_deep_probe_v1",
        "mert_large_probe_v1", "mert_large_probe_v2", "mert_large_probe_v3",
        "mert_large_flat_v1",
    ]
    # Also check diverse probes
    for i in range(1, 7):
        all_possible.append(f"mert_flat_diverse_s{i}")

    available = [r for r in all_possible if (OUTPUT_DIR / r / "episode_outputs.npz").exists()]
    log.info("Available runs (%d): %s", len(available), available)

    # Print embedding dims
    for run in available:
        d = load_run(run)
        print(f"  {run:35s}  embed_dim={d['test_embeddings'].shape[1]}")

    # ── Configuration grid ──
    configs = [
        {"hidden_dims": (512, 256), "dropout": 0.5, "lr": 0.001, "weight_decay": 0.001,
         "batch_size": 256, "epochs": 80, "patience": 15, "label_smoothing": 0.1, "mixup_alpha": 0.3},
        {"hidden_dims": (768, 384), "dropout": 0.5, "lr": 0.0005, "weight_decay": 0.001,
         "batch_size": 256, "epochs": 80, "patience": 15, "label_smoothing": 0.1, "mixup_alpha": 0.3},
        {"hidden_dims": (512, 256, 128), "dropout": 0.4, "lr": 0.001, "weight_decay": 0.0005,
         "batch_size": 128, "epochs": 100, "patience": 20, "label_smoothing": 0.1, "mixup_alpha": 0.2},
        {"hidden_dims": (1024, 512), "dropout": 0.6, "lr": 0.0003, "weight_decay": 0.002,
         "batch_size": 256, "epochs": 80, "patience": 15, "label_smoothing": 0.15, "mixup_alpha": 0.4},
    ]

    # ── Backbone combinations to try ──
    combos = [
        # CNN + MERT-95M (cross-backbone)
        ["cnn_improved", "mert_flat_probe_v1"],
        # CNN + MERT-95M + MERT-330M (triple backbone)
        ["cnn_improved", "mert_flat_probe_v1", "mert_large_probe_v1"],
        # All MERT variants (max diversity within MERT)
        ["mert_flat_probe_v1", "mert_large_probe_v1"],
        # CNN + best of each MERT size
        ["cnn_improved", "mert_flat_probe_v1", "mert_large_probe_v1", "mert_probe_wide"],
        # Maximal: CNN + multiple MERT
        ["cnn_improved", "mert_probe_wide", "mert_flat_probe_v1",
         "mert_large_probe_v1", "mert_large_flat_v1"],
    ]

    # Filter to available runs
    combos = [[r for r in combo if r in available] for combo in combos]
    combos = [c for c in combos if len(c) >= 2]
    # Deduplicate
    seen = set()
    unique_combos = []
    for c in combos:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_combos.append(c)

    all_results = []

    for combo in unique_combos:
        print(f"\n{'=' * 80}")
        print(f"BACKBONE COMBO: {combo}")
        print(f"{'=' * 80}")

        for ci, cfg in enumerate(configs):
            # Run 3 seeds for reliability
            accs, f1s = [], []
            for seed in [42, 123, 456]:
                torch.manual_seed(seed)
                np.random.seed(seed)
                acc, f1, val_acc = train_fusion(combo, cfg, device)
                accs.append(acc)
                f1s.append(f1)

            mean_acc = np.mean(accs)
            mean_f1 = np.mean(f1s)
            std_acc = np.std(accs)
            print(f"  Config {ci}: hidden={cfg['hidden_dims']} drop={cfg['dropout']} "
                  f"lr={cfg['lr']}  acc={mean_acc:.4f}±{std_acc:.4f} f1={mean_f1:.4f}")

            all_results.append({
                "backbones": combo,
                "config_idx": ci,
                "hidden_dims": list(cfg["hidden_dims"]),
                "dropout": cfg["dropout"],
                "lr": cfg["lr"],
                "test_acc_mean": float(mean_acc),
                "test_acc_std": float(std_acc),
                "test_f1_mean": float(mean_f1),
                "individual_accs": [float(a) for a in accs],
            })

    # ── Summary ──
    print(f"\n{'=' * 80}")
    print("OVERALL RANKING — HYBRID FUSION")
    print(f"{'=' * 80}")
    all_results.sort(key=lambda x: -x["test_acc_mean"])
    for i, r in enumerate(all_results[:15]):
        print(f"  {i+1:2d}. acc={r['test_acc_mean']:.4f}±{r['test_acc_std']:.4f} "
              f"f1={r['test_f1_mean']:.4f}  {r['backbones']}  h={r['hidden_dims']}")

    # Save
    out_path = OUTPUT_DIR / "hybrid_fusion_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    log.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
