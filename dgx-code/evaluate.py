"""Standalone evaluation script with track-level scoring and confusion matrix.

Usage:
    python evaluate.py --run outputs/mert_lora_330m

Reads config.json + best.pt from the run directory, evaluates on the test split,
and writes eval_metrics.json + confusion matrix plot.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.amp import autocast
from torch.utils.data import DataLoader

from data.dataset import FMA_GENRES, IDX_TO_GENRE, NUM_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def build_model_from_config(cfg: dict):
    """Reconstruct the model from saved config."""
    from train import build_model
    return build_model(cfg)


def build_test_loader(cfg: dict):
    """Build the test dataloader."""
    from train import build_dataset
    test_ds = build_dataset(cfg, "test")
    return DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )


def compute_metrics(labels: list[int], preds: list[int]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    accuracy = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    per_genre = {
        IDX_TO_GENRE[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(NUM_CLASSES)
    }
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(f1.mean()),
        "per_genre": per_genre,
    }


def aggregate_to_tracks(probs, labels, track_ids):
    sums, counts, label_of = {}, {}, {}
    for p, l, t in zip(probs, labels, track_ids):
        if t not in sums:
            sums[t] = np.zeros_like(p)
            counts[t] = 0
            label_of[t] = l
        sums[t] += p
        counts[t] += 1

    ordered = sorted(sums)
    track_labels = [label_of[t] for t in ordered]
    track_preds = [int((sums[t] / counts[t]).argmax()) for t in ordered]
    return track_labels, track_preds


def plot_confusion_matrix(labels, preds, run_name: str, out_dir: Path):
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(FMA_GENRES, rotation=45, ha="right")
    ax.set_yticklabels(FMA_GENRES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {run_name} (track-level)")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im)
    fig.tight_layout()
    path = out_dir / f"{run_name}_confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def evaluate(run_dir: Path):
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model_from_config(cfg)
    # trainable_weights.pt holds only fine-tuned params; frozen ones come from the pretrained checkpoint
    state = torch.load(run_dir / "trainable_weights.pt", map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    loader = build_test_loader(cfg)
    use_amp = torch.cuda.is_available()

    all_probs, all_labels, all_track_ids = [], [], []
    with torch.no_grad():
        for waveforms, labels, track_ids in loader:
            waveforms = waveforms.to(device)
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(waveforms)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())
            all_track_ids.extend(track_ids.tolist())

    # Episode-level
    episode_preds = [int(p.argmax()) for p in all_probs]
    clip_metrics = compute_metrics(all_labels, episode_preds)

    # Track-level
    track_labels, track_preds = aggregate_to_tracks(all_probs, all_labels, all_track_ids)
    track_metrics = compute_metrics(track_labels, track_preds)

    # Confusion matrix
    fig_path = plot_confusion_matrix(track_labels, track_preds, cfg["run_name"], run_dir)

    metrics = {
        "run_name": cfg["run_name"],
        "clip_level": clip_metrics,
        "track_level": track_metrics,
        "n_clips": len(all_labels),
        "n_tracks": len(track_labels),
        "confusion_matrix_plot": str(fig_path),
    }
    with open(run_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("clip-level  (n=%d): accuracy=%.4f macro_f1=%.4f", len(all_labels), clip_metrics["accuracy"], clip_metrics["macro_f1"])
    log.info("track-level (n=%d): accuracy=%.4f macro_f1=%.4f", len(track_labels), track_metrics["accuracy"], track_metrics["macro_f1"])
    log.info("Wrote %s", run_dir / "eval_metrics.json")

    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    evaluate(args.run)


if __name__ == "__main__":
    main()
