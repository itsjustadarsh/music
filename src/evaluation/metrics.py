"""Compute full evaluation metrics for a trained run: accuracy, macro-F1,
per-genre precision/recall, and a confusion matrix plot.

Usage:
    python -m src.evaluation.metrics --run mert_flat_probe_v1

Reads outputs/<run>/config.json + best.pt, evaluates on the FMA-small test
split, writes outputs/<run>/eval_metrics.json and docs/figures/<run>_confusion_matrix.png.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.config import DATA_DIR, DOCS_DIR, FMA_GENRES, OUTPUT_DIR, PROCESSED_DIR
from src.models.cnn import GenreCNN
from src.models.common import IDX_TO_GENRE, NUM_CLASSES, get_device
from src.training.dataset import LogMelDataset, MertEmbeddingDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"
FIGURES_DIR = DOCS_DIR / "figures"


def compute_metrics(all_labels: list[int], all_preds: list[int]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    macro_f1 = f1.mean()

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
        "macro_f1": float(macro_f1),
        "per_genre": per_genre,
    }


def aggregate_to_tracks(
    probs: "np.ndarray", labels: list[int], track_ids: list[int]
) -> tuple[list[int], list[int]]:
    """Mean-pool per-episode probabilities within a track -> one prediction per track.

    Published FMA-small results score whole tracks, not the 3s sub-clips we train on,
    so comparing our per-clip numbers against them understates us. We average
    probabilities rather than taking a majority vote: when episodes disagree, the
    vote throws away the confidence that breaks the tie.

    Returns (track_labels, track_preds) aligned by track, ordered by track_id.
    """
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    label_of: dict[int, int] = {}
    for p_row, label, tid in zip(probs, labels, track_ids):
        if tid not in sums:
            sums[tid] = np.zeros_like(p_row)
            counts[tid] = 0
            label_of[tid] = label
        elif label_of[tid] != label:  # a track carries exactly one genre
            raise ValueError(f"Track {tid} has conflicting labels {label_of[tid]} and {label}")
        sums[tid] += p_row
        counts[tid] += 1

    ordered = sorted(sums)
    track_labels = [label_of[t] for t in ordered]
    track_preds = [int((sums[t] / counts[t]).argmax()) for t in ordered]
    return track_labels, track_preds


def plot_confusion_matrix(all_labels: list[int], all_preds: list[int], run_name: str) -> Path:
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(FMA_GENRES, rotation=45, ha="right")
    ax.set_yticklabels(FMA_GENRES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix -- {run_name}")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"{run_name}_confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def evaluate_run(run_name: str) -> dict:
    run_dir = OUTPUT_DIR / run_name
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)

    device = get_device()

    # A run must be scored on the same split it was trained under. Defaulting to
    # "official" here would score a clip-random model against the artist-filtered
    # test set, producing a number that is neither protocol and silently mislabelled
    # as artist-filtered downstream.
    split_mode = cfg.get("split_mode", "official")

    if cfg["model"] == "cnn":
        model = GenreCNN(num_classes=NUM_CLASSES)
        test_ds = LogMelDataset(MANIFEST_PATH, PROCESSED_DIR, "test", augment=False, return_track_ids=True, split_mode=split_mode)
        loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)
    elif cfg["model"] == "resnet":
        from src.models.resnet_cnn import GenreResNet

        model = GenreResNet(
            num_classes=NUM_CLASSES,
            dropout=cfg.get("dropout", 0.4),
            embed_dim=cfg.get("embed_dim", 256),
        )
        test_ds = LogMelDataset(MANIFEST_PATH, PROCESSED_DIR, "test", augment=False, return_track_ids=True, split_mode=split_mode)
        loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)
    elif cfg["model"] == "mert_flat_probe":
        from src.models.mert_flat_probe import MertFlatProbe

        mert_emb_dir = DATA_DIR / "embeddings" / "mert"
        mert_manifest = mert_emb_dir / "manifest.csv"
        model = MertFlatProbe(
            num_classes=NUM_CLASSES,
            proj_size=cfg.get("proj_size", 512),
            dropout=cfg.get("dropout", 0.5),
        )
        test_ds = MertEmbeddingDataset(mert_manifest, mert_emb_dir, "test", return_track_ids=True, split_mode=split_mode)
        loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)
    elif cfg["model"] == "mert_deep_probe":
        from src.models.mert_deep_probe import MertDeepProbe

        mert_emb_dir = DATA_DIR / "embeddings" / "mert"
        mert_manifest = mert_emb_dir / "manifest.csv"
        model = MertDeepProbe(
            num_classes=NUM_CLASSES,
            proj_size=cfg.get("proj_size", 512),
            n_groups=cfg.get("n_groups", 3),
            n_residual_blocks=cfg.get("n_residual_blocks", 2),
            dropout=cfg.get("dropout", 0.4),
        )
        test_ds = MertEmbeddingDataset(mert_manifest, mert_emb_dir, "test", return_track_ids=True, split_mode=split_mode)
        loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)
    elif cfg["model"] == "mert_large_probe":
        from src.models.mert_probe import MertProbe
        from src.models.mert_embed_large import MERT_LARGE_HIDDEN_SIZE, MERT_LARGE_NUM_LAYERS

        mert_large_dir = DATA_DIR / "embeddings" / "mert_large"
        mert_large_manifest = mert_large_dir / "manifest.csv"
        model = MertProbe(
            num_classes=NUM_CLASSES,
            num_layers=MERT_LARGE_NUM_LAYERS,
            hidden_size=MERT_LARGE_HIDDEN_SIZE,
            proj_size=cfg.get("proj_size", 512),
            dropout=cfg.get("dropout", 0.3),
            learn_layer_weights=cfg.get("learn_layer_weights", True),
            layer_index=cfg.get("layer_index"),
        )
        test_ds = MertEmbeddingDataset(mert_large_manifest, mert_large_dir, "test", return_track_ids=True, split_mode=split_mode)
        loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)
    elif cfg["model"] == "mert_large_flat_probe":
        from src.models.mert_flat_probe import MertFlatProbe
        from src.models.mert_embed_large import MERT_LARGE_HIDDEN_SIZE, MERT_LARGE_NUM_LAYERS

        mert_large_dir = DATA_DIR / "embeddings" / "mert_large"
        mert_large_manifest = mert_large_dir / "manifest.csv"
        model = MertFlatProbe(
            num_classes=NUM_CLASSES,
            num_layers=MERT_LARGE_NUM_LAYERS,
            hidden_size=MERT_LARGE_HIDDEN_SIZE,
            proj_size=cfg.get("proj_size", 512),
            dropout=cfg.get("dropout", 0.5),
        )
        test_ds = MertEmbeddingDataset(mert_large_manifest, mert_large_dir, "test", return_track_ids=True, split_mode=split_mode)
        loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)
    elif cfg["model"] == "mert_large_deep_probe":
        from src.models.mert_deep_probe import MertDeepProbe
        from src.models.mert_embed_large import MERT_LARGE_HIDDEN_SIZE, MERT_LARGE_NUM_LAYERS

        mert_large_dir = DATA_DIR / "embeddings" / "mert_large"
        mert_large_manifest = mert_large_dir / "manifest.csv"
        model = MertDeepProbe(
            num_classes=NUM_CLASSES,
            num_layers=MERT_LARGE_NUM_LAYERS,
            hidden_size=MERT_LARGE_HIDDEN_SIZE,
            proj_size=cfg.get("proj_size", 512),
            n_groups=cfg.get("n_groups", 5),
            n_residual_blocks=cfg.get("n_residual_blocks", 2),
            dropout=cfg.get("dropout", 0.4),
        )
        test_ds = MertEmbeddingDataset(mert_large_manifest, mert_large_dir, "test", return_track_ids=True, split_mode=split_mode)
        loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)
    elif cfg["model"] == "densenet":
        from src.models.densenet import GenreDenseNet

        model = GenreDenseNet(
            num_classes=NUM_CLASSES,
            dropout=cfg.get("dropout", 0.3),
            pretrained=cfg.get("pretrained", True),
        )
        test_ds = LogMelDataset(MANIFEST_PATH, PROCESSED_DIR, "test", augment=False, return_track_ids=True, split_mode=split_mode)
        loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)
    elif cfg["model"] == "dual_cnn":
        from src.models.dual_cnn import DualCNN

        model = DualCNN(num_classes=NUM_CLASSES, dropout=cfg.get("dropout", 0.3))
        test_ds = LogMelDataset(MANIFEST_PATH, PROCESSED_DIR, "test", augment=False, return_track_ids=True, split_mode=split_mode)
        loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)
    elif cfg["model"] == "cnn_bigru":
        from src.models.cnn_bigru import CNNBiGRU

        model = CNNBiGRU(
            num_classes=NUM_CLASSES,
            dropout=cfg.get("dropout", 0.3),
            gru_hidden=cfg.get("gru_hidden", 128),
            gru_layers=cfg.get("gru_layers", 2),
        )
        test_ds = LogMelDataset(MANIFEST_PATH, PROCESSED_DIR, "test", augment=False, return_track_ids=True, split_mode=split_mode)
        loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)
    elif cfg["model"] == "capsnet":
        from src.models.capsnet import CapsNet

        model = CapsNet(num_classes=NUM_CLASSES, num_routing=cfg.get("num_routing", 3))
        test_ds = LogMelDataset(MANIFEST_PATH, PROCESSED_DIR, "test", augment=False, return_track_ids=True, split_mode=split_mode)
        loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)
    elif cfg["model"] == "cnn_transformer":
        from src.models.cnn_transformer import CNNTransformer

        model = CNNTransformer(
            num_classes=NUM_CLASSES,
            dropout=cfg.get("dropout", 0.3),
            d_model=cfg.get("d_model", 256),
            nhead=cfg.get("nhead", 4),
            num_transformer_layers=cfg.get("num_transformer_layers", 2),
        )
        test_ds = LogMelDataset(MANIFEST_PATH, PROCESSED_DIR, "test", augment=False, return_track_ids=True, split_mode=split_mode)
        loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)
    elif cfg["model"] == "embedding_probe":
        from src.models.embedding_probe import EmbeddingProbe

        model = EmbeddingProbe(
            num_classes=NUM_CLASSES,
            embed_dim=cfg["embed_dim"],
            hidden_sizes=tuple(cfg.get("hidden_sizes", (512, 256))),
            dropout=cfg.get("dropout", 0.4),
        )
        embed_dir = DATA_DIR / "embeddings" / cfg["embedding_name"]
        test_ds = MertEmbeddingDataset(
            embed_dir / "manifest.csv", embed_dir, "test", return_track_ids=True, split_mode=split_mode
        )
        loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)
    else:
        raise ValueError(f"Unknown model type: {cfg['model']}")

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    model.to(device).eval()

    all_probs, all_preds, all_labels, all_track_ids = [], [], [], []
    with torch.no_grad():
        for inputs, labels, track_ids in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.extend(probs.argmax(axis=1).tolist())
            all_labels.extend(labels.tolist())
            all_track_ids.extend(track_ids.tolist())
    all_probs = np.concatenate(all_probs, axis=0)

    clip_metrics = compute_metrics(all_labels, all_preds)
    track_labels, track_preds = aggregate_to_tracks(all_probs, all_labels, all_track_ids)
    track_metrics = compute_metrics(track_labels, track_preds)

    # The confusion matrix shows the track-level result -- that is the number the
    # paper reports and the one comparable to published work.
    fig_path = plot_confusion_matrix(track_labels, track_preds, run_name)

    metrics = {
        "run_name": run_name,
        "clip_level": clip_metrics,
        "track_level": track_metrics,
        "n_clips": len(all_labels),
        "n_tracks": len(track_labels),
        "confusion_matrix_plot": str(fig_path),
    }
    with open(run_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(
        "clip-level  (n=%d): accuracy=%.4f macro_f1=%.4f",
        len(all_labels), clip_metrics["accuracy"], clip_metrics["macro_f1"],
    )
    log.info(
        "track-level (n=%d): accuracy=%.4f macro_f1=%.4f",
        len(track_labels), track_metrics["accuracy"], track_metrics["macro_f1"],
    )
    log.info("Wrote %s and %s", run_dir / "eval_metrics.json", fig_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run name (subdirectory of outputs/)")
    args = parser.parse_args()
    evaluate_run(args.run)


if __name__ == "__main__":
    main()
