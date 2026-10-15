"""Multi-model ensemble over DGX-trained runs.

Loads best.pt from each run, collects test-set probabilities, and combines
them via uniform or weighted averaging. Outputs track-level accuracy/F1.

Usage:
    python ensemble.py --runs outputs/mert_lora_330m outputs/ast_finetune outputs/htsat_genre
    python ensemble.py --runs outputs/mert_lora_330m outputs/mert_lora_95m --weights 0.6 0.4
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from data.dataset import FMA_GENRES, IDX_TO_GENRE, NUM_CLASSES
from evaluate import aggregate_to_tracks, compute_metrics, plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def collect_probs(run_dir: Path, device: torch.device) -> tuple[list, list, list]:
    """Load model from run dir and collect test-set probabilities."""
    from train import build_model, build_dataset

    with open(run_dir / "config.json") as f:
        cfg = json.load(f)

    model = build_model(cfg)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    model.to(device).eval()

    test_ds = build_dataset(cfg, "test")
    loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    all_probs, all_labels, all_track_ids = [], [], []
    with torch.no_grad():
        for waveforms, labels, track_ids in loader:
            waveforms = waveforms.to(device)
            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits = model(waveforms)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.tolist())
            all_track_ids.extend(track_ids.tolist())

    return np.concatenate(all_probs), all_labels, all_track_ids


def ensemble(run_dirs: list[Path], weights: list[float] | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if weights is None:
        weights = [1.0 / len(run_dirs)] * len(run_dirs)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    log.info("Ensembling %d models with weights: %s", len(run_dirs), weights)

    all_run_probs = []
    ref_labels, ref_track_ids = None, None

    for run_dir, w in zip(run_dirs, weights):
        log.info("  Loading %s (weight=%.3f)...", run_dir.name, w)
        probs, labels, track_ids = collect_probs(run_dir, device)
        all_run_probs.append(probs * w)

        if ref_labels is None:
            ref_labels, ref_track_ids = labels, track_ids
        else:
            assert labels == ref_labels, "Label mismatch between runs — different test splits?"

    # Combine probabilities
    combined_probs = sum(all_run_probs)

    # Episode-level
    episode_preds = combined_probs.argmax(axis=1).tolist()
    clip_metrics = compute_metrics(ref_labels, episode_preds)

    # Track-level
    track_labels, track_preds = aggregate_to_tracks(
        combined_probs, ref_labels, ref_track_ids
    )
    track_metrics = compute_metrics(track_labels, track_preds)

    # Output
    run_names = [d.name for d in run_dirs]
    ensemble_name = "ensemble_" + "+".join(run_names[:3])
    out_dir = Path("outputs") / ensemble_name
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(track_labels, track_preds, ensemble_name, out_dir)

    results = {
        "ensemble_name": ensemble_name,
        "runs": [str(d) for d in run_dirs],
        "weights": weights,
        "clip_level": clip_metrics,
        "track_level": track_metrics,
        "n_clips": len(ref_labels),
        "n_tracks": len(track_labels),
    }
    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info("ENSEMBLE track-level: accuracy=%.4f macro_f1=%.4f (%d tracks)",
             track_metrics["accuracy"], track_metrics["macro_f1"], len(track_labels))

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    args = parser.parse_args()
    ensemble(args.runs, args.weights)


if __name__ == "__main__":
    main()
