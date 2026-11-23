"""Multi-model ensemble over DGX-trained runs.

Reads each run's test_probs.npz (written by train.py), mean-pools clip
probabilities to track level per run, then combines runs by uniform or
weighted averaging of track probabilities. Track-level combination lets
models with different clip lengths (3s / 10s / full-track) be ensembled.

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

from evaluate import compute_metrics, plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def track_probs(probs: np.ndarray, labels: np.ndarray, track_ids: np.ndarray):
    """Mean clip probabilities per track -> {track_id: (prob, label)}."""
    out = {}
    for t in np.unique(track_ids):
        m = track_ids == t
        out[int(t)] = (probs[m].mean(axis=0), int(labels[m][0]))
    return out


def load_run(run_dir: Path):
    d = np.load(run_dir / "test_probs.npz")
    return track_probs(d["probs"], d["labels"], d["track_ids"])


def ensemble_from_track_probs(per_run: dict[str, dict], weights: list[float] | None = None,
                              out_root: Path = Path("outputs")):
    names = list(per_run)
    if weights is None:
        weights = [1.0 / len(names)] * len(names)
    else:
        weights = [w / sum(weights) for w in weights]
    log.info("Ensembling %s with weights %s", names, [round(w, 3) for w in weights])

    common = sorted(set.intersection(*(set(r) for r in per_run.values())))
    combined = sum(w * np.stack([per_run[n][t][0] for t in common]) for n, w in zip(names, weights))
    labels = [per_run[names[0]][t][1] for t in common]
    preds = combined.argmax(axis=1).tolist()
    track_metrics = compute_metrics(labels, preds)

    ensemble_name = "ensemble_" + "+".join(names)
    out_dir = out_root / ensemble_name
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(labels, preds, ensemble_name, out_dir)
    results = {
        "ensemble_name": ensemble_name,
        "runs": names,
        "weights": weights,
        "track_level": track_metrics,
        "n_tracks": len(common),
    }
    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("ENSEMBLE track-level: accuracy=%.4f macro_f1=%.4f (%d tracks)",
             track_metrics["accuracy"], track_metrics["macro_f1"], len(common))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    args = parser.parse_args()
    per_run = {d.name: load_run(d) for d in args.runs}
    ensemble_from_track_probs(per_run, args.weights)


if __name__ == "__main__":
    main()
