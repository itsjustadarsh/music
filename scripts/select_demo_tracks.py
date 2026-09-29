#!/usr/bin/env python3
"""Select one real FMA track per genre for the demo files.

For each of the 8 genres, finds the test-set track the model is most confident
about (highest softmax probability for the correct class), then converts it
from mp3 to wav and saves it to demo-files/.

Usage:
    python scripts/select_demo_tracks.py [--run cnn_tuned_lr]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FMA_SMALL_AUDIO_DIR, OUTPUT_DIR, PROCESSED_DIR, SAMPLE_RATE
from src.data.splits import track_id_to_path
from src.models.cnn import GenreCNN
from src.models.common import IDX_TO_GENRE, NUM_CLASSES, get_device
from src.training.dataset import LogMelDataset


DEMO_DIR = Path(__file__).resolve().parent.parent / "demo-files"


def main(run_name: str = "cnn_tuned_lr"):
    run_dir = OUTPUT_DIR / run_name
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)

    if cfg["model"] != "cnn":
        print(f"Only CNN models supported for demo selection, got {cfg['model']}")
        sys.exit(1)

    device = get_device()
    model = GenreCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    model.to(device).eval()

    # Load test set with track IDs
    manifest = PROCESSED_DIR / "manifest.csv"
    test_ds = LogMelDataset(manifest, PROCESSED_DIR, "test", augment=False, return_track_ids=True)

    # Collect per-track probabilities
    from collections import defaultdict
    track_probs = defaultdict(list)
    track_labels = {}

    from torch.utils.data import DataLoader

    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    print(f"Running inference on {len(test_ds)} test episodes...")
    with torch.no_grad():
        for inputs, labels, tids in loader:
            x = inputs.to(device)  # (B, 1, n_mels, time) — channel dim already from dataset
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
            for p, label, tid in zip(probs, labels.tolist(), tids.tolist()):
                track_probs[tid].append(p)
                track_labels[tid] = label

    # Average probabilities per track
    best_per_genre: dict[int, tuple[int, float]] = {}  # genre_idx -> (track_id, confidence)
    for tid, prob_list in track_probs.items():
        avg_probs = np.mean(prob_list, axis=0)
        label = track_labels[tid]
        conf = float(avg_probs[label])

        if label not in best_per_genre or conf > best_per_genre[label][1]:
            best_per_genre[label] = (tid, conf)

    # Convert and save
    DEMO_DIR.mkdir(exist_ok=True)
    print(f"\nSelected demo tracks (model: {run_name}):\n")
    print(f"  {'Genre':<15} {'Track ID':>8} {'Confidence':>10} {'File'}")
    print(f"  {'-----':<15} {'--------':>8} {'----------':>10} {'----'}")

    for genre_idx in sorted(best_per_genre):
        tid, conf = best_per_genre[genre_idx]
        genre = IDX_TO_GENRE[genre_idx]
        genre_safe = genre.lower().replace("-", "_")

        src_path = track_id_to_path(tid, FMA_SMALL_AUDIO_DIR)
        dst_path = DEMO_DIR / f"{genre_safe}_fma_{tid:06d}.wav"

        try:
            y, sr = librosa.load(src_path, sr=SAMPLE_RATE, mono=True)
            # Take the first 10 seconds for a reasonable demo file size
            max_samples = SAMPLE_RATE * 10
            if len(y) > max_samples:
                y = y[:max_samples]
            sf.write(str(dst_path), y, SAMPLE_RATE)
            print(f"  {genre:<15} {tid:>8} {conf:>10.1%} {dst_path.name}")
        except Exception as exc:
            print(f"  {genre:<15} {tid:>8} {conf:>10.1%} FAILED: {exc}")

    print(f"\nDone. Demo files saved to {DEMO_DIR}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="cnn_tuned_lr")
    args = parser.parse_args()
    main(args.run)
