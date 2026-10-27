"""Confusion matrix difference between artist-filtered and clip-random protocols.

Shows which genre pairs' confusion changes most across protocols, revealing
what leakage actually "fixes" (recording recognition, not genre recognition).

Usage:
    python scripts/confusion_matrix_diff.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from sklearn.metrics import confusion_matrix

ROOT = Path("/Users/ady/Code/clgprjcts/music")
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

GENRES = ["Electronic", "Experimental", "Folk", "Hip-Hop",
          "Instrumental", "International", "Pop", "Rock"]

# Paired runs: (label, artist-filtered run, clip-random run)
PAIRED_RUNS = [
    ("MERT-95M flat", "mert_flat_probe_v1", "mert_flat_probe_v1_cliprandom"),
    ("MERT-95M probe", "mert_probe_tuned", "mert_probe_cliprandom"),
    ("MERT-95M deep", "mert_deep_probe_v1", "mert_deep_probe_v1_cliprandom"),
    ("MERT-95M attn", "mert_attn_probe_v1", "mert_attn_probe_v1_cliprandom"),
    ("MERT-330M", "mert_large_probe_v1", "mert_large_probe_v1_cliprandom"),
    ("BYOL-A", "byola_probe_v1", "byola_probe_v1_cliprandom"),
    ("PANNs", "panns_probe_v1", "panns_probe_v1_cliprandom"),
    ("CLAP", "clap_probe_v1", "clap_probe_v1_cliprandom"),
    ("CNN", "cnn_improved", "cnn_improved_cliprandom"),
]


def load_track_predictions(run_name: str):
    """Load episode_outputs.npz and aggregate to track-level predictions."""
    npz_path = OUTPUTS / run_name / "episode_outputs.npz"
    if not npz_path.exists():
        return None, None
    data = np.load(npz_path)
    logits = data["test_logits"]
    labels = data["test_labels"]
    track_ids = data["test_track_ids"]

    probs = softmax(logits, axis=1)
    unique_tracks = np.unique(track_ids)
    preds, trues = [], []
    for tid in unique_tracks:
        mask = track_ids == tid
        mean_prob = probs[mask].mean(axis=0)
        preds.append(mean_prob.argmax())
        trues.append(labels[mask][0])
    return np.array(preds), np.array(trues)


def main():
    # Accumulate normalized confusion matrices across all paired runs
    af_cms = []
    cr_cms = []

    for label, af_run, cr_run in PAIRED_RUNS:
        af_preds, af_trues = load_track_predictions(af_run)
        cr_preds, cr_trues = load_track_predictions(cr_run)
        if af_preds is None or cr_preds is None:
            print(f"  WARN: missing data for {label}")
            continue

        af_cm = confusion_matrix(af_trues, af_preds, labels=np.arange(8))
        cr_cm = confusion_matrix(cr_trues, cr_preds, labels=np.arange(8))

        # Normalize by row (true class)
        af_cm_norm = af_cm.astype(float) / af_cm.sum(axis=1, keepdims=True).clip(min=1)
        cr_cm_norm = cr_cm.astype(float) / cr_cm.sum(axis=1, keepdims=True).clip(min=1)

        af_cms.append(af_cm_norm)
        cr_cms.append(cr_cm_norm)
        print(f"  Loaded {label}: AF acc={af_cm.trace()/af_cm.sum():.3f}, "
              f"CR acc={cr_cm.trace()/cr_cm.sum():.3f}")

    if not af_cms:
        print("ERROR: no paired runs found")
        return

    # Average across all models
    mean_af = np.mean(af_cms, axis=0)
    mean_cr = np.mean(cr_cms, axis=0)
    diff = mean_cr - mean_af  # positive = clip-random has higher confusion/accuracy

    # --- Figure: Confusion matrix difference ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel a: Artist-filtered mean confusion
    im0 = axes[0].imshow(mean_af, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("(a) Artist-filtered\n(mean across models)")
    for i in range(8):
        for j in range(8):
            color = "white" if mean_af[i, j] > 0.5 else "black"
            axes[0].text(j, i, f"{mean_af[i, j]:.2f}", ha="center", va="center",
                         fontsize=7, color=color)

    # Panel b: Clip-random mean confusion
    im1 = axes[1].imshow(mean_cr, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("(b) Clip-random\n(mean across models)")
    for i in range(8):
        for j in range(8):
            color = "white" if mean_cr[i, j] > 0.5 else "black"
            axes[1].text(j, i, f"{mean_cr[i, j]:.2f}", ha="center", va="center",
                         fontsize=7, color=color)

    # Panel c: Difference (clip-random minus artist-filtered)
    vmax = max(abs(diff).max(), 0.05)
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[2].set_title("(c) Difference\n(clip-random − artist-filtered)")
    for i in range(8):
        for j in range(8):
            axes[2].text(j, i, f"{diff[i, j]:+.2f}", ha="center", va="center",
                         fontsize=7, color="black")

    for ax in axes:
        ax.set_xticks(np.arange(8))
        ax.set_yticks(np.arange(8))
        ax.set_xticklabels(GENRES, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(GENRES, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"Confusion Matrix Shift Under Protocol Leakage\n"
                 f"(averaged over {len(af_cms)} architectures)", fontsize=12)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = FIGURES / f"confusion_diff.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)

    # Save numerical results
    results = {
        "n_models": len(af_cms),
        "mean_af_diagonal": [float(mean_af[i, i]) for i in range(8)],
        "mean_cr_diagonal": [float(mean_cr[i, i]) for i in range(8)],
        "diagonal_inflation": [float(diff[i, i]) for i in range(8)],
        "genres": GENRES,
        "largest_off_diagonal_changes": [],
    }

    # Find largest off-diagonal changes
    for i in range(8):
        for j in range(8):
            if i != j:
                results["largest_off_diagonal_changes"].append({
                    "true": GENRES[i], "pred": GENRES[j],
                    "af": float(mean_af[i, j]), "cr": float(mean_cr[i, j]),
                    "diff": float(diff[i, j]),
                })
    results["largest_off_diagonal_changes"].sort(key=lambda x: abs(x["diff"]), reverse=True)
    results["largest_off_diagonal_changes"] = results["largest_off_diagonal_changes"][:10]

    out_path = OUTPUTS / "confusion_diff_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
