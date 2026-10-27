"""Confidence calibration analysis across evaluation protocols.

Computes ECE (Expected Calibration Error), Brier score, and generates
reliability diagrams for all runs with cached episode_outputs.npz.

Key finding: does protocol leakage inflate model confidence, not just accuracy?

Usage:
    python scripts/calibration_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

OUTPUT_DIR = Path("outputs")
FIGURES_DIR = Path("docs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 8
GENRES = ["Electronic", "Experimental", "Folk", "Hip-Hop",
          "Instrumental", "International", "Pop", "Rock"]


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Expected Calibration Error with binned statistics for reliability diagrams."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accs[i] = accuracies[mask].mean()
            bin_confs[i] = confidences[mask].mean()
            bin_counts[i] = mask.sum()

    ece = np.sum(bin_counts / len(labels) * np.abs(bin_accs - bin_confs))
    return ece, bin_accs, bin_confs, bin_counts


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score."""
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1
    return ((probs - one_hot) ** 2).sum(axis=1).mean()


def aggregate_to_tracks(probs, labels, track_ids):
    """Mean-pool episode probs to track level."""
    sums, counts, label_of = {}, {}, {}
    for p, l, t in zip(probs, labels, track_ids):
        if t not in sums:
            sums[t] = np.zeros_like(p)
            counts[t] = 0
            label_of[t] = l
        sums[t] += p
        counts[t] += 1
    ordered = sorted(sums)
    track_probs = np.array([sums[t] / counts[t] for t in ordered])
    track_labels = np.array([label_of[t] for t in ordered])
    return track_probs, track_labels


def temperature_scale(logits: np.ndarray, labels: np.ndarray, n_iter: int = 50) -> float:
    """Find optimal temperature via grid search on validation set."""
    best_t, best_ece = 1.0, float("inf")
    for t in np.linspace(0.1, 5.0, n_iter):
        scaled_probs = softmax(logits / t, axis=1)
        ece, _, _, _ = compute_ece(scaled_probs, labels)
        if ece < best_ece:
            best_ece = ece
            best_t = t
    return best_t


def load_run(run_dir: Path) -> dict | None:
    """Load episode outputs and config for a run."""
    npz_path = run_dir / "episode_outputs.npz"
    cfg_path = run_dir / "config.json"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    cfg = json.load(open(cfg_path)) if cfg_path.exists() else {}
    split_mode = cfg.get("split_mode", "official")
    return {
        "name": run_dir.name,
        "split_mode": split_mode,
        "test_logits": data["test_logits"],
        "test_labels": data["test_labels"],
        "test_track_ids": data["test_track_ids"],
        "val_logits": data["val_logits"],
        "val_labels": data["val_labels"],
    }


def main():
    # Discover all runs with episode_outputs
    runs = []
    for run_dir in sorted(OUTPUT_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        run = load_run(run_dir)
        if run is not None:
            runs.append(run)

    print(f"Found {len(runs)} runs with episode outputs")

    # Compute calibration metrics for each run
    results = []
    for run in runs:
        logits = run["test_logits"]
        labels = run["test_labels"]
        track_ids = run["test_track_ids"]
        probs = softmax(logits, axis=1)

        # Episode-level calibration
        ep_ece, ep_accs, ep_confs, ep_counts = compute_ece(probs, labels)
        ep_brier = compute_brier(probs, labels)

        # Track-level calibration
        track_probs, track_labels = aggregate_to_tracks(probs, labels, track_ids)
        tr_ece, tr_accs, tr_confs, tr_counts = compute_ece(track_probs, track_labels)
        tr_brier = compute_brier(track_probs, track_labels)

        # Temperature scaling (fit on val, report on test)
        val_logits = run["val_logits"]
        val_labels = run["val_labels"]
        opt_temp = temperature_scale(val_logits, val_labels)
        scaled_probs = softmax(logits / opt_temp, axis=1)
        scaled_track_probs, _ = aggregate_to_tracks(scaled_probs, labels, track_ids)
        tr_ece_scaled, _, _, _ = compute_ece(scaled_track_probs, track_labels)

        # Mean confidence
        mean_conf = probs.max(axis=1).mean()
        track_mean_conf = track_probs.max(axis=1).mean()
        track_acc = (track_probs.argmax(axis=1) == track_labels).mean()

        results.append({
            "name": run["name"],
            "split_mode": run["split_mode"],
            "track_accuracy": float(track_acc),
            "track_ece": float(tr_ece),
            "track_brier": float(tr_brier),
            "track_ece_scaled": float(tr_ece_scaled),
            "opt_temperature": float(opt_temp),
            "mean_track_confidence": float(track_mean_conf),
            "episode_ece": float(ep_ece),
            "episode_brier": float(ep_brier),
            "mean_episode_confidence": float(mean_conf),
            "reliability_accs": tr_accs.tolist(),
            "reliability_confs": tr_confs.tolist(),
            "reliability_counts": tr_counts.tolist(),
        })

        print(f"  {run['name']:45s} split={run['split_mode']:12s} "
              f"acc={track_acc:.3f} ECE={tr_ece:.4f} Brier={tr_brier:.4f} "
              f"conf={track_mean_conf:.3f} T*={opt_temp:.2f}")

    # Save full results
    out_path = OUTPUT_DIR / "calibration_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # === FIGURES ===

    # Group by split mode for comparison
    by_split = {}
    for r in results:
        by_split.setdefault(r["split_mode"], []).append(r)

    # --- Figure 1: ECE comparison bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel a: ECE by split mode (best run per architecture)
    split_colors = {"official": "#2196F3", "random_track": "#FF9800", "random_clip": "#F44336"}
    split_labels = {"official": "Artist-filtered", "random_track": "Random-track", "random_clip": "Clip-random"}

    for split_mode, split_runs in sorted(by_split.items()):
        eces = [r["track_ece"] for r in split_runs]
        accs = [r["track_accuracy"] for r in split_runs]
        color = split_colors.get(split_mode, "gray")
        label = split_labels.get(split_mode, split_mode)
        axes[0].scatter(accs, eces, c=color, label=label, alpha=0.7, s=60, edgecolors="k", linewidth=0.5)

    axes[0].set_xlabel("Track-Level Accuracy")
    axes[0].set_ylabel("Expected Calibration Error (ECE)")
    axes[0].set_title("(a) Accuracy vs Calibration by Protocol")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel b: Mean confidence vs accuracy
    for split_mode, split_runs in sorted(by_split.items()):
        confs = [r["mean_track_confidence"] for r in split_runs]
        accs = [r["track_accuracy"] for r in split_runs]
        color = split_colors.get(split_mode, "gray")
        label = split_labels.get(split_mode, split_mode)
        axes[1].scatter(accs, confs, c=color, label=label, alpha=0.7, s=60, edgecolors="k", linewidth=0.5)

    # Perfect calibration line
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    axes[1].set_xlabel("Track-Level Accuracy")
    axes[1].set_ylabel("Mean Confidence")
    axes[1].set_title("(b) Confidence vs Accuracy by Protocol")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_by_protocol.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "calibration_by_protocol.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved calibration_by_protocol figure")

    # --- Figure 2: Reliability diagrams for representative models ---
    # Pick best artist-filtered run for reliability diagram
    official_runs = [r for r in results if r["split_mode"] == "official"]
    if official_runs:
        # Sort by accuracy, pick top 3
        official_runs.sort(key=lambda r: r["track_accuracy"], reverse=True)
        top_runs = official_runs[:min(4, len(official_runs))]

        fig, axes = plt.subplots(1, len(top_runs), figsize=(4 * len(top_runs), 4))
        if len(top_runs) == 1:
            axes = [axes]

        for ax, r in zip(axes, top_runs):
            accs = np.array(r["reliability_accs"])
            confs = np.array(r["reliability_confs"])
            counts = np.array(r["reliability_counts"])
            mask = counts > 0

            bar_width = 1.0 / 15
            bin_centers = np.linspace(bar_width / 2, 1 - bar_width / 2, 15)

            ax.bar(bin_centers[mask], accs[mask], width=bar_width, alpha=0.7,
                   color="#2196F3", edgecolor="k", linewidth=0.5, label="Accuracy")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            short_name = r["name"].replace("mert_", "M-").replace("_probe", "P")
            ax.set_title(f"{short_name}\nECE={r['track_ece']:.3f}", fontsize=9)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            ax.grid(True, alpha=0.2)

        fig.suptitle("Reliability Diagrams (Artist-Filtered, Track-Level)", fontsize=12)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "reliability_diagrams.pdf", dpi=150, bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "reliability_diagrams.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved reliability_diagrams figure")

    # --- Figure 3: Temperature scaling summary ---
    if official_runs:
        fig, ax = plt.subplots(figsize=(10, 5))
        names = [r["name"] for r in official_runs[:15]]
        ece_before = [r["track_ece"] for r in official_runs[:15]]
        ece_after = [r["track_ece_scaled"] for r in official_runs[:15]]
        temps = [r["opt_temperature"] for r in official_runs[:15]]

        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w/2, ece_before, w, label="Before temp. scaling", color="#F44336", alpha=0.7)
        ax.bar(x + w/2, ece_after, w, label="After temp. scaling", color="#4CAF50", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("mert_", "").replace("_probe", "P")[:20] for n in names],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("ECE")
        ax.set_title("Temperature Scaling Effect (Artist-Filtered Runs)")
        ax.legend()
        ax.grid(True, alpha=0.2, axis="y")

        # Add temperature values as annotations
        for i, t in enumerate(temps):
            ax.annotate(f"T={t:.1f}", (x[i], max(ece_before[i], ece_after[i]) + 0.005),
                       ha="center", fontsize=6, color="gray")

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "temperature_scaling.pdf", dpi=150, bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "temperature_scaling.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved temperature_scaling figure")

    # --- Summary statistics ---
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    for split_mode, split_runs in sorted(by_split.items()):
        eces = [r["track_ece"] for r in split_runs]
        briers = [r["track_brier"] for r in split_runs]
        confs = [r["mean_track_confidence"] for r in split_runs]
        accs = [r["track_accuracy"] for r in split_runs]
        overconf = [r["mean_track_confidence"] - r["track_accuracy"] for r in split_runs]
        print(f"\n  {split_labels.get(split_mode, split_mode)} ({len(split_runs)} runs):")
        print(f"    ECE:            {np.mean(eces):.4f} ± {np.std(eces):.4f}")
        print(f"    Brier:          {np.mean(briers):.4f} ± {np.std(briers):.4f}")
        print(f"    Mean confidence:{np.mean(confs):.4f} ± {np.std(confs):.4f}")
        print(f"    Mean accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"    Overconfidence: {np.mean(overconf):.4f} ± {np.std(overconf):.4f}")


if __name__ == "__main__":
    main()
