"""Episode-level temporal analysis within FMA-small tracks.

Analyzes how genre classification varies across the 10 three-second
episodes of each 30-second track:
- Per-episode accuracy (which positions are most/least informative?)
- Inter-episode agreement per track (genre ambiguity measure)
- Per-genre temporal profiles

Usage:
    python scripts/temporal_analysis.py
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

GENRES = ["Electronic", "Experimental", "Folk", "Hip-Hop",
          "Instrumental", "International", "Pop", "Rock"]


def load_run(run_dir: Path) -> dict | None:
    npz_path = run_dir / "episode_outputs.npz"
    cfg_path = run_dir / "config.json"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    cfg = json.load(open(cfg_path)) if cfg_path.exists() else {}
    return {
        "name": run_dir.name,
        "split_mode": cfg.get("split_mode", "official"),
        "test_logits": data["test_logits"],
        "test_labels": data["test_labels"],
        "test_track_ids": data["test_track_ids"],
        "test_subclip_idx": data["test_subclip_idx"],
    }


def main():
    # Load all runs
    runs = []
    for run_dir in sorted(OUTPUT_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        run = load_run(run_dir)
        if run is not None and run["split_mode"] == "official":
            runs.append(run)

    print(f"Found {len(runs)} artist-filtered runs with episode outputs")

    # Pick representative runs (best per architecture family)
    # For now, use all runs for aggregation
    best_runs = {
        "MERT-95M flat": "mert_flat_probe_v1",
        "BYOL-A": "byola_probe_v1",
        "PANNs": "panns_probe_v1",
        "CNN": "cnn_improved",
        "MERT-95M probe": "mert_probe_tuned",
        "MERT-330M": "mert_large_probe_v1",
    }

    rep_runs = [r for r in runs if r["name"] in best_runs.values()]
    print(f"Using {len(rep_runs)} representative runs for detailed analysis")

    # === Analysis 1: Accuracy by episode position ===
    print("\n--- Accuracy by Episode Position ---")
    position_data = {}  # name -> {pos: accuracy}

    for run in rep_runs:
        logits = run["test_logits"]
        labels = run["test_labels"]
        subclip = run["test_subclip_idx"]
        preds = logits.argmax(axis=1)

        pos_acc = {}
        for pos in range(10):
            mask = subclip == pos
            if mask.sum() > 0:
                pos_acc[pos] = (preds[mask] == labels[mask]).mean()
        position_data[run["name"]] = pos_acc

    # Print table
    print(f"{'Position':>8s}", end="")
    for name in best_runs.values():
        if name in position_data:
            print(f"  {name[:15]:>15s}", end="")
    print()
    for pos in range(10):
        print(f"  {pos:>6d}", end="")
        for name in best_runs.values():
            if name in position_data:
                print(f"  {position_data[name].get(pos, 0):.3f}          ", end="")
        print()

    # === Analysis 2: Inter-episode agreement per track ===
    print("\n--- Inter-Episode Agreement ---")

    # Use best single model for detailed track analysis
    best = next((r for r in runs if r["name"] == "mert_flat_probe_v1"), runs[0])
    logits = best["test_logits"]
    labels = best["test_labels"]
    track_ids = best["test_track_ids"]
    subclip = best["test_subclip_idx"]
    probs = softmax(logits, axis=1)
    preds = logits.argmax(axis=1)

    # Group episodes by track
    tracks = {}
    for i in range(len(labels)):
        tid = track_ids[i]
        if tid not in tracks:
            tracks[tid] = {"preds": [], "labels": labels[i], "probs": []}
        tracks[tid]["preds"].append(preds[i])
        tracks[tid]["probs"].append(probs[i])

    # Compute agreement for each track
    agreements = []
    entropies = []
    correct_flags = []
    genre_labels = []
    for tid, t in tracks.items():
        ep_preds = np.array(t["preds"])
        mode_pred = np.bincount(ep_preds, minlength=8).argmax()
        agree_frac = (ep_preds == mode_pred).mean()
        agreements.append(agree_frac)

        # Prediction entropy (how spread are episode predictions?)
        mean_probs = np.mean(t["probs"], axis=0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
        entropies.append(entropy)

        # Is the track correctly classified?
        track_pred = np.mean(t["probs"], axis=0).argmax()
        correct_flags.append(track_pred == t["labels"])
        genre_labels.append(t["labels"])

    agreements = np.array(agreements)
    entropies = np.array(entropies)
    correct_flags = np.array(correct_flags)
    genre_labels = np.array(genre_labels)

    print(f"Mean episode agreement: {agreements.mean():.3f} ± {agreements.std():.3f}")
    print(f"Mean prediction entropy: {entropies.mean():.3f} ± {entropies.std():.3f}")
    print(f"\nAgreement distribution:")
    for thresh in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        n = (agreements >= thresh).sum()
        acc = correct_flags[agreements >= thresh].mean() if n > 0 else 0
        print(f"  >= {thresh:.0%} agreement: {n:4d} tracks ({n/len(agreements):.1%}), "
              f"accuracy = {acc:.1%}")

    # Low agreement tracks
    low_agree = agreements < 0.5
    high_agree = agreements >= 0.9
    print(f"\nLow agreement (<50%): {low_agree.sum()} tracks, "
          f"accuracy = {correct_flags[low_agree].mean():.1%}" if low_agree.sum() > 0 else "")
    print(f"High agreement (>=90%): {high_agree.sum()} tracks, "
          f"accuracy = {correct_flags[high_agree].mean():.1%}")

    # Per-genre agreement
    print(f"\nPer-genre mean agreement:")
    for g in range(8):
        mask = genre_labels == g
        if mask.sum() > 0:
            print(f"  {GENRES[g]:15s}: agreement={agreements[mask].mean():.3f}, "
                  f"entropy={entropies[mask].mean():.3f}, "
                  f"accuracy={correct_flags[mask].mean():.3f}")

    # === FIGURES ===

    # --- Figure 1: Accuracy by episode position ---
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(rep_runs)))
    for i, (label, name) in enumerate(best_runs.items()):
        if name in position_data:
            positions = sorted(position_data[name].keys())
            accs = [position_data[name][p] for p in positions]
            ax.plot(positions, accs, "o-", color=colors[i], label=label, linewidth=1.5, markersize=5)

    ax.set_xlabel("Episode Position (0 = first 3s, 9 = last 3s)")
    ax.set_ylabel("Episode-Level Accuracy")
    ax.set_title("Genre Classification Accuracy by Temporal Position")
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"{i*3}-{(i+1)*3}s" for i in range(10)], fontsize=8)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "temporal_accuracy.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "temporal_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved temporal_accuracy figure")

    # --- Figure 2: Agreement histogram ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel a: histogram of agreement
    axes[0].hist(agreements, bins=20, color="#2196F3", alpha=0.7, edgecolor="k")
    axes[0].axvline(agreements.mean(), color="red", linestyle="--", label=f"Mean = {agreements.mean():.2f}")
    axes[0].set_xlabel("Episode Agreement (fraction)")
    axes[0].set_ylabel("Number of Tracks")
    axes[0].set_title("(a) Inter-Episode Agreement Distribution")
    axes[0].legend()

    # Panel b: accuracy conditioned on agreement
    n_bins = 10
    agree_bins = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_centers = []
    bin_sizes = []
    for i in range(n_bins):
        mask = (agreements >= agree_bins[i]) & (agreements < agree_bins[i + 1])
        if i == n_bins - 1:  # include right edge
            mask = (agreements >= agree_bins[i]) & (agreements <= agree_bins[i + 1])
        if mask.sum() > 5:
            bin_accs.append(correct_flags[mask].mean())
            bin_centers.append((agree_bins[i] + agree_bins[i + 1]) / 2)
            bin_sizes.append(mask.sum())

    axes[1].bar(bin_centers, bin_accs, width=0.08, color="#4CAF50", alpha=0.7, edgecolor="k")
    axes[1].set_xlabel("Episode Agreement")
    axes[1].set_ylabel("Track-Level Accuracy")
    axes[1].set_title("(b) Accuracy Conditioned on Episode Agreement")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Best model: {best['name']}", fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "episode_agreement.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "episode_agreement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved episode_agreement figure")

    # --- Figure 3: Per-genre temporal profile ---
    fig, ax = plt.subplots(figsize=(10, 6))
    genre_colors = plt.cm.tab10(np.linspace(0, 1, 8))

    for g in range(8):
        g_mask = labels == g
        pos_accs = []
        for pos in range(10):
            p_mask = (subclip == pos) & g_mask
            if p_mask.sum() > 0:
                pos_accs.append((preds[p_mask] == labels[p_mask]).mean())
            else:
                pos_accs.append(0)
        ax.plot(range(10), pos_accs, "o-", color=genre_colors[g],
                label=GENRES[g], linewidth=1.5, markersize=4)

    ax.set_xlabel("Episode Position (0 = first 3s, 9 = last 3s)")
    ax.set_ylabel("Episode-Level Accuracy")
    ax.set_title(f"Per-Genre Temporal Accuracy Profile ({best['name']})")
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"{i*3}-{(i+1)*3}s" for i in range(10)], fontsize=8)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "genre_temporal_profile.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "genre_temporal_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved genre_temporal_profile figure")

    # Save results
    results = {
        "position_accuracy": {name: {str(k): float(v) for k, v in pa.items()}
                              for name, pa in position_data.items()},
        "agreement_stats": {
            "mean": float(agreements.mean()),
            "std": float(agreements.std()),
            "median": float(np.median(agreements)),
        },
        "per_genre_agreement": {
            GENRES[g]: {
                "mean_agreement": float(agreements[genre_labels == g].mean()),
                "mean_entropy": float(entropies[genre_labels == g].mean()),
                "accuracy": float(correct_flags[genre_labels == g].mean()),
            }
            for g in range(8)
        },
    }
    with open(OUTPUT_DIR / "temporal_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUTPUT_DIR / 'temporal_analysis.json'}")


if __name__ == "__main__":
    main()
