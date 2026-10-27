"""Backbone error diversity analysis.

Computes pairwise error correlation between classifiers to explain:
1. Why uniform ensembling works
2. Why learned fusion (ESCF) fails
3. Which backbones are complementary

Usage:
    python scripts/error_diversity.py
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


def aggregate_to_tracks(logits, labels, track_ids):
    probs = softmax(logits, axis=1)
    sums, counts, label_of = {}, {}, {}
    for p, l, t in zip(probs, labels, track_ids):
        if t not in sums:
            sums[t] = np.zeros_like(p)
            counts[t] = 0
            label_of[t] = l
        sums[t] += p
        counts[t] += 1
    ordered = sorted(sums)
    track_preds = np.array([(sums[t] / counts[t]).argmax() for t in ordered])
    track_labels = np.array([label_of[t] for t in ordered])
    return track_labels, track_preds, ordered


def cohens_kappa(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """Cohen's kappa between two classifiers' correctness vectors."""
    n = len(correct_a)
    # 2x2 contingency: both correct, A only, B only, both wrong
    n11 = (correct_a & correct_b).sum()
    n10 = (correct_a & ~correct_b).sum()
    n01 = (~correct_a & correct_b).sum()
    n00 = (~correct_a & ~correct_b).sum()

    po = (n11 + n00) / n  # observed agreement
    pa = (n11 + n10) / n  # P(A correct)
    pb = (n11 + n01) / n  # P(B correct)
    pe = pa * pb + (1 - pa) * (1 - pb)  # expected agreement

    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def q_statistic(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """Yule's Q statistic — a diversity measure for classifier pairs.
    Q = 1 means identical errors; Q = -1 means maximally diverse; Q = 0 means independent."""
    n11 = (correct_a & correct_b).sum()
    n10 = (correct_a & ~correct_b).sum()
    n01 = (~correct_a & correct_b).sum()
    n00 = (~correct_a & ~correct_b).sum()

    num = n11 * n00 - n01 * n10
    den = n11 * n00 + n01 * n10
    if den == 0:
        return 0.0
    return num / den


def disagreement_measure(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """Fraction of samples where classifiers disagree on correctness."""
    return ((correct_a != correct_b).sum()) / len(correct_a)


def main():
    # Best run per architecture family (artist-filtered only)
    target_runs = {
        "MERT-95M flat": "mert_flat_probe_v1",
        "MERT-95M weighted": "mert_probe_tuned",
        "MERT-95M deep": "mert_deep_probe_v2",
        "MERT-95M attn": "mert_attn_probe_v2",
        "MERT-330M": "mert_large_probe_v1",
        "MERT-330M flat": "mert_large_flat_v1",
        "BYOL-A": "byola_probe_v1",
        "PANNs": "panns_probe_v1",
        "CLAP": "clap_probe_v1",
        "CNN": "cnn_improved",
    }

    # Load runs
    models = {}
    for label, run_name in target_runs.items():
        run_dir = OUTPUT_DIR / run_name
        npz_path = run_dir / "episode_outputs.npz"
        if not npz_path.exists():
            print(f"  SKIP {run_name} (no episode_outputs)")
            continue
        data = np.load(npz_path)
        track_labels, track_preds, track_order = aggregate_to_tracks(
            data["test_logits"], data["test_labels"], data["test_track_ids"]
        )
        correct = (track_preds == track_labels)
        models[label] = {
            "run_name": run_name,
            "correct": correct,
            "preds": track_preds,
            "labels": track_labels,
            "track_ids": track_order,
            "accuracy": float(correct.mean()),
        }

    names = list(models.keys())
    n = len(names)
    print(f"Loaded {n} models for diversity analysis")

    # === Pairwise metrics ===
    kappa_matrix = np.zeros((n, n))
    q_matrix = np.zeros((n, n))
    disagree_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            ci = models[names[i]]["correct"]
            cj = models[names[j]]["correct"]
            kappa_matrix[i, j] = cohens_kappa(ci, cj)
            q_matrix[i, j] = q_statistic(ci, cj)
            disagree_matrix[i, j] = disagreement_measure(ci, cj)

    # Print kappa matrix
    print("\n--- Cohen's Kappa Matrix ---")
    print(f"{'':>20s}", end="")
    for name in names:
        print(f"  {name[:10]:>10s}", end="")
    print()
    for i, name in enumerate(names):
        print(f"  {name[:18]:>18s}", end="")
        for j in range(n):
            print(f"  {kappa_matrix[i, j]:10.3f}", end="")
        print()

    # Mean off-diagonal kappa
    mask = ~np.eye(n, dtype=bool)
    mean_kappa = kappa_matrix[mask].mean()
    mean_q = q_matrix[mask].mean()
    mean_disagree = disagree_matrix[mask].mean()
    print(f"\nMean pairwise kappa: {mean_kappa:.3f}")
    print(f"Mean pairwise Q:     {mean_q:.3f}")
    print(f"Mean disagreement:   {mean_disagree:.3f}")

    # === Backbone family analysis ===
    # Are different backbone families more diverse than same-backbone variants?
    mert_names = [n for n in names if "MERT" in n]
    other_names = [n for n in names if "MERT" not in n]

    within_mert = []
    cross_family = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            k = kappa_matrix[i, j]
            if names[i] in mert_names and names[j] in mert_names:
                within_mert.append(k)
            elif names[i] in other_names or names[j] in other_names:
                cross_family.append(k)

    if within_mert and cross_family:
        print(f"\nWithin-MERT kappa:   {np.mean(within_mert):.3f} ± {np.std(within_mert):.3f} (n={len(within_mert)})")
        print(f"Cross-family kappa:  {np.mean(cross_family):.3f} ± {np.std(cross_family):.3f} (n={len(cross_family)})")

    # === Error overlap analysis ===
    print("\n--- Error Overlap (Best 4 Models) ---")
    top4 = sorted(models.items(), key=lambda x: x[1]["accuracy"], reverse=True)[:4]
    top4_names = [t[0] for t in top4]
    print(f"Top 4: {top4_names}")

    # Tracks that ALL top 4 get wrong
    all_wrong = np.ones(len(top4[0][1]["correct"]), dtype=bool)
    any_wrong = np.zeros(len(top4[0][1]["correct"]), dtype=bool)
    for name, m in top4:
        all_wrong &= ~m["correct"]
        any_wrong |= ~m["correct"]

    print(f"All 4 wrong:     {all_wrong.sum():4d} tracks ({all_wrong.mean():.1%})")
    print(f"At least 1 wrong:{any_wrong.sum():4d} tracks ({any_wrong.mean():.1%})")
    print(f"All 4 correct:   {(~any_wrong).sum():4d} tracks ({(~any_wrong).mean():.1%})")

    # Per-model unique errors (only that model gets it wrong, others right)
    for name, m in top4:
        others_correct = np.ones(len(m["correct"]), dtype=bool)
        for name2, m2 in top4:
            if name2 != name:
                others_correct &= m2["correct"]
        unique_errors = (~m["correct"]) & others_correct
        print(f"  {name:20s}: {unique_errors.sum():3d} unique errors")

    # === FIGURES ===

    # --- Figure 1: Kappa heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im1 = axes[0].imshow(kappa_matrix, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    axes[0].set_xticklabels([n[:12] for n in names], rotation=45, ha="right", fontsize=7)
    axes[0].set_yticklabels([n[:12] for n in names], fontsize=7)
    for i in range(n):
        for j in range(n):
            axes[0].text(j, i, f"{kappa_matrix[i, j]:.2f}", ha="center", va="center", fontsize=6)
    axes[0].set_title("(a) Cohen's Kappa (Error Agreement)")
    fig.colorbar(im1, ax=axes[0], shrink=0.8)

    # Panel b: Q-statistic heatmap
    im2 = axes[1].imshow(q_matrix, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[1].set_xticks(range(n))
    axes[1].set_yticks(range(n))
    axes[1].set_xticklabels([n[:12] for n in names], rotation=45, ha="right", fontsize=7)
    axes[1].set_yticklabels([n[:12] for n in names], fontsize=7)
    for i in range(n):
        for j in range(n):
            axes[1].text(j, i, f"{q_matrix[i, j]:.2f}", ha="center", va="center", fontsize=6)
    axes[1].set_title("(b) Yule's Q (Error Correlation)")
    fig.colorbar(im2, ax=axes[1], shrink=0.8)

    fig.suptitle("Backbone Error Diversity Analysis", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_diversity_heatmap.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "error_diversity_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved error_diversity_heatmap figure")

    # --- Figure 2: Diversity vs accuracy scatter ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n):
        mean_kappa_i = np.mean([kappa_matrix[i, j] for j in range(n) if j != i])
        acc_i = models[names[i]]["accuracy"]
        ax.scatter(mean_kappa_i, acc_i, s=80, edgecolors="k", linewidth=0.5, zorder=5)
        ax.annotate(names[i], (mean_kappa_i, acc_i), fontsize=7,
                   xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Mean Pairwise Kappa (higher = less diverse)")
    ax.set_ylabel("Track-Level Accuracy")
    ax.set_title("Individual Accuracy vs Error Diversity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "diversity_vs_accuracy.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "diversity_vs_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved diversity_vs_accuracy figure")

    # Save results
    save_data = {
        "models": {name: {"run_name": m["run_name"], "accuracy": m["accuracy"]}
                   for name, m in models.items()},
        "kappa_matrix": kappa_matrix.tolist(),
        "q_matrix": q_matrix.tolist(),
        "disagreement_matrix": disagree_matrix.tolist(),
        "model_names": names,
        "summary": {
            "mean_kappa": float(mean_kappa),
            "mean_q": float(mean_q),
            "mean_disagreement": float(mean_disagree),
        },
    }
    with open(OUTPUT_DIR / "error_diversity.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved results to {OUTPUT_DIR / 'error_diversity.json'}")


if __name__ == "__main__":
    main()
