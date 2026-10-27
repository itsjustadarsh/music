"""Multi-genre soft-label analysis using FMA's genres_all metadata.

FMA tracks have multiple genre tags in genres_all, but published papers use
only genre_top. This script measures:
1. How many FMA-small tracks are genuinely multi-genre
2. Whether multi-genre tracks are harder to classify
3. Whether multi-genre tracks are disproportionately "helped" by leakage

Usage:
    python scripts/multigenre_analysis.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax

OUTPUT_DIR = Path("outputs")
FIGURES_DIR = Path("docs/figures")
DATA_DIR = Path("data")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

GENRES = ["Electronic", "Experimental", "Folk", "Hip-Hop",
          "Instrumental", "International", "Pop", "Rock"]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}


def load_fma_metadata():
    """Load FMA tracks metadata with genres_all."""
    tracks_path = DATA_DIR / "raw" / "fma_metadata" / "tracks.csv"
    genres_path = DATA_DIR / "raw" / "fma_metadata" / "genres.csv"

    # Load genre hierarchy
    genres_df = pd.read_csv(genres_path, index_col=0)
    genre_id_to_title = genres_df["title"].to_dict()

    # Load tracks (multi-level header)
    tracks = pd.read_csv(tracks_path, header=[0, 1], index_col=0)

    # Get FMA-small tracks
    subset = tracks[("set", "subset")]
    small_mask = subset == "small"
    small_tracks = tracks[small_mask]

    result = {}
    for tid in small_tracks.index:
        genre_top = small_tracks.loc[tid, ("track", "genre_top")]
        genres_all_str = small_tracks.loc[tid, ("track", "genres_all")]
        genres_str = small_tracks.loc[tid, ("track", "genres")]

        # Parse genres_all (stored as string repr of list)
        try:
            genres_all_ids = ast.literal_eval(str(genres_all_str))
        except (ValueError, SyntaxError):
            genres_all_ids = []

        try:
            genre_ids = ast.literal_eval(str(genres_str))
        except (ValueError, SyntaxError):
            genre_ids = []

        # Convert IDs to genre titles
        genres_all_titles = [genre_id_to_title.get(gid, f"Unknown-{gid}")
                             for gid in genres_all_ids]
        genre_titles = [genre_id_to_title.get(gid, f"Unknown-{gid}")
                        for gid in genre_ids]

        # Count top-level genres that match our 8 classes
        top_genres_in_track = set()
        for title in genres_all_titles:
            if title in GENRE_TO_IDX:
                top_genres_in_track.add(title)

        result[tid] = {
            "genre_top": genre_top,
            "genres_all_titles": genres_all_titles,
            "genre_titles": genre_titles,
            "n_genres_all": len(genres_all_ids),
            "n_top_genres": len(top_genres_in_track),
            "top_genres": sorted(top_genres_in_track),
        }

    return result


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
    track_probs = np.array([sums[t] / counts[t] for t in ordered])
    track_preds = track_probs.argmax(axis=1)
    track_labels = np.array([label_of[t] for t in ordered])
    return track_labels, track_preds, track_probs, ordered


def main():
    # Load metadata
    print("Loading FMA metadata...")
    metadata = load_fma_metadata()
    print(f"Loaded metadata for {len(metadata)} FMA-small tracks")

    # Distribution of genre counts
    n_genres_all = [m["n_genres_all"] for m in metadata.values()]
    n_top_genres = [m["n_top_genres"] for m in metadata.values()]

    print(f"\n--- Genre Tag Distribution ---")
    print(f"genres_all count distribution:")
    for n in sorted(set(n_genres_all)):
        count = n_genres_all.count(n)
        print(f"  {n} genre tags: {count} tracks ({count/len(metadata):.1%})")

    print(f"\nTop-level (our 8 classes) count distribution:")
    for n in sorted(set(n_top_genres)):
        count = n_top_genres.count(n)
        print(f"  {n} top genres: {count} tracks ({count/len(metadata):.1%})")

    # Load best model predictions
    best_run = "mert_flat_probe_v1"
    npz_path = OUTPUT_DIR / best_run / "episode_outputs.npz"
    data = np.load(npz_path)
    track_labels, track_preds, track_probs, track_order = aggregate_to_tracks(
        data["test_logits"], data["test_labels"], data["test_track_ids"]
    )

    # Match predictions to metadata
    single_genre_correct = []
    multi_genre_correct = []
    single_genre_entropy = []
    multi_genre_entropy = []
    single_genre_conf = []
    multi_genre_conf = []

    n_test_multi = 0
    n_test_single = 0
    genre_count_vs_correct = {}  # n_top_genres -> list of (correct, confidence)

    for i, tid in enumerate(track_order):
        if tid not in metadata:
            continue
        m = metadata[tid]
        correct = (track_preds[i] == track_labels[i])
        prob = track_probs[i]
        confidence = prob.max()
        entropy = -np.sum(prob * np.log(prob + 1e-10))

        n_top = m["n_top_genres"]
        genre_count_vs_correct.setdefault(n_top, []).append(
            (correct, confidence, entropy))

        if n_top <= 1:
            single_genre_correct.append(correct)
            single_genre_entropy.append(entropy)
            single_genre_conf.append(confidence)
            n_test_single += 1
        else:
            multi_genre_correct.append(correct)
            multi_genre_entropy.append(entropy)
            multi_genre_conf.append(confidence)
            n_test_multi += 1

    print(f"\n--- Test Set Multi-Genre Analysis ({best_run}) ---")
    print(f"Single top-genre tracks: {n_test_single} ({n_test_single/(n_test_single+n_test_multi):.1%})")
    print(f"Multi top-genre tracks:  {n_test_multi} ({n_test_multi/(n_test_single+n_test_multi):.1%})")

    if single_genre_correct and multi_genre_correct:
        sg_acc = np.mean(single_genre_correct)
        mg_acc = np.mean(multi_genre_correct)
        print(f"\nSingle-genre accuracy:  {sg_acc:.3f} (n={len(single_genre_correct)})")
        print(f"Multi-genre accuracy:   {mg_acc:.3f} (n={len(multi_genre_correct)})")
        print(f"Difference:             {sg_acc - mg_acc:+.3f}")
        print(f"\nSingle-genre mean conf: {np.mean(single_genre_conf):.3f}")
        print(f"Multi-genre mean conf:  {np.mean(multi_genre_conf):.3f}")
        print(f"Single-genre entropy:   {np.mean(single_genre_entropy):.3f}")
        print(f"Multi-genre entropy:    {np.mean(multi_genre_entropy):.3f}")

    # Per genre-count breakdown
    print(f"\n--- Accuracy by Number of Top-Level Genre Tags ---")
    for n_genres in sorted(genre_count_vs_correct.keys()):
        entries = genre_count_vs_correct[n_genres]
        accs = [e[0] for e in entries]
        confs = [e[1] for e in entries]
        if len(accs) >= 5:
            print(f"  {n_genres} genres: acc={np.mean(accs):.3f}, "
                  f"conf={np.mean(confs):.3f}, n={len(accs)}")

    # Most common multi-genre combinations
    print(f"\n--- Most Common Multi-Genre Combinations ---")
    combo_counts = {}
    for m in metadata.values():
        if m["n_top_genres"] > 1:
            combo = tuple(m["top_genres"])
            combo_counts[combo] = combo_counts.get(combo, 0) + 1

    for combo, count in sorted(combo_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {' + '.join(combo):40s}: {count:4d} tracks")

    # === FIGURES ===

    # --- Figure 1: Accuracy by genre count ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel a: accuracy by genre count
    genre_counts = sorted(genre_count_vs_correct.keys())
    gc_accs = []
    gc_sizes = []
    for gc in genre_counts:
        entries = genre_count_vs_correct[gc]
        if len(entries) >= 5:
            gc_accs.append(np.mean([e[0] for e in entries]))
            gc_sizes.append(len(entries))
        else:
            gc_accs.append(None)
            gc_sizes.append(0)

    valid = [(gc, acc, sz) for gc, acc, sz in zip(genre_counts, gc_accs, gc_sizes) if acc is not None]
    if valid:
        gcs, accs, sizes = zip(*valid)
        bars = axes[0].bar(range(len(gcs)), accs, color="#2196F3", alpha=0.7, edgecolor="k")
        axes[0].set_xticks(range(len(gcs)))
        axes[0].set_xticklabels([str(gc) for gc in gcs])
        axes[0].set_xlabel("Number of Top-Level Genre Tags")
        axes[0].set_ylabel("Track-Level Accuracy")
        axes[0].set_title("(a) Accuracy by Genre Ambiguity")
        # Add count labels
        for bar, sz in zip(bars, sizes):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"n={sz}", ha="center", fontsize=7)
        axes[0].grid(True, alpha=0.3, axis="y")

    # Panel b: confidence vs genre count
    if valid:
        gc_confs = []
        for gc in gcs:
            entries = genre_count_vs_correct[gc]
            gc_confs.append(np.mean([e[1] for e in entries]))
        axes[1].bar(range(len(gcs)), gc_confs, color="#FF9800", alpha=0.7, edgecolor="k")
        axes[1].set_xticks(range(len(gcs)))
        axes[1].set_xticklabels([str(gc) for gc in gcs])
        axes[1].set_xlabel("Number of Top-Level Genre Tags")
        axes[1].set_ylabel("Mean Confidence")
        axes[1].set_title("(b) Model Confidence by Genre Ambiguity")
        axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Multi-Genre Analysis ({best_run})", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "multigenre_accuracy.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "multigenre_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved multigenre_accuracy figure")

    # --- Figure 2: Per-genre multi-genre fraction ---
    fig, ax = plt.subplots(figsize=(10, 5))
    genre_multi_frac = {}
    for g in GENRES:
        total = sum(1 for m in metadata.values() if m["genre_top"] == g)
        multi = sum(1 for m in metadata.values() if m["genre_top"] == g and m["n_top_genres"] > 1)
        genre_multi_frac[g] = multi / total if total > 0 else 0

    sorted_genres = sorted(genre_multi_frac.items(), key=lambda x: -x[1])
    gnames, fracs = zip(*sorted_genres)
    bars = ax.bar(range(len(gnames)), fracs, color="#9C27B0", alpha=0.7, edgecolor="k")
    ax.set_xticks(range(len(gnames)))
    ax.set_xticklabels(gnames, rotation=30, ha="right")
    ax.set_ylabel("Fraction with Multiple Top-Level Genre Tags")
    ax.set_title("Genre Ambiguity by Primary Genre")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "multigenre_by_genre.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "multigenre_by_genre.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved multigenre_by_genre figure")

    # Save results
    save_data = {
        "total_tracks": len(metadata),
        "genre_count_distribution": {str(n): n_genres_all.count(n) for n in sorted(set(n_genres_all))},
        "top_genre_count_distribution": {str(n): n_top_genres.count(n) for n in sorted(set(n_top_genres))},
        "accuracy_by_genre_count": {
            str(gc): {
                "accuracy": float(np.mean([e[0] for e in genre_count_vs_correct[gc]])),
                "mean_confidence": float(np.mean([e[1] for e in genre_count_vs_correct[gc]])),
                "n": len(genre_count_vs_correct[gc]),
            }
            for gc in sorted(genre_count_vs_correct.keys())
            if len(genre_count_vs_correct[gc]) >= 5
        },
        "single_vs_multi": {
            "single_genre_accuracy": float(np.mean(single_genre_correct)) if single_genre_correct else None,
            "multi_genre_accuracy": float(np.mean(multi_genre_correct)) if multi_genre_correct else None,
            "n_single": len(single_genre_correct),
            "n_multi": len(multi_genre_correct),
        },
        "top_combos": {" + ".join(combo): count
                       for combo, count in sorted(combo_counts.items(), key=lambda x: -x[1])[:20]},
        "genre_multi_fraction": genre_multi_frac,
    }
    with open(OUTPUT_DIR / "multigenre_analysis.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved results to {OUTPUT_DIR / 'multigenre_analysis.json'}")


if __name__ == "__main__":
    main()
