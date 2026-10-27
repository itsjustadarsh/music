"""Track difficulty profiling across evaluation protocols.

Computes per-track "difficulty" (fraction of models classifying correctly)
under artist-filtered vs clip-random, showing that leakage disproportionately
helps hard tracks — evidence that models recognize recordings, not genres.

Usage:
    python scripts/track_difficulty_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

ROOT = Path("/Users/ady/Code/clgprjcts/music")
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

GENRES = ["Electronic", "Experimental", "Folk", "Hip-Hop",
          "Instrumental", "International", "Pop", "Rock"]

# Paired runs
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


def get_track_correctness(run_name: str) -> dict[int, int] | None:
    """Return {track_id: 1/0 correct} for a run."""
    npz_path = OUTPUTS / run_name / "episode_outputs.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    logits = data["test_logits"]
    labels = data["test_labels"]
    track_ids = data["test_track_ids"]
    probs = softmax(logits, axis=1)

    tracks = {}
    for i in range(len(labels)):
        tid = int(track_ids[i])
        if tid not in tracks:
            tracks[tid] = {"probs": [], "label": int(labels[i])}
        tracks[tid]["probs"].append(probs[i])

    result = {}
    for tid, t in tracks.items():
        mean_prob = np.mean(t["probs"], axis=0)
        result[tid] = int(mean_prob.argmax() == t["label"])
    return result


def main():
    # Collect per-track correctness across models for both protocols
    af_correctness = {}  # {track_id: [0/1, 0/1, ...]}
    cr_correctness = {}
    track_genres = {}
    n_models = 0

    for label, af_run, cr_run in PAIRED_RUNS:
        af = get_track_correctness(af_run)
        cr = get_track_correctness(cr_run)
        if af is None or cr is None:
            print(f"  WARN: missing data for {label}")
            continue

        n_models += 1
        # Get genres from AF run
        if not track_genres:
            data = np.load(OUTPUTS / af_run / "episode_outputs.npz")
            labels = data["test_labels"]
            track_ids = data["test_track_ids"]
            for i in range(len(labels)):
                track_genres[int(track_ids[i])] = int(labels[i])

        for tid, correct in af.items():
            af_correctness.setdefault(tid, []).append(correct)
        for tid, correct in cr.items():
            cr_correctness.setdefault(tid, []).append(correct)

        print(f"  Loaded {label}")

    if n_models == 0:
        print("ERROR: no paired runs found")
        return

    # Compute difficulty scores (fraction of models correct)
    common_tracks = set(af_correctness.keys()) & set(cr_correctness.keys())
    print(f"\n{n_models} models, {len(common_tracks)} common test tracks")

    af_difficulty = {}  # track_id -> fraction correct (higher = easier)
    cr_difficulty = {}
    for tid in common_tracks:
        af_difficulty[tid] = np.mean(af_correctness[tid])
        cr_difficulty[tid] = np.mean(cr_correctness[tid])

    af_scores = np.array([af_difficulty[tid] for tid in sorted(common_tracks)])
    cr_scores = np.array([cr_difficulty[tid] for tid in sorted(common_tracks)])
    genres = np.array([track_genres.get(tid, -1) for tid in sorted(common_tracks)])

    # --- Figure: Track difficulty scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel a: Scatter plot
    ax = axes[0]
    jitter = np.random.RandomState(42).normal(0, 0.015, size=len(af_scores))
    genre_colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for g in range(8):
        mask = genres == g
        ax.scatter(af_scores[mask] + jitter[mask], cr_scores[mask] + jitter[mask],
                   c=[genre_colors[g]], s=12, alpha=0.5, label=GENRES[g])

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="x=y (no change)")
    ax.set_xlabel("Difficulty under Artist-filtered\n(fraction of models correct)")
    ax.set_ylabel("Difficulty under Clip-random\n(fraction of models correct)")
    ax.set_title("(a) Per-Track Difficulty: Honest vs Leaky")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Panel b: Difficulty-binned improvement
    ax = axes[1]
    n_bins = 5
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_improvements = []
    bin_counts = []
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (af_scores >= bin_edges[i]) & (af_scores < bin_edges[i + 1])
        else:
            mask = (af_scores >= bin_edges[i]) & (af_scores <= bin_edges[i + 1])
        if mask.sum() > 0:
            improvement = cr_scores[mask].mean() - af_scores[mask].mean()
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_improvements.append(improvement)
            bin_counts.append(mask.sum())

    colors = ["#d65f5f" if v > 0 else "#4878cf" for v in bin_improvements]
    bars = ax.bar(bin_centers, bin_improvements, width=0.18, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)

    # Track count annotations
    for bc, bi, bn in zip(bin_centers, bin_improvements, bin_counts):
        ax.text(bc, bi + 0.005 if bi > 0 else bi - 0.015,
                f"n={bn}", ha="center", fontsize=8, color="gray")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Honest Difficulty Bin\n(fraction of models correct under artist-filtered)")
    ax.set_ylabel("Mean Improvement from Leakage\n(clip-random − artist-filtered)")
    ax.set_title("(b) Leakage Benefit by Track Difficulty")
    ax.set_xticks(bin_centers)
    ax.set_xticklabels([f"{bin_edges[i]:.1f}–{bin_edges[i+1]:.1f}" for i in range(len(bin_centers))],
                       fontsize=9)

    fig.suptitle(f"Track Difficulty Profiling ({n_models} models, {len(common_tracks)} tracks)",
                 fontsize=12)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = FIGURES / f"track_difficulty.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)

    # Save results
    # Per-genre analysis
    genre_results = {}
    for g in range(8):
        mask = genres == g
        if mask.sum() > 0:
            genre_results[GENRES[g]] = {
                "n_tracks": int(mask.sum()),
                "af_mean_difficulty": float(af_scores[mask].mean()),
                "cr_mean_difficulty": float(cr_scores[mask].mean()),
                "mean_improvement": float((cr_scores[mask] - af_scores[mask]).mean()),
                "hard_track_count_af": int((af_scores[mask] == 0).sum()),
                "hard_track_count_cr": int((cr_scores[mask] == 0).sum()),
            }

    results = {
        "n_models": n_models,
        "n_tracks": len(common_tracks),
        "overall_af_mean": float(af_scores.mean()),
        "overall_cr_mean": float(cr_scores.mean()),
        "overall_improvement": float((cr_scores - af_scores).mean()),
        "hard_tracks_af": int((af_scores == 0).sum()),
        "hard_tracks_cr": int((cr_scores == 0).sum()),
        "easy_tracks_af": int((af_scores == 1).sum()),
        "easy_tracks_cr": int((cr_scores == 1).sum()),
        "per_genre": genre_results,
        "difficulty_bins": [
            {"bin": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
             "improvement": float(imp), "n_tracks": int(cnt)}
            for i, (imp, cnt) in enumerate(zip(bin_improvements, bin_counts))
        ],
    }

    out_path = OUTPUTS / "track_difficulty_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
