"""Representation collapse analysis: within-track embedding similarity as leakage indicator.

For each backbone, groups clips by track_id, computes mean pairwise cosine
similarity within each track, and correlates with protocol inflation.

Usage:
    python scripts/representation_collapse.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine

ROOT = Path("/Users/ady/Code/clgprjcts/music")
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

EMBED_DIR = ROOT / "data" / "embeddings"
MANIFEST = ROOT / "data" / "processed" / "manifest.csv"

# Backbones with their embedding dirs and known inflation values (from existing results)
BACKBONES = {
    "MERT-95M": {
        "embed_dir": "mert",
        "shape": "3d",  # (N, 13, 768) — use mean across layers
        "af_run": "mert_flat_probe_v1",
        "cr_run": "mert_flat_probe_v1_cliprandom",
    },
    "CLAP": {
        "embed_dir": "clap",
        "shape": "2d",  # (N, 512)
        "af_run": "clap_probe_v1",
        "cr_run": "clap_probe_v1_cliprandom",
    },
    "BYOL-A": {
        "embed_dir": "byola",
        "shape": "2d",  # (N, 2048)
        "af_run": "byola_probe_v1",
        "cr_run": "byola_probe_v1_cliprandom",
    },
    "PANNs": {
        "embed_dir": "panns",
        "shape": "2d",  # (N, 2048)
        "af_run": "panns_probe_v1",
        "cr_run": "panns_probe_v1_cliprandom",
    },
}


def load_accuracy(run_name: str) -> float | None:
    for fname in ("eval_metrics.json", "summary.json"):
        path = OUTPUTS / run_name / fname
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            if fname == "eval_metrics.json":
                acc = d.get("track_level", {}).get("accuracy")
            else:
                acc = d.get("test_accuracy")
            if acc is not None:
                return acc
    return None


def load_backbone_embeddings(backbone_cfg: dict) -> dict[int, np.ndarray]:
    """Load all embeddings from disk, grouped by track_id.

    Returns dict: track_id -> (num_episodes, embed_dim)
    """
    embed_root = EMBED_DIR / backbone_cfg["embed_dir"]
    tracks = {}

    for split in ("train", "val", "test"):
        split_dir = embed_root / split
        if not split_dir.exists():
            continue
        for genre_dir in sorted(split_dir.iterdir()):
            if not genre_dir.is_dir():
                continue
            for npy_file in sorted(genre_dir.glob("*.npy")):
                track_id = int(npy_file.stem)
                arr = np.load(npy_file)  # (num_episodes, ...)
                if backbone_cfg["shape"] == "3d":
                    # (N, 13, 768) -> mean across layers -> (N, 768)
                    arr = arr.mean(axis=1)
                tracks[track_id] = arr.astype(np.float32)

    return tracks


def compute_within_track_similarity(tracks: dict[int, np.ndarray], max_tracks: int = 2000) -> np.ndarray:
    """Compute mean pairwise cosine similarity of episodes within each track."""
    sims = []
    track_ids = sorted(tracks.keys())
    if len(track_ids) > max_tracks:
        rng = np.random.RandomState(42)
        track_ids = rng.choice(track_ids, max_tracks, replace=False).tolist()

    for tid in track_ids:
        embs = tracks[tid]  # (num_episodes, dim)
        if len(embs) < 2:
            continue
        # Normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embs_norm = embs / norms
        # Pairwise cosine sim = dot product of normalized vectors
        sim_matrix = embs_norm @ embs_norm.T
        # Take upper triangle (excluding diagonal)
        n = len(embs)
        upper_idx = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[upper_idx]
        sims.append(np.mean(pairwise_sims))

    return np.array(sims)


def compute_between_track_similarity(tracks: dict[int, np.ndarray],
                                      n_pairs: int = 5000) -> np.ndarray:
    """Compute cosine similarity between random pairs of clips from different tracks."""
    track_ids = sorted(tracks.keys())
    rng = np.random.RandomState(42)

    sims = []
    for _ in range(n_pairs):
        t1, t2 = rng.choice(len(track_ids), 2, replace=False)
        embs1 = tracks[track_ids[t1]]
        embs2 = tracks[track_ids[t2]]
        i1 = rng.randint(len(embs1))
        i2 = rng.randint(len(embs2))
        e1 = embs1[i1]
        e2 = embs2[i2]
        cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
        sims.append(cos_sim)

    return np.array(sims)


def main():
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    backbone_names = []
    within_means = []
    between_means = []
    inflations = []
    colors = ["#4878cf", "#b47cc7", "#6acc65", "#d65f5f"]

    for i, (name, cfg) in enumerate(BACKBONES.items()):
        print(f"\nProcessing {name}...")
        tracks = load_backbone_embeddings(cfg)
        print(f"  Loaded {len(tracks)} tracks")

        within_sims = compute_within_track_similarity(tracks)
        between_sims = compute_between_track_similarity(tracks)

        within_mean = float(np.mean(within_sims))
        between_mean = float(np.mean(between_sims))
        collapse_ratio = within_mean / max(between_mean, 1e-8)

        af_acc = load_accuracy(cfg["af_run"])
        cr_acc = load_accuracy(cfg["cr_run"])
        inflation = ((cr_acc - af_acc) * 100) if (af_acc is not None and cr_acc is not None) else None

        print(f"  Within-track sim: {within_mean:.4f}")
        print(f"  Between-track sim: {between_mean:.4f}")
        print(f"  Collapse ratio: {collapse_ratio:.2f}")
        if inflation is not None:
            print(f"  Protocol inflation: {inflation:+.1f} pp")

        backbone_names.append(name)
        within_means.append(within_mean)
        between_means.append(between_mean)
        if inflation is not None:
            inflations.append(inflation)
        else:
            inflations.append(0.0)

        results[name] = {
            "within_track_sim_mean": within_mean,
            "within_track_sim_std": float(np.std(within_sims)),
            "between_track_sim_mean": between_mean,
            "between_track_sim_std": float(np.std(between_sims)),
            "collapse_ratio": collapse_ratio,
            "af_accuracy": af_acc,
            "cr_accuracy": cr_acc,
            "inflation_pp": inflation,
            "n_tracks": len(tracks),
        }

        # Panel a: violin/box of within vs between
        ax = axes[0]
        pos = i * 2
        bp_w = ax.boxplot([within_sims], positions=[pos], widths=0.6,
                          patch_artist=True, showfliers=False)
        bp_b = ax.boxplot([between_sims[:len(within_sims)]], positions=[pos + 0.7], widths=0.6,
                          patch_artist=True, showfliers=False)
        bp_w["boxes"][0].set_facecolor(colors[i])
        bp_w["boxes"][0].set_alpha(0.8)
        bp_b["boxes"][0].set_facecolor(colors[i])
        bp_b["boxes"][0].set_alpha(0.3)

    # Panel a formatting
    ax = axes[0]
    positions = []
    labels = []
    for i, name in enumerate(backbone_names):
        positions.extend([i * 2, i * 2 + 0.7])
        labels.extend([f"{name}\n(within)", f"{name}\n(between)"])
    ax.set_xticks([i * 2 + 0.35 for i in range(len(backbone_names))])
    ax.set_xticklabels(backbone_names, fontsize=9)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("(a) Within-Track vs Between-Track Embedding Similarity")
    ax.grid(True, alpha=0.3, axis="y")
    # Add legend
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor="gray", alpha=0.8), Patch(facecolor="gray", alpha=0.3)],
              ["Within-track", "Between-track"], fontsize=9)

    # Panel b: collapse ratio vs inflation scatter
    ax = axes[1]
    collapse_ratios = [within_means[i] / max(between_means[i], 1e-8) for i in range(len(backbone_names))]
    for i, name in enumerate(backbone_names):
        ax.scatter(collapse_ratios[i], inflations[i], s=120, color=colors[i],
                   zorder=5, edgecolors="black", linewidth=0.8)
        ax.annotate(name, (collapse_ratios[i], inflations[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)

    # Trend line
    if len(collapse_ratios) >= 2:
        z = np.polyfit(collapse_ratios, inflations, 1)
        x_line = np.linspace(min(collapse_ratios) * 0.95, max(collapse_ratios) * 1.05, 50)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="gray", alpha=0.5)
        corr = np.corrcoef(collapse_ratios, inflations)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax.transAxes,
                fontsize=10, va="top")

    ax.set_xlabel("Collapse Ratio (within / between similarity)")
    ax.set_ylabel("Protocol Inflation (pp)")
    ax.set_title("(b) Representation Collapse vs Protocol Inflation")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Representation Collapse as a Leakage Indicator", fontsize=12)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = FIGURES / f"representation_collapse.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)

    out_path = OUTPUTS / "representation_collapse.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
