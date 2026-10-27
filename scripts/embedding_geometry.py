"""Embedding space geometry: t-SNE visualization of protocol leakage.

Produces 2×3 grid of t-SNE projections for MERT-95M and CLAP, showing:
  (a) colored by genre
  (b) colored by train/test split under clip-random, same-track clips connected
  (c) colored by train/test split under artist-filtered

Usage:
    python scripts/embedding_geometry.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

ROOT = Path("/Users/ady/Code/clgprjcts/music")
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

MANIFEST = ROOT / "data" / "processed" / "manifest.csv"
EMBED_DIR = ROOT / "data" / "embeddings"

RANDOM_CLIP_SEED = 42
RANDOM_CLIP_FRACTIONS = (0.8, 0.1, 0.1)

FMA_GENRES = ["Electronic", "Experimental", "Folk", "Hip-Hop",
              "Instrumental", "International", "Pop", "Rock"]
GENRE_COLORS = {
    "Electronic": "#1f77b4", "Experimental": "#ff7f0e", "Folk": "#2ca02c",
    "Hip-Hop": "#d62728", "Instrumental": "#9467bd", "International": "#8c564b",
    "Pop": "#e377c2", "Rock": "#7f7f7f",
}


def assign_random_clip_splits(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_CLIP_SEED)
    order = rng.permutation(len(df))
    n_train = int(RANDOM_CLIP_FRACTIONS[0] * len(df))
    n_val = int(RANDOM_CLIP_FRACTIONS[1] * len(df))
    assigned = np.empty(len(df), dtype=object)
    assigned[order[:n_train]] = "train"
    assigned[order[n_train:n_train + n_val]] = "val"
    assigned[order[n_train + n_val:]] = "test"
    out = df.copy()
    out["split"] = assigned
    return out


def load_subsample(embed_name: str, manifest: pd.DataFrame, n_clips: int = 2000,
                   shape_3d: bool = False, seed: int = 42):
    """Load a subsample of clips from train+test, return embeddings, metadata."""
    # Only keep train and test
    df = manifest[manifest["split"].isin(["train", "test"])].reset_index(drop=True)

    # Subsample
    rng = np.random.RandomState(seed)
    if len(df) > n_clips:
        idx = rng.choice(len(df), n_clips, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    embeddings = []
    genres = []
    track_ids = []
    splits = []
    valid_mask = []

    embed_root = EMBED_DIR / embed_name

    for _, row in df.iterrows():
        genre = row["genre"]
        track_id = int(row["track_id"])
        subclip_idx = int(row["subclip_idx"])

        # Find the file (may be in any original split dir)
        npy_path = None
        for orig_split in ("train", "val", "test"):
            candidate = embed_root / orig_split / genre / f"{track_id:06d}.npy"
            if candidate.exists():
                npy_path = candidate
                break

        if npy_path is None:
            valid_mask.append(False)
            embeddings.append(np.zeros(768 if shape_3d else 512))
            genres.append(genre)
            track_ids.append(track_id)
            splits.append(row["split"])
            continue

        arr = np.load(npy_path)
        if subclip_idx >= len(arr):
            valid_mask.append(False)
            embeddings.append(np.zeros(arr.shape[-1] if not shape_3d else 768))
            genres.append(genre)
            track_ids.append(track_id)
            splits.append(row["split"])
            continue

        emb = arr[subclip_idx]
        if shape_3d:
            emb = emb.mean(axis=0)  # (13, 768) -> (768,)

        valid_mask.append(True)
        embeddings.append(emb.astype(np.float32))
        genres.append(genre)
        track_ids.append(track_id)
        splits.append(row["split"])

    valid_mask = np.array(valid_mask)
    embeddings = np.array(embeddings)[valid_mask]
    genres = np.array(genres)[valid_mask]
    track_ids = np.array(track_ids)[valid_mask]
    splits = np.array(splits)[valid_mask]

    return embeddings, genres, track_ids, splits


def draw_same_track_links(ax, coords, track_ids, splits, max_links: int = 200):
    """Draw lines connecting same-track clips across train/test splits."""
    train_mask = splits == "train"
    test_mask = splits == "test"

    # Find track_ids that span both splits
    train_tids = set(track_ids[train_mask])
    test_tids = set(track_ids[test_mask])
    shared_tids = train_tids & test_tids

    links_drawn = 0
    for tid in sorted(shared_tids):
        if links_drawn >= max_links:
            break
        train_pts = coords[(track_ids == tid) & train_mask]
        test_pts = coords[(track_ids == tid) & test_mask]
        # Connect first train point to first test point
        if len(train_pts) > 0 and len(test_pts) > 0:
            ax.plot([train_pts[0, 0], test_pts[0, 0]],
                    [train_pts[0, 1], test_pts[0, 1]],
                    color="black", alpha=0.15, lw=0.5, zorder=1)
            links_drawn += 1

    return len(shared_tids)


def main():
    manifest_official = pd.read_csv(MANIFEST)
    manifest_cr = assign_random_clip_splits(manifest_official.copy())

    backbones = {
        "MERT-95M": {"embed_name": "mert", "shape_3d": True},
        "CLAP": {"embed_name": "clap", "shape_3d": False},
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    results = {}

    for row_idx, (bb_name, bb_cfg) in enumerate(backbones.items()):
        print(f"\nProcessing {bb_name}...")

        # Load AF subsample
        embs_af, genres_af, tids_af, splits_af = load_subsample(
            bb_cfg["embed_name"], manifest_official, n_clips=2000,
            shape_3d=bb_cfg["shape_3d"], seed=42)
        print(f"  AF: {len(embs_af)} clips loaded")

        # Load CR subsample (same tracks where possible)
        embs_cr, genres_cr, tids_cr, splits_cr = load_subsample(
            bb_cfg["embed_name"], manifest_cr, n_clips=2000,
            shape_3d=bb_cfg["shape_3d"], seed=42)
        print(f"  CR: {len(embs_cr)} clips loaded")

        # t-SNE on AF embeddings
        print("  Running t-SNE (AF)...")
        tsne_af = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        coords_af = tsne_af.fit_transform(embs_af)

        # t-SNE on CR embeddings
        print("  Running t-SNE (CR)...")
        tsne_cr = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        coords_cr = tsne_cr.fit_transform(embs_cr)

        # Panel (row, 0): Genre coloring (using AF embeddings)
        ax = axes[row_idx, 0]
        for genre in FMA_GENRES:
            mask = genres_af == genre
            if mask.sum() > 0:
                ax.scatter(coords_af[mask, 0], coords_af[mask, 1], s=8, alpha=0.6,
                           color=GENRE_COLORS[genre], label=genre, zorder=2)
        ax.set_title(f"{bb_name}: Genre Clusters")
        if row_idx == 0:
            ax.legend(fontsize=6, loc="upper right", ncol=2, markerscale=2)
        ax.set_xticks([])
        ax.set_yticks([])

        # Panel (row, 1): CR split coloring + same-track links
        ax = axes[row_idx, 1]
        train_mask_cr = splits_cr == "train"
        test_mask_cr = splits_cr == "test"
        ax.scatter(coords_cr[train_mask_cr, 0], coords_cr[train_mask_cr, 1],
                   s=8, alpha=0.4, color="#4878cf", label="Train", zorder=2)
        ax.scatter(coords_cr[test_mask_cr, 0], coords_cr[test_mask_cr, 1],
                   s=8, alpha=0.4, color="#d65f5f", label="Test", zorder=2)
        n_shared = draw_same_track_links(ax, coords_cr, tids_cr, splits_cr, max_links=150)
        ax.set_title(f"{bb_name}: Clip-Random Split ({n_shared} shared tracks)")
        ax.legend(fontsize=8, loc="upper right", markerscale=2)
        ax.set_xticks([])
        ax.set_yticks([])

        # Panel (row, 2): AF split coloring + same-track links (should be ~0)
        ax = axes[row_idx, 2]
        train_mask_af = splits_af == "train"
        test_mask_af = splits_af == "test"
        ax.scatter(coords_af[train_mask_af, 0], coords_af[train_mask_af, 1],
                   s=8, alpha=0.4, color="#4878cf", label="Train", zorder=2)
        ax.scatter(coords_af[test_mask_af, 0], coords_af[test_mask_af, 1],
                   s=8, alpha=0.4, color="#d65f5f", label="Test", zorder=2)
        n_shared_af = draw_same_track_links(ax, coords_af, tids_af, splits_af, max_links=150)
        ax.set_title(f"{bb_name}: Artist-Filtered Split ({n_shared_af} shared tracks)")
        ax.legend(fontsize=8, loc="upper right", markerscale=2)
        ax.set_xticks([])
        ax.set_yticks([])

        results[bb_name] = {
            "n_clips_af": int(len(embs_af)),
            "n_clips_cr": int(len(embs_cr)),
            "n_shared_tracks_cr": n_shared,
            "n_shared_tracks_af": n_shared_af,
        }

    fig.suptitle("Embedding Space Geometry Under Protocol Variation", fontsize=14, y=0.98)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = FIGURES / f"embedding_tsne.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)

    out_path = OUTPUTS / "embedding_geometry.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
