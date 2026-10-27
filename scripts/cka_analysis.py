"""Centered Kernel Alignment (CKA) analysis between backbone representations.

Computes:
1. Inter-model CKA: pairwise similarity between MERT, BYOL-A, PANNs embeddings
2. Layer-wise CKA within MERT-95M: 13x13 heatmap showing layer redundancy
3. (Optional) CKA between leaked vs honest probe representations

Usage:
    python scripts/cka_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
FIGURES_DIR = Path("docs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two representation matrices.

    X: (n_samples, p) — representations from model A
    Y: (n_samples, q) — representations from model B
    Returns CKA similarity in [0, 1].

    Uses the efficient formulation from Kornblith et al. (2019):
    CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    with linear kernels K = XX^T, L = YY^T.
    """
    n = X.shape[0]
    assert Y.shape[0] == n

    # Cast to float64 to avoid overflow with large embedding dims
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    # Center the representations
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Numerically stable CKA via Frobenius norm formulation
    # CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    YtX = Y.T @ X
    cross = np.linalg.norm(YtX, "fro") ** 2

    XtX = X.T @ X
    self_x = np.linalg.norm(XtX, "fro")

    YtY = Y.T @ Y
    self_y = np.linalg.norm(YtY, "fro")

    return float(cross / (self_x * self_y + 1e-10))


def load_backbone_embeddings(backbone: str, split: str = "test", max_tracks: int = 800) -> tuple[np.ndarray, list]:
    """Load mean-pooled embeddings for a backbone. Returns (n_tracks, embed_dim).

    FMA embeddings are stored as one .npy per TRACK with shape (n_episodes, ...).
    The manifest has one row per episode, all pointing to the same .npy file.
    We load each unique .npy once and mean-pool across episodes.
    """
    import pandas as pd
    emb_dir = DATA_DIR / "embeddings" / backbone
    manifest = pd.read_csv(emb_dir / "manifest.csv")
    manifest = manifest[manifest["split"] == split]

    # Get unique (track_id, embedding_path) pairs
    unique_tracks = manifest.drop_duplicates("track_id")[["track_id", "embedding_path"]].sort_values("track_id")

    track_ids = []
    pooled = []
    for _, row in unique_tracks.iterrows():
        tid = int(row["track_id"])
        path = emb_dir / row["embedding_path"]
        emb = np.load(path)  # (n_episodes, ...) for all backbones
        # Mean-pool across episodes (axis 0)
        pooled.append(emb.mean(axis=0))
        track_ids.append(tid)
        if len(track_ids) >= max_tracks:
            break

    return np.array(pooled), track_ids


def load_mert_layer_embeddings(split: str = "test", max_tracks: int = 800) -> tuple[np.ndarray, list]:
    """Load per-layer MERT embeddings. Returns (n_tracks, n_layers, hidden_dim)."""
    import pandas as pd
    emb_dir = DATA_DIR / "embeddings" / "mert"
    manifest = pd.read_csv(emb_dir / "manifest.csv")
    manifest = manifest[manifest["split"] == split]

    unique_tracks = manifest.drop_duplicates("track_id")[["track_id", "embedding_path"]].sort_values("track_id")

    track_ids = []
    pooled = []
    for _, row in unique_tracks.iterrows():
        tid = int(row["track_id"])
        path = emb_dir / row["embedding_path"]
        emb = np.load(path)  # (n_episodes, n_layers, hidden_dim)
        pooled.append(emb.mean(axis=0))  # (n_layers, hidden_dim)
        track_ids.append(tid)
        if len(track_ids) >= max_tracks:
            break

    return np.array(pooled), track_ids


def main():
    print("=" * 60)
    print("CKA Representation Analysis")
    print("=" * 60)

    # === Part 1: Inter-model CKA ===
    print("\n--- Loading backbone embeddings ---")

    backbones = {}
    for name in ["mert", "byola", "panns", "mert_large", "clap"]:
        emb_dir = DATA_DIR / "embeddings" / name
        if not emb_dir.exists():
            print(f"  SKIP {name} (not found)")
            continue
        try:
            emb, tids = load_backbone_embeddings(name, "test")
            # Flatten multi-dimensional embeddings
            if emb.ndim > 2:
                emb = emb.reshape(emb.shape[0], -1)
            backbones[name] = {"embeddings": emb, "track_ids": tids}
            print(f"  {name}: shape={emb.shape}")
        except Exception as e:
            print(f"  ERROR {name}: {e}")

    if len(backbones) < 2:
        print("Need at least 2 backbones for CKA analysis")
        return

    # Align track IDs across backbones
    common_tids = set(backbones[list(backbones.keys())[0]]["track_ids"])
    for name, data in backbones.items():
        common_tids &= set(data["track_ids"])
    common_tids = sorted(common_tids)
    print(f"\n  Common tracks across all backbones: {len(common_tids)}")

    # Reindex to common tracks
    aligned = {}
    for name, data in backbones.items():
        tid_to_idx = {t: i for i, t in enumerate(data["track_ids"])}
        indices = [tid_to_idx[t] for t in common_tids]
        aligned[name] = data["embeddings"][indices]

    # Compute pairwise CKA
    names = list(aligned.keys())
    n = len(names)
    cka_matrix = np.zeros((n, n))

    print("\n--- Inter-Model CKA Matrix ---")
    for i in range(n):
        for j in range(n):
            cka_matrix[i, j] = linear_cka(aligned[names[i]], aligned[names[j]])

    # Pretty print
    display_names = {
        "mert": "MERT-95M", "mert_large": "MERT-330M",
        "byola": "BYOL-A", "panns": "PANNs", "clap": "CLAP"
    }
    print(f"{'':>12s}", end="")
    for name in names:
        print(f"  {display_names.get(name, name):>10s}", end="")
    print()
    for i, name in enumerate(names):
        print(f"  {display_names.get(name, name):>10s}", end="")
        for j in range(n):
            print(f"  {cka_matrix[i, j]:10.3f}", end="")
        print()

    # === Part 2: Layer-wise CKA within MERT-95M ===
    print("\n--- Layer-wise CKA within MERT-95M ---")
    try:
        mert_layers, mert_tids = load_mert_layer_embeddings("test")
        n_layers = mert_layers.shape[1]
        print(f"  MERT-95M: {mert_layers.shape[0]} tracks, {n_layers} layers, {mert_layers.shape[2]} dims")

        layer_cka = np.zeros((n_layers, n_layers))
        for i in range(n_layers):
            for j in range(n_layers):
                layer_cka[i, j] = linear_cka(mert_layers[:, i, :], mert_layers[:, j, :])

        print("\nLayer CKA matrix (13x13):")
        print(f"{'':>4s}", end="")
        for j in range(n_layers):
            print(f" L{j:02d}", end="")
        print()
        for i in range(n_layers):
            print(f"L{i:02d} ", end="")
            for j in range(n_layers):
                print(f" {layer_cka[i, j]:.2f}", end="")
            print()
    except Exception as e:
        print(f"  ERROR loading MERT layers: {e}")
        layer_cka = None
        n_layers = 0

    # === FIGURES ===

    # --- Figure 1: Inter-model CKA heatmap ---
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cka_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    dnames = [display_names.get(name, name) for name in names]
    ax.set_xticklabels(dnames, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(dnames, fontsize=10)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cka_matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, color="white" if cka_matrix[i, j] > 0.7 else "black")
    ax.set_title("Inter-Model CKA Similarity\n(Test Set, Track-Level Mean Pooling)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Linear CKA")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cka_inter_model.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "cka_inter_model.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved cka_inter_model figure")

    # --- Figure 2: MERT-95M layer-wise CKA ---
    if layer_cka is not None:
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(layer_cka, cmap="YlOrRd", vmin=0, vmax=1)
        layer_labels = ["Conv"] + [f"L{i}" for i in range(1, n_layers)]
        ax.set_xticks(range(n_layers))
        ax.set_yticks(range(n_layers))
        ax.set_xticklabels(layer_labels, fontsize=8)
        ax.set_yticklabels(layer_labels, fontsize=8)
        for i in range(n_layers):
            for j in range(n_layers):
                ax.text(j, i, f"{layer_cka[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if layer_cka[i, j] > 0.7 else "black")
        ax.set_title("MERT-95M Layer-wise CKA Similarity\n(13 layers: 1 conv + 12 transformer)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Linear CKA")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "cka_mert_layers.pdf", dpi=150, bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "cka_mert_layers.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved cka_mert_layers figure")

    # --- Figure 3: Combined panel ---
    if layer_cka is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                        gridspec_kw={"width_ratios": [1, 1.3]})

        # Panel a: inter-model
        im1 = ax1.imshow(cka_matrix, cmap="YlOrRd", vmin=0, vmax=1)
        ax1.set_xticks(range(n))
        ax1.set_yticks(range(n))
        ax1.set_xticklabels(dnames, rotation=45, ha="right", fontsize=9)
        ax1.set_yticklabels(dnames, fontsize=9)
        for i in range(n):
            for j in range(n):
                ax1.text(j, i, f"{cka_matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if cka_matrix[i, j] > 0.7 else "black")
        ax1.set_title("(a) Inter-Model CKA", fontsize=11)
        fig.colorbar(im1, ax=ax1, shrink=0.8)

        # Panel b: layer-wise
        im2 = ax2.imshow(layer_cka, cmap="YlOrRd", vmin=0, vmax=1)
        ax2.set_xticks(range(n_layers))
        ax2.set_yticks(range(n_layers))
        ax2.set_xticklabels(layer_labels, fontsize=7)
        ax2.set_yticklabels(layer_labels, fontsize=7)
        for i in range(n_layers):
            for j in range(n_layers):
                ax2.text(j, i, f"{layer_cka[i, j]:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if layer_cka[i, j] > 0.7 else "black")
        ax2.set_title("(b) MERT-95M Layer-wise CKA", fontsize=11)
        fig.colorbar(im2, ax=ax2, shrink=0.8)

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "cka_combined.pdf", dpi=150, bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "cka_combined.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved cka_combined figure")

    # Save results
    save_data = {
        "inter_model": {
            "model_names": [display_names.get(n, n) for n in names],
            "cka_matrix": cka_matrix.tolist(),
        },
    }
    if layer_cka is not None:
        save_data["mert_layer_wise"] = {
            "n_layers": n_layers,
            "layer_names": ["Conv"] + [f"Transformer-{i}" for i in range(1, n_layers)],
            "cka_matrix": layer_cka.tolist(),
        }
    with open(OUTPUT_DIR / "cka_analysis.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved results to {OUTPUT_DIR / 'cka_analysis.json'}")


if __name__ == "__main__":
    main()
