"""Nearest-neighbor identity recovery analysis.

For each test clip, finds its nearest neighbor in the training set by embedding
cosine distance. Under clip-random splits, many test clips have a same-track NN
in training (= information leakage). Under artist-filtered splits, this should
be ~0%.

Usage:
    python scripts/nn_identity_recovery.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/ady/Code/clgprjcts/music")
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

MANIFEST = ROOT / "data" / "processed" / "manifest.csv"
EMBED_DIR = ROOT / "data" / "embeddings"

# Split mode seeds (must match dataset.py)
RANDOM_CLIP_SEED = 42
RANDOM_CLIP_FRACTIONS = (0.8, 0.1, 0.1)


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


def load_embeddings_for_split(embed_name: str, manifest: pd.DataFrame,
                               split: str, shape_3d: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings for a given split, return (embeddings, track_ids)."""
    split_df = manifest[manifest["split"] == split].reset_index(drop=True)
    embeddings = []
    track_ids = []

    embed_root = EMBED_DIR / embed_name

    for _, row in split_df.iterrows():
        genre = row["genre"]
        track_id = int(row["track_id"])
        subclip_idx = int(row["subclip_idx"])

        npy_path = embed_root / row["split"] / genre / f"{track_id:06d}.npy"
        # Under re-split, the file may be in a different original split dir
        if not npy_path.exists():
            for orig_split in ("train", "val", "test"):
                alt_path = embed_root / orig_split / genre / f"{track_id:06d}.npy"
                if alt_path.exists():
                    npy_path = alt_path
                    break

        if not npy_path.exists():
            continue

        arr = np.load(npy_path)  # (num_episodes, ...)
        if subclip_idx >= len(arr):
            continue

        emb = arr[subclip_idx]
        if shape_3d:
            emb = emb.mean(axis=0)  # (13, 768) -> (768,)
        embeddings.append(emb.astype(np.float32))
        track_ids.append(track_id)

    return np.array(embeddings), np.array(track_ids)


def compute_nn_leak_rate(train_embs: np.ndarray, train_tids: np.ndarray,
                          test_embs: np.ndarray, test_tids: np.ndarray,
                          batch_size: int = 500) -> tuple[float, np.ndarray]:
    """For each test clip, find its NN in training set. Return leak rate and per-clip NN match."""
    # Normalize
    train_norms = np.linalg.norm(train_embs, axis=1, keepdims=True)
    train_norms = np.maximum(train_norms, 1e-8)
    train_normed = train_embs / train_norms

    test_norms = np.linalg.norm(test_embs, axis=1, keepdims=True)
    test_norms = np.maximum(test_norms, 1e-8)
    test_normed = test_embs / test_norms

    same_track = np.zeros(len(test_embs), dtype=bool)

    # Process in batches to manage memory
    for start in range(0, len(test_embs), batch_size):
        end = min(start + batch_size, len(test_embs))
        batch = test_normed[start:end]  # (batch, dim)
        sims = batch @ train_normed.T  # (batch, n_train)
        nn_idx = np.argmax(sims, axis=1)
        nn_tids = train_tids[nn_idx]
        same_track[start:end] = nn_tids == test_tids[start:end]

    leak_rate = float(np.mean(same_track))
    return leak_rate, same_track


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


BACKBONES = {
    "MERT-95M": {
        "embed_name": "mert",
        "shape_3d": True,
        "af_run": "mert_flat_probe_v1",
        "cr_run": "mert_flat_probe_v1_cliprandom",
    },
    "CLAP": {
        "embed_name": "clap",
        "shape_3d": False,
        "af_run": "clap_probe_v1",
        "cr_run": "clap_probe_v1_cliprandom",
    },
    "BYOL-A": {
        "embed_name": "byola",
        "shape_3d": False,
        "af_run": "byola_probe_v1",
        "cr_run": "byola_probe_v1_cliprandom",
    },
    "PANNs": {
        "embed_name": "panns",
        "shape_3d": False,
        "af_run": "panns_probe_v1",
        "cr_run": "panns_probe_v1_cliprandom",
    },
}


def main():
    # Load manifest
    manifest_official = pd.read_csv(MANIFEST)
    manifest_cr = assign_random_clip_splits(manifest_official.copy())

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = ["#4878cf", "#b47cc7", "#6acc65", "#d65f5f"]

    backbone_names = []
    af_leak_rates = []
    cr_leak_rates = []
    inflations = []

    for i, (name, cfg) in enumerate(BACKBONES.items()):
        print(f"\n{'='*40}")
        print(f"Processing {name}...")

        # Artist-filtered (official) split
        print("  Loading AF train embeddings...")
        af_train_embs, af_train_tids = load_embeddings_for_split(
            cfg["embed_name"], manifest_official, "train", cfg["shape_3d"])
        print(f"  AF train: {len(af_train_embs)} clips")

        print("  Loading AF test embeddings...")
        af_test_embs, af_test_tids = load_embeddings_for_split(
            cfg["embed_name"], manifest_official, "test", cfg["shape_3d"])
        print(f"  AF test: {len(af_test_embs)} clips")

        if len(af_train_embs) > 0 and len(af_test_embs) > 0:
            af_leak, _ = compute_nn_leak_rate(af_train_embs, af_train_tids,
                                               af_test_embs, af_test_tids)
        else:
            af_leak = 0.0
        print(f"  AF NN leak rate: {af_leak*100:.1f}%")

        # Clip-random split
        print("  Loading CR train embeddings...")
        cr_train_embs, cr_train_tids = load_embeddings_for_split(
            cfg["embed_name"], manifest_cr, "train", cfg["shape_3d"])
        print(f"  CR train: {len(cr_train_embs)} clips")

        print("  Loading CR test embeddings...")
        cr_test_embs, cr_test_tids = load_embeddings_for_split(
            cfg["embed_name"], manifest_cr, "test", cfg["shape_3d"])
        print(f"  CR test: {len(cr_test_embs)} clips")

        if len(cr_train_embs) > 0 and len(cr_test_embs) > 0:
            cr_leak, _ = compute_nn_leak_rate(cr_train_embs, cr_train_tids,
                                               cr_test_embs, cr_test_tids)
        else:
            cr_leak = 0.0
        print(f"  CR NN leak rate: {cr_leak*100:.1f}%")

        # Inflation
        af_acc = load_accuracy(cfg["af_run"])
        cr_acc = load_accuracy(cfg["cr_run"])
        inflation = ((cr_acc - af_acc) * 100) if (af_acc is not None and cr_acc is not None) else 0.0

        backbone_names.append(name)
        af_leak_rates.append(af_leak * 100)
        cr_leak_rates.append(cr_leak * 100)
        inflations.append(inflation)

        results[name] = {
            "af_nn_leak_rate": af_leak,
            "cr_nn_leak_rate": cr_leak,
            "af_accuracy": af_acc,
            "cr_accuracy": cr_acc,
            "inflation_pp": inflation,
        }

    # Panel a: Grouped bars — AF vs CR leak rate per backbone
    ax = axes[0]
    x = np.arange(len(backbone_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, af_leak_rates, width, color="#4878cf", alpha=0.85,
                   label="Artist-filtered", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, cr_leak_rates, width, color="#d65f5f", alpha=0.85,
                   label="Clip-random", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(backbone_names, fontsize=9)
    ax.set_ylabel("NN Identity Leak Rate (%)")
    ax.set_title("(a) Nearest-Neighbor Same-Track Recovery Rate")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=8)

    # Panel b: Scatter — CR leak rate vs inflation
    ax = axes[1]
    for i, name in enumerate(backbone_names):
        ax.scatter(cr_leak_rates[i], inflations[i], s=120, color=colors[i],
                   zorder=5, edgecolors="black", linewidth=0.8)
        ax.annotate(name, (cr_leak_rates[i], inflations[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)

    if len(cr_leak_rates) >= 2:
        z = np.polyfit(cr_leak_rates, inflations, 1)
        x_line = np.linspace(min(cr_leak_rates) * 0.95, max(cr_leak_rates) * 1.05, 50)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="gray", alpha=0.5)
        corr = np.corrcoef(cr_leak_rates, inflations)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax.transAxes,
                fontsize=10, va="top")

    ax.set_xlabel("Clip-Random NN Leak Rate (%)")
    ax.set_ylabel("Protocol Inflation (pp)")
    ax.set_title("(b) NN Leak Rate vs Protocol Inflation")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Nearest-Neighbor Recording Identity Recovery", fontsize=12)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = FIGURES / f"nn_identity_recovery.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)

    out_path = OUTPUTS / "nn_identity_recovery.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
