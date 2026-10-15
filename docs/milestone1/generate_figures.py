#!/usr/bin/env python3
"""Generate publication-quality figures for the milestone 1 report."""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# IEEE two-column widths (inches)
COL_W = 3.5
FULL_W = 7.16

GENRES = ["Electronic", "Experimental", "Folk", "Hip-Hop",
          "Instrumental", "International", "Pop", "Rock"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


# ── Figure 1: Protocol leakage grouped bar chart ──────────────────────

def fig_leakage_bars():
    """Grouped bar chart: 3 protocols per architecture."""
    data = [
        ("MERT Probe",       56.1, 69.2, 93.6),
        ("MERT Flat",        58.2, 70.2, 90.2),
        ("MERT Deep",        55.5, 68.5, 92.2),
        ("MERT Attn",        54.6, 69.8, 95.7),
        ("MERT-330M",        56.6, 70.9, 96.0),
        ("GenreCNN",         55.1, None, 69.2),
        ("BYOL-A",           57.4, 69.9, 84.5),
        ("PANNs",            57.4, 67.2, 73.4),
    ]
    labels = [d[0] for d in data]
    artist = [d[1] for d in data]
    random = [d[2] if d[2] else 0 for d in data]
    clip   = [d[3] for d in data]

    x = np.arange(len(labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(FULL_W, 2.4))
    b1 = ax.bar(x - w, artist, w, label="Artist-filtered", color="#2166ac", zorder=3)
    b2 = ax.bar(x,     random, w, label="Random-track",    color="#fdae61", zorder=3)
    b3 = ax.bar(x + w, clip,   w, label="Clip-random",     color="#d73027", zorder=3)

    # Mark missing GenreCNN random-track
    ax.bar(x[5], 0, w, color="none", edgecolor="#999", linestyle="--", linewidth=0.5)

    ax.set_ylabel("Track-level accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(40, 100)
    ax.yaxis.set_major_locator(mtick.MultipleLocator(10))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="upper left", framealpha=0.9)

    # Annotate inflation on clip-random bars
    for i, (a, c) in enumerate(zip(artist, clip)):
        delta = c - a
        ax.text(x[i] + w, c + 0.8, f"+{delta:.0f}", ha="center", va="bottom",
                fontsize=6, color="#d73027", fontweight="bold")

    fig.savefig(FIGDIR / "leakage_bars.pdf")
    fig.savefig(FIGDIR / "leakage_bars.png")
    plt.close(fig)
    print("  leakage_bars.pdf")


# ── Figure 2: Per-genre F1 heatmap (best models) ─────────────────────

def fig_genre_heatmap():
    """Heatmap of per-genre F1 for the top architectures."""
    runs = [
        ("Ensemble",          "ensemble_uniform"),
        ("MERT-95M Flat",     "mert_flat_probe_v1"),
        ("BYOL-A",            "byola_probe_v1"),
        ("PANNs CNN14",       "panns_probe_v1"),
        ("MERT-330M",         "mert_large_probe_v1"),
        ("MERT-95M Probe",    "mert_probe_tuned"),
        ("MERT-95M Deep",     "mert_deep_probe_v2"),
        ("GenreCNN",          "cnn_improved"),
        ("DenseNet121",       "densenet_v1"),
        ("XGBoost",           "xgboost_mert"),
    ]

    matrix = []
    valid_runs = []
    for label, run in runs:
        path = OUTPUTS / run / "eval_metrics.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        pg = d["track_level"]["per_genre"]
        row = []
        for g in GENRES:
            v = pg[g]
            row.append((v.get("f1", v.get("accuracy", 0))) * 100)

        matrix.append(row)
        valid_runs.append(label)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(COL_W + 0.6, 3.0))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=20, vmax=90)
    ax.set_xticks(range(len(GENRES)))
    ax.set_xticklabels([g[:4] for g in GENRES], rotation=45, ha="right")
    ax.set_yticks(range(len(valid_runs)))
    ax.set_yticklabels(valid_runs)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val < 40 or val > 75 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=6, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("F1 (%)", fontsize=7)
    ax.set_title("Per-Genre F1 by Architecture")
    fig.savefig(FIGDIR / "genre_heatmap.pdf")
    fig.savefig(FIGDIR / "genre_heatmap.png")
    plt.close(fig)
    print("  genre_heatmap.pdf")


# ── Figure 3: Training curves (representative models) ────────────────

def fig_training_curves():
    """Val F1 learning curves for representative architectures."""
    runs = [
        ("MERT Flat Probe",   "mert_flat_probe_v1",  "#2166ac"),
        ("BYOL-A Probe",      "byola_probe_v1",      "#1b7837"),
        ("PANNs Probe",       "panns_probe_v1",      "#762a83"),
        ("GenreCNN",          "cnn_improved",         "#e66101"),
        ("DenseNet121",       "densenet_v1",          "#b2182b"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 2.2))

    for label, run, color in runs:
        csv = OUTPUTS / run / "metrics.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        ax1.plot(df["epoch"], df["val_f1"] * 100, label=label, color=color, linewidth=1)
        ax2.plot(df["epoch"], df["train_loss"], color=color, linewidth=1)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Macro-F1 (%)")
    ax1.set_ylim(20, 65)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right", fontsize=6)
    ax1.set_title("(a) Validation F1")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training loss")
    ax2.grid(alpha=0.3)
    ax2.set_title("(b) Training loss")

    fig.savefig(FIGDIR / "training_curves.pdf")
    fig.savefig(FIGDIR / "training_curves.png")
    plt.close(fig)
    print("  training_curves.pdf")


# ── Figure 4: Architecture comparison horizontal bars ─────────────────

def fig_architecture_bars():
    """Horizontal bar chart of all 13 architectures ranked by accuracy."""
    data = [
        ("XGBoost on MERT",      52.5),
        ("DenseNet121",           53.9),
        ("MERT-330M Deep",       54.9),
        ("GenreCNN",             55.1),
        ("MERT-95M Attn",       55.2),
        ("MERT-95M Deep",       56.0),
        ("MERT-95M Probe",      56.4),
        ("MERT-330M Flat",      56.5),
        ("MERT-330M Probe",     56.6),
        ("PANNs CNN14",         57.4),
        ("BYOL-A + DNN",        57.4),
        ("MERT-95M Flat",       58.2),
        ("Ensemble",            58.5),
    ]

    labels = [d[0] for d in data]
    accs = [d[1] for d in data]
    colors = ["#2166ac" if a >= 57 else "#67a9cf" if a >= 55 else "#d1e5f0" for a in accs]

    fig, ax = plt.subplots(figsize=(COL_W, 3.2))
    y = range(len(labels))
    bars = ax.barh(y, accs, color=colors, edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Track-level accuracy (%)")
    ax.set_xlim(48, 62)
    ax.grid(axis="x", alpha=0.3, zorder=0)

    for i, v in enumerate(accs):
        ax.text(v + 0.15, i, f"{v:.1f}", va="center", fontsize=6.5)

    ax.set_title("Best Per-Architecture (Artist-Filtered)")
    fig.savefig(FIGDIR / "architecture_bars.pdf")
    fig.savefig(FIGDIR / "architecture_bars.png")
    plt.close(fig)
    print("  architecture_bars.pdf")


# ── Figure 5: Seed stability box/scatter ──────────────────────────────

def fig_seed_stability():
    """Show accuracy spread across seeds/configs for key architectures."""
    groups = {
        "MERT Flat\n(7 runs)": [
            "mert_flat_probe_v1", "mert_flat_probe_v2",
            "mert_flat_diverse_s1", "mert_flat_diverse_s2",
            "mert_flat_diverse_s3", "mert_flat_diverse_s4",
            "mert_flat_diverse_s5", "mert_flat_diverse_s6",
        ],
        "MERT Probe\n(6 runs)": [
            "mert_probe_baseline", "mert_probe_tuned", "mert_probe_wide",
            "mert_probe_seed1", "mert_probe_seed2", "mert_probe_seed3",
        ],
        "MERT Attn\n(3 runs)": [
            "mert_attn_probe_v1", "mert_attn_probe_v2", "mert_attn_probe_v3",
        ],
        "GenreCNN\n(4 runs)": [
            "cnn_baseline", "cnn_augmented", "cnn_tuned_lr", "cnn_improved",
        ],
    }

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))

    positions = []
    all_accs = []
    labels = []
    for i, (label, runs) in enumerate(groups.items()):
        accs = []
        for run in runs:
            path = OUTPUTS / run / "eval_metrics.json"
            if path.exists():
                d = json.loads(path.read_text())
                accs.append(d["track_level"]["accuracy"] * 100)
        if accs:
            positions.append(i)
            all_accs.append(accs)
            labels.append(label)

    bp = ax.boxplot(all_accs, positions=positions, widths=0.5,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="#d73027", markersize=4),
                    medianprops=dict(color="#2166ac", linewidth=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor("#d1e5f0")
        patch.set_edgecolor("#2166ac")

    # Scatter individual points
    for i, accs in enumerate(all_accs):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(accs))
        ax.scatter([positions[i] + j for j in jitter], accs,
                   s=15, color="#2166ac", alpha=0.6, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Run-to-Run Variability")
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(FIGDIR / "seed_stability.pdf")
    fig.savefig(FIGDIR / "seed_stability.png")
    plt.close(fig)
    print("  seed_stability.pdf")


# ── Figure 6: Confusion matrix for ensemble ───────────────────────────

def fig_confusion_matrix():
    """Confusion matrix for the best model (ensemble_uniform)."""
    path = OUTPUTS / "ensemble_uniform" / "eval_metrics.json"
    d = json.loads(path.read_text())
    pg = d["track_level"]["per_genre"]

    # Reconstruct confusion matrix from per-genre metrics
    # We need the actual confusion matrix data - check if it exists
    # Build approximate from precision/recall/support
    n = len(GENRES)
    # Use the existing confusion matrix PNG if available, or compute from predictions
    cm_path = ROOT / "docs" / "figures" / "ensemble_uniform_confusion_matrix.png"
    if cm_path.exists():
        # Load and re-render in publication style
        img = plt.imread(str(cm_path))
        fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.85))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("Ensemble Confusion Matrix (Track-Level)", fontsize=9)
        fig.savefig(FIGDIR / "confusion_matrix.pdf")
        fig.savefig(FIGDIR / "confusion_matrix.png")
        plt.close(fig)
        print("  confusion_matrix.pdf (from existing)")
    else:
        print("  confusion_matrix.pdf SKIPPED (no source image)")


# ── Figure 7: Literature placement scatter ────────────────────────────

def fig_literature_scatter():
    """Scatter plot showing published results vs our artist-filtered line."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))

    # Published results (x=year, y=accuracy)
    published = [
        (2018, 56.9, "Transfer CNN"),
        (2018, 54.8, "FMA baseline re-run"),
        (2019, 68.9, "DenseNet+aug"),
        (2021, 56.4, "CNN/RNN ens."),
        (2022, 60.9, "PANNs (open-set)"),
        (2025, 68.5, "Dual res. CNN"),
        (2025, 64.3, "BYOL-A+DNN"),
        (2025, 90.2, "TS-Resformer"),
        (2026, 89.4, "CT-GateNet"),
    ]

    years = [p[0] for p in published]
    accs  = [p[1] for p in published]
    names = [p[2] for p in published]

    ax.scatter(years, accs, s=30, color="#d73027", zorder=5, label="Published (protocol unstated)")

    # Our result
    ax.scatter([2026], [58.5], s=60, color="#2166ac", marker="*", zorder=6,
              label="Ours (artist-filtered)")

    # Reference line: our best
    ax.axhline(58.5, color="#2166ac", linestyle="--", alpha=0.5, linewidth=0.8)

    # Annotate a few
    offsets = {
        "TS-Resformer": (-15, 5), "CT-GateNet": (-15, -12),
        "DenseNet+aug": (5, 5), "BYOL-A+DNN": (5, -10),
        "FMA baseline re-run": (5, -10),
    }
    for yr, acc, name in published:
        if name in offsets:
            dx, dy = offsets[name]
            ax.annotate(name, (yr, acc), xytext=(dx, dy),
                       textcoords="offset points", fontsize=5.5, color="#666")

    ax.set_xlabel("Year")
    ax.set_ylabel("Reported accuracy (%)")
    ax.set_ylim(45, 98)
    ax.set_xlim(2017, 2027)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=6)
    ax.set_title("Published FMA-small Results Over Time")

    fig.savefig(FIGDIR / "literature_scatter.pdf")
    fig.savefig(FIGDIR / "literature_scatter.png")
    plt.close(fig)
    print("  literature_scatter.pdf")


if __name__ == "__main__":
    print("Generating figures...")
    fig_leakage_bars()
    fig_genre_heatmap()
    fig_training_curves()
    fig_architecture_bars()
    fig_seed_stability()
    fig_confusion_matrix()
    fig_literature_scatter()
    print("Done. Figures saved to", FIGDIR)
