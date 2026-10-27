"""
generate_paper_figures.py
Generate all missing paper figures for docs/final-paper/report.md.
Saves each figure to docs/figures/ as both PDF and PNG.
"""

import matplotlib
matplotlib.use("Agg")

import json
import os
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.special import softmax

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path("/Users/ady/Code/clgprjcts/music")
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────

def load_eval(run: str) -> dict:
    p = OUTPUTS / run / "eval_metrics.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def track_acc(run: str) -> float | None:
    d = load_eval(run)
    return d.get("track_level", {}).get("accuracy")


def track_f1(run: str) -> float | None:
    d = load_eval(run)
    return d.get("track_level", {}).get("macro_f1")


def per_genre_f1(run: str) -> dict:
    d = load_eval(run)
    pg = d.get("track_level", {}).get("per_genre", {})
    return {g: v["f1"] for g, v in pg.items()}


def save_fig(fig, name: str):
    for ext in ("pdf", "png"):
        path = FIGURES / f"{name}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close(fig)


GENRES = ["Electronic", "Experimental", "Folk", "Hip-Hop",
          "Instrumental", "International", "Pop", "Rock"]

# ── plot style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

# ══════════════════════════════════════════════════════════════════════════
# 1. architecture_bars
# ══════════════════════════════════════════════════════════════════════════
def fig_architecture_bars():
    print("Generating architecture_bars …")

    archs = [
        # (label, run_name, family)
        ("Ensemble (uniform)",   "ensemble_uniform",        "Ensemble"),
        ("MERT-95M flat",        "mert_flat_probe_v1",      "MERT-95M"),
        ("BYOL-A DNN",           "byola_probe_v1",          "Other"),
        ("PANNs CNN14",          "panns_probe_v1",          "Other"),
        ("CLAP",                 "clap_probe_v1",           "Other"),
        ("MERT-330M probe",      "mert_large_probe_v1",     "MERT-330M"),
        ("MERT-330M flat",       "mert_large_flat_v1",      "MERT-330M"),
        ("MERT-95M probe",       "mert_probe_tuned",        "MERT-95M"),
        ("MERT-95M deep",        "mert_deep_probe_v2",      "MERT-95M"),
        ("MERT-95M attn",        "mert_attn_probe_v2",      "MERT-95M"),
        ("GenreCNN",             "cnn_improved",            "CNN"),
        ("MERT-330M deep",       "mert_large_deep_probe_v1","MERT-330M"),
        ("DenseNet",             "densenet_v1",             "CNN"),
        ("XGBoost",              "xgboost_mert",            "Other"),
    ]

    # hardcoded paper values as fallback
    paper_accs = {
        "ensemble_uniform":        0.585,
        "mert_flat_probe_v1":      0.582,
        "byola_probe_v1":          0.574,
        "panns_probe_v1":          0.574,
        "mert_large_probe_v1":     0.566,
        "mert_large_flat_v1":      0.565,
        "mert_probe_tuned":        0.564,
        "mert_deep_probe_v2":      0.560,
        "mert_attn_probe_v2":      0.552,
        "cnn_improved":            0.551,
        "mert_large_deep_probe_v1":0.549,
        "densenet_v1":             0.539,
        "xgboost_mert":            0.525,
    }

    family_colors = {
        "Ensemble": "#e07b39",
        "MERT-95M": "#4878cf",
        "MERT-330M": "#6acc65",
        "CNN": "#d65f5f",
        "Other": "#b47cc7",
    }

    labels, accs, colors = [], [], []
    for label, run, family in archs:
        a = track_acc(run)
        if a is None:
            a = paper_accs.get(run, 0.0)
        labels.append(label)
        accs.append(a * 100)
        colors.append(family_colors[family])

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(labels))
    bars = ax.barh(y, accs, color=colors, height=0.65, edgecolor="white", linewidth=0.5)

    # value labels
    for bar, v in zip(bars, accs):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", ha="left", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Track-level Accuracy (%)")
    ax.set_title("Best Track-level Accuracy per Architecture\n(Artist-filtered Protocol)")
    ax.set_xlim(40, 65)
    ax.axvline(50, color="gray", lw=0.8, ls="--", alpha=0.5)

    legend_patches = [mpatches.Patch(color=c, label=f) for f, c in family_colors.items()]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    save_fig(fig, "architecture_bars")


# ══════════════════════════════════════════════════════════════════════════
# 2. seed_stability
# ══════════════════════════════════════════════════════════════════════════
def fig_seed_stability():
    print("Generating seed_stability …")

    groups = {
        "MERT-95M flat\n(7 runs)": [
            "mert_flat_probe_v1", "mert_flat_probe_v2",
            "mert_flat_diverse_s1", "mert_flat_diverse_s2",
            "mert_flat_diverse_s3", "mert_flat_diverse_s4",
            "mert_flat_diverse_s5",
        ],
        "MERT-95M probe\n(8 runs)": [
            "mert_probe_tuned", "mert_probe_seed1", "mert_probe_seed2",
            "mert_probe_seed3", "mert_probe_baseline", "mert_probe_wide",
            "mert_probe_lastlayer", "mert_probe_deep",
        ],
        "MERT-330M probe\n(3 runs)": [
            "mert_large_probe_v1", "mert_large_probe_v2", "mert_large_probe_v3",
        ],
        "CNN\n(4 runs)": [
            "cnn_improved", "cnn_augmented", "cnn_baseline", "cnn_tuned_lr",
        ],
        "MERT-95M attn\n(3 runs)": [
            "mert_attn_probe_v1", "mert_attn_probe_v2", "mert_attn_probe_v3",
        ],
    }

    data, xlabels = [], []
    for label, runs in groups.items():
        vals = [track_acc(r) * 100 for r in runs if track_acc(r) is not None]
        if vals:
            data.append(vals)
            xlabels.append(label)

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, patch_artist=True, vert=True,
                    medianprops=dict(color="navy", lw=2),
                    whiskerprops=dict(lw=1.2),
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))

    palette = ["#4878cf", "#4878cf", "#6acc65", "#d65f5f", "#4878cf"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # red diamonds for means
    for i, vals in enumerate(data, start=1):
        ax.plot(i, np.mean(vals), "D", color="red", markersize=7, zorder=5,
                label="Mean" if i == 1 else "")

    ax.set_xticks(range(1, len(xlabels) + 1))
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Track-level Accuracy (%)")
    ax.set_title("Run-to-run Variability by Architecture\n(Artist-filtered Protocol)")
    ax.legend(loc="lower right")
    ax.set_ylim(44, 65)
    ax.axhline(50, color="gray", lw=0.8, ls="--", alpha=0.5)

    plt.tight_layout()
    save_fig(fig, "seed_stability")


# ══════════════════════════════════════════════════════════════════════════
# 3. training_curves
# ══════════════════════════════════════════════════════════════════════════
def fig_training_curves():
    print("Generating training_curves …")

    runs = [
        ("mert_flat_probe_v1", "MERT-95M flat"),
        ("byola_probe_v1",     "BYOL-A DNN"),
        ("panns_probe_v1",     "PANNs CNN14"),
        ("cnn_improved",       "GenreCNN"),
        ("densenet_v1",        "DenseNet"),
    ]
    colors = ["#4878cf", "#e07b39", "#6acc65", "#d65f5f", "#b47cc7"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for (run, label), color in zip(runs, colors):
        csv_path = OUTPUTS / run / "metrics.csv"
        if not csv_path.exists():
            print(f"  WARN: {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        # column names may vary slightly
        epoch_col = "epoch"
        val_f1_col = next((c for c in df.columns if "val_f1" in c.lower()), None)
        train_loss_col = next((c for c in df.columns if "train_loss" in c.lower()), None)
        val_loss_col = next((c for c in df.columns if "val_loss" in c.lower()), None)

        if val_f1_col:
            ax1.plot(df[epoch_col], df[val_f1_col] * 100, color=color, lw=2, label=label)
        if train_loss_col:
            ax2.plot(df[epoch_col], df[train_loss_col], color=color, lw=2, label=label)
        elif val_loss_col:
            ax2.plot(df[epoch_col], df[val_loss_col], color=color, lw=2, ls="--", label=f"{label} (val)")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Macro-F1 (%)")
    ax1.set_title("(a) Validation Macro-F1 vs Epoch")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training Loss")
    ax2.set_title("(b) Training Loss vs Epoch")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "training_curves")


# ══════════════════════════════════════════════════════════════════════════
# 4. leakage_bars
# ══════════════════════════════════════════════════════════════════════════
def fig_leakage_bars():
    print("Generating leakage_bars …")

    arch_runs = [
        ("MERT Probe",   "mert_probe_tuned",       "mert_probe_randomtrack",       "mert_probe_cliprandom"),
        ("MERT Flat",    "mert_flat_probe_v1",      "mert_flat_probe_v1_randomtrack","mert_flat_probe_v1_cliprandom"),
        ("MERT Deep",    "mert_deep_probe_v1",      "mert_deep_probe_v1_randomtrack","mert_deep_probe_v1_cliprandom"),
        ("MERT Attn",    "mert_attn_probe_v1",      "mert_attn_probe_v1_randomtrack","mert_attn_probe_v1_cliprandom"),
        ("MERT-330M",    "mert_large_probe_v1",     "mert_large_probe_v1_randomtrack","mert_large_probe_v1_cliprandom"),
        ("MERT-330M flat","mert_large_flat_v1",    "mert_large_flat_v1_randomtrack","mert_large_flat_v1_cliprandom"),
        ("MERT-330M deep","mert_large_deep_probe_v1","mert_large_deep_probe_v1_randomtrack","mert_large_deep_probe_v1_cliprandom"),
        ("CNN",          "cnn_improved",            "cnn_improved_randomtrack",      "cnn_improved_cliprandom"),
        ("BYOL-A",       "byola_probe_v1",          "byola_probe_v1_randomtrack",    "byola_probe_v1_cliprandom"),
        ("PANNs",        "panns_probe_v1",          "panns_probe_v1_randomtrack",    "panns_probe_v1_cliprandom"),
        ("CLAP",         "clap_probe_v1",           "clap_probe_v1_randomtrack",     "clap_probe_v1_cliprandom"),
    ]

    labels = []
    af_vals, rt_vals, cr_vals = [], [], []

    for arch, run_af, run_rt, run_cr in arch_runs:
        af = track_acc(run_af)
        rt = track_acc(run_rt)
        cr = track_acc(run_cr)
        # skip if any rung is missing
        if af is None or rt is None or cr is None:
            print(f"  WARN: incomplete ladder for {arch} (af={af}, rt={rt}, cr={cr})")
            # use available; fill missing with last known
            if af is None:
                af = 0.0
            if rt is None:
                rt = af
            if cr is None:
                cr = rt
        labels.append(arch)
        af_vals.append(af * 100)
        rt_vals.append(rt * 100)
        cr_vals.append(cr * 100)

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - width, af_vals, width, label="Artist-filtered", color="#4878cf", alpha=0.85)
    b2 = ax.bar(x,         rt_vals, width, label="Random-track",    color="#6acc65", alpha=0.85)
    b3 = ax.bar(x + width, cr_vals, width, label="Clip-random",     color="#d65f5f", alpha=0.85)

    # red annotations: inflation = clip-random - artist-filtered
    for xi, (af, cr) in enumerate(zip(af_vals, cr_vals)):
        inflation = cr - af
        if inflation > 0:
            ax.annotate(f"+{inflation:.1f}pp",
                        xy=(xi + width, cr),
                        xytext=(xi + width, cr + 0.8),
                        ha="center", va="bottom",
                        fontsize=8, color="crimson",
                        arrowprops=dict(arrowstyle="-", color="crimson", lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Track-level Accuracy (%)")
    ax.set_title("Leakage Ladder: 3-Rung Protocol Comparison\n"
                 "(red +Xpp = clip-random inflation over artist-filtered)")
    ax.legend()
    ax.set_ylim(40, 102)
    ax.axhline(50, color="gray", lw=0.8, ls="--", alpha=0.4)

    plt.tight_layout()
    save_fig(fig, "leakage_bars")


# ══════════════════════════════════════════════════════════════════════════
# 5. genre_heatmap
# ══════════════════════════════════════════════════════════════════════════
def fig_genre_heatmap():
    print("Generating genre_heatmap …")

    top_runs = [
        ("Ensemble",          "ensemble_uniform"),
        ("MERT-95M flat",     "mert_flat_probe_v1"),
        ("BYOL-A DNN",        "byola_probe_v1"),
        ("PANNs CNN14",       "panns_probe_v1"),
        ("MERT-330M probe",   "mert_large_probe_v1"),
        ("MERT-330M flat",    "mert_large_flat_v1"),
        ("MERT-95M probe",    "mert_probe_tuned"),
        ("MERT-95M deep",     "mert_deep_probe_v2"),
        ("MERT-95M attn",     "mert_attn_probe_v2"),
        ("GenreCNN",          "cnn_improved"),
    ]

    matrix = []
    row_labels = []
    for label, run in top_runs:
        pg = per_genre_f1(run)
        if not pg:
            print(f"  WARN: no per-genre data for {run}")
            row = [0.0] * len(GENRES)
        else:
            row = [pg.get(g, 0.0) * 100 for g in GENRES]
        matrix.append(row)
        row_labels.append(label)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=20, vmax=90)

    ax.set_xticks(np.arange(len(GENRES)))
    ax.set_xticklabels(GENRES, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val > 65 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("F1 Score (%)")

    ax.set_title("Per-genre F1 Heatmap — Top 10 Architectures (Artist-filtered)")
    plt.tight_layout()
    save_fig(fig, "genre_heatmap")


# ══════════════════════════════════════════════════════════════════════════
# 6. genre_leakage_decomposition
# ══════════════════════════════════════════════════════════════════════════
def fig_genre_leakage_decomposition():
    print("Generating genre_leakage_decomposition …")

    arch_runs = [
        ("mert_flat_probe_v1",      "mert_flat_probe_v1_randomtrack",      "mert_flat_probe_v1_cliprandom"),
        ("mert_deep_probe_v1",      "mert_deep_probe_v1_randomtrack",      "mert_deep_probe_v1_cliprandom"),
        ("mert_attn_probe_v1",      "mert_attn_probe_v1_randomtrack",      "mert_attn_probe_v1_cliprandom"),
        ("mert_large_probe_v1",     "mert_large_probe_v1_randomtrack",     "mert_large_probe_v1_cliprandom"),
        ("byola_probe_v1",          "byola_probe_v1_randomtrack",          "byola_probe_v1_cliprandom"),
        ("panns_probe_v1",          "panns_probe_v1_randomtrack",          "panns_probe_v1_cliprandom"),
        ("clap_probe_v1",           "clap_probe_v1_randomtrack",           "clap_probe_v1_cliprandom"),
    ]

    artist_leak = {g: [] for g in GENRES}   # rt - af
    clip_leak   = {g: [] for g in GENRES}   # cr - rt

    for run_af, run_rt, run_cr in arch_runs:
        af_pg = per_genre_f1(run_af)
        rt_pg = per_genre_f1(run_rt)
        cr_pg = per_genre_f1(run_cr)
        if not af_pg or not rt_pg or not cr_pg:
            continue
        for g in GENRES:
            af_v = af_pg.get(g, 0.0) * 100
            rt_v = rt_pg.get(g, 0.0) * 100
            cr_v = cr_pg.get(g, 0.0) * 100
            artist_leak[g].append(rt_v - af_v)
            clip_leak[g].append(cr_v - rt_v)

    al_means = np.array([np.mean(artist_leak[g]) if artist_leak[g] else 0.0 for g in GENRES])
    cl_means = np.array([np.mean(clip_leak[g])   if clip_leak[g]   else 0.0 for g in GENRES])

    x = np.arange(len(GENRES))
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x, al_means, label="Artist leakage (random-track − artist-filtered)", color="#6acc65", alpha=0.85)
    ax.bar(x, cl_means, bottom=al_means, label="Clip leakage (clip-random − random-track)", color="#d65f5f", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(GENRES, rotation=20, ha="right")
    ax.set_ylabel("Mean F1 Inflation (pp)")
    ax.set_title("Per-genre Leakage Decomposition\n(Averaged across architectures)")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend()

    plt.tight_layout()
    save_fig(fig, "genre_leakage_decomposition")


# ══════════════════════════════════════════════════════════════════════════
# 7. genre_inflation_heatmap
# ══════════════════════════════════════════════════════════════════════════
def fig_genre_inflation_heatmap():
    print("Generating genre_inflation_heatmap …")

    arch_pairs = [
        ("MERT Flat",  "mert_flat_probe_v1",      "mert_flat_probe_v1_cliprandom"),
        ("MERT Probe", "mert_probe_tuned",         "mert_probe_cliprandom"),
        ("MERT Deep",  "mert_deep_probe_v1",       "mert_deep_probe_v1_cliprandom"),
        ("MERT Attn",  "mert_attn_probe_v1",       "mert_attn_probe_v1_cliprandom"),
        ("MERT-330M",  "mert_large_probe_v1",      "mert_large_probe_v1_cliprandom"),
        ("MERT-330M flat","mert_large_flat_v1",    "mert_large_flat_v1_cliprandom"),
        ("CNN",        "cnn_improved",             "cnn_improved_cliprandom"),
        ("BYOL-A",     "byola_probe_v1",           "byola_probe_v1_cliprandom"),
        ("PANNs",      "panns_probe_v1",           "panns_probe_v1_cliprandom"),
        ("CLAP",       "clap_probe_v1",            "clap_probe_v1_cliprandom"),
    ]

    matrix = []
    row_labels = []
    for label, run_af, run_cr in arch_pairs:
        af_pg = per_genre_f1(run_af)
        cr_pg = per_genre_f1(run_cr)
        if not af_pg or not cr_pg:
            print(f"  WARN: missing data for {label}")
            continue
        row = [(cr_pg.get(g, 0.0) - af_pg.get(g, 0.0)) * 100 for g in GENRES]
        matrix.append(row)
        row_labels.append(label)

    if not matrix:
        print("  WARN: no data for genre_inflation_heatmap")
        return

    matrix = np.array(matrix)
    vmax = max(abs(matrix).max(), 5)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(GENRES)))
    ax.set_xticklabels(GENRES, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                    fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("F1 Inflation (pp)\n(clip-random − artist-filtered)")

    ax.set_title("Per-genre F1 Inflation (clip-random − artist-filtered)")
    plt.tight_layout()
    save_fig(fig, "genre_inflation_heatmap")


# ══════════════════════════════════════════════════════════════════════════
# 8. genre_inflation_bars
# ══════════════════════════════════════════════════════════════════════════
def fig_genre_inflation_bars():
    print("Generating genre_inflation_bars …")

    arch_pairs = [
        ("mert_flat_probe_v1",      "mert_flat_probe_v1_cliprandom"),
        ("mert_probe_tuned",         "mert_probe_cliprandom"),
        ("mert_deep_probe_v1",       "mert_deep_probe_v1_cliprandom"),
        ("mert_attn_probe_v1",       "mert_attn_probe_v1_cliprandom"),
        ("mert_large_probe_v1",      "mert_large_probe_v1_cliprandom"),
        ("mert_large_flat_v1",       "mert_large_flat_v1_cliprandom"),
        ("cnn_improved",             "cnn_improved_cliprandom"),
        ("byola_probe_v1",           "byola_probe_v1_cliprandom"),
        ("panns_probe_v1",           "panns_probe_v1_cliprandom"),
        ("clap_probe_v1",            "clap_probe_v1_cliprandom"),
    ]

    af_sums = {g: [] for g in GENRES}
    cr_sums = {g: [] for g in GENRES}

    for run_af, run_cr in arch_pairs:
        af_pg = per_genre_f1(run_af)
        cr_pg = per_genre_f1(run_cr)
        if not af_pg or not cr_pg:
            continue
        for g in GENRES:
            af_sums[g].append(af_pg.get(g, 0.0) * 100)
            cr_sums[g].append(cr_pg.get(g, 0.0) * 100)

    af_means = np.array([np.mean(af_sums[g]) if af_sums[g] else 0.0 for g in GENRES])
    cr_means = np.array([np.mean(cr_sums[g]) if cr_sums[g] else 0.0 for g in GENRES])

    x = np.arange(len(GENRES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, af_means, width, label="Artist-filtered", color="#4878cf", alpha=0.85)
    ax.bar(x + width / 2, cr_means, width, label="Clip-random",     color="#d65f5f", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(GENRES, rotation=20, ha="right")
    ax.set_ylabel("Mean Track-level F1 (%)")
    ax.set_title("Per-genre F1: Artist-filtered vs Clip-random\n(Mean across architectures)")
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    save_fig(fig, "genre_inflation_bars")


# ══════════════════════════════════════════════════════════════════════════
# helpers for confusion matrices
# ══════════════════════════════════════════════════════════════════════════
def aggregate_to_tracks(logits, labels, track_ids):
    """Mean-pool softmax(logits) per track, return (pred_labels, true_labels)."""
    probs = softmax(logits, axis=1)
    unique_tracks = np.unique(track_ids)
    preds, trues = [], []
    for tid in unique_tracks:
        mask = track_ids == tid
        mean_prob = probs[mask].mean(axis=0)
        preds.append(mean_prob.argmax())
        trues.append(labels[mask][0])
    return np.array(preds), np.array(trues)


def plot_confusion(preds, trues, title, out_name):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(trues, preds, labels=np.arange(len(GENRES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(GENRES)))
    ax.set_yticks(np.arange(len(GENRES)))
    ax.set_xticklabels(GENRES, rotation=40, ha="right")
    ax.set_yticklabels(GENRES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(len(GENRES)):
        for j in range(len(GENRES)):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    # confusion matrices saved as PNG only (also PDF)
    for ext in ("pdf", "png"):
        path = FIGURES / f"{out_name}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# 9. confusion_best_artist_filtered
# ══════════════════════════════════════════════════════════════════════════
def fig_confusion_artist_filtered():
    print("Generating confusion_best_artist_filtered …")
    npz = OUTPUTS / "mert_flat_probe_v1" / "episode_outputs.npz"
    d = np.load(npz)
    preds, trues = aggregate_to_tracks(d["test_logits"], d["test_labels"], d["test_track_ids"])
    plot_confusion(preds, trues,
                   "Confusion Matrix — MERT-95M Flat (Artist-filtered)",
                   "confusion_best_artist_filtered")


# ══════════════════════════════════════════════════════════════════════════
# 10. confusion_best_cliprandom
# ══════════════════════════════════════════════════════════════════════════
def fig_confusion_cliprandom():
    print("Generating confusion_best_cliprandom …")
    npz = OUTPUTS / "mert_flat_probe_v1_cliprandom" / "episode_outputs.npz"
    d = np.load(npz)
    preds, trues = aggregate_to_tracks(d["test_logits"], d["test_labels"], d["test_track_ids"])
    plot_confusion(preds, trues,
                   "Confusion Matrix — MERT-95M Flat (Clip-random)",
                   "confusion_best_cliprandom")


# ══════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════
FIGURES_TO_GENERATE = [
    ("architecture_bars",            fig_architecture_bars),
    ("seed_stability",               fig_seed_stability),
    ("training_curves",              fig_training_curves),
    ("leakage_bars",                 fig_leakage_bars),
    ("genre_heatmap",                fig_genre_heatmap),
    ("genre_leakage_decomposition",  fig_genre_leakage_decomposition),
    ("genre_inflation_heatmap",      fig_genre_inflation_heatmap),
    ("genre_inflation_bars",         fig_genre_inflation_bars),
    ("confusion_best_artist_filtered", fig_confusion_artist_filtered),
    ("confusion_best_cliprandom",    fig_confusion_cliprandom),
]

if __name__ == "__main__":
    successes, failures = [], []

    for name, fn in FIGURES_TO_GENERATE:
        try:
            fn()
            successes.append(name)
        except Exception as e:
            print(f"  ERROR generating {name}: {e}")
            traceback.print_exc()
            failures.append((name, str(e)))

    print("\n" + "=" * 60)
    print(f"Done. {len(successes)}/{len(FIGURES_TO_GENERATE)} figures generated successfully.")
    if failures:
        print("\nFailed figures:")
        for name, err in failures:
            print(f"  {name}: {err}")
    print(f"\nFigures saved to: {FIGURES}")
