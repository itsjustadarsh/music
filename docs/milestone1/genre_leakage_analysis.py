#!/usr/bin/env python3
"""Genre-specific protocol leakage analysis."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
FIGDIR = Path(__file__).resolve().parent / "figures"

GENRES = ["Electronic", "Experimental", "Folk", "Hip-Hop",
          "Instrumental", "International", "Pop", "Rock"]

# Mapping: (artist-filtered, random-track, clip-random)
MODELS = {
    "MERT Probe":  ("mert_probe_baseline",      "mert_probe_randomtrack",            "mert_probe_cliprandom"),
    "MERT Flat":   ("mert_flat_probe_v1",        "mert_flat_probe_v1_randomtrack",    "mert_flat_probe_v1_cliprandom"),
    "MERT Deep":   ("mert_deep_probe_v1",        "mert_deep_probe_v1_randomtrack",    "mert_deep_probe_v1_cliprandom"),
    "MERT Attn":   ("mert_attn_probe_v1",        "mert_attn_probe_v1_randomtrack",    "mert_attn_probe_v1_cliprandom"),
    "MERT-330M":   ("mert_large_probe_v1",       "mert_large_probe_v1_randomtrack",   "mert_large_probe_v1_cliprandom"),
    "BYOL-A":      ("byola_probe_v1",            "byola_probe_v1_randomtrack",        "byola_probe_v1_cliprandom"),
    "PANNs":       ("panns_probe_v1",            "panns_probe_v1_randomtrack",        "panns_probe_v1_cliprandom"),
    "GenreCNN":    ("cnn_improved",              "cnn_improved_randomtrack",           "cnn_improved_cliprandom"),
}

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

COL_W = 3.5
FULL_W = 7.16


def load_genre_f1(run_name):
    """Load per-genre F1 from eval_metrics.json."""
    p = OUT / run_name / "eval_metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    pg = d["track_level"]["per_genre"]
    return {g: pg[g].get("f1", pg[g].get("accuracy", 0)) for g in GENRES}


def main():
    # Collect per-genre F1 for each model x protocol
    af_all = {g: [] for g in GENRES}
    rt_all = {g: [] for g in GENRES}
    cr_all = {g: [] for g in GENRES}

    print("Per-model genre-specific inflation (F1, pp):\n")
    print(f"{'Model':12s}  {'Genre':14s}  {'AF':>5s}  {'RT':>5s}  {'CR':>5s}  {'RT-AF':>6s}  {'CR-AF':>6s}")
    print("-" * 70)

    for model_name, (af_run, rt_run, cr_run) in MODELS.items():
        af = load_genre_f1(af_run)
        rt = load_genre_f1(rt_run) if rt_run else None
        cr = load_genre_f1(cr_run) if cr_run else None

        if af is None:
            continue

        for g in GENRES:
            af_all[g].append(af[g])
            if rt:
                rt_all[g].append(rt[g])
            if cr:
                cr_all[g].append(cr[g])

            rt_val = rt[g] if rt else float("nan")
            cr_val = cr[g] if cr else float("nan")
            rt_delta = (rt_val - af[g]) * 100 if rt else float("nan")
            cr_delta = (cr_val - af[g]) * 100 if cr else float("nan")

            print(f"{model_name:12s}  {g:14s}  {af[g]*100:5.1f}  {rt_val*100:5.1f}  {cr_val*100:5.1f}  {rt_delta:+6.1f}  {cr_delta:+6.1f}")
        print()

    # Compute mean inflation per genre
    print("\n=== MEAN GENRE-SPECIFIC INFLATION (across all models) ===\n")
    print(f"{'Genre':14s}  {'AF':>5s}  {'RT':>5s}  {'CR':>5s}  {'RT-AF':>6s}  {'CR-AF':>6s}")
    print("-" * 55)

    genre_af_mean = {}
    genre_cr_mean = {}
    genre_rt_mean = {}
    genre_inflation = {}

    for g in GENRES:
        af_m = np.mean(af_all[g]) * 100
        rt_m = np.mean(rt_all[g]) * 100 if rt_all[g] else float("nan")
        cr_m = np.mean(cr_all[g]) * 100 if cr_all[g] else float("nan")
        genre_af_mean[g] = af_m
        genre_rt_mean[g] = rt_m
        genre_cr_mean[g] = cr_m
        genre_inflation[g] = cr_m - af_m
        print(f"{g:14s}  {af_m:5.1f}  {rt_m:5.1f}  {cr_m:5.1f}  {rt_m - af_m:+6.1f}  {cr_m - af_m:+6.1f}")

    # Sort genres by inflation
    sorted_genres = sorted(GENRES, key=lambda g: genre_inflation[g], reverse=True)
    print("\nRanked by total inflation (CR - AF):")
    for i, g in enumerate(sorted_genres, 1):
        print(f"  {i}. {g}: +{genre_inflation[g]:.1f} pp")

    # ── Figure: Genre inflation heatmap ──────────────────────────────
    # Rows = models, Cols = genres, Values = CR-AF inflation
    models_with_both = []
    inflation_matrix = []
    for model_name, (af_run, rt_run, cr_run) in MODELS.items():
        af = load_genre_f1(af_run)
        cr = load_genre_f1(cr_run) if cr_run else None
        if af and cr:
            row = [(cr[g] - af[g]) * 100 for g in GENRES]
            inflation_matrix.append(row)
            models_with_both.append(model_name)

    inflation_matrix = np.array(inflation_matrix)

    fig, ax = plt.subplots(figsize=(COL_W + 0.6, 2.8))
    im = ax.imshow(inflation_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=60)
    ax.set_xticks(range(len(GENRES)))
    ax.set_xticklabels([g[:4] for g in GENRES], rotation=45, ha="right")
    ax.set_yticks(range(len(models_with_both)))
    ax.set_yticklabels(models_with_both)

    for i in range(inflation_matrix.shape[0]):
        for j in range(inflation_matrix.shape[1]):
            val = inflation_matrix[i, j]
            color = "white" if val > 40 else "black"
            ax.text(j, i, f"+{val:.0f}", ha="center", va="center",
                    fontsize=6, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Inflation (pp)", fontsize=7)
    ax.set_title("Genre-Specific Protocol Inflation (Clip-Random vs Artist-Filtered)")
    fig.savefig(FIGDIR / "genre_inflation_heatmap.pdf")
    fig.savefig(FIGDIR / "genre_inflation_heatmap.png")
    plt.close(fig)
    print("\n  genre_inflation_heatmap.pdf")

    # ── Figure: Genre inflation bar chart (mean across models) ───────
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))

    x = np.arange(len(GENRES))
    w = 0.35
    af_vals = [genre_af_mean[g] for g in GENRES]
    cr_vals = [genre_cr_mean[g] for g in GENRES]
    infl_vals = [genre_inflation[g] for g in GENRES]

    ax.bar(x - w/2, af_vals, w, label="Artist-filtered", color="#2166ac", zorder=3)
    ax.bar(x + w/2, cr_vals, w, label="Clip-random", color="#d73027", zorder=3)

    for i, (af, cr, inf) in enumerate(zip(af_vals, cr_vals, infl_vals)):
        ax.annotate(f"+{inf:.0f}", xy=(x[i] + w/2, cr), xytext=(0, 3),
                   textcoords="offset points", ha="center", fontsize=6,
                   color="#d73027", fontweight="bold")

    ax.set_ylabel("Mean F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(GENRES, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="upper left", fontsize=6)
    ax.set_title("Per-Genre Inflation (Mean Across Architectures)")

    fig.savefig(FIGDIR / "genre_inflation_bars.pdf")
    fig.savefig(FIGDIR / "genre_inflation_bars.png")
    plt.close(fig)
    print("  genre_inflation_bars.pdf")

    # ── Figure: Stacked decomposition (artist→random-track→clip-random) ──
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))

    # Sort by total inflation
    order = sorted(range(len(GENRES)), key=lambda i: genre_inflation[GENRES[i]], reverse=True)
    sorted_g = [GENRES[i] for i in order]

    af_s = [genre_af_mean[g] for g in sorted_g]
    rt_s = [genre_rt_mean[g] for g in sorted_g]
    cr_s = [genre_cr_mean[g] for g in sorted_g]

    x = np.arange(len(sorted_g))
    # Decompose: base (AF), artist leak (RT-AF), clip leak (CR-RT)
    base = af_s
    artist_leak = [rt - af for rt, af in zip(rt_s, af_s)]
    clip_leak = [cr - rt for cr, rt in zip(cr_s, rt_s)]

    ax.bar(x, base, color="#2166ac", label="Artist-filtered (base)", zorder=3)
    ax.bar(x, artist_leak, bottom=base, color="#fdae61", label="+ Artist leak", zorder=3)
    ax.bar(x, clip_leak, bottom=[b + a for b, a in zip(base, artist_leak)],
           color="#d73027", label="+ Clip leak", zorder=3)

    ax.set_ylabel("F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_g, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="upper right", fontsize=6)
    ax.set_title("Leakage Decomposition by Genre")

    fig.savefig(FIGDIR / "genre_leakage_decomposition.pdf")
    fig.savefig(FIGDIR / "genre_leakage_decomposition.png")
    plt.close(fig)
    print("  genre_leakage_decomposition.pdf")


if __name__ == "__main__":
    main()
