"""Layer-wise leakage fingerprinting for MERT-95M.

Reads eval_metrics.json from 39 single-layer MERT probes (13 layers x 3 protocols)
and plots inflation as a function of transformer depth.

Usage:
    python scripts/layer_leakage_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/ady/Code/clgprjcts/music")
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

NUM_LAYERS = 13


def load_accuracy(run_name: str) -> float | None:
    # Prefer eval_metrics.json (track-level), fall back to summary.json (episode-level)
    eval_path = OUTPUTS / run_name / "eval_metrics.json"
    if eval_path.exists():
        with open(eval_path) as f:
            d = json.load(f)
        acc = d.get("track_level", {}).get("accuracy")
        if acc is not None:
            return acc
    summary_path = OUTPUTS / run_name / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            d = json.load(f)
        return d.get("test_accuracy")


def main():
    af_accs = []
    rt_accs = []
    cr_accs = []

    for layer in range(NUM_LAYERS):
        af = load_accuracy(f"mert_layer{layer}")
        rt = load_accuracy(f"mert_layer{layer}_randomtrack")
        cr = load_accuracy(f"mert_layer{layer}_cliprandom")

        if af is None or cr is None:
            print(f"  WARN: missing data for layer {layer} (af={af}, rt={rt}, cr={cr})")
            af_accs.append(np.nan)
            rt_accs.append(np.nan if rt is None else rt)
            cr_accs.append(np.nan)
            continue

        af_accs.append(af)
        rt_accs.append(rt if rt is not None else np.nan)
        cr_accs.append(cr)
        print(f"  Layer {layer:2d}: AF={af*100:.1f}%  RT={rt*100:.1f}%  CR={cr*100:.1f}%  "
              f"inflation={((cr-af)*100):+.1f} pp")

    af_arr = np.array(af_accs)
    rt_arr = np.array(rt_accs)
    cr_arr = np.array(cr_accs)

    valid = ~np.isnan(af_arr) & ~np.isnan(cr_arr)
    if valid.sum() < 5:
        print(f"ERROR: only {valid.sum()} layers have complete data")
        return

    inflation = (cr_arr - af_arr) * 100  # clip-random minus artist-filtered
    rt_inflation = (rt_arr - af_arr) * 100  # random-track minus artist-filtered

    # --- Figure 1: Accuracy by layer and protocol ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel a: Accuracy vs layer
    ax = axes[0]
    layers = np.arange(NUM_LAYERS)
    ax.plot(layers[valid], af_arr[valid] * 100, "o-", color="#4878cf", lw=2, ms=6,
            label="Artist-filtered")
    ax.plot(layers[valid], rt_arr[valid] * 100, "s-", color="#6acc65", lw=2, ms=6,
            label="Random-track")
    ax.plot(layers[valid], cr_arr[valid] * 100, "^-", color="#d65f5f", lw=2, ms=6,
            label="Clip-random")

    # Fill between AF and CR
    ax.fill_between(layers[valid], af_arr[valid] * 100, cr_arr[valid] * 100,
                    alpha=0.15, color="#d65f5f")

    ax.set_xlabel("MERT-95M Transformer Layer")
    ax.set_ylabel("Track-level Accuracy (%)")
    ax.set_title("(a) Single-Layer Probe Accuracy by Protocol")
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{i}" for i in range(NUM_LAYERS)], fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel b: Inflation by layer
    ax = axes[1]
    ax.bar(layers[valid] - 0.15, inflation[valid], width=0.3, color="#d65f5f", alpha=0.85,
           label="Clip-random inflation")
    if not np.all(np.isnan(rt_inflation)):
        rt_valid = valid & ~np.isnan(rt_arr)
        ax.bar(layers[rt_valid] + 0.15, rt_inflation[rt_valid], width=0.3, color="#6acc65",
               alpha=0.85, label="Random-track inflation")

    ax.set_xlabel("MERT-95M Transformer Layer")
    ax.set_ylabel("Inflation over Artist-filtered (pp)")
    ax.set_title("(b) Per-Layer Protocol Inflation")
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{i}" for i in range(NUM_LAYERS)], fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Layer-Wise Leakage Fingerprinting (MERT-95M)", fontsize=12)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = FIGURES / f"layer_leakage.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)

    # Save results
    results = {
        "layers": list(range(NUM_LAYERS)),
        "artist_filtered": [float(x) if not np.isnan(x) else None for x in af_arr],
        "random_track": [float(x) if not np.isnan(x) else None for x in rt_arr],
        "clip_random": [float(x) if not np.isnan(x) else None for x in cr_arr],
        "inflation_cr": [float(x) if not np.isnan(x) else None for x in inflation],
        "inflation_rt": [float(x) if not np.isnan(x) else None for x in rt_inflation],
        "max_inflation_layer": int(np.nanargmax(inflation)),
        "min_inflation_layer": int(np.nanargmin(inflation)),
        "max_af_layer": int(np.nanargmax(af_arr)),
        "mean_inflation": float(np.nanmean(inflation)),
    }

    out_path = OUTPUTS / "layer_leakage_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {out_path}")

    print(f"\nSummary:")
    print(f"  Max AF accuracy: layer {results['max_af_layer']} ({af_arr[results['max_af_layer']]*100:.1f}%)")
    print(f"  Max inflation: layer {results['max_inflation_layer']} ({inflation[results['max_inflation_layer']]:+.1f} pp)")
    print(f"  Min inflation: layer {results['min_inflation_layer']} ({inflation[results['min_inflation_layer']]:+.1f} pp)")
    print(f"  Mean inflation: {results['mean_inflation']:+.1f} pp")


if __name__ == "__main__":
    main()
