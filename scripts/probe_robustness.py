"""Probe architecture robustness analysis.

Tests whether protocol inflation is consistent across different probe head
architectures (linear, 1-layer MLP, 2-layer MLP, larger MLP).

Usage:
    python scripts/probe_robustness.py
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

PROBE_TYPES = ["linear", "mlp64", "mlp128_64", "mlp256_128"]
PROBE_LABELS = ["Linear", "MLP-64", "MLP-128-64", "MLP-256-128"]

BACKBONES = {
    "CLAP": {
        "af_prefix": "clap_probe_",
        "cr_prefix": "clap_probe_",
        "cr_suffix": "_cliprandom",
    },
    "MERT-95M (flat)": {
        "af_prefix": "mert_flat_",
        "cr_prefix": "mert_flat_",
        "cr_suffix": "_cliprandom",
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


def main():
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors_af = "#4878cf"
    colors_cr = "#d65f5f"

    for ax_idx, (bb_name, bb_cfg) in enumerate(BACKBONES.items()):
        ax = axes[ax_idx]
        af_accs = []
        cr_accs = []
        inflations = []

        for probe_type in PROBE_TYPES:
            af_run = f"{bb_cfg['af_prefix']}{probe_type}"
            cr_run = f"{bb_cfg['cr_prefix']}{probe_type}{bb_cfg['cr_suffix']}"

            af_acc = load_accuracy(af_run)
            cr_acc = load_accuracy(cr_run)

            if af_acc is not None and cr_acc is not None:
                inflation = (cr_acc - af_acc) * 100
                print(f"  {bb_name} {probe_type}: AF={af_acc*100:.1f}%  CR={cr_acc*100:.1f}%  "
                      f"inflation={inflation:+.1f} pp")
            else:
                inflation = None
                print(f"  {bb_name} {probe_type}: AF={af_acc}  CR={cr_acc}")

            af_accs.append(af_acc)
            cr_accs.append(cr_acc)
            inflations.append(inflation)

        results[bb_name] = {
            "probe_types": PROBE_TYPES,
            "af_accuracies": [float(a) if a is not None else None for a in af_accs],
            "cr_accuracies": [float(a) if a is not None else None for a in cr_accs],
            "inflations": [float(x) if x is not None else None for x in inflations],
        }

        # Plot grouped bars
        x = np.arange(len(PROBE_TYPES))
        width = 0.35

        af_vals = [a * 100 if a is not None else 0 for a in af_accs]
        cr_vals = [a * 100 if a is not None else 0 for a in cr_accs]

        bars_af = ax.bar(x - width/2, af_vals, width, color=colors_af, alpha=0.85,
                         label="Artist-filtered", edgecolor="black", linewidth=0.5)
        bars_cr = ax.bar(x + width/2, cr_vals, width, color=colors_cr, alpha=0.85,
                         label="Clip-random", edgecolor="black", linewidth=0.5)

        # Add inflation labels on top
        for j, inf in enumerate(inflations):
            if inf is not None:
                mid_x = x[j]
                top_y = max(af_vals[j], cr_vals[j]) + 1.5
                ax.text(mid_x, top_y, f"+{inf:.1f}pp", ha="center", va="bottom",
                        fontsize=8, color="#d65f5f", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(PROBE_LABELS, fontsize=9)
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title(f"{bb_name}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 100)

    fig.suptitle("Probe Architecture Robustness: Inflation Persists Across Probe Types", fontsize=12)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = FIGURES / f"probe_robustness.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)

    # Summary stats
    all_inflations = []
    for bb_name, bb_res in results.items():
        for inf in bb_res["inflations"]:
            if inf is not None:
                all_inflations.append(inf)

    if all_inflations:
        results["summary"] = {
            "mean_inflation_pp": float(np.mean(all_inflations)),
            "std_inflation_pp": float(np.std(all_inflations)),
            "min_inflation_pp": float(min(all_inflations)),
            "max_inflation_pp": float(max(all_inflations)),
            "n_probes": len(all_inflations),
        }
        print(f"\nSummary across all probes:")
        print(f"  Mean inflation: {np.mean(all_inflations):+.1f} pp")
        print(f"  Std inflation: {np.std(all_inflations):.1f} pp")
        print(f"  Range: [{min(all_inflations):+.1f}, {max(all_inflations):+.1f}] pp")

    out_path = OUTPUTS / "probe_robustness.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
