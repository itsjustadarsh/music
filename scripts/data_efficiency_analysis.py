"""Data efficiency × protocol interaction analysis.

Reads eval_metrics.json from subsampled training runs and plots accuracy
vs training fraction under artist-filtered vs clip-random evaluation.

Usage:
    python scripts/data_efficiency_analysis.py
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

FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]

MODELS = {
    "CLAP": {
        "af_runs": ["clap_frac10", "clap_frac25", "clap_frac50", "clap_frac75", "clap_probe_v1"],
        "cr_runs": ["clap_frac10_cliprandom", "clap_frac25_cliprandom", "clap_frac50_cliprandom",
                     "clap_frac75_cliprandom", "clap_probe_v1_cliprandom"],
        "color": "#b47cc7",
    },
    "MERT-95M flat": {
        "af_runs": ["mert_flat_frac10", "mert_flat_frac25", "mert_flat_frac50",
                     "mert_flat_frac75", "mert_flat_probe_v1"],
        "cr_runs": ["mert_flat_frac10_cliprandom", "mert_flat_frac25_cliprandom",
                     "mert_flat_frac50_cliprandom", "mert_flat_frac75_cliprandom",
                     "mert_flat_probe_v1_cliprandom"],
        "color": "#4878cf",
    },
}


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
    results = {"models": {}}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for model_name, model_cfg in MODELS.items():
        af_accs = []
        cr_accs = []

        for i, frac in enumerate(FRACTIONS):
            af = load_accuracy(model_cfg["af_runs"][i])
            cr = load_accuracy(model_cfg["cr_runs"][i])
            af_accs.append(af)
            cr_accs.append(cr)
            if af is not None and cr is not None:
                print(f"  {model_name} frac={frac:.0%}: AF={af*100:.1f}%  CR={cr*100:.1f}%  "
                      f"inflation={((cr-af)*100):+.1f} pp")
            else:
                print(f"  {model_name} frac={frac:.0%}: AF={af}  CR={cr}")

        color = model_cfg["color"]

        # Panel a: Accuracy vs fraction
        ax = axes[0]
        af_valid = [(f, a) for f, a in zip(FRACTIONS, af_accs) if a is not None]
        cr_valid = [(f, a) for f, a in zip(FRACTIONS, cr_accs) if a is not None]
        if af_valid:
            ax.plot([f for f, _ in af_valid], [a * 100 for _, a in af_valid],
                    "o-", color=color, lw=2, ms=7, label=f"{model_name} (AF)")
        if cr_valid:
            ax.plot([f for f, _ in cr_valid], [a * 100 for _, a in cr_valid],
                    "s--", color=color, lw=2, ms=7, label=f"{model_name} (CR)")

        # Panel b: Inflation vs fraction
        ax = axes[1]
        inflations = [(f, (cr - af) * 100) for f, af, cr in zip(FRACTIONS, af_accs, cr_accs)
                      if af is not None and cr is not None]
        if inflations:
            ax.plot([f for f, _ in inflations], [inf for _, inf in inflations],
                    "D-", color=color, lw=2, ms=7, label=model_name)

        results["models"][model_name] = {
            "fractions": FRACTIONS,
            "af_accuracies": [float(a) if a is not None else None for a in af_accs],
            "cr_accuracies": [float(a) if a is not None else None for a in cr_accs],
            "inflations": [float((cr - af) * 100) if af is not None and cr is not None else None
                           for af, cr in zip(af_accs, cr_accs)],
        }

    # Format panels
    axes[0].set_xlabel("Training Data Fraction")
    axes[0].set_ylabel("Track-level Accuracy (%)")
    axes[0].set_title("(a) Accuracy vs Training Data Fraction")
    axes[0].legend(fontsize=8)
    axes[0].set_xticks(FRACTIONS)
    axes[0].set_xticklabels([f"{f:.0%}" for f in FRACTIONS])
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Training Data Fraction")
    axes[1].set_ylabel("Inflation (clip-random − artist-filtered, pp)")
    axes[1].set_title("(b) Protocol Inflation vs Training Data Fraction")
    axes[1].legend(fontsize=9)
    axes[1].set_xticks(FRACTIONS)
    axes[1].set_xticklabels([f"{f:.0%}" for f in FRACTIONS])
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Data Efficiency × Protocol Interaction", fontsize=12)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = FIGURES / f"data_efficiency.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)

    out_path = OUTPUTS / "data_efficiency_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
