"""Cross-protocol temporal consistency analysis.

Compares inter-episode agreement under artist-filtered vs clip-random evaluation.
Under leaky evaluation, models recognize recordings so all 10 episodes agree;
under honest evaluation, genre ambiguity causes disagreement.

Usage:
    python scripts/temporal_consistency_protocol.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

ROOT = Path("/Users/ady/Code/clgprjcts/music")
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

GENRES = ["Electronic", "Experimental", "Folk", "Hip-Hop",
          "Instrumental", "International", "Pop", "Rock"]

# Paired runs with all 3 protocols
PAIRED_RUNS = [
    ("MERT-95M flat", "mert_flat_probe_v1", "mert_flat_probe_v1_randomtrack", "mert_flat_probe_v1_cliprandom"),
    ("MERT-95M probe", "mert_probe_tuned", "mert_probe_randomtrack", "mert_probe_cliprandom"),
    ("MERT-95M deep", "mert_deep_probe_v1", "mert_deep_probe_v1_randomtrack", "mert_deep_probe_v1_cliprandom"),
    ("MERT-95M attn", "mert_attn_probe_v1", "mert_attn_probe_v1_randomtrack", "mert_attn_probe_v1_cliprandom"),
    ("MERT-330M", "mert_large_probe_v1", "mert_large_probe_v1_randomtrack", "mert_large_probe_v1_cliprandom"),
    ("BYOL-A", "byola_probe_v1", "byola_probe_v1_randomtrack", "byola_probe_v1_cliprandom"),
    ("PANNs", "panns_probe_v1", "panns_probe_v1_randomtrack", "panns_probe_v1_cliprandom"),
    ("CLAP", "clap_probe_v1", "clap_probe_v1_randomtrack", "clap_probe_v1_cliprandom"),
    ("CNN", "cnn_improved", "cnn_improved_randomtrack", "cnn_improved_cliprandom"),
]


def compute_track_consistency(run_name: str) -> dict | None:
    """Compute per-track episode agreement for a run."""
    npz_path = OUTPUTS / run_name / "episode_outputs.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    logits = data["test_logits"]
    labels = data["test_labels"]
    track_ids = data["test_track_ids"]
    probs = softmax(logits, axis=1)
    preds = logits.argmax(axis=1)

    tracks = {}
    for i in range(len(labels)):
        tid = track_ids[i]
        if tid not in tracks:
            tracks[tid] = {"preds": [], "label": labels[i], "probs": []}
        tracks[tid]["preds"].append(preds[i])
        tracks[tid]["probs"].append(probs[i])

    agreements = []
    correct_flags = []
    genre_labels = []
    for tid, t in tracks.items():
        ep_preds = np.array(t["preds"])
        mode_pred = np.bincount(ep_preds, minlength=8).argmax()
        agree_frac = (ep_preds == mode_pred).mean()
        agreements.append(agree_frac)

        track_pred = np.mean(t["probs"], axis=0).argmax()
        correct_flags.append(int(track_pred == t["label"]))
        genre_labels.append(int(t["label"]))

    return {
        "agreements": agreements,
        "correct_flags": correct_flags,
        "genre_labels": genre_labels,
        "mean_agreement": float(np.mean(agreements)),
        "accuracy": float(np.mean(correct_flags)),
    }


def main():
    results = {"models": []}
    af_agreements_all = []
    rt_agreements_all = []
    cr_agreements_all = []

    for label, af_run, rt_run, cr_run in PAIRED_RUNS:
        af = compute_track_consistency(af_run)
        rt = compute_track_consistency(rt_run)
        cr = compute_track_consistency(cr_run)

        if af is None or cr is None:
            print(f"  WARN: missing data for {label}")
            continue

        af_agreements_all.extend(af["agreements"])
        if rt is not None:
            rt_agreements_all.extend(rt["agreements"])
        cr_agreements_all.extend(cr["agreements"])

        model_result = {
            "name": label,
            "artist_filtered": {"mean_agreement": af["mean_agreement"], "accuracy": af["accuracy"]},
            "clip_random": {"mean_agreement": cr["mean_agreement"], "accuracy": cr["accuracy"]},
        }
        if rt is not None:
            model_result["random_track"] = {"mean_agreement": rt["mean_agreement"], "accuracy": rt["accuracy"]}

        results["models"].append(model_result)
        print(f"  {label}: AF agree={af['mean_agreement']:.3f} acc={af['accuracy']:.3f}  "
              f"CR agree={cr['mean_agreement']:.3f} acc={cr['accuracy']:.3f}")

    if not results["models"]:
        print("ERROR: no paired runs found")
        return

    af_arr = np.array(af_agreements_all)
    cr_arr = np.array(cr_agreements_all)
    rt_arr = np.array(rt_agreements_all) if rt_agreements_all else None

    results["summary"] = {
        "af_mean_agreement": float(af_arr.mean()),
        "af_std_agreement": float(af_arr.std()),
        "cr_mean_agreement": float(cr_arr.mean()),
        "cr_std_agreement": float(cr_arr.std()),
    }
    if rt_arr is not None:
        results["summary"]["rt_mean_agreement"] = float(rt_arr.mean())
        results["summary"]["rt_std_agreement"] = float(rt_arr.std())

    print(f"\nAggregated ({len(results['models'])} models):")
    print(f"  AF mean agreement: {af_arr.mean():.3f} +/- {af_arr.std():.3f}")
    if rt_arr is not None:
        print(f"  RT mean agreement: {rt_arr.mean():.3f} +/- {rt_arr.std():.3f}")
    print(f"  CR mean agreement: {cr_arr.mean():.3f} +/- {cr_arr.std():.3f}")

    # --- Figure 1: Violin/box plot of agreement distributions by protocol ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel a: Overlapping histograms
    ax = axes[0]
    bins = np.linspace(0.2, 1.0, 25)
    ax.hist(af_arr, bins=bins, alpha=0.6, color="#4878cf", label="Artist-filtered", density=True)
    ax.hist(cr_arr, bins=bins, alpha=0.6, color="#d65f5f", label="Clip-random", density=True)
    ax.axvline(af_arr.mean(), color="#4878cf", ls="--", lw=2, label=f"AF mean={af_arr.mean():.2f}")
    ax.axvline(cr_arr.mean(), color="#d65f5f", ls="--", lw=2, label=f"CR mean={cr_arr.mean():.2f}")
    ax.set_xlabel("Episode Agreement (fraction of 10 episodes agreeing)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Episode Agreement Distribution by Protocol")
    ax.legend(fontsize=8)

    # Panel b: Per-model agreement comparison
    ax = axes[1]
    model_names = [m["name"] for m in results["models"]]
    af_means = [m["artist_filtered"]["mean_agreement"] for m in results["models"]]
    cr_means = [m["clip_random"]["mean_agreement"] for m in results["models"]]
    x = np.arange(len(model_names))
    width = 0.35
    ax.bar(x - width/2, af_means, width, label="Artist-filtered", color="#4878cf", alpha=0.85)
    ax.bar(x + width/2, cr_means, width, label="Clip-random", color="#d65f5f", alpha=0.85)

    # Annotation: delta
    for xi, (af_m, cr_m) in enumerate(zip(af_means, cr_means)):
        delta = cr_m - af_m
        ax.annotate(f"+{delta:.2f}", xy=(xi + width/2, cr_m),
                    xytext=(xi + width/2, cr_m + 0.01),
                    ha="center", fontsize=7, color="crimson")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Mean Episode Agreement")
    ax.set_title("(b) Per-Model Agreement by Protocol")
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 1.05)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = FIGURES / f"temporal_consistency.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)

    # Save results
    out_path = OUTPUTS / "temporal_consistency_protocol.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
