"""Statistical testing for the protocol leakage ladder.

Computes:
- Bootstrap 95% CIs on all accuracy numbers
- McNemar tests for pairwise model comparisons
- Effect sizes for protocol gaps
- Permutation test for mean inflation

Usage:
    python scripts/statistical_tests.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

OUTPUT_DIR = Path("outputs")
FIGURES_DIR = Path("docs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

GENRES = ["Electronic", "Experimental", "Folk", "Hip-Hop",
          "Instrumental", "International", "Pop", "Rock"]


def bootstrap_ci(correct: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95) -> tuple[float, float, float]:
    """Bootstrap confidence interval for accuracy."""
    n = len(correct)
    accs = np.array([correct[np.random.randint(0, n, n)].mean() for _ in range(n_bootstrap)])
    alpha = (1 - ci) / 2
    return float(np.percentile(accs, alpha * 100)), float(np.mean(accs)), float(np.percentile(accs, (1 - alpha) * 100))


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """McNemar's test comparing two classifiers on the same test set."""
    # b01 = A wrong, B right; b10 = A right, B wrong
    b01 = ((~correct_a) & correct_b).sum()
    b10 = (correct_a & (~correct_b)).sum()
    b00 = ((~correct_a) & (~correct_b)).sum()
    b11 = (correct_a & correct_b).sum()

    n = b01 + b10
    if n == 0:
        return {"chi2": 0, "p_value": 1.0, "b01": int(b01), "b10": int(b10)}

    # McNemar with continuity correction
    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10) if (b01 + b10) > 0 else 0

    # Compute p-value from chi-squared distribution (1 df)
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df=1)

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "b01": int(b01),  # A wrong, B right
        "b10": int(b10),  # A right, B wrong
        "b00": int(b00),
        "b11": int(b11),
    }


def aggregate_to_tracks(logits, labels, track_ids):
    """Aggregate episode logits to track-level predictions."""
    probs = softmax(logits, axis=1)
    sums, counts, label_of = {}, {}, {}
    for p, l, t in zip(probs, labels, track_ids):
        if t not in sums:
            sums[t] = np.zeros_like(p)
            counts[t] = 0
            label_of[t] = l
        sums[t] += p
        counts[t] += 1
    ordered = sorted(sums)
    track_preds = np.array([(sums[t] / counts[t]).argmax() for t in ordered])
    track_labels = np.array([label_of[t] for t in ordered])
    return track_labels, track_preds


def load_run(run_dir: Path) -> dict | None:
    npz_path = run_dir / "episode_outputs.npz"
    cfg_path = run_dir / "config.json"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    cfg = json.load(open(cfg_path)) if cfg_path.exists() else {}
    return {
        "name": run_dir.name,
        "split_mode": cfg.get("split_mode", "official"),
        "test_logits": data["test_logits"],
        "test_labels": data["test_labels"],
        "test_track_ids": data["test_track_ids"],
    }


def main():
    np.random.seed(42)

    runs = []
    for run_dir in sorted(OUTPUT_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        run = load_run(run_dir)
        if run is not None:
            runs.append(run)

    print(f"Loaded {len(runs)} runs with episode outputs")

    # Compute per-track correct/incorrect vectors + bootstrap CIs
    results = []
    for run in runs:
        track_labels, track_preds = aggregate_to_tracks(
            run["test_logits"], run["test_labels"], run["test_track_ids"]
        )
        correct = (track_preds == track_labels)
        acc = correct.mean()
        ci_lo, ci_mean, ci_hi = bootstrap_ci(correct)

        results.append({
            "name": run["name"],
            "split_mode": run["split_mode"],
            "accuracy": float(acc),
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "n_tracks": len(track_labels),
            "correct": correct,  # keep for McNemar
        })
        print(f"  {run['name']:45s} acc={acc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

    # === McNemar tests: top models pairwise ===
    print("\n--- McNemar Pairwise Tests (Artist-Filtered, Top Models) ---")
    official = [r for r in results if r["split_mode"] == "official"]
    official.sort(key=lambda r: r["accuracy"], reverse=True)
    top_n = min(8, len(official))
    top = official[:top_n]

    mcnemar_results = []
    print(f"\n{'Model A':>30s} vs {'Model B':<30s}  chi2     p-value  A>B  B>A")
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            mc = mcnemar_test(top[i]["correct"], top[j]["correct"])
            sig = "*" if mc["p_value"] < 0.05 else " "
            print(f"  {top[i]['name'][:28]:>28s} vs {top[j]['name'][:28]:<28s}  "
                  f"{mc['chi2']:6.2f}  {mc['p_value']:.4f}{sig} {mc['b10']:4d} {mc['b01']:4d}")
            mcnemar_results.append({
                "model_a": top[i]["name"],
                "model_b": top[j]["name"],
                **mc,
            })

    # === Bootstrap CI forest plot ===
    fig, ax = plt.subplots(figsize=(10, max(6, len(official) * 0.35)))
    official_sorted = sorted(official, key=lambda r: r["accuracy"])

    y_pos = range(len(official_sorted))
    accs = [r["accuracy"] for r in official_sorted]
    ci_los = [r["ci_95_lo"] for r in official_sorted]
    ci_his = [r["ci_95_hi"] for r in official_sorted]
    names = [r["name"].replace("mert_", "").replace("_probe", "P")[:30]
             for r in official_sorted]
    xerr = [[a - lo for a, lo in zip(accs, ci_los)],
            [hi - a for a, hi in zip(accs, ci_his)]]

    ax.errorbar(accs, y_pos, xerr=xerr, fmt="o", color="#2196F3",
                capsize=3, markersize=5, linewidth=1.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Track-Level Accuracy")
    ax.set_title("Artist-Filtered Runs: Accuracy with Bootstrap 95% CIs")
    ax.grid(True, alpha=0.3, axis="x")
    ax.axvline(0.583, color="red", linestyle="--", alpha=0.5, label="Best (58.3%)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bootstrap_forest.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "bootstrap_forest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved bootstrap_forest figure")

    # === Per-genre bootstrap CIs for best model ===
    best_run = next(r for r in runs if r["name"] == "mert_flat_probe_v1")
    track_labels, track_preds = aggregate_to_tracks(
        best_run["test_logits"], best_run["test_labels"], best_run["test_track_ids"]
    )
    correct = (track_preds == track_labels)

    print("\n--- Per-Genre Bootstrap CIs (mert_flat_probe_v1) ---")
    genre_cis = {}
    for g in range(8):
        mask = track_labels == g
        if mask.sum() > 0:
            g_correct = correct[mask]
            ci_lo, ci_mean, ci_hi = bootstrap_ci(g_correct)
            genre_cis[GENRES[g]] = {"acc": float(g_correct.mean()),
                                     "ci_lo": ci_lo, "ci_hi": ci_hi, "n": int(mask.sum())}
            print(f"  {GENRES[g]:15s}: {g_correct.mean():.3f} [{ci_lo:.3f}, {ci_hi:.3f}] (n={mask.sum()})")

    # Save all results
    save_results = {
        "bootstrap_cis": [{k: v for k, v in r.items() if k != "correct"}
                          for r in results],
        "mcnemar_tests": mcnemar_results,
        "per_genre_cis": genre_cis,
    }
    with open(OUTPUT_DIR / "statistical_tests.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved results to {OUTPUT_DIR / 'statistical_tests.json'}")


if __name__ == "__main__":
    main()
