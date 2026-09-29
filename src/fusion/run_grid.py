"""Run the full ESCF ablation grid at multiple seeds and build the results table.

Every row sees identical cached backbone outputs and identical splits, so differences
between rows are attributable to the fusion mechanism and nothing else. Rows A5-A7 and
A10 are the ones that decide whether the proposed method is real: if A8 does not clear
them with separated error bars, the paper must say so.

Usage:
    python -m src.fusion.run_grid --seeds 5
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np

from src.config import OUTPUT_DIR
from src.evaluation.significance import mcnemar_test, paired_bootstrap
from src.fusion.train_fusion import FUSION_DIR, train_fusion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROPOSED_ROW = "A8_escf_cosine"

ROWS = [
    ("A1_cnn_meanpool",      dict(variant="cnn_only",    use_attention=False, schedule="constant")),
    ("A2_cnn_attention",     dict(variant="cnn_only",    use_attention=True,  schedule="constant")),
    ("A3_mert_meanpool",     dict(variant="mert_only",   use_attention=False, schedule="constant")),
    ("A4_mert_attention",    dict(variant="mert_only",   use_attention=True,  schedule="constant")),
    ("A5_fixed_ensemble",    dict(variant="fixed",       use_attention=False, schedule="constant")),
    ("A6_global_learned",    dict(variant="global",      use_attention=False, schedule="constant")),
    ("A7_conditional_nocurr",dict(variant="conditional", use_attention=True,  schedule="constant")),
    (PROPOSED_ROW,           dict(variant="conditional", use_attention=True,  schedule="cosine")),
    ("A9a_escf_linear",      dict(variant="conditional", use_attention=True,  schedule="linear")),
    ("A9b_escf_step",        dict(variant="conditional", use_attention=True,  schedule="step")),
    ("A9c_escf_supervisor",  dict(variant="conditional", use_attention=True,  schedule="supervisor")),
]

# Rows that are baselines rather than variants of the proposal -- the comparison set
# the headline claim has to beat.
BASELINE_ROWS = {
    "A1_cnn_meanpool", "A2_cnn_attention", "A3_mert_meanpool", "A4_mert_attention",
    "A5_fixed_ensemble", "A6_global_learned", "A7_conditional_nocurr",
}


def search_fixed_alpha(base_cfg: dict, seed: int) -> float:
    """Pick the global ensemble weight on validation -- never on test."""
    best_alpha, best_f1 = 0.5, -1.0
    for alpha in np.round(np.arange(0.0, 1.01, 0.05), 2):
        cfg = {**base_cfg, "seed": seed, "fixed_alpha": float(alpha), "row": "_alpha_search", "epochs": 1}
        summary = train_fusion(cfg)
        if summary["val_macro_f1"] > best_f1:
            best_f1, best_alpha = summary["val_macro_f1"], float(alpha)
    log.info("A5 grid search picked alpha=%.2f (val macro-F1 %.4f)", best_alpha, best_f1)
    return best_alpha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", default="cnn_tuned_lr")
    parser.add_argument("--run-b", default="mert_probe_baseline")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    base = {"run_a": args.run_a, "run_b": args.run_b, "epochs": args.epochs}
    results: dict[str, list[dict]] = {}

    for row_name, row_cfg in ROWS:
        cfgs = {**base, **row_cfg}
        if row_cfg["variant"] == "fixed":
            cfgs["fixed_alpha"] = search_fixed_alpha(cfgs, seed=0)

        runs = []
        for seed in range(args.seeds):
            summary = train_fusion({**cfgs, "seed": seed, "row": row_name})
            runs.append(summary)
            log.info(
                "%-24s seed=%d  test_acc=%.4f  test_macro_f1=%.4f",
                row_name, seed, summary["test_accuracy"], summary["test_macro_f1"],
            )
        results[row_name] = runs

    aggregate = {}
    for row_name, runs in results.items():
        acc = np.array([r["test_accuracy"] for r in runs])
        f1 = np.array([r["test_macro_f1"] for r in runs])
        aggregate[row_name] = {
            "accuracy_mean": float(acc.mean()), "accuracy_std": float(acc.std(ddof=1) if len(acc) > 1 else 0.0),
            "macro_f1_mean": float(f1.mean()), "macro_f1_std": float(f1.std(ddof=1) if len(f1) > 1 else 0.0),
            "seeds": len(runs),
        }

    # Significance of the proposal against the strongest baseline, seed 0 vs seed 0
    # (paired tests need per-track predictions from one run each).
    best_baseline = max(BASELINE_ROWS, key=lambda r: aggregate[r]["macro_f1_mean"])
    labels = np.load(FUSION_DIR / f"{PROPOSED_ROW}_seed0" / "test_labels.npy")
    preds_proposed = np.load(FUSION_DIR / f"{PROPOSED_ROW}_seed0" / "test_preds.npy")
    preds_baseline = np.load(FUSION_DIR / f"{best_baseline}_seed0" / "test_preds.npy")

    mcnemar = mcnemar_test(labels, preds_proposed, preds_baseline)
    diff, ci, p_boot = paired_bootstrap(labels, preds_proposed, preds_baseline)
    significance = {
        "proposed": PROPOSED_ROW,
        "strongest_baseline": best_baseline,
        "mcnemar_p": mcnemar.p_value,
        "mcnemar_detail": mcnemar.detail,
        "macro_f1_difference": diff,
        "bootstrap_ci_95": ci,
        "bootstrap_p": p_boot,
    }

    out = {"aggregate": aggregate, "significance": significance, "run_a": args.run_a, "run_b": args.run_b}
    FUSION_DIR.mkdir(parents=True, exist_ok=True)
    with open(FUSION_DIR / "ablation_results.json", "w") as f:
        json.dump(out, f, indent=2)

    lines = [
        "# ESCF ablation grid",
        "",
        f"Backbones: `{args.run_a}` (A) and `{args.run_b}` (B). "
        f"Official artist-filtered split, track-level scoring, {args.seeds} seeds (mean +/- std).",
        "",
        "| Row | Test accuracy | Test macro-F1 |",
        "|---|---|---|",
    ]
    for row_name, _ in ROWS:
        a = aggregate[row_name]
        lines.append(
            f"| {row_name} | {a['accuracy_mean']:.4f} +/- {a['accuracy_std']:.4f} "
            f"| {a['macro_f1_mean']:.4f} +/- {a['macro_f1_std']:.4f} |"
        )
    lines += [
        "",
        f"Strongest baseline: **{best_baseline}**. "
        f"McNemar p = {mcnemar.p_value:.4g} ({mcnemar.detail}). "
        f"Macro-F1 difference {diff:+.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}], bootstrap p = {p_boot:.4g}.",
    ]
    (FUSION_DIR / "ablation_table.md").write_text("\n".join(lines) + "\n")
    log.info("Wrote %s", FUSION_DIR / "ablation_table.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
