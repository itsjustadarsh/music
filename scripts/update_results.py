"""Patches docs/RESULTS.md in place with real numbers from outputs/*/eval_metrics.json.

Replaces the content between the OUR_MODELS_TABLE, BASELINE_TABLE, and SUMMARY_LINE
marker comments in docs/RESULTS.md -- everything else in the file (the published
baselines, the "what this comparison does and doesn't claim" section, etc.) is left
untouched. Safe to re-run any time after training more runs or re-evaluating.

Usage:
    python scripts/update_results.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR  # noqa: E402
from src.evaluation.compare_baselines import PUBLISHED_BASELINES, load_our_results  # noqa: E402

RESULTS_MD = PROJECT_ROOT / "docs" / "RESULTS.md"


def find_all_run_names() -> list[str]:
    if not OUTPUT_DIR.exists():
        return []
    return sorted(
        d.name for d in OUTPUT_DIR.iterdir()
        if d.is_dir() and (d / "eval_metrics.json").exists() and (d / "config.json").exists()
    )


def best_per_model_type(run_names: list[str]) -> dict[str, dict]:
    """Returns {'cnn': best_result_dict_or_None, 'ast': best_result_dict_or_None}."""
    best: dict[str, dict] = {}
    for run_name in run_names:
        cfg = json.loads((OUTPUT_DIR / run_name / "config.json").read_text())
        metrics = json.loads((OUTPUT_DIR / run_name / "eval_metrics.json").read_text())
        model_type = cfg["model"]
        track = metrics.get("track_level", metrics)
        entry = {"run_name": run_name, "accuracy_pct": track["accuracy"] * 100, "macro_f1_pct": track["macro_f1"] * 100}
        if model_type not in best or entry["macro_f1_pct"] > best[model_type]["macro_f1_pct"]:
            best[model_type] = entry
    return best


def replace_block(text: str, marker: str, new_content: str) -> str:
    pattern = re.compile(
        rf"(<!-- {marker}_START -->\n).*?(\n<!-- {marker}_END -->)", re.DOTALL
    )
    if not pattern.search(text):
        raise RuntimeError(f"Could not find {marker} markers in {RESULTS_MD} -- was it edited?")
    return pattern.sub(lambda m: m.group(1) + new_content + m.group(2), text)


def main() -> None:
    run_names = find_all_run_names()
    if not run_names:
        print("No evaluated runs found under outputs/ -- train and evaluate at least one run first.")
        return

    best = best_per_model_type(run_names)
    all_results = load_our_results(run_names)

    our_table = ["| Model | Config | Test Accuracy | Test Macro-F1 |", "|---|---|---|---|"]
    labels = {"cnn": "GenreCNN", "ast": "ASTGenreClassifier"}
    for model_type, label in labels.items():
        if model_type in best:
            b = best[model_type]
            our_table.append(f"| {label} | `{b['run_name']}` | {b['accuracy_pct']:.1f}% | {b['macro_f1_pct']:.1f}% |")
        else:
            our_table.append(f"| {label} | (not yet trained/evaluated) | TBD | TBD |")

    baseline_table = ["| Model | Accuracy | Macro-F1 | Source |", "|---|---|---|---|"]
    for model_type, label in labels.items():
        if model_type in best:
            b = best[model_type]
            baseline_table.append(f"| **Ours -- {label}** ({b['run_name']}) | **{b['accuracy_pct']:.1f}%** | **{b['macro_f1_pct']:.1f}%** | this work |")
    for b in sorted(PUBLISHED_BASELINES, key=lambda x: -x.accuracy_pct):
        f1_str = f"{b.macro_f1_pct:.1f}%" if b.macro_f1_pct is not None else "--"
        baseline_table.append(f"| {b.citation} ({b.year}) | {b.accuracy_pct:.1f}% | {f1_str} | {b.source_url} |")

    best_acc = max((b["accuracy_pct"] for b in best.values()), default=0.0)
    n_beaten = sum(1 for pub in PUBLISHED_BASELINES if best_acc > pub.accuracy_pct)
    best_label = max(best.items(), key=lambda kv: kv[1]["accuracy_pct"])[0] if best else None
    summary_line = (
        f"**Result**: our best model ({labels.get(best_label, best_label)}, {best_acc:.1f}% accuracy) "
        f"beats {n_beaten} of {len(PUBLISHED_BASELINES)} published FMA-small baselines listed above."
        if best_label else "No trained/evaluated runs yet."
    )

    status_line = (
        "> **Status: populated from a real training run.** Regenerate this file any time "
        "after training/evaluating more runs with `python scripts/update_results.py`."
    )

    text = RESULTS_MD.read_text()
    text = replace_block(text, "STATUS", status_line)
    text = replace_block(text, "OUR_MODELS_TABLE", "\n".join(our_table))
    text = replace_block(text, "BASELINE_TABLE", "\n".join(baseline_table))
    text = replace_block(text, "SUMMARY_LINE", summary_line)
    RESULTS_MD.write_text(text)
    print(f"Updated {RESULTS_MD} with {len(run_names)} evaluated run(s).")
    print(summary_line)


if __name__ == "__main__":
    main()
