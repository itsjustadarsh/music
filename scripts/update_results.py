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

LABELS = {
    "cnn": "GenreCNN",
    "ast": "ASTGenreClassifier",
    "densenet": "DenseNet121 (Bian et al. repro)",
    "dual_cnn": "Dual residual CNN (Ashraf et al. repro)",
    "cnn_bigru": "CNN+BiGRU (Kostrzewa et al. repro)",
    "capsnet": "CapsNet",
    "cnn_transformer": "CNN+Transformer (CT-GateNet repro)",
    "embedding_probe:byola": "BYOL-A + DNN (Rai & Bhattacharjee repro)",
    "embedding_probe:panns": "PANNs CNN14 probe (Liu et al. repro)",
    "mert_probe": "MERT-95M probe",
    "mert_flat_probe": "MERT-95M flat probe",
    "mert_deep_probe": "MERT-95M deep probe",
    "mert_attn_probe": "MERT-95M attention probe",
    "mert_large_probe": "MERT-330M probe",
    "mert_large_flat_probe": "MERT-330M flat probe",
    "mert_large_deep_probe": "MERT-330M deep probe",
    "xgboost_mert": "XGBoost on MERT embeddings",
}


def find_all_run_names() -> list[str]:
    """Only runs on the official artist-filtered split.

    Runs trained under `random_clip` or `random_track` exist solely as the protocol-ladder
    control arm. Letting one into this table would print a leaked number under the heading
    "official artist-filtered" -- the exact substitution this project exists to document.
    """
    if not OUTPUT_DIR.exists():
        return []
    names = []
    for d in sorted(OUTPUT_DIR.iterdir()):
        if not (d.is_dir() and (d / "eval_metrics.json").exists() and (d / "config.json").exists()):
            continue
        cfg = json.loads((d / "config.json").read_text())
        if cfg.get("split_mode", "official") != "official":
            continue
        names.append(d.name)
    return names


def best_per_model_type(run_names: list[str]) -> dict[str, dict]:
    """Returns {model_type: best_result_by_macro_f1} over the given runs."""
    best: dict[str, dict] = {}
    for run_name in run_names:
        cfg = json.loads((OUTPUT_DIR / run_name / "config.json").read_text())
        metrics = json.loads((OUTPUT_DIR / run_name / "eval_metrics.json").read_text())
        model_type = cfg["model"]
        # BYOL-A and PANNs share the generic "embedding_probe" model type; keying on it
        # alone would collapse two different backbones into one row and hide whichever
        # scored lower.
        if model_type == "embedding_probe":
            model_type = f"embedding_probe:{cfg['embedding_name']}"
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
    for model_type, b in sorted(best.items(), key=lambda kv: -kv[1]["macro_f1_pct"]):
        label = LABELS.get(model_type, model_type)
        our_table.append(f"| {label} | `{b['run_name']}` | {b['accuracy_pct']:.1f}% | {b['macro_f1_pct']:.1f}% |")

    baseline_table = [
        "| Model | Accuracy | Macro-F1 | Stated protocol | Source |",
        "|---|---|---|---|---|",
    ]
    # One row for our single best artist-filtered model. Listing every architecture here
    # would pad the registry with repeats of the same split and obscure the comparison.
    if best:
        top_type, top = max(best.items(), key=lambda kv: kv[1]["accuracy_pct"])
        baseline_table.append(
            f"| **Ours -- {LABELS.get(top_type, top_type)}** ({top['run_name']}) | "
            f"**{top['accuracy_pct']:.1f}%** | **{top['macro_f1_pct']:.1f}%** | "
            "**official artist-filtered, track-level** | this work |"
        )
    for b in sorted(PUBLISHED_BASELINES, key=lambda x: -x.accuracy_pct):
        f1_str = f"{b.macro_f1_pct:.1f}%" if b.macro_f1_pct is not None else "--"
        baseline_table.append(
            f"| {b.citation} ({b.year}) | {b.accuracy_pct:.1f}% | {f1_str} "
            f"| {b.stated_protocol} | {b.source_url} |"
        )

    best_acc = max((b["accuracy_pct"] for b in best.values()), default=0.0)
    best_label = max(best.items(), key=lambda kv: kv[1]["accuracy_pct"])[0] if best else None
    # Deliberately NOT "beats N of M". Ranking against baselines whose protocol is
    # undisclosed would treat incomparable numbers as comparable, which is the exact
    # failure this project documents. We compare only against artist-filtered results.
    comparable = [b for b in PUBLISHED_BASELINES if b.protocol == "artist_filtered"]
    best_comparable = max((b.accuracy_pct for b in comparable), default=0.0)
    n_undisclosed = sum(1 for b in PUBLISHED_BASELINES if b.protocol == "unspecified")
    summary_line = (
        f"**Result**: our best model ({LABELS.get(best_label, best_label)}, "
        f"{best_acc:.1f}% accuracy) against the only artist-filtered published number, "
        f"{best_comparable:.1f}%, is {best_acc - best_comparable:+.1f}pp. The other "
        f"{n_undisclosed} of {len(PUBLISHED_BASELINES)} external results do not disclose "
        "their split, so they are not comparable in either direction -- see "
        "`outputs/protocol_leakage_table.md` for the measured cost of each protocol choice."
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
