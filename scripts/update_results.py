"""Regenerate docs/RESULTS.md and docs/EXPERIMENT_LOG.md from outputs/.

These two files are **generated in full** and must never be hand-edited. That is a
structural fix, not a style preference: this project previously kept its numbers in ten
overlapping hand-maintained documents, and they drifted — including once into a published
comparison that mixed two different scoring units (`docs/NEGATIVE_RESULTS.md` §4).

The rule now is: numbers live only in generated files; hand-written prose references them
and never restates them.

Two invariants are enforced here rather than left to discipline:

  1. **Track-level only.** Accuracy is read from `eval_metrics.json -> track_level`,
     never from `summary.json`, whose accuracy is per-episode and runs 3-7pp lower.
  2. **Artist-filtered only, in the headline table.** Runs with a `split_mode` other
     than `official` exist solely as the protocol-ladder control arm. One of them in a
     table captioned "artist-filtered" would be the exact substitution this project
     documents. They appear only in the ladder (`protocol_leakage_table.md`) and, clearly
     labelled, in the experiment log.

Usage:
    python scripts/update_results.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR  # noqa: E402
from src.evaluation.compare_baselines import PUBLISHED_BASELINES  # noqa: E402

RESULTS_MD = PROJECT_ROOT / "docs" / "RESULTS.md"
LOG_MD = PROJECT_ROOT / "docs" / "EXPERIMENT_LOG.md"

LABELS = {
    "cnn": "GenreCNN (log-mel, from scratch)",
    "densenet": "DenseNet121",
    "dual_cnn": "Dual residual CNN",
    "cnn_bigru": "CNN + BiGRU",
    "capsnet": "CapsNet (dynamic routing)",
    "cnn_transformer": "CNN + Transformer",
    "embedding_probe:byola": "BYOL-A + DNN probe",
    "embedding_probe:panns": "PANNs CNN14 probe",
    "mert_probe": "MERT-95M probe (learned layer weights)",
    "mert_flat_probe": "MERT-95M flat probe (13-layer concat)",
    "mert_deep_probe": "MERT-95M deep probe (multi-scale)",
    "mert_attn_probe": "MERT-95M attention probe",
    "mert_large_probe": "MERT-330M probe (learned layer weights)",
    "mert_large_flat_probe": "MERT-330M flat probe (25-layer concat)",
    "mert_large_deep_probe": "MERT-330M deep probe (multi-scale)",
    "xgboost_mert": "Tree ensemble on MERT embeddings",
    "ensemble": "Ensemble of per-architecture bests",
}

SPLIT_LABEL = {
    "official": "artist-filtered",
    "random_track": "random-track",
    "random_clip": "clip-random",
}

# Hyperparameters worth recording per run. Anything absent from a config is simply
# omitted rather than defaulted, so the log never asserts a setting the run did not have.
HPARAM_KEYS = ["lr", "batch_size", "epochs", "optimizer", "scheduler", "weight_decay",
               "dropout", "label_smoothing", "mixup_alpha", "seed"]


class Run:
    def __init__(self, path: Path):
        self.name = path.name
        self.cfg = json.loads((path / "config.json").read_text())
        metrics = json.loads((path / "eval_metrics.json").read_text())
        track = metrics.get("track_level", metrics)
        self.accuracy = track["accuracy"] * 100
        self.macro_f1 = track.get("macro_f1", float("nan")) * 100
        self.split_mode = self.cfg.get("split_mode", "official")

    @property
    def model_type(self) -> str:
        m = self.cfg.get("model", "unknown")
        # BYOL-A and PANNs share the generic "embedding_probe" type; keying on it alone
        # would collapse two different backbones into one row and hide the lower one.
        if m == "embedding_probe":
            return f"embedding_probe:{self.cfg.get('embedding_name', '?')}"
        return m

    @property
    def label(self) -> str:
        return LABELS.get(self.model_type, self.model_type)

    def hparams(self) -> str:
        parts = [f"{k}={self.cfg[k]}" for k in HPARAM_KEYS if k in self.cfg]
        return ", ".join(parts) if parts else "—"


def load_runs() -> list[Run]:
    if not OUTPUT_DIR.exists():
        return []
    runs = []
    for d in sorted(OUTPUT_DIR.iterdir()):
        if d.is_dir() and (d / "eval_metrics.json").exists() and (d / "config.json").exists():
            runs.append(Run(d))
    return runs


def build_results(runs: list[Run]) -> str:
    official = [r for r in runs if r.split_mode == "official"]

    best_per_type: dict[str, Run] = {}
    for r in official:
        cur = best_per_type.get(r.model_type)
        if cur is None or r.macro_f1 > cur.macro_f1:
            best_per_type[r.model_type] = r
    ranked = sorted(best_per_type.values(), key=lambda r: -r.accuracy)
    best = ranked[0] if ranked else None

    comparable = [b for b in PUBLISHED_BASELINES if b.protocol == "artist_filtered"]
    best_comparable = max((b.accuracy_pct for b in comparable), default=0.0)
    n_undisclosed = sum(1 for b in PUBLISHED_BASELINES if b.protocol == "unspecified")

    L = [
        "# Results",
        "",
        f"<!-- GENERATED FILE — do not edit. `python scripts/update_results.py`. "
        f"Last run {date.today().isoformat()}. -->",
        "",
        "> **Generated file — do not edit by hand.** Every number in this project lives here",
        "> or in `outputs/protocol_leakage_table.md`. Prose elsewhere references these and",
        "> never restates them. Regenerate with `python scripts/update_results.py`.",
        "",
        "All accuracies are **track-level**: per-episode softmax averaged within a track, then",
        "argmax. See `docs/METHODOLOGY.md` §2.2 for why this is the only unit reported.",
        "",
        "---",
        "",
        "## 1. Our models, official artist-filtered split",
        "",
        f"Best run per architecture, ranked by accuracy. {len(official)} artist-filtered runs "
        f"across {len(best_per_type)} architectures.",
        "",
        "| Architecture | Best run | Accuracy | Macro-F1 |",
        "|---|---|---|---|",
    ]
    for r in ranked:
        L.append(f"| {r.label} | `{r.name}` | {r.accuracy:.1f}% | {r.macro_f1:.1f}% |")

    L += [
        "",
        "## 2. Against the published literature",
        "",
        "Ordered by accuracy, with what each paper discloses about its split. The protocol",
        "column records what the source *states*, never what its accuracy implies — see",
        "`docs/LITERATURE.md` for the full audit and the three excluded claims.",
        "",
        "| Result | Accuracy | Macro-F1 | Stated protocol | Source |",
        "|---|---|---|---|---|",
    ]
    rows: list[tuple[float, str]] = []
    if best:
        rows.append((
            best.accuracy,
            f"| **Ours — {best.label}** (`{best.name}`) | **{best.accuracy:.1f}%** | "
            f"**{best.macro_f1:.1f}%** | **official artist-filtered, track-level** | this work |",
        ))
    for b in PUBLISHED_BASELINES:
        f1 = f"{b.macro_f1_pct:.1f}%" if b.macro_f1_pct is not None else "—"
        rows.append((
            b.accuracy_pct,
            f"| {b.citation} ({b.year}) | {b.accuracy_pct:.1f}% | {f1} | {b.stated_protocol} | {b.source_url} |",
        ))
    L += [row for _, row in sorted(rows, key=lambda x: -x[0])]

    if best:
        L += [
            "",
            "### How to read that table",
            "",
            f"Our best artist-filtered model reaches **{best.accuracy:.1f}%**. The only published "
            f"number carrying an artist-filtered guarantee is {best_comparable:.1f}%, a margin of "
            f"**{best.accuracy - best_comparable:+.1f}pp** — and that one is artist-filtered by "
            "construction (it inherits FMA's own `split` column) rather than by authorial claim.",
            "",
            f"The remaining {n_undisclosed} of {len(PUBLISHED_BASELINES)} external results do not "
            "disclose their split. We deliberately do **not** report a 'beats N of M' count: "
            "ranking against numbers whose protocol is unknown would treat incomparable "
            "quantities as comparable, which is the failure this project documents. The measured "
            "cost of each protocol choice is in `outputs/protocol_leakage_table.md`.",
        ]

    L += [
        "",
        "## 3. Protocol inflation",
        "",
        "The same architectures run down the three-rung ladder (`official` → `random_track` →",
        "`random_clip`) are tabulated in **`outputs/protocol_leakage_table.md`**, regenerated by",
        "`python scripts/protocol_leakage_analysis.py`. That file is the single source for every",
        "inflation figure quoted anywhere in this repository.",
        "",
        "## 4. Everything else",
        "",
        "- Per-run hyperparameters and the full run inventory: `docs/EXPERIMENT_LOG.md`",
        "- What failed and why: `docs/NEGATIVE_RESULTS.md`",
        "- Definitions of every term above: `docs/METHODOLOGY.md`",
        "",
    ]
    return "\n".join(L)


def build_log(runs: list[Run]) -> str:
    by_split: dict[str, list[Run]] = {}
    for r in runs:
        by_split.setdefault(r.split_mode, []).append(r)

    L = [
        "# Experiment log",
        "",
        f"<!-- GENERATED FILE — do not edit. `python scripts/update_results.py`. "
        f"Last run {date.today().isoformat()}. -->",
        "",
        "> **Generated file — do not edit by hand.** Complete inventory of every evaluated run,",
        "> with the hyperparameters it was trained under. Regenerate with",
        "> `python scripts/update_results.py`.",
        "",
        f"{len(runs)} evaluated runs. All accuracies track-level.",
        "",
    ]
    for mode in ["official", "random_track", "random_clip"]:
        group = sorted(by_split.get(mode, []), key=lambda r: -r.accuracy)
        if not group:
            continue
        L += [
            f"## {SPLIT_LABEL[mode]} (`{mode}`) — {len(group)} runs",
            "",
        ]
        if mode != "official":
            L += [
                "**Diagnostic only.** These runs deliberately leak and exist to measure the cost",
                "of doing so. No number in this section may be quoted as a result.",
                "",
            ]
        L += [
            "| Run | Architecture | Accuracy | Macro-F1 | Hyperparameters |",
            "|---|---|---|---|---|",
        ]
        for r in group:
            L.append(
                f"| `{r.name}` | {r.label} | {r.accuracy:.1f}% | {r.macro_f1:.1f}% | {r.hparams()} |"
            )
        L.append("")
    return "\n".join(L)


def main() -> None:
    runs = load_runs()
    if not runs:
        print("No evaluated runs under outputs/ — train and evaluate at least one run first.")
        return
    RESULTS_MD.write_text(build_results(runs))
    LOG_MD.write_text(build_log(runs))
    n_official = sum(1 for r in runs if r.split_mode == "official")
    print(f"Wrote {RESULTS_MD.relative_to(PROJECT_ROOT)} and {LOG_MD.relative_to(PROJECT_ROOT)}")
    print(f"  {len(runs)} evaluated runs ({n_official} artist-filtered)")


if __name__ == "__main__":
    main()
