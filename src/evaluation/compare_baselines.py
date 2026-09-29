"""Compare our trained models against published FMA-small genre classification results.

The table below was built by directly reading primary sources (arXiv PDFs, journal
articles) rather than trusting secondary summaries -- several numbers that initially
looked plausible (e.g. a claimed 89-99% accuracy on FMA-small) could not be verified
against an actual paper and were excluded rather than guessed at. Where the original
FMA paper's own genre-classification number is cited, two different secondary sources
report slightly different values (54.8% vs 58.0%), likely reflecting different
feature/classifier configurations from the FMA repo's own baselines notebook; both are
kept as separate rows with their respective citing source, since we can't adjudicate
between them without re-running the original notebook.

Usage:
    python -m src.evaluation.compare_baselines --runs cnn_augmented ast_baseline

Reads outputs/<run>/eval_metrics.json for each run given, and writes
outputs/comparison_table.md (or .json) with the full, un-cherry-picked comparison
-- every baseline is listed regardless of whether we beat it.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass

from src.config import OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PublishedBaseline:
    citation: str
    year: int
    method: str
    accuracy_pct: float
    macro_f1_pct: float | None
    source_url: str
    note: str = ""
    # How the number was measured. "artist_filtered" = FMA's official split, where an
    # artist appears in only one of train/val/test. "clip_random" = clips shuffled into
    # splits, so clips of one track appear in both train and test. "unspecified" = the
    # paper does not say; we mark it rather than guess, in keeping with this module's
    # policy of excluding what cannot be verified against a primary source.
    protocol: str = "unspecified"
    eval_unit: str = "unspecified"  # "clip" | "track" | "unspecified"


# Verified 2026-09-11 by reading each primary source directly (see module docstring
# for the methodology and what was excluded as unverifiable).
PUBLISHED_BASELINES: list[PublishedBaseline] = [
    PublishedBaseline(
        citation="Defferrard et al., FMA baseline (as cited by Bian et al. 2019)",
        year=2017,
        method="Hand-crafted audio features + SVM",
        accuracy_pct=54.8,
        macro_f1_pct=None,
        source_url="https://arxiv.org/pdf/1906.11620",
        note="Original FMA dataset paper's own baseline; two papers cite slightly "
        "different values for it (see next row) -- both kept, unresolved discrepancy.",
        protocol="artist_filtered",
        eval_unit="track",
    ),
    PublishedBaseline(
        citation="Defferrard et al., FMA baseline (as cited by Rai & Bhattacharjee 2026)",
        year=2017,
        method="Hand-crafted audio features + SVM",
        accuracy_pct=58.0,
        macro_f1_pct=None,
        source_url="https://arxiv.org/pdf/2603.13871",
        protocol="artist_filtered",
        eval_unit="track",
    ),
    PublishedBaseline(
        citation="Lee, Lee, Park & Lee, transfer learning CNN (as cited by Bian et al. 2019)",
        year=2019,
        method="Transfer learning CNN from user-item log data",
        accuracy_pct=51.2,
        macro_f1_pct=None,
        source_url="https://arxiv.org/pdf/1906.11620",
        protocol="unspecified",
        eval_unit="unspecified",
    ),
    PublishedBaseline(
        citation="Park et al., transfer learning CNN (as cited by Bian et al. 2019)",
        year=2018,
        method="Transfer learning CNN using artist labels",
        accuracy_pct=56.8,
        macro_f1_pct=None,
        source_url="https://arxiv.org/pdf/1906.11620",
        protocol="unspecified",
        eval_unit="unspecified",
    ),
    PublishedBaseline(
        citation="Liu, DeMori & Abayomi, CNN14 (PANNs)",
        year=2022,
        method="AudioSet-pretrained CNN14, closed-set fine-tune",
        accuracy_pct=60.9,
        macro_f1_pct=None,
        source_url="https://arxiv.org/html/2209.07548v2",
        protocol="unspecified",
        eval_unit="track",
    ),
    PublishedBaseline(
        citation="Rai & Bhattacharjee, BYOL-A + DNN",
        year=2026,
        method="BYOL-A pretrained audio embeddings + tuned DNN classifier",
        accuracy_pct=64.3,
        macro_f1_pct=None,
        source_url="https://arxiv.org/pdf/2603.13871",
        protocol="unspecified",
        eval_unit="unspecified",
    ),
    PublishedBaseline(
        citation="Bian, Wang, Zhuang et al., DenseNet + augmentation",
        year=2019,
        method="DenseNet + time-overlap/pitch-shift augmentation + SVM stacking",
        accuracy_pct=68.9,
        macro_f1_pct=None,
        source_url="https://arxiv.org/pdf/1906.11620",
        protocol="unspecified",
        eval_unit="unspecified",
    ),
    PublishedBaseline(
        citation="Ashraf, Abid, Raza et al., dual residual CNN",
        year=2025,
        method="Dual CNN with modified residual learning + hybrid pooling",
        accuracy_pct=68.8,
        macro_f1_pct=70.55,
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12520411/",
        protocol="clip_random",
        eval_unit="clip",
    ),
]


def load_our_results(run_names: list[str]) -> list[dict]:
    results = []
    for run_name in run_names:
        eval_path = OUTPUT_DIR / run_name / "eval_metrics.json"
        if not eval_path.exists():
            log.warning("No eval_metrics.json for run %r (run src.evaluation.metrics first); skipping", run_name)
            continue
        metrics = json.loads(eval_path.read_text())
        # eval_metrics.json nests clip_level/track_level since track-level scoring was
        # added; fall back to the flat shape so older result files still load.
        track = metrics.get("track_level", metrics)
        clip = metrics.get("clip_level", track)
        results.append(
            {
                "run_name": run_name,
                "accuracy_pct": track.get("accuracy", 0) * 100,
                "macro_f1_pct": track.get("macro_f1", 0) * 100,
                "clip_accuracy_pct": clip.get("accuracy", 0) * 100,
                "clip_macro_f1_pct": clip.get("macro_f1", 0) * 100,
            }
        )
    return results


def _baseline_row(b: PublishedBaseline) -> str:
    f1_str = f"{b.macro_f1_pct:.1f}%" if b.macro_f1_pct is not None else "--"
    return (
        f"| {b.citation} ({b.year}) -- {b.method} | {b.accuracy_pct:.1f}% | {f1_str} "
        f"| {b.protocol} | {b.eval_unit} | {b.source_url} |"
    )


def build_comparison_table(our_results: list[dict]) -> str:
    """Two tables, because these numbers are not all measuring the same thing.

    Published FMA-small accuracies are reported under at least two incompatible
    protocols. FMA's official split applies an artist filter, so no artist -- let alone
    track -- spans train and test. Ashraf et al. (2025) instead split 3-second clips
    8:1:1, which places clips of the same track on both sides; FMA's own paper warns
    that this "leads to over-optimistic accuracy". Ranking all of them in one column
    would silently compare a leakage-inflated number against a clean one, so the
    strict-protocol table below is the one our headline claim is made against, and the
    leakage-matched replication is reported separately and labelled as such.
    """
    strict = [b for b in PUBLISHED_BASELINES if b.protocol != "clip_random"]
    leaky = [b for b in PUBLISHED_BASELINES if b.protocol == "clip_random"]

    header = "| Model | Accuracy | Macro-F1 | Split protocol | Eval unit | Source |"
    divider = "|---|---|---|---|---|---|"

    lines = [
        "## Strict protocol (FMA official artist-filtered split)",
        "",
        "Our numbers are track-level: per-episode probabilities averaged within a track,",
        "which is the unit published work reports. Baselines whose protocol the source",
        "does not state are marked `unspecified` rather than assumed comparable.",
        "",
        header,
        divider,
    ]
    for r in sorted(our_results, key=lambda x: -x["accuracy_pct"]):
        lines.append(
            f"| **Ours -- {r['run_name']}** | **{r['accuracy_pct']:.1f}%** | "
            f"**{r['macro_f1_pct']:.1f}%** | artist_filtered | track | this work |"
        )
    for b in sorted(strict, key=lambda x: -x.accuracy_pct):
        lines.append(_baseline_row(b))

    best = max((r["accuracy_pct"] for r in our_results), default=0.0)
    beaten = sum(1 for b in strict if best > b.accuracy_pct)
    lines += [
        "",
        f"Our best run beats {beaten} of {len(strict)} baselines in this table on accuracy "
        f"(every baseline listed, nothing omitted).",
        "",
        "## Leakage-matched protocol (clip-level random split)",
        "",
        "Reported only to quantify the premium that same-track leakage buys. These",
        "numbers are **not** a like-for-like claim against the strict table above, and",
        "must not be quoted as a headline result.",
        "",
        header,
        divider,
    ]
    for b in sorted(leaky, key=lambda x: -x.accuracy_pct):
        lines.append(_baseline_row(b))

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Run names to include (subdirs of outputs/)")
    parser.add_argument("--out", default=str(OUTPUT_DIR / "comparison_table.md"))
    args = parser.parse_args()

    our_results = load_our_results(args.runs)
    if not our_results:
        raise SystemExit("No evaluated runs found -- run src.evaluation.metrics on at least one run first.")

    table = build_comparison_table(our_results)
    with open(args.out, "w") as f:
        f.write(table)
    log.info("Wrote comparison table to %s", args.out)
    print(table)


if __name__ == "__main__":
    main()
