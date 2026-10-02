"""Compare our trained models against published FMA-small genre classification results.

Every row below was read from the paper's own text, not from a secondary summary, and
the `protocol` field records what the paper SAYS rather than what its accuracy implies
-- inferring the latter from the former would beg the question this project exists to
answer. Numbers that could not be traced to a primary source are excluded, with the
reason recorded in EXCLUDED_CLAIMS below rather than silently dropped.

One correction is worth calling out, because it was wrong here for a long time. This
module used to carry two rows for "Defferrard et al.", at 54.8% and 58.0%, on the theory
that two secondary sources had cited different configurations of FMA's own baselines
notebook. That was wrong in both directions:

  - 58.0% does not exist. The FMA paper proposes "single top genre on the balanced small
    subset" as problem 1 and never reports it; its only accuracy table (Table 6) is
    captioned "...on the MEDIUM subset" -- 25,000 tracks, 16 genres -- and 58 is the
    MLP/all-features cell there. The repo's executed baselines.ipynb (outputs branch)
    subsets to `<= 'medium'` and prints 19,922/2,505/2,573 over 16 genres. Issue
    mdeff/fma#69 (Dec 2023) asks the authors for small-subset baselines and is
    unanswered. The 58.0 attribution traces to Rai & Bhattacharjee's Tables 1 and 3,
    cited to no page.
  - 54.8% is real but is not Defferrard's. It is Park et al. 2018, Table 4, row
    `SVM [7] -- 0.5482 dagger`, footnoted "this result was obtained using the provided
    code and dataset in [7]" -- i.e. Park's own re-run of FMA's released baseline code
    on fma_small. It is artist-filtered because that code reads FMA's `split` column,
    not because any author claims it is. Bian et al. later reprint it as
    "SVM (Defferrard, 2017) 54.8", which is how the attribution drifted.

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
    # How the number was measured, as a machine-readable bucket:
    #   "artist_filtered"  FMA's official split; an artist appears in only one partition
    #   "random_track"     random split over whole tracks; artists span splits
    #   "clip_random"      clips shuffled into splits, so one track lands on both sides
    #   "unspecified"      the paper does not say
    # We mark rather than guess. A high accuracy is not evidence of a protocol here;
    # treating it as such would assume the conclusion this project is testing.
    protocol: str = "unspecified"
    eval_unit: str = "unspecified"  # "clip" | "track" | "unspecified"
    # Verbatim or near-verbatim of what the paper discloses, for the prose column.
    stated_protocol: str = "no split description at all"


# Verified 2026-09-11 by reading each primary source directly (see module docstring
# for the methodology and what was excluded as unverifiable).
PUBLISHED_BASELINES: list[PublishedBaseline] = [
    PublishedBaseline(
        citation="Features+SVM, re-run by Park et al. of FMA's released baseline code",
        year=2018,
        method="Hand-crafted audio features + SVM (FMA's own baselines code)",
        accuracy_pct=54.8,
        macro_f1_pct=None,
        source_url="https://arxiv.org/abs/1710.06648",
        note="Park Table 4, row `SVM [7] -- 0.5482 dagger`. Commonly misattributed to "
        "Defferrard et al., who never published any fma_small accuracy.",
        protocol="artist_filtered",
        eval_unit="track",
        stated_protocol="artist-filtered *by construction* -- the code reads FMA's "
        "`split` column -- but not asserted in prose",
    ),
    PublishedBaseline(
        citation="Park, Lee, Park, Ha & Nam, transfer learning from artist labels",
        year=2018,
        method="Transfer learning CNN using artist labels",
        accuracy_pct=56.9,
        macro_f1_pct=None,
        source_url="https://arxiv.org/abs/1710.06648",
        protocol="unspecified",
        eval_unit="unspecified",
        stated_protocol="none for FMA; the same paper *does* specify an artist-separated "
        "split for GTZAN, so the awareness is present and the disclosure is not",
    ),
    PublishedBaseline(
        citation="Kostrzewa, Kaminski & Brzeski, CNN/RNN/CRNN ensembles",
        year=2021,
        method="CNN, RNN and CRNN ensembles on one mel-spectrogram per song",
        accuracy_pct=56.4,
        macro_f1_pct=54.9,
        source_url="https://doi.org/10.1007/978-3-030-77961-0_6",
        protocol="unspecified",
        eval_unit="track",
        stated_protocol='"divided into three independent subsets in an 80:10:10 ratio", '
        "mechanism unstated. One spectrogram per song, so clip leakage is structurally "
        "impossible; artist overlap is the open question",
    ),
    PublishedBaseline(
        citation="Liu, DeMori & Abayomi, CNN14 (PANNs), open-set study",
        year=2022,
        method="AudioSet-pretrained CNN14, closed-set sub-metric of an open-set experiment",
        accuracy_pct=60.9,
        macro_f1_pct=None,
        source_url="https://arxiv.org/html/2209.07548v2",
        note="60.9 is the closed-set sub-metric Acc_c from their Table 3, whose test set "
        "also contains unknown classes drawn from fma_medium -- defensible to read as an "
        "8-way fma_small number, but not exactly what they measured.",
        protocol="unspecified",
        eval_unit="track",
        stated_protocol="partitions *are* described and the 6,400/800 counts match FMA's "
        "official sizes; whether those splits are artist-conditional is never stated",
    ),
    PublishedBaseline(
        citation="K. Rai & M. Bhattacharjee, BYOL-A + DNN (PReMI)",
        year=2025,
        method="BYOL-A pretrained audio embeddings + tuned DNN classifier",
        accuracy_pct=64.3,
        macro_f1_pct=None,
        source_url="https://arxiv.org/pdf/2603.13871",
        protocol="unspecified",
        eval_unit="track",
        stated_protocol='none -- the strings "split", "artist", "fold", "partition", '
        '"random" and "stratif" do not occur in the paper',
    ),
    PublishedBaseline(
        citation="Bian, Wang, Zhuang et al., DenseNet + augmentation",
        year=2019,
        method="DenseNet + time-overlap/pitch-shift augmentation + SVM stacking",
        accuracy_pct=68.9,
        macro_f1_pct=None,
        source_url="https://arxiv.org/pdf/1906.11620",
        protocol="unspecified",
        eval_unit="track",
        stated_protocol='none -- no split sentence anywhere. Scoring is track-level: '
        'segment predictions averaged into "a single song-level prediction"',
    ),
    PublishedBaseline(
        citation="Ashraf, Abid, Raza et al., dual residual CNN",
        year=2025,
        method="Dual CNN with modified residual learning + hybrid pooling",
        accuracy_pct=68.5,
        macro_f1_pct=70.55,
        source_url="https://doi.org/10.1371/journal.pone.0333808",
        protocol="clip_random",
        eval_unit="clip",
        stated_protocol='clip-random, stated outright: "we segmented all music files into '
        '3-second clips and split them into training, validation, and test sets using an '
        '8:1:1 ratio" -- segmentation precedes splitting',
    ),
    PublishedBaseline(
        citation="Ma et al., CT-GateNet",
        year=2026,
        method="CNN + Transformer with gated channel-spatial attention",
        accuracy_pct=89.42,
        macro_f1_pct=None,
        source_url="https://doi.org/10.1371/journal.pone.0344606",
        protocol="unspecified",
        eval_unit="unspecified",
        stated_protocol='"all datasets were evenly split into training, validation and '
        'test sets at an 8:1:1 ratio" -- the paper never says 8:1:1 *of what*',
    ),
    PublishedBaseline(
        citation="Zhang, TS-Resformer",
        year=2025,
        method="ResNet + Transformer multimodal fusion",
        accuracy_pct=90.23,
        macro_f1_pct=None,
        source_url="https://doi.org/10.3389/fnbot.2025.1568811",
        protocol="random_track",
        eval_unit="clip",
        stated_protocol="random 8:1:1 over *songs*, so the 12s/50%-overlap segmentation "
        "sits inside each partition rather than across them. Track-level random; artists "
        "unaddressed",
    ),
]


# Numbers that circulate as FMA-small records but do not survive checking against a
# primary source. Kept here, with the reason, rather than silently dropped -- a reader
# who has seen these elsewhere deserves to know why they are absent from the table.
EXCLUDED_CLAIMS: list[tuple[str, float, str]] = [
    (
        "Defferrard et al., Features+SVM (the field's assumed artist-filtered baseline)",
        58.0,
        "Does not exist. The FMA paper proposes 'single top genre on the balanced small "
        "subset' as problem 1 and never reports it; its only accuracy table (Table 6) is "
        "captioned '...on the medium subset' (25,000 tracks, 16 genres), where 58 is the "
        "MLP/all-features cell. The repo's executed baselines.ipynb subsets to "
        "`<= 'medium'`. Issue mdeff/fma#69 (Dec 2023) asks for small-subset baselines and "
        "is unanswered. Attribution traces to Rai & Bhattacharjee Tables 1 and 3, "
        "cited to no page.",
    ),
    (
        "CapsNet + UIGMM (Scientific Reports 2024)",
        68.87,
        "That paper runs only on ISMIR2004, GTZAN and Extended Ballroom. The 68.87% "
        "appears in its related-work discussion of Ashraf et al. and is not its own result.",
    ),
    (
        "DPADMM-G, parallel structured sparse SVM",
        97.27,
        "A binary two-genre task on FMA's precomputed feature vectors: 'we randomly "
        "selected two categories from the eight categories in fma_small... 800 samples as "
        "the training set and the remaining 200 as the test set'. Not 8-way audio "
        "classification.",
    ),
    (
        "Lee, Lee, Park & Lee, transfer learning CNN",
        51.2,
        "Carried here for a long time on the strength of a row in Bian et al.'s comparison "
        "table. The underlying citation could not be resolved to a paper reporting an "
        "fma_small number; dropped rather than guessed at.",
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
        f"| {b.stated_protocol} | {b.eval_unit} | {b.source_url} |"
    )


def build_comparison_table(our_results: list[dict]) -> str:
    """One registry, ordered by accuracy, with what each paper discloses alongside.

    We deliberately do not sort published numbers into "clean" and "leaky" buckets and
    rank ourselves against the clean one. That framing assumes we can tell which bucket
    each paper belongs in, and the whole finding here is that we usually cannot: of the
    external works in this registry, none states artist-filtered evaluation in prose,
    one states clip-random, one states a track-level random split, and the rest give a
    ratio or nothing. Sorting by inferred protocol would launder an assumption into a
    result.

    So: one table, accuracy-ordered, with a prose column recording what the paper
    actually says. The reader can see for themselves which rows are comparable to ours.
    `outputs/protocol_leakage_table.md` (from scripts/protocol_leakage_analysis.py)
    carries the measured cost of each protocol choice.
    """
    header = "| Model | Accuracy | Macro-F1 | Stated protocol | Eval unit | Source |"
    divider = "|---|---|---|---|---|---|"

    lines = [
        "## Published FMA-small results, with what each paper discloses",
        "",
        "Our numbers are track-level: per-episode probabilities averaged within a track,",
        "which is the unit published work reports. The `Stated protocol` column records",
        "what each source's own text says -- never what its accuracy implies, since",
        "inferring one from the other would assume the conclusion.",
        "",
        header,
        divider,
    ]
    rows: list[tuple[float, str]] = [
        (
            r["accuracy_pct"],
            f"| **Ours -- {r['run_name']}** | **{r['accuracy_pct']:.1f}%** | "
            f"**{r['macro_f1_pct']:.1f}%** | **official artist-filtered** | track | this work |",
        )
        for r in our_results
    ]
    rows += [(b.accuracy_pct, _baseline_row(b)) for b in PUBLISHED_BASELINES]
    lines += [row for _, row in sorted(rows, key=lambda x: -x[0])]

    n_af = sum(1 for b in PUBLISHED_BASELINES if b.protocol == "artist_filtered")
    n_cr = sum(1 for b in PUBLISHED_BASELINES if b.protocol == "clip_random")
    n_rt = sum(1 for b in PUBLISHED_BASELINES if b.protocol == "random_track")
    n_un = sum(1 for b in PUBLISHED_BASELINES if b.protocol == "unspecified")
    best = max((r["accuracy_pct"] for r in our_results), default=0.0)
    comparable = [b for b in PUBLISHED_BASELINES if b.protocol == "artist_filtered"]
    best_comparable = max((b.accuracy_pct for b in comparable), default=0.0)

    lines += [
        "",
        f"Of {len(PUBLISHED_BASELINES)} external results: {n_af} artist-filtered (and that "
        f"one only by construction, not by authorial claim), {n_cr} stated clip-random, "
        f"{n_rt} stated track-level random, {n_un} undisclosed.",
        "",
        f"Our best run is {best:.1f}%. The only artist-filtered number it can honestly be "
        f"compared against is {best_comparable:.1f}%, a margin of "
        f"{best - best_comparable:+.1f}pp. We do not claim a state of the art: the higher "
        "numbers in this table have undisclosed protocols, and the measured spread between "
        "protocols (see `outputs/protocol_leakage_table.md`) is larger than the spread "
        "between the architectures.",
        "",
        "## Circulating claims excluded from the registry",
        "",
        "Kept here with reasons rather than silently dropped -- a reader who has seen these",
        "elsewhere deserves to know why they are absent.",
        "",
        "| Claim | Reported | Why excluded |",
        "|---|---|---|",
    ]
    for claim, pct, reason in EXCLUDED_CLAIMS:
        lines.append(f"| {claim} | {pct:.2f}% | {reason} |")

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
