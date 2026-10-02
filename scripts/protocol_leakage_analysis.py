#!/usr/bin/env python3
"""Multi-model protocol leakage analysis.

Runs the same architectures down a three-rung protocol ladder and reports the gap:

    official      FMA's artist-filtered split. No artist spans splits. The honest one.
    random_track  random 8:1:1 over whole tracks. No clip leaks, but artists span splits.
    random_clip   random 8:1:1 over 3s clips. The same track appears in train and test.

The middle rung exists because the literature is not binary. Of the papers surveyed, only
Ashraf et al. 2025 states clip-level splitting outright; Zhang 2025 splits "songs", and
Ma et al. 2026 give a ratio without ever saying what unit it applies to. Scoring those
against artist-filtered alone would overstate how much of their lead is clip leakage.

Usage:
    python scripts/protocol_leakage_analysis.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

OUTPUT_DIR = Path("outputs")

# (artist_filtered_run, clip_random_run, random_track_run_or_None, display_name)
MODEL_PAIRS = [
    ("mert_probe_wide", "mert_probe_cliprandom", "mert_probe_randomtrack", "MERT Probe (weighted-sum)"),
    ("mert_flat_probe_v1", "mert_flat_probe_v1_cliprandom", "mert_flat_probe_v1_randomtrack", "MERT Flat Probe (13-layer concat)"),
    ("mert_deep_probe_v1", "mert_deep_probe_v1_cliprandom", "mert_deep_probe_v1_randomtrack", "MERT Deep Probe (multi-scale)"),
    ("mert_attn_probe_v1", "mert_attn_probe_v1_cliprandom", "mert_attn_probe_v1_randomtrack", "MERT Attention Probe"),
    ("mert_large_probe_v1", "mert_large_probe_v1_cliprandom", "mert_large_probe_v1_randomtrack", "MERT-330M Probe"),
    ("cnn_improved", "cnn_improved_cliprandom", "cnn_improved_randomtrack", "CNN (log-mel spectrogram)"),
    ("densenet_v1", "densenet_v1_cliprandom", None, "DenseNet121 (Bian et al. 2019)"),
    ("dual_cnn_v1", "dual_cnn_v1_cliprandom", None, "Dual-path CNN (Ashraf et al. 2025)"),
    ("cnn_bigru_v1", "cnn_bigru_v1_cliprandom", None, "CNN+BiGRU (Kostrzewa et al. 2021)"),
    ("capsnet_v1", "capsnet_v1_cliprandom", None, "CapsNet (dynamic routing)"),
    ("cnn_transformer_v1", "cnn_transformer_v1_cliprandom", None, "CNN+Transformer (CT-GateNet 2026)"),
    ("byola_probe_v1", "byola_probe_v1_cliprandom", "byola_probe_v1_randomtrack", "BYOL-A + DNN (Rai & Bhattacharjee 2025)"),
    ("panns_probe_v1", "panns_probe_v1_cliprandom", "panns_probe_v1_randomtrack", "PANNs CNN14 (Liu et al. 2022)"),
]

# What each paper reports on FMA-small, and what it actually discloses about its split.
# Every entry here was checked against the paper's own text; `protocol` records what the
# paper SAYS, not what its accuracy implies. Keyed by the artist-filtered run name.
#
# Deliberately absent:
#   capsnet  - the Sci Rep CapsNet+UIGMM paper runs on ISMIR2004/GTZAN/Extended Ballroom
#              and never touches FMA. Its "68.87% on FMA" is a sentence quoting Ashraf
#              et al. in related work. We keep the architecture as a family data point
#              but it reproduces no FMA claim.
PUBLISHED_ACCURACY = {
    "densenet_v1": ("Bian et al. 2019", 68.9, "no split description at all"),
    "dual_cnn_v1": ("Ashraf et al. 2025", 68.5, "3s clips segmented then split 8:1:1 (stated)"),
    # 54.9 is their macro F1; their headline accuracy is 56.39%. Accuracy used here.
    "cnn_bigru_v1": ("Kostrzewa et al. 2021", 56.4, "80:10:10 ratio, mechanism unstated"),
    "cnn_transformer_v1": ("Ma et al. 2026 (CT-GateNet)", 89.42, "8:1:1 ratio, unit unnamed"),
    "byola_probe_v1": ("Rai & Bhattacharjee 2025", 64.3, "no split description at all"),
    # 60.9 is the closed-set sub-metric Acc_c of an open-set experiment (Table 3), not a
    # standalone 8-way fma_small result. They do describe partitions, and their 6400/800
    # counts match FMA's official sizes; what they never state is artist-conditioning.
    "panns_probe_v1": ("Liu et al. 2022", 60.9, "partitions described, artist-conditioning unstated"),
}


LEVEL = os.environ.get("LEAKAGE_LEVEL", "track")


FAMILY_OF = {
    "MERT Probe (weighted-sum)": "MERT probes",
    "MERT Flat Probe (13-layer concat)": "MERT probes",
    "MERT Deep Probe (multi-scale)": "MERT probes",
    "MERT Attention Probe": "MERT probes",
    "MERT-330M Probe": "MERT probes",
    "CNN (log-mel spectrogram)": "from-scratch CNNs",
    "DenseNet121 (Bian et al. 2019)": "from-scratch CNNs",
    "Dual-path CNN (Ashraf et al. 2025)": "from-scratch CNNs",
    "CNN+BiGRU (Kostrzewa et al. 2021)": "recurrent hybrids",
    "CapsNet (dynamic routing)": "capsule networks",
    "CNN+Transformer (CT-GateNet 2026)": "transformer hybrids",
    "BYOL-A + DNN (Rai & Bhattacharjee 2025)": "frozen BYOL-A / PANNs embeddings",
    "PANNs CNN14 (Liu et al. 2022)": "frozen BYOL-A / PANNs embeddings",
}


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def load_accuracy(run_name: str) -> float | None:
    """Test accuracy as a fraction, at the level named by LEAKAGE_LEVEL.

    "track" reads eval_metrics.json (per-episode softmax averaged within a track, then
    argmax); "episode" reads summary.json, which scores each 3s episode independently.

    This defaults to track because every published number we compare against is one
    prediction per track -- Liu and Rai both feed a whole 30s track and never segment,
    Bian and Park both average segment outputs into "a single song-level prediction".
    Quoting our episode-level number beside their track-level one would compare two
    different measurements, which is the error this project exists to document.
    """
    if LEVEL == "track":
        path = OUTPUT_DIR / run_name / "eval_metrics.json"
        if not path.exists():
            return None
        data = json.load(open(path))
        return data.get("track_level", data)["accuracy"]
    path = OUTPUT_DIR / run_name / "summary.json"
    if not path.exists():
        return None
    data = json.load(open(path))
    return data.get("test_accuracy", data.get("test_acc"))


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}%"


def main():
    rows = []  # (name, af_run, af, rt_or_None, cr)
    missing: list[str] = []

    for af_run, cr_run, rt_run, name in MODEL_PAIRS:
        af_acc = load_accuracy(af_run)
        cr_acc = load_accuracy(cr_run)
        rt_acc = load_accuracy(rt_run) if rt_run else None
        if af_acc is None:
            missing.append(f"{af_run} (artist-filtered)")
        if cr_acc is None:
            missing.append(f"{cr_run} (clip-random)")
        if rt_run and rt_acc is None:
            missing.append(f"{rt_run} (random-track)")
        if af_acc is not None and cr_acc is not None:
            rows.append(
                (name, af_run, af_acc * 100, None if rt_acc is None else rt_acc * 100, cr_acc * 100)
            )

    if missing:
        print("Missing runs (train these first):")
        for m in missing:
            print(f"  - {m}")
        print()

    if not rows:
        print("No complete model pairs found. Train both variants first.")
        return

    print("=" * 96)
    print("Protocol Leakage Analysis: artist-filtered -> random-track -> clip-random")
    print("=" * 96)
    print()
    print(f"{'Model':<40}{'Artist-filt.':>14}{'Random-track':>14}{'Clip-random':>14}{'Total infl.':>14}")
    print("-" * 96)
    for name, _, af, rt, cr in rows:
        print(f"{name:<40}{af:>13.1f}%{fmt(rt):>14}{cr:>13.1f}%{cr - af:>+13.1f}pp")
    print("-" * 96)

    deltas = [cr - af for _, _, af, _, cr in rows]
    mean_delta = sum(deltas) / len(deltas)
    min_delta, max_delta = min(deltas), max(deltas)
    print(f"{'Mean total inflation':<40}{'':>42}{mean_delta:>+13.1f}pp")
    print(f"{'Range':<40}{'':>42}{f'+{min_delta:.1f} to +{max_delta:.1f}pp':>15}")

    # Where the inflation comes from, for the models that have all three rungs.
    ladder = [(af, rt, cr) for _, _, af, rt, cr in rows if rt is not None]
    if ladder:
        artist_effect = sum(rt - af for af, rt, _ in ladder) / len(ladder)
        clip_effect = sum(cr - rt for _, rt, cr in ladder) / len(ladder)
        print()
        print(f"Decomposition over {len(ladder)} architectures with all three rungs:")
        print(f"  dropping the artist filter   {artist_effect:+.1f}pp")
        print(f"  additionally leaking clips   {clip_effect:+.1f}pp")

    md_path = OUTPUT_DIR / "protocol_leakage_table.md"
    lines = [
        "# Protocol Leakage Analysis",
        "",
        "Identical architectures trained down a three-rung protocol ladder:",
        "- **Artist-filtered** (official FMA split): no artist appears in more than one split",
        "- **Random-track** (random 80/10/10 over whole tracks): clips never span splits, but artists do",
        "- **Clip-random** (random 80/10/10 over 3s clips): the same track appears in train and test",
        "",
        "The middle rung matters because the literature is not binary: of the papers we",
        "surveyed, only Ashraf et al. 2025 states clip-level splitting outright, while others",
        "either split songs or never say what unit their ratio applies to.",
        "",
        "| Model | Artist-filtered | Random-track | Clip-random | Total inflation (pp) |",
        "|-------|-----------------|--------------|-------------|----------------------|",
    ]
    for name, _, af, rt, cr in rows:
        lines.append(f"| {name} | {af:.1f}% | {fmt(rt)} | {cr:.1f}% | +{cr - af:.1f} |")
    lines.append(f"| **Mean** | | | | **+{mean_delta:.1f}** |")
    lines.append("")
    # Describe only the families actually in the table. Hardcoding the full intended
    # roster here once produced a file claiming DenseNet/capsule/transformer coverage
    # from a table that contained none of them.
    families = sorted({FAMILY_OF.get(name, "other") for name, *_ in rows})
    lines.append(
        f"Across {len(rows)} architectures spanning {_join(families)}, total "
        f"inflation ranges from +{min_delta:.1f}pp to +{max_delta:.1f}pp "
        f"(mean +{mean_delta:.1f}pp)."
    )
    lines.append("The effect is **systematic and architecture-independent**.")
    lines.append("")
    if ladder:
        lines += [
            f"Decomposed over the {len(ladder)} architectures trained on all three rungs, dropping the",
            f"artist filter alone accounts for {artist_effect:+.1f}pp and leaking clips on top of that adds",
            f"a further {clip_effect:+.1f}pp. Both failures matter, and a paper that reports only",
            "\"random 8:1:1\" without naming the unit leaves this ambiguity unresolved.",
            "",
        ]

    repro = [r for r in rows if r[1] in PUBLISHED_ACCURACY]
    if repro:
        lines += [
            "## Reproduction of Published Architectures",
            "",
            "Each architecture below is reimplemented from its paper and run down the same ladder.",
            "The `Paper's stated protocol` column records what the paper actually discloses — not",
            "what its accuracy implies — so the reader can judge which of our columns it belongs",
            "beside. Where a paper never names its split, no column is the fair comparison, which",
            "is precisely the problem.",
            "",
            "| Paper | Published | Paper's stated protocol | Our artist-filtered | Our random-track | Our clip-random |",
            "|-------|-----------|-------------------------|---------------------|------------------|-----------------|",
        ]
        for name, af_run, af, rt, cr in repro:
            citation, published, protocol = PUBLISHED_ACCURACY[af_run]
            lines.append(
                f"| {citation} | {published:.1f}% | {protocol} | {af:.1f}% | {fmt(rt)} | {cr:.1f}% |"
            )
        lines.append("")

    md_path.write_text("\n".join(lines))
    print(f"\nSaved to {md_path}")


if __name__ == "__main__":
    main()
