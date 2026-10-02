# Results

<!-- STATUS_START -->
> **Status: populated from a real training run.** Regenerate this file any time after training/evaluating more runs with `python scripts/update_results.py`.
<!-- STATUS_END -->

## Our models

Best single run per architecture, **official artist-filtered split only** — runs trained under
`random_clip` or `random_track` are the protocol-ladder control arm and are deliberately excluded
here so that no leaked number can be read as an artist-filtered one. Ensembles over these runs
(best: 60.3% meta-ensemble) are reported separately in [`FULL_RESULTS.md`](FULL_RESULTS.md).

<!-- OUR_MODELS_TABLE_START -->
| Model | Config | Test Accuracy | Test Macro-F1 |
|---|---|---|---|
| MERT-95M flat probe | `mert_flat_probe_v1` | 58.2% | 57.8% |
| BYOL-A + DNN (Rai & Bhattacharjee repro) | `byola_probe_v1` | 57.4% | 57.0% |
| PANNs CNN14 probe (Liu et al. repro) | `panns_probe_v1` | 57.4% | 56.6% |
| MERT-330M probe | `mert_large_probe_v1` | 56.6% | 56.5% |
| MERT-95M deep probe | `mert_deep_probe_v2` | 56.0% | 55.9% |
| MERT-330M flat probe | `mert_large_flat_v1` | 56.5% | 55.9% |
| MERT-95M probe | `mert_probe_tuned` | 56.4% | 55.7% |
| MERT-330M deep probe | `mert_large_deep_probe_v1` | 54.9% | 55.5% |
| MERT-95M attention probe | `mert_attn_probe_v2` | 55.2% | 54.4% |
| GenreCNN | `cnn_improved` | 55.1% | 54.2% |
| DenseNet121 (Bian et al. repro) | `densenet_v1` | 53.9% | 52.2% |
| XGBoost on MERT embeddings | `xgboost_mert` | 52.5% | 51.4% |
<!-- OUR_MODELS_TABLE_END -->

Per-genre precision/recall and confusion matrices are written to
`docs/figures/<run_name>_confusion_matrix.png` and `outputs/<run_name>/eval_metrics.json` by
`python -m src.evaluation.metrics --run <run_name>`.

## Comparison against published FMA-small results

The table below lists **every** published FMA-small result we could verify by reading the primary
source directly (9 baselines). The `Stated protocol` column records what each paper's own text
discloses -- never what its accuracy implies. Four widely-cited numbers are *excluded* because the
primary source does not support them, including the oft-quoted "58.0% (Defferrard et al.)", which
appears in no version of the FMA paper or its repository; see `src/evaluation/compare_baselines.py`
(`EXCLUDED_CLAIMS`) and `study/evaluation_protocols.md` for each exclusion and its reason.

<!-- BASELINE_TABLE_START -->
| Model | Accuracy | Macro-F1 | Stated protocol | Source |
|---|---|---|---|---|
| **Ours -- MERT-95M flat probe** (mert_flat_probe_v1) | **58.2%** | **57.8%** | **official artist-filtered, track-level** | this work |
| Zhang, TS-Resformer (2025) | 90.2% | -- | random 8:1:1 over *songs*, so the 12s/50%-overlap segmentation sits inside each partition rather than across them. Track-level random; artists unaddressed | https://doi.org/10.3389/fnbot.2025.1568811 |
| Ma et al., CT-GateNet (2026) | 89.4% | -- | "all datasets were evenly split into training, validation and test sets at an 8:1:1 ratio" -- the paper never says 8:1:1 *of what* | https://doi.org/10.1371/journal.pone.0344606 |
| Bian, Wang, Zhuang et al., DenseNet + augmentation (2019) | 68.9% | -- | none -- no split sentence anywhere. Scoring is track-level: segment predictions averaged into "a single song-level prediction" | https://arxiv.org/pdf/1906.11620 |
| Ashraf, Abid, Raza et al., dual residual CNN (2025) | 68.5% | 70.5% | clip-random, stated outright: "we segmented all music files into 3-second clips and split them into training, validation, and test sets using an 8:1:1 ratio" -- segmentation precedes splitting | https://doi.org/10.1371/journal.pone.0333808 |
| K. Rai & M. Bhattacharjee, BYOL-A + DNN (PReMI) (2025) | 64.3% | -- | none -- the strings "split", "artist", "fold", "partition", "random" and "stratif" do not occur in the paper | https://arxiv.org/pdf/2603.13871 |
| Liu, DeMori & Abayomi, CNN14 (PANNs), open-set study (2022) | 60.9% | -- | partitions *are* described and the 6,400/800 counts match FMA's official sizes; whether those splits are artist-conditional is never stated | https://arxiv.org/html/2209.07548v2 |
| Park, Lee, Park, Ha & Nam, transfer learning from artist labels (2018) | 56.9% | -- | none for FMA; the same paper *does* specify an artist-separated split for GTZAN, so the awareness is present and the disclosure is not | https://arxiv.org/abs/1710.06648 |
| Kostrzewa, Kaminski & Brzeski, CNN/RNN/CRNN ensembles (2021) | 56.4% | 54.9% | "divided into three independent subsets in an 80:10:10 ratio", mechanism unstated. One spectrogram per song, so clip leakage is structurally impossible; artist overlap is the open question | https://doi.org/10.1007/978-3-030-77961-0_6 |
| Features+SVM, re-run by Park et al. of FMA's released baseline code (2018) | 54.8% | -- | artist-filtered *by construction* -- the code reads FMA's `split` column -- but not asserted in prose | https://arxiv.org/abs/1710.06648 |
<!-- BASELINE_TABLE_END -->

<!-- SUMMARY_LINE_START -->
**Result**: our best model (MERT-95M flat probe, 58.2% accuracy) against the only artist-filtered published number, 54.8%, is +3.5pp. The other 6 of 9 external results do not disclose their split, so they are not comparable in either direction -- see `outputs/protocol_leakage_table.md` for the measured cost of each protocol choice.
<!-- SUMMARY_LINE_END -->

## What this comparison does and doesn't claim

Genre-classification accuracy on FMA-small is a small, well-studied benchmark where the papers
above used varying compute budgets, augmentation strategies, and in some cases ensembling; a
laptop-scale single-model run is not expected to be the strict state of the art. The project's
actual contribution, argued in full in `docs/PAPER_DRAFT.md`, is the **complete agentic pipeline**
-- classification, mood-tagging grounded in a validated psychological model, and an automated
curator/playlist-creation agent -- which no compared FMA genre-classification paper attempts.
