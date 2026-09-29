# Results

<!-- STATUS_START -->
> **Status: populated from a real training run.** Regenerate this file any time after training/evaluating more runs with `python scripts/update_results.py`.
<!-- STATUS_END -->

## Our models (fill in after training)

<!-- OUR_MODELS_TABLE_START -->
| Model | Config | Test Accuracy | Test Macro-F1 |
|---|---|---|---|
| GenreCNN | `cnn_tuned_lr` | 52.9% | 52.2% |
| GenreCNN + label smoothing + mixup | `cnn_improved` | 55.1% | 54.2% |
| MertProbe (frozen MERT-v1-95M) | `mert_probe_wide` | 56.1% | 55.3% |
| **ESCF Fusion (CNN + MERT)** | `A5_fixed_ensemble` | **58.6%** | **58.1%** |
<!-- OUR_MODELS_TABLE_END -->

Per-genre precision/recall and confusion matrices are written to
`docs/figures/<run_name>_confusion_matrix.png` and `outputs/<run_name>/eval_metrics.json` by
`python -m src.evaluation.metrics --run <run_name>`.

## Comparison against published FMA-small results

The table below lists **every** published FMA-small result we could verify by reading the primary
source directly (8 baselines; see `src/evaluation/compare_baselines.py` for the full methodology
and for numbers we found but explicitly excluded as unverifiable). Nothing is omitted regardless of
whether our model beats it.

<!-- BASELINE_TABLE_START -->
| Model | Accuracy | Macro-F1 | Source |
|---|---|---|---|
| **Ours -- ESCF Fusion** (CNN + MERT probe) | **58.6%** | **58.1%** | this work |
| Bian, Wang, Zhuang et al., DenseNet + augmentation (2019) | 68.9% | -- | https://arxiv.org/pdf/1906.11620 |
| Ashraf, Abid, Raza et al., dual residual CNN (2025) | 68.8% | 70.5% | https://pmc.ncbi.nlm.nih.gov/articles/PMC12520411/ |
| Rai & Bhattacharjee, BYOL-A + DNN (2026) | 64.3% | -- | https://arxiv.org/pdf/2603.13871 |
| Liu, DeMori & Abayomi, CNN14 (PANNs) (2022) | 60.9% | -- | https://arxiv.org/html/2209.07548v2 |
| Defferrard et al., FMA baseline (as cited by Rai & Bhattacharjee 2026) (2017) | 58.0% | -- | https://arxiv.org/pdf/2603.13871 |
| Park et al., transfer learning CNN (as cited by Bian et al. 2019) (2018) | 56.8% | -- | https://arxiv.org/pdf/1906.11620 |
| Defferrard et al., FMA baseline (as cited by Bian et al. 2019) (2017) | 54.8% | -- | https://arxiv.org/pdf/1906.11620 |
| Lee, Lee, Park & Lee, transfer learning CNN (as cited by Bian et al. 2019) (2019) | 51.2% | -- | https://arxiv.org/pdf/1906.11620 |
<!-- BASELINE_TABLE_END -->

<!-- SUMMARY_LINE_START -->
**Result**: our best model (ESCF fusion, 58.6% accuracy) beats 4 of 8 published FMA-small baselines listed above, including the FMA paper's own baseline (58.0%).
<!-- SUMMARY_LINE_END -->

## What this comparison does and doesn't claim

Genre-classification accuracy on FMA-small is a small, well-studied benchmark where the papers
above used varying compute budgets, augmentation strategies, and in some cases ensembling; a
laptop-scale single-model run is not expected to be the strict state of the art. The project's
actual contribution, argued in full in `docs/PAPER_DRAFT.md`, is the **complete agentic pipeline**
-- classification, mood-tagging grounded in a validated psychological model, and an automated
curator/playlist-creation agent -- which no compared FMA genre-classification paper attempts.
