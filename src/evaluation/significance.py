"""Paired significance tests for comparing two models on the same test tracks.

Both models are evaluated on exactly the same tracks, so paired tests are the right
family: they condition on the shared difficulty of the test set instead of treating
the two accuracy numbers as independent draws. An unpaired comparison of two
accuracies computed on identical data overstates the uncertainty and is a standard
reviewer objection.

- `mcnemar_test`: exact binomial test on the tracks where the two models disagree.
  The classical choice for paired classifiers on one test set.
- `paired_bootstrap`: resamples tracks with replacement to get a CI on the metric
  difference. Works for macro-F1, which McNemar cannot handle since it is not a
  per-track win/loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score


@dataclass
class TestResult:
    statistic: float
    p_value: float
    detail: str

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def mcnemar_test(labels, preds_a, preds_b) -> TestResult:
    """Exact McNemar on the discordant pairs: does A get right what B gets wrong?"""
    labels, preds_a, preds_b = map(np.asarray, (labels, preds_a, preds_b))
    correct_a, correct_b = preds_a == labels, preds_b == labels

    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    n = only_a + only_b
    if n == 0:
        return TestResult(0.0, 1.0, "models agree on every track")

    # exact binomial, two-sided -- preferred over the chi-square approximation when
    # the discordant count is small
    p = float(stats.binomtest(only_a, n, 0.5).pvalue)
    return TestResult(float(only_a - only_b), p, f"A-only correct: {only_a}, B-only correct: {only_b}")


def paired_bootstrap(
    labels, preds_a, preds_b, metric: str = "macro_f1", n_resamples: int = 10000, seed: int = 0
) -> tuple[float, tuple[float, float], float]:
    """Bootstrap the metric difference (A - B) over tracks.

    Returns (observed difference, 95% CI, two-sided p-value). The p-value is the
    fraction of resamples whose difference crosses zero, doubled.
    """
    labels, preds_a, preds_b = map(np.asarray, (labels, preds_a, preds_b))

    def score(idx, preds):
        if metric == "accuracy":
            return float(np.mean(preds[idx] == labels[idx]))
        return float(f1_score(labels[idx], preds[idx], average="macro", zero_division=0))

    full = np.arange(len(labels))
    observed = score(full, preds_a) - score(full, preds_b)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, len(labels), len(labels))
        diffs[i] = score(idx, preds_a) - score(idx, preds_b)

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return observed, (float(lo), float(hi)), min(1.0, p)
