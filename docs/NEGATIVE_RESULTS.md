# Negative results

Things we built, measured, and could not make work. They are recorded here rather than
dropped, because each one closed off a direction and the closing is itself information —
and because a paper about honest reporting that quietly deletes its failures would be
self-refuting.

> **This is the one hand-written file that contains numbers.** Everywhere else, numbers
> live in generated files (`docs/RESULTS.md`, `outputs/protocol_leakage_table.md`) and
> prose only references them. The exception is deliberate: the runs behind this page have
> been deleted, so nothing can regenerate these figures. They are transcribed verbatim
> from the artifacts at the moment of deletion, with the commit that still contains them
> named below.

---

## 1. ESCF — Epoch-Scheduled Conditional Fusion

**What it was.** A learned fusion of two backbones (`cnn_improved` and
`mert_probe_wide`): a per-episode sigmoid gate α mixing their logits, a per-track
attention β over the ten episodes of a track, and a curriculum λ_t interpolating from a
fixed 0.5 prior toward the learned gate over training. The idea was that *which* backbone
to trust should depend on the clip, and that the model should earn that freedom gradually
rather than start with it.

**Why it was dropped.** It loses to averaging the two backbones with fixed 50/50 weights.
Not narrowly, and not within noise.

Official artist-filtered split, track-level scoring, 5 seeds (mean ± std):

| Row | Test accuracy | Test macro-F1 |
|---|---|---|
| A1_cnn_meanpool | 0.5475 ± 0.0000 | 0.5398 ± 0.0000 |
| A2_cnn_attention | 0.5470 ± 0.0048 | 0.5424 ± 0.0051 |
| A3_mert_meanpool | 0.5713 ± 0.0000 | 0.5665 ± 0.0000 |
| A4_mert_attention | 0.5643 ± 0.0073 | 0.5581 ± 0.0071 |
| **A5_fixed_ensemble** | **0.5863 ± 0.0000** | **0.5807 ± 0.0000** |
| A6_global_learned | 0.5805 ± 0.0007 | 0.5753 ± 0.0008 |
| A7_conditional_nocurr | 0.5622 ± 0.0058 | 0.5569 ± 0.0048 |
| A8_escf_cosine | 0.5578 ± 0.0062 | 0.5533 ± 0.0064 |
| A9a_escf_linear | 0.5650 ± 0.0013 | 0.5608 ± 0.0009 |
| A9b_escf_step | 0.5790 ± 0.0099 | 0.5740 ± 0.0101 |
| A9c_escf_supervisor | 0.5610 ± 0.0058 | 0.5552 ± 0.0055 |

Best ESCF variant (A9b) vs. the fixed ensemble (A5): McNemar p = 0.01055 (A-only correct
32, B-only correct 57); macro-F1 difference −0.0305, 95% CI [−0.0543, −0.0070], bootstrap
p = 0.0106. **The test is significant against us.** Every curriculum shape we tried —
cosine, linear, step, and a learned supervisor — lands below the fixed ensemble.

## 2. Learned attention over episodes

Independently of fusion, pooling a track's ten episodes with learned attention is worse
than averaging them. Both pairs in the grid above show it: A2 (0.5470) < A1 (0.5475) for
the CNN, and A4 (0.5643) < A3 (0.5713) for MERT. It also shows in the main runs, where
`mert_attn_probe_v1` is the weakest MERT probe we trained.

## 3. The lesson these two share

**Anything learned on top of frozen features overfits at this data scale.** Training is
6,400 tracks and model selection runs on 800 validation tracks; a gate, an attention
weight, or a learned ensemble weighting has enough freedom to fit that validation set and
not enough signal to generalise past it. A5 (fixed weights, ±0.0000 across seeds) beating
A6 (learned weights, 0.5805) is the cleanest statement of it: the *only* difference is
whether the combination weights were free, and freeing them costs accuracy.

This is a constraint on everything that follows, not just a post-mortem. Accuracy gains
have to come from better features or better-conditioned inputs, not from new learned
combiners on top of fixed ones. Where this repo still ensembles, it uses fixed uniform
weights for that reason.

Note also the seed columns: A5 varies ±0.0000 while A8 varies ±0.0062. Seed noise alone
is the same size as several of the differences being chased in that grid. Any future claim
of a 1-point improvement here needs multiple seeds before it means anything.

---

## 4. A withdrawn claim, kept visible on purpose

An earlier revision of this project claimed that our BYOL-A and PANNs reimplementations
reproduced their published numbers, and used that as evidence those papers had been
evaluated under an artist-filtered protocol.

The claim was wrong, and it was wrong in a specific way worth preserving: it compared our
**episode-level** accuracy against their **track-level** accuracy. Our episode-level
figures run 5–7pp below our track-level ones, and that gap was doing all the work.

| | published | ours, episode-level | ours, track-level |
|---|---|---|---|
| BYOL-A (Rai & Bhattacharjee 2025) | 64.3% | 63.2% — reported as a "match" | **69.9% (+5.6pp)** |
| PANNs CNN14 (Liu et al. 2022) | 60.9% | 60.9% — reported as "exact" | **67.2% (+6.3pp)** |

Measured consistently, at track level, both of our random-track runs *overshoot* the
published numbers. The reproduction claim does not survive.

**Why this page keeps it.** The paper's argument is that an unstated measurement choice
can move a benchmark number by more than the architectures being compared. We made exactly
that error, against ourselves, and it produced a confident false conclusion. It is a
better illustration of the thesis than any example we could have borrowed.

Two rules follow, and they are enforced in the generators rather than by discipline:

- Every reported accuracy in this project is **track-level** — per-episode softmax averaged
  within a track, then argmax. `eval_metrics.json` → `track_level.accuracy`. A
  `summary.json` accuracy is episode-level and must never appear beside a published number.
- **Do not re-introduce a "reproduced within X pp" claim** for any paper. Our
  reimplementations undershoot the published numbers at artist-filtered and overshoot at
  random-track, so the published numbers land *between* our rungs. That is consistent with
  several different explanations, including ones where the paper did nothing wrong and our
  probe head is simply weaker than their model. The architecture-fidelity confound is as
  large as the effect, so the experiment cannot adjudicate and we stopped trying.

---

## 5. Where the deleted code lives

`src/fusion/`, `outputs/fusion/` (57 run directories, 5 seeds × 11 rows) and
`scripts/train_hybrid_fusion.py` were removed once the result above was settled. They
remain in git history at commit `9152717` and its parent. Nothing on this page needs them
to be re-run; it needs them to be findable, which they are.
