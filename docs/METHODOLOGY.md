# Methodology

How every number in this project is produced. **No results live here** — see
`docs/RESULTS.md` and `outputs/protocol_leakage_table.md`, which are generated. This file
defines what those numbers mean.

---

## 1. Data

**FMA-small** (Defferrard et al., 2017): 8,000 tracks, 8 balanced genres, ~30s per track.
See `docs/DATA_CARD.md` for the full card and `src/data/splits.py` for the loading code.

Each track is cut into ten non-overlapping **3-second episodes**, standard practice in the
FMA/GTZAN genre literature: 3s windows give ~10× more training examples per track while
still carrying genre-discriminative signal. This choice is not free, and it is revisited in
§5 — genre is partly a long-timescale property and 3s may be a ceiling on what any model
here can see.

| | tracks | episodes |
|---|---|---|
| train | 6,400 | ~64,000 |
| validation | 800 | ~8,000 |
| test | 800 | ~8,000 |

---

## 2. The two choices that decide what "FMA-small accuracy" means

Two independent decisions determine what an accuracy number on this dataset actually
measures, and the literature does not make either one consistently
(`docs/LITERATURE.md`). Stating both is a precondition for any comparison.

### 2.1 Split protocol — and why we use the artist-filtered one

FMA's official split applies an **artist filter**: every track by a given artist lands in
exactly one of train/validation/test. The FMA authors are explicit that dropping this is
harmful — using songs from the same artist on both sides "leads to over-optimistic
accuracy".

The reason is that an artist is a bundle of confounds that are *not genre*: a production
signature, a mastering chain, a recording space, a particular drum machine, a singer's
timbre, often a single mixing engineer across an entire catalogue. A model rewarded for
recognising any of those is being scored on a task adjacent to the one we claim to be
measuring. A classifier that identifies the artist and recalls their genre has solved
retrieval, not classification — and it will fail on the first track by an artist it has
never heard, which is the only case anyone cares about in deployment.

This is not binary, and treating it as binary would mis-score most of the literature. Most
papers state a ratio ("random 8:1:1") without naming the **unit** it applies to, and the
unit is where the leakage lives. So we run a **three-rung ladder**, implemented as
`split_mode` in `src/training/dataset.py`:

| Mode | Definition | What leaks into test |
|---|---|---|
| `official` | FMA's shipped artist-filtered split | nothing |
| `random_track` | random 8:1:1 over whole tracks, seed 42 | artist and album identity |
| `random_clip` | random 8:1:1 over individual 3s episodes, seed 42 | artist, album, **and the recording itself** |

The middle rung is the one the literature does not report, and it is the one that makes
the ladder informative. Without it, a paper stating "random 8:1:1" can only be scored
against artist-filtered, which overstates its exposure if it actually split whole tracks.
With it, the two failures separate and can be priced independently.

Why `random_clip` leaks so severely: episode 2 of a track (0:06–0:09) and episode 3
(0:09–0:12) share a microphone, a room, a mix, a performance, usually a chord progression,
and an identical frequency response. A model does not need to know what Folk sounds like to
label episode 3 correctly if it has memorised episode 2 — it needs to recognise *that
specific recording*. The task silently becomes audio fingerprinting.

**Only `official` may produce a headline number.** The other two rungs are diagnostic
instruments. They are run on identical data, identical hyperparameters and identical epoch
budgets, so the split is the only varying factor
(`scripts/protocol_leakage_analysis.py`, `scripts/run_ladder_queue.sh`).

### 2.2 Evaluation unit — track-level, everywhere

We train on 3-second episodes, but a **track** is the thing being classified. Scoring per
episode answers a different question than scoring per track, and published work generally
reports the latter.

We aggregate by averaging per-episode probabilities within a track and taking the argmax
(`aggregate_to_tracks` in `src/evaluation/metrics.py`). Averaging beats majority voting
because a vote discards exactly the confidence that should break ties between disagreeing
episodes.

Both levels are written to each run's `eval_metrics.json`. **Every number this project
reports is the track-level one, including in the leakage ladder**, because every published
figure we compare against is one prediction per track: Liu et al. and Rai & Bhattacharjee
feed a whole track and never segment; Bian et al. and Park et al. average segment outputs
into "a single song-level prediction".

The gap between the two levels is 3–7 accuracy points depending on the model, so it is not
a rounding detail. Quoting an episode-level number beside a published track-level one once
produced a confident false conclusion in this very project; it is documented in
`docs/NEGATIVE_RESULTS.md` §4 rather than quietly corrected. **Two results can share a
split and still be incomparable if one scores clips and the other scores songs.**

This rule is enforced mechanically, not by discipline: `scripts/protocol_leakage_analysis.py`
and `scripts/update_results.py` read `eval_metrics.json → track_level.accuracy` and never
`summary.json`, whose accuracy is episode-level. `LEAKAGE_LEVEL=episode` regenerates the
episode-level view for inspection, but it is never the reported one.

---

## 3. Models

All architectures share one config-driven entrypoint, `src/training/train.py`, so that
checkpointing, logging, early stopping and evaluation are identical across them — a
precondition for the ladder, where any difference other than the split would confound the
measurement.

**From-scratch spectrogram models** (log-mel, 128 bins, `src/models/`):
`GenreCNN` (4-block CNN, ~260K params), `GenreDenseNet` (DenseNet121, after Bian et al.),
`DualCNN` (after Ashraf et al.), `CNNBiGRU` (after Kostrzewa et al.), `CapsNet` (dynamic
routing), `CNNTransformer` (after CT-GateNet).

**Frozen-embedding probes** (`src/models/embedding_probe.py` and the MERT probes). All
backbone features are extracted once and cached to `data/embeddings/`, so a probe run
touches no audio and no backbone — which is what makes the ladder affordable:

| Backbone | Cached shape per episode | Pretraining |
|---|---|---|
| MERT-v1-95M | 13 × 768 (all layers) | 20k hours of music, self-supervised |
| MERT-v1-330M | 25 × 1024 (all layers) | as above, larger |
| BYOL-A (AudioNTT2020) | 2048 | general audio, self-supervised |
| PANNs CNN14 | 2048 | AudioSet, supervised |

Probe heads over these differ in how they consume the layer axis: learned weighted sum,
flat concatenation of all layers, multi-scale grouping, or attention over episodes.

**Non-neural**: a tree-ensemble classifier over the same MERT features
(`scripts/train_xgboost.py`), included on the ladder because an architecture family that
shares nothing with the neural models is the strongest available test of whether protocol
inflation is architecture-independent.

### Reimplementation fidelity, stated plainly

The spectrogram models above are reimplementations from published descriptions, and they
are **not** faithful in every engineering detail — we drop the SVM stacking in Bian et al.
and the gating in CT-GateNet, among others. They are on the ladder as *architecture
families*, to show that protocol inflation is not confined to one model class.

They are explicitly **not** evidence about any paper's protocol. Our reimplementations
undershoot the published numbers at `official` and overshoot at `random_track`, so the
published figures land between our rungs — a pattern consistent with several explanations,
including ones in which the paper did nothing wrong and our head is simply weaker than
their model. The architecture-fidelity confound is the same size as the effect, so the
comparison cannot adjudicate, and this project does not use it to. See
`docs/NEGATIVE_RESULTS.md` §4.

---

## 4. Training

One YAML config per run (`src/training/configs/*.yaml`), one entrypoint. Per-epoch metrics
go to `outputs/<run>/metrics.csv`; the checkpoint with the best **validation macro-F1** is
kept as `best.pt` and is the one evaluated on test. Early stopping on the same criterion.

Apple Silicon MPS is auto-detected, with `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported ops
fall back to CPU rather than crashing. **Runs are strictly sequential**: two concurrent jobs
contending for the single MPS device took DenseNet from 350s/epoch to 1800s/epoch.

**Augmentation.** SpecAugment (Park et al. 2019) — random frequency and time masking applied
to cached log-mel spectrograms, train split only (`src/training/augment.py`). Mixup on the
batch where configured. Label smoothing where configured.

**Seeds.** Seed noise on this dataset is comparable in size to several of the effects worth
chasing (`docs/NEGATIVE_RESULTS.md` §3). Any claimed improvement of ~1pp requires multiple
seeds before it means anything, and single-seed differences at that scale are reported as
such.

---

## 5. Known limitations of this methodology

- **3-second context.** Every embedding in this project is extracted from 3s windows. Genre
  is partly a long-timescale property — form, instrumentation turnover, section changes —
  so the window may be a ceiling on achievable accuracy rather than a neutral choice. It is
  a controlled variable, not a validated one.
- **Reimplementation fidelity**, as stated in §3.
- **8 coarse genres.** Many FMA tracks are genuinely multi-genre, which caps what any
  classifier can reach against single `genre_top` labels.
- **One dataset.** Everything here is FMA-small. The ladder measures protocol inflation on
  this benchmark; whether the magnitudes transfer to GTZAN or MagnaTagATune is untested.
