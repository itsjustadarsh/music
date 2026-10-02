# Methodology

## 1. Genre classification

### 1.1 Data & split

Trained and evaluated exclusively on **FMA-small** (8,000 tracks, 8 balanced genres), using the
dataset's own official train/validation/test split from `tracks.csv` (not a random resplit) so
accuracy numbers are comparable to published FMA-small results. See `docs/DATA_CARD.md` for
dataset details and `src/data/splits.py` for the split-loading code.

Each 30-second track is cut into ten non-overlapping 3-second sub-clips (standard practice in
FMA/GTZAN-style genre classification literature, since 3s windows give ~10x more training examples
per track while still containing enough signal to be genre-discriminative). At inference time on a
new (non-FMA) clip, the same 3-second windowing is applied (`src/api/main.py` takes the first 3s of
an uploaded file; a production system would average predictions over multiple windows).

### 1.1b Evaluation protocol (and why published FMA-small numbers are not all comparable)

Two independent choices decide what an "FMA-small accuracy" actually means, and the
literature does not make them consistently. Stating both is a prerequisite for any
comparison in `docs/RESULTS.md`.

**Split protocol.** FMA's official split applies an *artist filter*: an artist's tracks
appear in exactly one of train/validation/test. The FMA paper is explicit that dropping
this is harmful -- using songs from the same artist in both train and test "leads to
over-optimistic accuracy", because the model can recognise the artist's production
signature instead of the genre. Ashraf et al. (2025), who report 68.5%, instead split
*3-second clips* 8:1:1, which places clips of the **same track** on both sides of the
split -- a stronger form of the same leakage.

This is not a binary, and treating it as one would mis-score most of the literature.
Most papers state a ratio ("random 8:1:1") without naming the *unit* it applies to, so
we run a three-rung ladder (`split_mode` in `src/training/dataset.py`):

| Mode | Definition | What leaks |
|---|---|---|
| `official` | FMA's shipped artist-filtered split | nothing |
| `random_track` | random 8:1:1 over whole tracks, seed 42 | artist and album identity |
| `random_clip` | random 8:1:1 over individual 3s clips, seed 42 | artist, album, **and the recording itself** |

Only `official` may produce a headline number; the other two are diagnostic instruments,
run on identical data, hyperparameters and epoch budgets so the only varying factor is the
split (`scripts/protocol_leakage_analysis.py`). Across eight architectures the two leakier
rungs cost +12.8pp and a further +20.0pp respectively.

**Evaluation unit.** We train on 3-second episodes but a *track* is the thing being
classified. Scoring per episode answers a different and easier-to-get-wrong question
than scoring per track, and published work generally reports the latter. We aggregate by
averaging per-episode probabilities within a track and taking the argmax
(`aggregate_to_tracks` in `src/evaluation/metrics.py`); averaging beats majority voting
because a vote discards the confidence that should break ties between disagreeing
episodes. Both numbers are written to `eval_metrics.json`; the track-level one is what
the paper reports, **including in the leakage ladder**, because every published figure we
compare against is one prediction per track. The gap is 3-7 accuracy points depending on
the model, so it is not a rounding detail: quoting an episode-level number beside a
published track-level one once produced a spurious "reproduced within 1.1pp" claim in this
project, which we withdrew (see `study/evaluation_protocols.md` §8). Two results can share
a split and still be incomparable if one scores clips and the other scores songs.

`src/evaluation/compare_baselines.py` therefore records a `protocol` and `eval_unit` for
every published baseline and emits two separate tables. Where a paper does not state its
protocol, it is marked `unspecified` rather than assumed comparable.

### 1.2 Models

**Baseline -- `GenreCNN` (`src/models/cnn.py`).** A 4-block 2D CNN (32→64→128→128 channels, each
block conv+batchnorm+relu+maxpool) over 128-bin log-mel spectrograms, global-average-pooled and
fed through a small classifier head. ~260K parameters -- deliberately modest so a full training run
finishes in well under an hour on a laptop.

**Modern architecture -- `ASTGenreClassifier` (`src/models/modern_model.py`).** A pretrained Audio
Spectrogram Transformer (Gong, Chung & Glass, "AST: Audio Spectrogram Transformer," Interspeech
2021), checkpoint `MIT/ast-finetuned-audioset-10-10-0.4593` (pretrained on AudioSet), fine-tuned
with a fresh 8-class classification head. Optionally freezes the first N transformer encoder layers
(`freeze_encoder_layers` config) to reduce the number of trainable parameters and speed up
fine-tuning. Transfer learning from AudioSet is what makes it feasible to get a transformer to a
competitive accuracy within a laptop-scale fine-tuning budget (a handful of epochs), rather than
needing to pretrain a transformer from scratch on only 8,000 tracks.

### 1.2b Epoch-Scheduled Conditional Fusion (ESCF)

The two backbones see the same audio very differently: `GenreCNN` is trained from
scratch on 6,400 tracks of log-mel spectrogram, while the MERT probe reads a frozen
self-supervised encoder pretrained on 20k hours of music. They therefore fail on
different material -- and, we argue, on different *parts of the same track*. A dense
percussive passage and a sparse harmonic one are not equally legible to the two. If that
holds, the weight given to each backbone should not be one number per dataset, nor even
one per track, but a function of the episode being classified.

For a track `T` with episodes `e_1..e_n` on the shared 3-second grid:

```
h_i     = [h_A_i ; h_B_i ; conf(z_A_i) ; conf(z_B_i)]
alpha_i = sigmoid(g(h_i))          # which backbone to trust, per episode
beta_i  = softmax_i(a(h_i))        # which episodes matter, per track
z_i     = alpha_i * z_A_i + (1 - alpha_i) * z_B_i
z_T     = sum_i beta_i * z_i
```

where `conf(.)` is (max probability, top-2 margin, negative entropy). Supervision is on
`z_T` against the track label, so the loss optimises exactly the quantity reported.

**The epoch curriculum.** The learned gate is unreliable early, when both the gate and
the attention are near-random. A schedule interpolates it against a fixed prior:

```
lambda_t    = schedule(t / T_total)
alpha_i(t)  = lambda_t * alpha_i + (1 - lambda_t) * alpha_prior
```

Training therefore begins as a plain fixed-weight ensemble and hands control to the
conditional gate as it becomes trustworthy. Note where the schedule acts: on *trust in
the gate*, not on whether a backbone participates. An epoch-weighted hybrid of the form
`w_t * m1 + (1 - w_t) * m2` with `w_t = max(0.8, t/T)` -- the form this design started
from -- drives `m2`'s contribution to exactly zero at the final epoch, so the hybrid
degenerates into `m1` at the moment it is evaluated, and `m1` never drops below weight
0.8 regardless of which model is actually stronger. Scheduling gate trust keeps both
backbones contributing at convergence while preserving the intent. That original curve is
retained as the `supervisor` schedule and compared empirically rather than dismissed
(`src/fusion/schedules.py`).

**Relation to prior work.** MATT (2022) applies multiple-instance attention over music
segments, but for a single model -- it weights `beta`, not `alpha`. The IISE Transactions
(2026) optimisation ensemble learns weights to combine per-segment probabilities, but the
weights are *global*, not conditioned on the segment's content. Standard ensembling learns
a global weight across models. ESCF's claim is the conjunction neither has: weights that
are **per-episode, input-conditional, and across heterogeneous backbones**. Rows A5-A7 of
the ablation grid collapse ESCF down to each of those prior formulations on identical data
and splits, so the contribution is measured rather than asserted; A8 must beat them with
separated error bars for the claim to stand.

**Cost.** Both backbones are frozen and their per-episode logits and embeddings are cached
once (`src/evaluation/cache_outputs.py`), so a fusion run touches no audio and no backbone.
That is what makes the grid affordable at 5 seeds per row, which in turn is what makes the
variance estimates and paired significance tests (`src/evaluation/significance.py`)
possible at all on laptop hardware.

### 1.3 Training

`src/training/train.py` is a single entrypoint parameterized by a YAML config
(`src/training/configs/*.yaml`) so both models share one training/checkpointing/logging code path.
Per-epoch metrics are logged to `outputs/<run_name>/metrics.csv`; the checkpoint with the best
validation macro-F1 is kept (`outputs/<run_name>/best.pt`) and used for final test-set evaluation
and for serving via the API. Training auto-detects Apple Silicon's MPS backend
(`torch.backends.mps.is_available()`), with `PYTORCH_ENABLE_MPS_FALLBACK=1` set by default so any
op MPS doesn't support silently falls back to CPU instead of crashing.

Several configurations were prepared (not just one default run) to actually search for a
competitive result, varying learning rate, optimizer, LR schedule, and augmentation:
`cnn_baseline`, `cnn_augmented`, `cnn_tuned_lr`, `ast_baseline`, `ast_partial_freeze_augmented`.

### 1.4 Augmentation

- **SpecAugment** (Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic
  Speech Recognition," Interspeech 2019) -- random frequency and time masking applied directly to
  cached log-mel spectrograms for the CNN (`src/training/augment.py::spec_augment`), train split only.
- **Pitch-shift / time-stretch** -- applied to the raw 16kHz waveform before AST feature extraction
  (`pitch_shift_waveform`, `time_stretch_waveform`), train split only, since these are naturally
  waveform-level transformations.

### 1.5 Evaluation

`src/evaluation/metrics.py` reports accuracy, macro-F1, per-genre precision/recall/F1/support, and
a confusion matrix plot for a given run, evaluated on the held-out FMA-small test split.
`src/evaluation/compare_baselines.py` places our results alongside >=8 published FMA-small results
gathered by directly reading primary sources (see that module's docstring for the verification
methodology and what was excluded as unverifiable) -- see `docs/RESULTS.md` for the full,
un-cherry-picked table.

## 2. Mood tagging

Grounded in **Russell's Circumplex Model of Affect** (Russell, J. A., "A circumplex model of
affect," *Journal of Personality and Social Psychology*, 39(6), 1980, 1161-1178), which places
affective states on a 2D plane of valence (pleasant/unpleasant) and arousal (activated/deactivated).

Spotify's `valence` feature maps directly onto the model's valence axis; `energy` is used as a
proxy for arousal (a standard substitution in music-emotion-recognition literature, since Spotify
does not expose a distinct "arousal" feature). Both are recentered from Spotify's native 0..1 range
to -1..1, and the resulting point is bucketed into one of 8 octants (happy, excited, tense, angry,
sad, tired, relaxed, content) plus a small-radius "neutral" zone at the center
(`src/mood/mood_tagger.py`).

`src/mood/echonest_crosscheck.py` cross-checks this mapping against FMA's `echonest.csv` subset
(Spotify-style features for a subset of FMA tracks that also carry a genre label), producing a
genre x mood distribution as a sanity check that the mapping is musically sensible (e.g. energetic
genres skewing toward excited/tense, ambient/instrumental genres skewing toward relaxed/content)
rather than a hard validation, since genre and mood are legitimately independent axes.

## 3. Curator agent

`src/curator/playlist_agent.py`'s `PlaylistCurator` implements a three-stage pipeline:

1. **Request resolution** -- accepts either structured `(genre, mood)` or a free-text prompt
   (`parse_prompt`, simple keyword matching against known genres and a mood-keyword table).
2. **Candidate ranking** -- filters the Spotify Tracks Dataset pool to the requested genre (falling
   back to the full pool if the genre isn't found, rather than returning empty), scores each
   candidate by a weighted combination of mood-target closeness (Euclidean distance in
   valence/energy space to the mood's centroid) and popularity, and keeps the top-scoring
   candidates.
3. **Flow ordering** -- greedily walks the ranked candidates, at each step picking the remaining
   track that minimizes a transition cost (normalized tempo delta + circular pitch-class distance
   between keys), while excluding the same artist as the immediately preceding track whenever an
   alternative exists.

This is the project's core novel contribution -- no FMA genre-classification paper we compared
against also builds a mood layer or an agentic curator/playlist-creation pipeline on top of its
classifier (see `docs/PAPER_DRAFT.md`, Discussion).

## 4. Spotify integration

`src/spotify/client.py` wraps `spotipy`: `SpotifyOAuth` handles the authorization-code flow (token
cached locally, gitignored), `search_track` does a best-effort text search per curated track and
returns `None` on no confident match (logged and skipped rather than failing the whole playlist),
and `create_playlist_from_tracks` creates a real playlist in the authenticated user's account and
adds all successfully matched tracks.
