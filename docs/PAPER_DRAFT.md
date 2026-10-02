# An Agentic Pipeline for Music Genre Classification, Mood Tagging, and Playlist Curation

> **SUPERSEDED — do not cite figures from this file.** This is the project's first
> draft, written before any training run, when the intended contribution was the
> classify → mood → curate pipeline. The project has since moved to a different
> thesis: a protocol audit of the FMA-small literature plus a hybrid-fusion
> baseline. **`docs/research.md` is the current paper.** Results here are still
> marked TBD, the related-work section does not reflect the verification pass
> recorded in `study/evaluation_protocols.md`, and the "eight published baselines"
> framing predates the finding that none of them states its split. Kept for
> provenance only.

## Abstract

We present a complete pipeline that goes beyond genre classification alone: a CNN and a
fine-tuned Audio Spectrogram Transformer (AST) are trained and compared on FMA-small using its
official train/validation/test split, benchmarked against eight published FMA-small results; a
mood-tagging layer grounded in Russell's Circumplex Model of Affect is built on top of Spotify-style
audio features and cross-checked against FMA's echonest subset; and an agentic curator assembles
genre- and mood-matched playlists with smooth tempo/key transitions and artist diversity, which can
be pushed as a real playlist to a user's Spotify account via the Spotify Web API. We argue that this
end-to-end system -- classification, affect modeling, and autonomous curation -- is the paper's
primary contribution, since no prior FMA genre-classification work builds the downstream
mood/curation layers.

## 1. Introduction

Music genre classification on the Free Music Archive (FMA) dataset (Defferrard et al., 2017) is a
well-studied benchmark, but published work stops at reporting a classification accuracy. In
practice, genre alone is a poor proxy for what a listener actually wants: two tracks in the same
genre can have very different moods, tempos, and energy levels. This work asks a broader question:
what would it take to go from "classify this track's genre" to "build me a playlist that matches
what I asked for"? We answer it with a three-layer system -- classify, tag mood, curate -- built and
evaluated end-to-end.

## 2. Related Work

- **FMA dataset and baselines.** Defferrard et al. (2017) introduced FMA and reported baseline
  genre-classification results using hand-crafted audio features with classical classifiers.
- **CNN and transfer-learning approaches on FMA.** Park et al. (2018) applied a CNN pretrained on
  artist-label classification; Bian et al. (2019) combined a DenseNet architecture with audio
  augmentation (time-overlap, pitch-shift) and SVM stacking, reporting the strongest accuracy among
  papers we could verify (68.9%).
- **Pretrained-representation approaches.** Liu, DeMori & Abayomi (2022) fine-tuned an
  AudioSet-pretrained CNN14 (PANNs); Rai & Bhattacharjee (2025) evaluated BYOL-A self-supervised
  embeddings with a downstream classifier.
- **Modern architecture comparison.** Ashraf et al. (2025) proposed a dual residual CNN with hybrid
  pooling, the most recent verified result in our comparison set.
- **Audio Spectrogram Transformer.** Gong, Chung & Glass (2021) introduced AST, a
  vision-transformer-style model pretrained on AudioSet, which we fine-tune as this work's "modern"
  architecture.
- **Affect modeling.** Russell (1980) proposed the circumplex model of affect (valence x arousal),
  widely used in music emotion recognition to map audio features to mood labels.

**Gap.** None of the above extend past classification accuracy into mood tagging or automated
playlist curation. This is the gap this work addresses.

## 3. Methodology

See `docs/METHODOLOGY.md` for full detail; summarized here:

- **Data.** FMA-small (8,000 tracks, 8 genres), official train/val/test split from `tracks.csv`.
  Each 30s track is windowed into ten 3s sub-clips.
- **Models.** (1) `GenreCNN`, a 4-block CNN over 128-bin log-mel spectrograms (~260K params). (2)
  `ASTGenreClassifier`, a fine-tuned `MIT/ast-finetuned-audioset-10-10-0.4593` checkpoint with a
  fresh classification head, optionally with partial layer freezing.
- **Training.** A single config-driven entrypoint (`src/training/train.py`) trains both
  architectures with Apple-Silicon MPS acceleration; five configurations were run varying learning
  rate, optimizer, LR schedule, and augmentation (SpecAugment for the CNN; pitch-shift/time-stretch
  for AST) to search for a competitive result rather than reporting a single default run.
- **Mood tagging.** Spotify `valence`/`energy` features mapped onto Russell's circumplex plane,
  bucketed into 8 mood octants plus a neutral zone, cross-checked against FMA's echonest feature
  subset for genre-mood consistency.
- **Curator agent.** A three-stage pipeline (candidate filtering, mood/popularity-weighted ranking,
  greedy tempo/key-aware ordering with an artist-diversity constraint) over the Spotify Tracks
  Dataset's ~114K tracks.
- **Serving.** A FastAPI service (`/classify`, `/mood`, `/generate-playlist`, `/push-playlist`)
  fronted by a minimal web demo; real playlist creation via the Spotify Web API (`spotipy`, OAuth).

## 4. Experiments

**Configurations.** `cnn_baseline`, `cnn_augmented`, `cnn_tuned_lr` (CNN); `ast_baseline`,
`ast_partial_freeze_augmented` (AST) -- see `src/training/configs/`.

**Metrics.** Accuracy, macro-F1, per-genre precision/recall, confusion matrices
(`src/evaluation/metrics.py`).

**Baselines compared against.** Eight published FMA-small results spanning 2017-2026, gathered by
reading each primary source directly rather than trusting secondary summaries (see
`src/evaluation/compare_baselines.py` for the two numbers we found but excluded as unverifiable).

*[TBD: hardware/wall-clock time per configuration once real training has run.]*

## 5. Results

*[TBD -- see `docs/RESULTS.md` for the live, auto-generated comparison table once training and
evaluation have been run via `FINALSTEPS.md`. Target: beat at least 4-5 of the 8 compared
baselines.]*

## 6. Discussion

The genuinely novel part of this work is not the classification accuracy in isolation -- FMA-small
is a small, thoroughly benchmarked dataset where marginal accuracy gains are unsurprising and where
published results already cluster in a narrow 51-69% band using widely varying compute and
augmentation budgets. What we believe is new is the combination: a validated psychological model
(Russell's circumplex) driving mood inference directly from the same audio-feature space the
classifier's training data lives adjacent to, feeding an agentic curator that produces a
listenable, DJ-ordered, artist-diverse playlist and can push it live to a real Spotify account. No
FMA genre-classification paper in our comparison set attempts any of the mood or curation layers.

## 7. Limitations

- FMA-small's 8 genres are coarse; many tracks are genuinely multi-genre, which caps the ceiling
  any classifier can reach on `genre_top` alone.
- The mood taxonomy substitutes Spotify's `energy` for "arousal" since Spotify does not expose a
  distinct arousal feature -- a common but imperfect proxy in the music-emotion-recognition
  literature.
- The curator's transition-smoothness heuristic (tempo + circular key distance) is a simple greedy
  heuristic, not a learned or globally optimal ordering.
- Track matching against the live Spotify catalog via text search can miss or mismatch tracks,
  particularly for less popular or re-released tracks; these are logged and skipped rather than
  silently substituted.
- Results in this draft are pending a real training run (see status note in `docs/RESULTS.md`).

## 8. Conclusion

We built and evaluated a full agentic pipeline from raw FMA-small audio to a Spotify playlist,
comparing a CNN baseline and a fine-tuned AST against eight published FMA-small results, and laid
out a mood-tagging and curation layer that -- to our knowledge -- no prior FMA genre-classification
work attempts. Future work includes learned (rather than heuristic) playlist ordering, richer mood
taxonomies validated against listener studies, and extending the classifier beyond FMA-small's 8
top-level genres to its full genre hierarchy.

## References

1. M. Defferrard, K. Benzi, P. Vandergheynst, X. Bresson, "FMA: A Dataset for Music Analysis,"
   *ISMIR*, 2017.
2. Y. Gong, Y.-A. Chung, J. Glass, "AST: Audio Spectrogram Transformer," *Interspeech*, 2021.
3. D. S. Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech
   Recognition," *Interspeech*, 2019.
4. J. A. Russell, "A Circumplex Model of Affect," *Journal of Personality and Social Psychology*,
   39(6), 1980, pp. 1161-1178.
5. W. Bian, J. Wang, B. Zhuang, J. Yang, S. Wang, J. Xiao, "Audio-Based Music Classification with
   DenseNet and Data Augmentation," *PAKDD*, 2019. https://arxiv.org/pdf/1906.11620
6. J. Park et al., "Representation Learning of Music Using Artist Labels," 2018 (as cited in [5]).
7. J. Lee, J. Lee, J. Park, J. Lee, "Enhancing Music Features by Knowledge Transfer from User-Item
   Log Data," 2019 (as cited in [5]).
8. Q. Liu, R. DeMori, T. Abayomi, "Open Set Recognition for Music Genre Classification," 2022,
   https://arxiv.org/html/2209.07548v2
9. Rai, Bhattacharjee, "Evaluating Pretrained General-Purpose Audio Representations for Music
   Genre Classification," 2026, https://arxiv.org/pdf/2603.13871
10. M. Ashraf, F. Abid, M. A. Raza et al., "Music genre classification with modified residual
   learning and dual neural network," *PLOS ONE*, 2025.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12520411/
