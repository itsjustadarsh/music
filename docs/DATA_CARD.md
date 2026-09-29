# Data Card

## FMA-small (Free Music Archive)

- **Source**: [github.com/mdeff/fma](https://github.com/mdeff/fma), Defferrard, Benzi, Vandergheynst,
  Bresson, "FMA: A Dataset for Music Analysis," ISMIR 2017.
- **Contents**: `fma_small.zip` (7.2 GiB) -- 8,000 tracks, 30 seconds each, 8 balanced genres
  (Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock; 1,000 tracks
  per genre). `fma_metadata.zip` (342 MiB) -- `tracks.csv` (metadata incl. the official
  train/validation/test split), `genres.csv`, `features.csv`, `echonest.csv` (Spotify-style audio
  features for a subset of tracks).
- **Integrity**: both archives are SHA1-verified against the hashes published in the FMA repo's
  README (`ade154f7...` for fma_small.zip, `f0df49ff...` for fma_metadata.zip; see
  `src/config.py` for the full values) before extraction -- see `src/data/fetch_fma.py`.
- **Official split**: taken verbatim from `tracks.csv`'s `('set', 'split')` column
  (`training`/`validation`/`test`), restricted to rows where `('set', 'subset') == 'small'`. We do
  **not** reshuffle -- this is what makes the accuracy comparison against published FMA-small
  papers valid (see `docs/RESULTS.md`).
- **Known caveats**: FMA-small contains a handful of corrupt/unreadable mp3 files (documented in
  the FMA repo's issues); `src/data/preprocess.py` logs and skips these rather than crashing.
- **License**: Individual tracks are released under a variety of Creative Commons licenses by
  their original artists on the Free Music Archive; see the FMA repo for details. This project
  uses the dataset for non-commercial research/education per its intended use.

## Spotify Tracks Dataset (Kaggle)

- **Source**: [kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset),
  downloaded via `kagglehub` in `src/data/fetch_spotify.py`.
- **Contents**: ~114,000 tracks with precomputed Spotify audio features (danceability, energy,
  valence, tempo, key, mode, popularity, `track_genre`, etc.). No raw audio.
- **Role in this project**: used only for the mood-tagging layer (`src/mood/mood_tagger.py`) and
  the playlist curator's track pool (`src/curator/playlist_agent.py`) -- **not** used to train the
  genre classifier, which is trained exclusively on FMA-small audio.

## Preprocessing

Each FMA-small 30-second track is split into ten non-overlapping 3-second sub-clips. For each
sub-clip we cache two representations to `data/processed/`:

- `logmel/<split>/<genre>/<track_id>_<i>.npy` -- a 128-bin log-mel spectrogram at 22,050 Hz
  (n_fft=2048, hop_length=512), consumed by the CNN baseline.
- `waveform16k/<split>/<genre>/<track_id>_<i>.npy` -- the same sub-clip resampled to 16 kHz,
  consumed by the AST model (which computes its own fbank features internally via
  `ASTFeatureExtractor`).

A `manifest.csv` records the mapping from cache files to `(track_id, genre, split, subclip_idx)`.

- `data/embeddings/mert/<split>/<genre>/<track_id>.npy` -- frozen MERT-v1-95M activations
  for the **same** 3-second spans, stored as `(n_episodes, 13, 768)` float16 (all 13 layers,
  mean-pooled over time). Episodes are read from the preprocessing manifest rather than
  recomputed, so MERT and log-mel episodes align exactly on `(track_id, subclip_idx)` --
  a precondition for the per-episode fusion in `src/fusion/`.

### Actual extraction outcome

All 8,000 FMA-small tracks were downloaded and preprocessed. **7,996 tracks** yielded
episodes; 4 were skipped as unreadable (`098569`, `099134`, `108925`, `133297` -- the
well-known corrupt/truncated files in the FMA-small archive). The result is 75,296
three-second episodes: 60,188 train / 7,544 validation / 7,564 test, and the genre
distribution stays balanced (941-951 test episodes per genre).

## Pretrained model licences

- **MERT-v1-95M** (`m-a-p/MERT-v1-95M`) is released under **CC-BY-NC-4.0**: non-commercial
  use only. It is used here as a frozen feature extractor for academic research, which is
  within that licence, but any commercial deployment of this pipeline would need to replace
  it. Noted here because the licence is more restrictive than the rest of the stack.
- **AST** (`MIT/ast-finetuned-audioset-10-10-0.4593`) is BSD-3-Clause.
- **FMA** audio is Creative Commons licensed per-track; see the FMA repository for details.

---

*Validation runs (row-count checks against the expected 8,000 FMA-small tracks and ~114,000
Spotify rows) are appended below automatically by `python -m src.data.validate` after each
download.*
