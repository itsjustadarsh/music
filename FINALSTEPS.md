# FINALSTEPS

Everything in this repo was built and smoke-tested (unit tests, a synthetic-data training run of
both models, a real end-to-end preprocessing run on a generated test mp3) but **the real 7.2GB
FMA-small download and the real multi-hour training run were not executed** -- that was a deliberate
scope decision so this build finishes in one sitting instead of tying up your machine for hours.

**Everything below is scripted** -- `scripts/` has one command per stage plus a `run_all.sh` that
chains them, so your manual work is really just: paste in credentials, then run 2-3 commands and
wait. Each script is also individually documented below if you want to run stages one at a time or
understand what it's doing; skip straight to the "TL;DR" if you just want the shortest path.

## TL;DR -- minimal manual steps

```bash
./scripts/00_setup.sh          # creates venv, installs deps, creates .env
# --- now edit .env: paste in your Kaggle key (required) and Spotify keys (optional, for later) ---
./scripts/run_all.sh           # fetch + preprocess + train (5 configs) + evaluate + update docs -- takes hours, let it run
./scripts/05_serve.sh          # starts the API and opens the demo frontend
```

That's it for genre classification + curator + mood tagging. Pushing a real playlist to Spotify in
the demo additionally needs the Spotify keys in `.env` (see table below) -- the first push opens a
browser window once for login, then it's remembered.

Every script is safe to re-run: each one skips work that's already done (existing checkpoints,
existing manifest, etc.) unless you pass `FORCE=1`, so if a step fails partway through (network
blip, laptop went to sleep) you can just run it again.

## 0. Setup

```bash
cd music-agent   # this directory
./scripts/00_setup.sh
```

This creates `.venv`, installs all dependencies, and creates `.env` from `.env.example` if it
doesn't exist yet, then prints a credential status report. Then open `.env` and fill in the
credentials below.

## 1. Credentials -- exactly what goes in `.env`

| Variable | Where to get it |
|---|---|
| `KAGGLE_USERNAME` | [kaggle.com](https://www.kaggle.com) → click your profile picture → **Settings** → **API** section → **Create New Token**. This downloads a `kaggle.json` file containing your username and key. |
| `KAGGLE_KEY` | Same `kaggle.json` file as above -- the `"key"` field. |
| `SPOTIFY_CLIENT_ID` | [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) → log in → **Create app** → after creating it, the Client ID is shown on the app's page. |
| `SPOTIFY_CLIENT_SECRET` | Same app page → **Settings** → **View client secret**. |
| `SPOTIFY_REDIRECT_URI` | You choose this -- it must also be added to the app's **Redirect URIs** in the Spotify dashboard settings. The default in `.env.example`, `http://127.0.0.1:8888/callback`, works fine for local use; just add that exact URI in the dashboard too. |

Alternative for Kaggle: instead of putting `KAGGLE_USERNAME`/`KAGGLE_KEY` in `.env`, you can place
the downloaded `kaggle.json` at `~/.kaggle/kaggle.json` (the standard Kaggle CLI location) and skip
those two `.env` lines entirely -- `src/data/fetch_spotify.py` checks both.

Never commit `.env` -- it's already in `.gitignore`.

## 2. Running it

**Fastest path**: `./scripts/run_all.sh` does everything in this section in one go (fetch,
preprocess, train all 5 configs, evaluate, update `docs/RESULTS.md`). It takes hours end to end
(the 7.2GB download + decoding ~8,000 mp3s + training 5 configs) -- run it in a terminal you can
leave open. What follows is the same pipeline broken into individually-runnable stages, for when
you want to run one piece, re-run just one step, or understand what's happening.

**(a) Fetch the data** (downloads ~7.5GB total; the FMA audio archive alone is 7.2GB, so this will
take a while depending on your connection):

```bash
./scripts/01_fetch_data.sh
# or, to skip the 7.2GB audio download and only get metadata + Spotify data:
SKIP_AUDIO=1 ./scripts/01_fetch_data.sh
```

Checks your Kaggle credentials first and fails fast with a clear message if they're missing.
Under the hood this runs `python -m src.data.fetch_fma`, `fetch_spotify`, then `validate` (which
appends row-count results to `docs/DATA_CARD.md`).

**(b) Preprocess FMA-small audio into cached spectrograms/waveforms**:

```bash
./scripts/02_preprocess.sh
# override worker count: WORKERS=8 ./scripts/02_preprocess.sh
```

Decodes all 8,000 tracks into 3-second sub-clips (~80,000 total) and caches two representations
per sub-clip under `data/processed/` -- audio decoding is the bottleneck here, not the model.
Skips automatically if `data/processed/manifest.csv` already exists (`FORCE=1` to redo).

**(c) Train all 5 model configs** (3 CNN, 2 AST -- shipped configs search for a good result rather
than relying on one default run, see `docs/METHODOLOGY.md` §1.3):

```bash
./scripts/03_train_all.sh
# or just a subset:
CONFIGS="cnn_baseline ast_baseline" ./scripts/03_train_all.sh
```

Each run writes to `outputs/<run_name>/` (checkpoint `best.pt`, `metrics.csv`, `summary.json`) and
is skipped if already trained (`FORCE=1` to retrain). CNN configs should each finish in well under
an hour on an M-series Mac; AST configs are slower per-epoch but have fewer epochs configured for
that reason -- if a run is taking too long, lower `epochs` in its yaml or raise `batch_size` if you
have memory headroom.

**(d) Evaluate and update the results docs**:

```bash
./scripts/04_evaluate_all.sh
```

Evaluates every trained run, builds the full comparison against 8 published FMA-small results
(`outputs/comparison_table.md`), and **automatically rewrites `docs/RESULTS.md`** with the real
numbers and an honest "beats N of 8" summary line -- no manual copy-pasting. Re-run any time after
training more configs; it's idempotent (`python scripts/update_results.py` on its own does the doc
update alone if you've already run evaluation).

**(e) Start the API + frontend**:

```bash
./scripts/05_serve.sh
```

Starts the API on `http://127.0.0.1:8000` and opens `frontend/index.html` in your browser. Checks
`/health` reports whether a trained model and the Spotify dataset were found; `/classify`
automatically picks whichever trained run has the best macro-F1. Type a prompt like "upbeat happy
pop" in the demo, click **Generate Playlist**, then enter a playlist name and click **Push to
Spotify** -- the first push opens a browser window for the Spotify login/consent flow (needs the
Spotify keys in `.env`); after that, the token is cached locally (`.spotify_token_cache`,
gitignored) and subsequent pushes won't need to re-authenticate.

## 3. How to change things

- **Swap in a different audio dataset**: replace `src/data/fetch_fma.py` and `src/data/splits.py`
  with equivalents for the new dataset, keeping the same output contract (`load_fma_small_tracks()`
  returning a DataFrame indexed by track id with `genre_top` and `split` columns) so
  `src/data/preprocess.py` and `src/training/train.py` don't need to change. Update `FMA_GENRES` in
  `src/config.py` to the new label set.
- **Adjust the model architecture**: `src/models/cnn.py` (`GenreCNN`) and
  `src/models/modern_model.py` (`ASTGenreClassifier`) are self-contained -- change layer counts,
  swap the AST checkpoint (`AST_CHECKPOINT` constant), etc. Add a new YAML under
  `src/training/configs/` to try it; `train.py` doesn't need code changes for new hyperparameter
  combinations, only for a genuinely new model type (add a branch in `train()`'s `if cfg["model"]`
  block).
- **Change the mood taxonomy**: edit `Mood` (the enum), `_OCTANT_ORDER`, and `NEUTRAL_RADIUS` in
  `src/mood/mood_tagger.py`. If you add/remove moods, also update `_MOOD_KEYWORDS` and
  `_MOOD_CENTERS` in `src/curator/playlist_agent.py` so the curator can still resolve prompts to
  the new taxonomy.
- **Point the curator at a different track pool**: `PlaylistCurator` only requires a pandas
  DataFrame with the columns listed in `REQUIRED_COLUMNS` in `src/curator/playlist_agent.py`. Load
  any track source (a different CSV, a database query, another API) into that shape and pass it to
  `PlaylistCurator(df)` -- `src/api/main.py`'s `get_curator()` is the one place that currently
  hardcodes the Spotify Tracks Dataset CSV path.

## 4. How to extend it

- **New API endpoint**: add a route in `src/api/main.py`, a request/response schema in
  `src/api/schemas.py`, and a test in `tests/test_api.py` following the existing pattern (mock
  `get_classifier`/`get_curator` via `monkeypatch.setattr(main, ...)` rather than hitting real
  models/data in tests).
- **New agent**: add a module under a new or existing `src/<area>/` package with a plain,
  dependency-light class or function (see `src/curator/playlist_agent.py` for the pattern -- no
  framework, just pandas + stdlib, so it's easy to unit test), then wire it into
  `src/api/main.py` if it needs to be served.
- **New evaluation metric**: add it to `compute_metrics()` in `src/evaluation/metrics.py`; it
  already writes whatever's in the returned dict to `eval_metrics.json`, so nothing else needs to
  change to have it show up there. If it should factor into the published-baseline comparison, also
  extend `PublishedBaseline` and the table-building logic in
  `src/evaluation/compare_baselines.py` (and, if you want it to show up in the auto-generated
  `docs/RESULTS.md`, the corresponding table-building code in `scripts/update_results.py`).

## 5. Troubleshooting

**"Permission denied" running a script.** The scripts are checked in with the executable bit set,
but if git or your OS dropped it, run `chmod +x scripts/*.sh` once and try again.

**MPS / device issues.** `src/models/common.py::get_device()` auto-picks MPS on Apple Silicon,
falling back to CPU. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically (in the same module) so
any operation MPS doesn't support falls back to CPU silently instead of crashing -- if training
looks unusually slow, check terminal output for repeated fallback warnings, which mean a
significant chunk of the model is quietly running on CPU. You can force CPU-only for debugging by
temporarily editing `get_device()`'s call site to pass `prefer="cpu"`.

**Spotify OAuth fails.** Most common cause: the `SPOTIFY_REDIRECT_URI` in `.env` doesn't exactly
match a URI registered in the app's Spotify Dashboard settings (including trailing slash and
port). Fix by making them identical, or add the `.env` one to the dashboard's Redirect URIs list.
If a stale/invalid token is cached, delete `.spotify_token_cache` and try again -- it will
re-trigger the browser login.

**Dataset download fails checksum verification.** `src/data/fetch_fma.py` deletes the corrupted
partial file automatically and raises an error -- just re-run the command. If it keeps failing
after multiple retries, FMA's hosted archive may have changed since this was written; compare the
current hash shown in the error against the value in the "Usage" section of
[github.com/mdeff/fma](https://github.com/mdeff/fma)'s README, and update `FMA_SMALL_ZIP_SHA1` /
`FMA_METADATA_ZIP_SHA1` in `src/config.py` if it genuinely changed.

**`echonest.csv` column structure.** `src/mood/echonest_crosscheck.py` assumes a 3-level header
(`('echonest', 'audio_features', 'valence'/'energy')`) based on FMA's documented format, but this
was written without the real file in hand -- if `load_echonest_features()` raises a `KeyError`,
open the file with `pandas.read_csv(path, index_col=0, header=[0,1,2], nrows=5)` and adjust the
column path in that function to match what you actually see.

**AST fine-tuning is slow or runs out of memory.** Lower `batch_size` in the AST config yaml, or
increase `freeze_encoder_layers` (fewer trainable parameters = faster steps, less memory, at some
cost to final accuracy).
