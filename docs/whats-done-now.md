# What's Done Now — Paper Summary (Plain English)

## What This Paper Is About

When researchers test music genre classification models on the FMA-small dataset (8,000 tracks, 8 genres), they get wildly different accuracy numbers depending on *how they split the data* into train and test sets. The problem: almost nobody says which split they used.

We measured exactly how much this matters: **+30.8 percentage points** on average. That means a model scoring 90% under a sloppy split might only score 60% under a fair one. The sloppy split lets the model "cheat" by recognizing *which recording* it's hearing rather than *which genre* it belongs to.

## The Three Ways to Split (The "Protocol Ladder")

1. **Artist-filtered (honest):** No artist appears in both train and test. The model can't memorize an artist's production style.
2. **Random-track:** Tracks are randomly assigned, but all clips from one track stay together. Partial leakage (same artist, different tracks).
3. **Clip-random (leaky):** Individual 3-second clips are randomly assigned. The same song can appear in both train and test. Maximum cheating.

## Key Results

- **82 training runs** across **21 architectures** and **8 backbone families**
- **Best honest result:** 62.0% (AST Probe, frozen backbone, artist-filtered, track-level)
- **Best ensemble:** 62.1% (uniform average of Round 2 models)
- **Same MERT model under leaky split:** 96.0% — a fake +41pp boost
- **Literature audit:** 0 out of 9 published papers state they used artist-filtered evaluation

## DGX Fine-Tuning Results (New)

### Round 1: Fine-Tuned Models
- AST-Finetune (freeze 6/12): 58.1%
- EfficientAT-Genre: 55.4%
- MERT-LoRA-330M (2M trainable params): 54.4%
- MERT-MultiCrop-330M: 53.9%
- MERT-Unfreeze-330M: 53.4%
- MERT-LoRA-95M: 52.9%
- HTS-AT from scratch: 47.4%

**Key finding:** Frozen AST Probe (62.0%) beats fine-tuned AST (58.1%). LoRA on MERT-330M (54.4%) underperforms frozen MERT-330M probe (56.6%). Fine-tuning is not the bottleneck — the protocol is.

### Round 2: DGX Probes + Ensembles
- AST Probe: 62.0% (new best single model)
- Fusion Probe (4 encoders): 60.3%
- AST-V2 (fine-tuned): 59.5%
- MERT-330M Probe (DGX): 57.9%
- Whisper Probe: 57.3%
- Uniform ensemble: 62.1% (beats Caruana greedy 61.8%)

## The 6 Diagnostics We Built

These are independent tests anyone can run to check if their evaluation is leaky:

1. **Calibration reversal** — Under leaky eval, models become *underconfident* (unusual). Temperature scaling parameter drops below 1.
2. **Temporal consistency inflation** — Inter-episode agreement jumps from 80% to 96%. No genre classifier should be that consistent.
3. **Difficulty profiling** — Hard tracks gain +61.5pp from leakage; easy tracks gain nothing (-1.1pp). Real improvements help all tracks equally.
4. **Representation collapse** — Within-track embedding similarity (0.83-0.93) vastly exceeds between-track similarity (0.30-0.77). The model is encoding recording identity.
5. **Nearest-neighbor identity recovery** — 57-90% of test clips have a same-track neighbor in training under clip-random. 0% under artist-filtered.
6. **Probe architecture invariance** — Linear, MLP-64, MLP-128-64, and MLP-256-128 probes all show the same inflation. It's not the classifier — it's the data split.

## Paper Stats
- ~30 pages (IEEE two-column format)
- 34 figures
- 9+ tables
- 63 references
- 21 architectures tested
- 82 training runs

## Current State
- PDF compiles cleanly
- All citations resolved
- All figures render
- Pushed to private GitHub repo
- Three rounds of multi-agent proofreading completed
- DGX fine-tuning results fully integrated
