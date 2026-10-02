
# Music Genre Classification on FMA-Small: Complete Results & Reproduction Guide

## Published Baselines (Comparison Papers)

All accuracy numbers below are on **FMA-small** (8 genres, ~8,000 tracks). Every row was read from the cited paper's own text. The `Stated protocol` column records what the paper *says*, never what its accuracy implies.

| # | Authors | Year | Method | Accuracy | Stated protocol | Source |
|---|---------|------|--------|----------|-----------------|--------|
| 1 | Zhang (TS-Resformer) | 2025 | ResNet + Transformer fusion | 90.23% | random 8:1:1 over **songs**; 12s/50%-overlap segments live inside each partition. Track-level random, artists unaddressed. | [doi:10.3389/fnbot.2025.1568811](https://doi.org/10.3389/fnbot.2025.1568811) |
| 2 | Ma et al. (CT-GateNet) | 2026 | CNN + Transformer, gated channel-spatial attention | 89.42% | "evenly split … at an 8:1:1 ratio" — **unit never named** | [doi:10.1371/journal.pone.0344606](https://doi.org/10.1371/journal.pone.0344606) |
| 3 | Bian, Wang, Zhuang et al. | 2019 | DenseNet + time-overlap/pitch-shift augmentation + SVM stacking | 68.9% | **none** — no split sentence anywhere. Scoring is track-level ("a single song-level prediction"). | [arXiv:1906.11620](https://arxiv.org/pdf/1906.11620) |
| 4 | Ashraf, Abid, Raza et al. | 2025 | Dual CNN, modified residual learning, hybrid pooling | 68.5% | **clip-random, stated**: "we segmented all music files into 3-second clips and split them into training, validation, and test sets using an 8:1:1 ratio" | [doi:10.1371/journal.pone.0333808](https://doi.org/10.1371/journal.pone.0333808) |
| 5 | K. Rai & M. Bhattacharjee | 2025 (PReMI) | BYOL-A pretrained audio embeddings + tuned DNN classifier | 64.3% | **none** — "split", "artist", "fold", "partition", "random" do not occur in the paper | [arXiv:2603.13871](https://arxiv.org/pdf/2603.13871) |
| 6 | K. Liu, J. DeMori & K. Abayomi | 2022 | AudioSet-pretrained CNN14 (PANNs) | 60.9%\* | partitions **are** described and the 6,400/800 counts match FMA's official sizes; whether they are artist-conditional is never stated | [arXiv:2209.07548v2](https://arxiv.org/html/2209.07548v2) |
| 7 | Park, Lee, Park, Ha & Nam | 2018 | Transfer learning CNN using artist labels | 56.9% | **none for FMA**; the same paper explicitly uses an artist-separated split for GTZAN | [arXiv:1710.06648](https://arxiv.org/abs/1710.06648) |
| 8 | Kostrzewa, Kaminski & Brzeski | 2021 | CNN / RNN / CRNN ensembles | 56.4% (54.9% macro-F1) | "80:10:10 ratio", selection mechanism unstated. One spectrogram per song, so clip leakage is structurally impossible; artist overlap is the open question. | [doi:10.1007/978-3-030-77961-0_6](https://doi.org/10.1007/978-3-030-77961-0_6) |
| 9 | Features+SVM, **re-run by Park et al.** | 2018 | FMA's own released baseline code | 54.8% | artist-filtered **by construction** — the code reads FMA's `split` column — but no author asserts it. Park Table 4, `SVM [7] — 0.5482 †`. | [arXiv:1710.06648](https://arxiv.org/abs/1710.06648) |

\* Liu et al.'s 60.9% is the closed-set sub-metric $Acc_c$ of an **open-set** experiment (their Table 3), whose test set also contains unknown classes from fma_medium. Reading it as a plain 8-way fma_small number is defensible but not exactly what they measured.

### Three results commonly listed as FMA-small records that do not survive checking

| Claim | Why it is excluded |
|-------|--------------------|
| **"Defferrard et al., 58.0%, artist-filtered"** | Does not exist. The FMA paper proposes "single top genre on the balanced small subset" as problem 1 and **never reports it**; its only accuracy table (Table 6) is captioned "…on the **medium** subset" (25,000 tracks, 16 genres), and 58 is the MLP/all-features cell *there*. The repository's executed `baselines.ipynb` subsets to `<= 'medium'` and prints 19,922/2,505/2,573 over 16 genres. Issue [mdeff/fma#69](https://github.com/mdeff/fma/issues/69) (Dec 2023) asks the authors for small-subset baselines and is unanswered. The 58.0 attribution appears to originate in Rai & Bhattacharjee's Tables 1 and 3, uncited to any page. **Do not cite it.** |
| **CapsNet + UIGMM, 68.87%** | That paper runs on ISMIR2004, GTZAN and Extended Ballroom only. The 68.87% appears in its *related-work* discussion of Ashraf et al. and is not a result of its own. |
| **DPADMM-G, 97.27%** | A **binary** two-genre task on FMA's precomputed feature vectors: "we randomly selected two categories from the eight categories in fma_small… 800 samples as the training set and the remaining 200 as the test set". Not 8-way audio classification. |

**Critical note on evaluation protocol**: of these nine genuine results, **not one states in prose that it used FMA's artist-filtered split**. Section E below shows the same architecture moving +14 to +41pp across the protocol ladder, so numbers whose protocol is undisclosed are not comparable to ours in either direction.

---

## Dataset & Evaluation Protocol

- **Dataset**: FMA-small — 8,000 tracks across 8 genres (Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock)
- **Split**: Official FMA artist-filtered split (no artist overlap between train/val/test)
- **Splits**: Train ~6,400 tracks (~60K episodes), Val 800 tracks (~7.5K episodes), Test 800 tracks (~7.5K episodes)
- **Episode**: 3-second non-overlapping subclip of each track
- **Evaluation unit**: **Track-level** — average per-episode softmax probabilities within a track, then argmax
- **Metrics**: Accuracy and macro-F1 (both track-level unless stated otherwise)

---

## A. CNN Models (log-mel spectrogram input)

| ID | Run Name | Architecture | Key Config | Track Acc | Track F1 | Best Epoch |
|----|----------|-------------|-----------|-----------|----------|------------|
| A1 | `cnn_baseline` | 4-block CNN (Conv2d+BN+ReLU+MaxPool), ~380K params | lr=0.001, bs=64 | 47.1% | 45.0% | 8 |
| A2 | `cnn_augmented` | Same + SpecAugment | lr=0.001, bs=64, augment_spec=true | 51.4% | 50.4% | 10 |
| A3 | `cnn_tuned_lr` | Same as A1, lower LR | lr=0.0003, bs=32, AdamW, cosine | 52.9% | 52.2% | 19 |
| A4 | `cnn_improved` | Same + label smoothing + mixup + SpecAugment | lr=0.0003, bs=32, LS=0.1, mixup=0.3 | **55.1%** | **54.2%** | 25 |

**Input**: 128-bin log-mel spectrograms computed from 3-second audio clips.

### Reproduction
```bash
# A1
python -m src.training.train --config src/training/configs/cnn_baseline.yaml
# A2
python -m src.training.train --config src/training/configs/cnn_augmented.yaml
# A3
python -m src.training.train --config src/training/configs/cnn_tuned_lr.yaml
# A4
python -m src.training.train --config src/training/configs/cnn_improved.yaml
```

---

## B. MERT-v1-95M Probe Models (frozen embeddings)

**Backbone**: [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) — music-specific self-supervised model, 12 transformer layers + 1 CNN feature layer = 13 layers x 768 hidden dimensions.

All probes train on **cached embeddings** (no backbone forward pass), so each trains in 1-3 minutes on a laptop.

### B.1 Learned-Weight Probes (MertProbe)

Architecture: Learned softmax layer weights across 13 layers → weighted sum → Linear(768→proj) → GELU → Dropout → Linear(proj→8)

| ID | Run Name | Key Config | Track Acc | Track F1 |
|----|----------|-----------|-----------|----------|
| B1 | `mert_probe_baseline` | lr=0.001, bs=256, proj=512, drop=0.3 | 55.9% | 54.9% |
| B2 | `mert_probe_lastlayer` | Last layer only (no learned weights) | 52.8% | 51.8% |
| B3 | `mert_probe_wide` | lr=0.0005, bs=128, proj=1024, drop=0.4 | 56.1% | 55.3% |
| B4 | `mert_probe_deep` | lr=0.0001, bs=256, proj=1024, drop=0.5, LS=0.15, mixup=0.4 | 55.0% | 54.4% |
| B5 | `mert_probe_tuned` | lr=0.0002, bs=64, proj=768, drop=0.5, LS=0.1 | **56.4%** | **55.7%** |

**Insight**: Learned layer weighting provides +3.1pp over last-layer-only (B5 vs B2).

### B.2 Seed Ablation (of B3 config)

| ID | Run Name | Seed | Track Acc | Track F1 |
|----|----------|------|-----------|----------|
| B6 | `mert_probe_seed1` | 1 | 53.5% | 52.9% |
| B7 | `mert_probe_seed2` | 2 | 55.3% | 55.0% |
| B8 | `mert_probe_seed3` | 3 | 54.5% | 54.4% |

**Mean across 3 extra seeds**: 54.4% ± 0.9%

### B.3 Flat Probes (MertFlatProbe)

Architecture: Concatenate all 13 layers (13×768 = 9,984 dims) → LayerNorm → Linear(9984→proj) → GELU → Dropout → Linear(proj→proj) → GELU → Dropout → Linear(proj→8). ~10.8M params.

| ID | Run Name | Key Config | Track Acc | Track F1 |
|----|----------|-----------|-----------|----------|
| B9 | `mert_flat_probe_v1` | lr=0.0003, bs=128, proj=512, drop=0.5, LS=0.1, mixup=0.3 | **58.3%** | **57.8%** |
| B10 | `mert_flat_probe_v2` | lr=0.0001, bs=64, proj=256, drop=0.6, LS=0.15, mixup=0.4 | 57.8% | 57.3% |

**Insight**: Flat probe is the strongest individual model. Concatenating all layers captures complementary information from different transformer depths.

### B.4 Deep Probes (MertDeepProbe)

Architecture: Multi-scale layer grouping (N groups) + learned group/global weights + residual blocks. ~4.7M params.

| ID | Run Name | Key Config | Track Acc | Track F1 |
|----|----------|-----------|-----------|----------|
| B11 | `mert_deep_probe_v1` | n_groups=3, n_res=2, proj=512, drop=0.4 | 55.5% | 55.0% |
| B12 | `mert_deep_probe_v2` | n_groups=4, n_res=3, proj=256, drop=0.5 | 56.0% | 55.9% |

**Insight**: Deep probe does not improve over the simpler flat probe — added complexity hurts.

### B.5 Attention Probes (MertAttentionProbe)

Architecture: Self-attention over 13 MERT layers + CLS query pooling.

| ID | Run Name | Key Config | Track Acc | Track F1 |
|----|----------|-----------|-----------|----------|
| B13 | `mert_attn_probe_v1` | n_heads=8, attn_layers=2, mlp=512 | 54.6% | 53.6% |
| B14 | `mert_attn_probe_v2` | lr=0.0005, attn_layers=4 | 55.3% | 54.4% |
| B15 | `mert_attn_probe_v3` | lr=0.0001, bs=64 | 53.5% | 52.8% |

**Insight**: Self-attention over only 13 tokens is counterproductive; weighted sum is optimal.

### Reproduction (all MERT-95M probes)
```bash
# Extract embeddings first (one-time, ~2 hours)
python -m src.data.extract_embeddings --workers 4

# Train any probe
python -m src.training.train --config src/training/configs/<config_name>.yaml

# Evaluate at track level
python -m src.evaluation.metrics --run <run_name>
```

---

## C. MERT-v1-330M Probe Models (frozen embeddings)

**Backbone**: [MERT-v1-330M](https://huggingface.co/m-a-p/MERT-v1-330M) — larger music SSL model, 24 transformer layers + 1 CNN = 25 layers x 1024 hidden.

| ID | Run Name | Architecture | Key Config | Track Acc | Track F1 |
|----|----------|-------------|-----------|-----------|----------|
| C1 | `mert_large_probe_v1` | MertProbe (25 layers) | lr=0.0005, proj=1024, drop=0.4 | **56.6%** | **56.5%** |
| C2 | `mert_large_probe_v2` | MertProbe | proj=2048, drop=0.5 | 55.0% | 55.0% |
| C3 | `mert_large_probe_v3` | MertProbe | proj=512, drop=0.3 | 56.3% | 56.2% |
| C4 | `mert_large_flat_v1` | MertFlatProbe (25,600-dim) | proj=512, drop=0.6 | 56.5% | 55.9% |
| C5 | `mert_large_deep_probe_v1` | MertDeepProbe (5 groups) | proj=768, drop=0.4 | 51.1% | 51.8% |

**Key finding**: MERT-330M (56.6% best) does **NOT** outperform MERT-95M flat probe (58.3%), despite 3.5x more parameters. The bottleneck is not feature quality but dataset/protocol difficulty. Simple architectures work better on the larger model — added complexity overfits.

### Reproduction
```bash
# Extract MERT-330M embeddings (one-time, ~3.5 hours)
python -m src.data.extract_mert_large_embeddings --workers 4

# Train
python -m src.training.train --config src/training/configs/mert_large_probe_v1.yaml
```

---

## D. Ensemble Models

### D1. ESCF Fusion (CNN + MERT-95M)
- **Method**: Epoch-Scheduled Conditional Fusion with cosine, linear, step schedules
- **Best variant**: A5 fixed ensemble (equal-weight average of CNN + MERT)
- **Result**: 58.6% track accuracy, 58.1% track F1 (5 seeds, mean)

### D2. Weighted Probability Ensemble (Nelder-Mead Optimized)
- **Method**: Optimize per-model weights on validation set using Nelder-Mead, evaluate weighted average on test
- **Best 3-model subset**: CNN_improved + MERT_probe_wide + MERT_flat_probe_v1
  - **Result**: 58.8% accuracy, 58.2% F1
- **Best 4-model subset**: CNN_improved + MERT_probe_wide + MERT_flat_probe_v1 + MERT_large_probe_v1
  - **Result**: 59.6% accuracy, 59.2% F1

### D3. Temperature-Calibrated Ensemble
- **Method**: Per-model temperature T_i optimized on val (min NLL), then weighted average of calibrated probabilities
- **Best result**: 59.5% accuracy (size-4, MERT-only models)
- **Insight**: Marginal gain (+0.25pp) over uncalibrated — the models are already well-calibrated

### D4. Genre-Conditional Ensemble
- **Method**: Per-genre weight vectors W_g optimized on val tracks of genre g, iterative refinement
- **Result**: 59.3% (best subset genre-conditional)
- **Insight**: With only ~100 val tracks per genre, per-genre weights overfit — global weights are better

### D5. Hybrid Feature Fusion (best ensemble approach)
- **Method**: Concatenate CNN + MERT embeddings (640 dim) + softmax probabilities from 8 models (64 dim) → train MLP (768→384) with mixup + label smoothing, average over K seeds
- **Architecture**: LayerNorm → Linear(704→768) → GELU → Dropout(0.5) → Linear(768→384) → GELU → Dropout(0.5) → Linear(384→8)
- **Fusion seed-ensemble (10 seeds)**: 60.0% accuracy, 59.9% F1
- **Meta-ensemble (10 fusion + 4 individual, Nelder-Mead)**: **60.3% accuracy, 60.2% F1** (overall best)
- **Best single seed**: 60.5% accuracy

### D6. Diverse Flat Probe Ensemble
- **Method**: Train 6 flat probes with different seeds (100-600) + loss functions (CE, Focal)
- **Individual results**: 54.4-57.4% track accuracy (high seed variance)
- **Insight**: Seed variance on this dataset is ~3pp, making small improvements hard to distinguish from noise

### Reproduction
```bash
# Cache all model outputs first
python -m src.evaluation.cache_outputs --run cnn_improved
python -m src.evaluation.cache_outputs --run mert_probe_wide
python -m src.evaluation.cache_outputs --run mert_flat_probe_v1
python -m src.evaluation.cache_outputs --run mert_large_probe_v1

# Run ensemble optimization
python scripts/mega_ensemble.py

# Calibrated ensemble
python scripts/calibrated_ensemble.py

# Genre-conditional ensemble
python scripts/genre_conditional_ensemble.py

# Hybrid fusion (best approach)
python scripts/train_hybrid_fusion.py

# Best meta-ensemble
python scripts/best_ensemble.py --seeds 10
```

---

## E. Protocol Leakage Analysis

### E1. Single-Model Comparison (Track-Level)

| Model | Artist-Filtered Acc | Clip-Random Acc | Inflation |
|-------|--------------------|-----------------| ----------|
| MERT probe (same config) | 56.1% | **93.6%** | **+37.5pp** |

This demonstrates that evaluation protocol matters far more than architecture. A 37pp gap means papers using unspecified/clip-random splits inflate their numbers dramatically. E3 generalises this to eight architectures.

### E2. The Three-Rung Protocol Ladder

A binary artist-filtered / clip-random comparison cannot score this literature, because most of the literature is not binary — only Ashraf et al. state clip-level splitting, Zhang states song-level splitting, and the rest state a ratio and nothing more. We therefore define three modes (`src/training/dataset.py`):

| Mode | Definition | What leaks |
|------|-----------|------------|
| `official` | FMA's artist-filtered split, shipped with the dataset | nothing |
| `random_track` | random 8:1:1 over whole tracks, seed 42 | artist and album identity |
| `random_clip` | random 8:1:1 over individual 3s clips, seed 42 | artist, album, **and the recording itself** |

`random_track` is constructed so no `track_id` spans partitions — verified at 60,263 / 7,521 / 7,512 episodes with zero tracks in more than one split.

### E3. Multi-Model Leakage Proof (Track-Level)

Eight architectures, identical data and scoring, run down the ladder.

**All numbers here are track-level** — per-episode softmax averaged within a track, then argmax. This is the measurement every published FMA-small number uses (Liu et al. and Rai & Bhattacharjee feed a whole 30s track and never segment; Bian et al. and Park et al. average segment outputs into one song-level prediction), so it is the only level at which our ladder can sit beside theirs. Episode-level figures live in each run's `summary.json`; regenerate the ladder at that level with `LEAKAGE_LEVEL=episode python scripts/protocol_leakage_analysis.py`.

| Model | `official` | `random_track` | `random_clip` | Total inflation |
|-------|-----------|----------------|---------------|-----------------|
| MERT Probe (weighted-sum) | 56.1% | 69.2% | 93.6% | +37.5pp |
| MERT Flat Probe (13-layer concat) | 58.2% | 70.2% | 90.2% | +32.0pp |
| MERT Deep Probe (multi-scale) | 55.5% | 68.5% | 92.2% | +36.7pp |
| MERT Attention Probe | 54.6% | 69.8% | 95.7% | +41.0pp |
| MERT-330M Probe | 56.6% | 70.9% | 96.0% | +39.4pp |
| CNN (log-mel spectrogram) | 55.1% | — | 69.2% | +14.1pp |
| BYOL-A + DNN | 57.4% | 69.9% | 84.5% | +27.2pp |
| PANNs CNN14 + DNN | 57.4% | 67.2% | 73.4% | +16.0pp |
| **Mean** | **56.4%** | **69.4%** | **86.9%** | **+30.5pp** |

*(The CNN's `random_track` run is still untrained; regenerate this table with `python scripts/protocol_leakage_analysis.py`, which writes `outputs/protocol_leakage_table.md`.)*

**Key findings**:
- Total inflation ranges from +14.1pp (log-mel CNN) to +41.0pp (MERT Attention Probe), mean +30.5pp.
- MERT-based models inflate most because their waveform-level representations capture timbre, production style and vocal characteristics — the same capability that makes them good music models lets them recognise a recording they have already seen. Representation quality and leakage susceptibility are the same axis.
- Even the log-mel CNN, which sees each clip independently, inflates by +14.1pp: coarse spectral statistics alone carry artist identity.
- Averaged over the seven architectures run on all three rungs, the two failures are of comparable size: dropping the artist filter alone costs **+12.8pp** and leaking clips on top adds a further **+20.0pp**. A paper reporting "random 8:1:1" without naming the unit has not disclosed which it incurred.
- The effect holds across log-mel CNNs, self-supervised music transformers, and AudioSet-pretrained audio CNNs, so it is a property of the protocol rather than of any model family.

### E4. Reproduction of Published Architectures

Each architecture reimplemented from its paper and run down the same ladder. The point is not that our reimplementation matches every engineering detail — we drop the SVM stacking in Bian et al. and the gating in CT-GateNet — but that the *protocol* accounts for the gap between a paper's headline and the artist-filtered reality.

| Paper | Published | Stated protocol | `official` | `random_track` | `random_clip` |
|-------|-----------|-----------------|-----------|----------------|---------------|
| Rai & Bhattacharjee, BYOL-A+DNN | 64.3% | none | 57.4% | **69.9%** | 84.5% |
| Liu et al., PANNs CNN14 | 60.9% | partitions described, artist-conditioning not | 57.4% | **67.2%** | 73.4% |
| Bian et al., DenseNet121 | 68.9% | none | 53.9% | — | *not yet trained* |
| Ashraf et al., Dual CNN | 68.5% | clip-random (stated) | *not yet trained* | — | *not yet trained* |
| Kostrzewa et al., CNN+BiGRU | 56.4% | ratio only | *not yet trained* | — | *not yet trained* |
| Ma et al., CT-GateNet | 89.42% | ratio only, unit unnamed | *not yet trained* | — | *not yet trained* |

Two rows are complete across all three rungs. Neither published figure is reachable under the artist-filtered protocol: both architectures score 57.4% there, against published claims of 64.3% and 60.9%. Both published figures sit below our random-track results and far below the clip-random ones.

**What this does not show.** We did not reproduce either number. At the middle rung our reimplementations overshoot by +5.6pp and +6.3pp; they differ from the originals in optimiser, schedule, head capacity and augmentation, so an exact match was never the right expectation. An earlier version of this document claimed a match to within 1.1pp — that claim came from comparing our *episode-level* accuracy against their *track-level* accuracy, which is the same unit mismatch this section criticises. It has been withdrawn.

**What it does show.** Each published figure falls inside the band its architecture occupies across undisclosed protocol choices, and outside the band it occupies under the disclosed one. A binary artist-filtered/clip-random framing would have to park 64.3% somewhere between 57.4% and 84.5% and call it "partial leakage", which explains nothing; the three-rung ladder locates it above the honest rung and below the maximally leaking one. Nothing in either paper rules out the leaking interpretation.

### Reproduction
```bash
# Artist-filtered (the only mode that may produce a headline number)
python -m src.training.train --config src/training/configs/mert_probe_wide.yaml

# Middle rung — tracks split at random, artists span splits
python -m src.training.train --config src/training/configs/byola_probe_v1_randomtrack.yaml

# Clip-random (maximally leaky, diagnostic only)
python -m src.training.train --config src/training/configs/mert_probe_cliprandom.yaml

# Regenerate the ladder tables
python scripts/protocol_leakage_analysis.py
```

---

## F. Classical ML on MERT Embeddings

All use track-averaged 9,984-dim flattened MERT-95M features. Train+Val combined (7,194 tracks), test (800 tracks).

| ID | Method | Track Acc | Track F1 |
|----|--------|-----------|----------|
| F1 | XGBoost | 52.5% | 51.4% |
| F2 | SVM (RBF, C=10) | **56.1%** | **56.3%** |
| F3 | LogReg (C=0.001) | 55.5% | 55.0% |
| F4 | ExtraTrees | 53.4% | 52.5% |

**Insight**: Classical SVM matches the neural probe on the same features — confirming that representation quality is the binding constraint, not the classifier.

---

## Summary of Key Findings

1. **Protocol matters more than architecture**: across eight architectures, total track-level inflation from `official` to `random_clip` is +14.1 to +41.0pp (mean +30.5pp) — larger than the entire span of accuracies this literature competes over. It decomposes into +12.8pp from dropping the artist filter and a further +20.0pp from leaking clips, so "random 8:1:1" without a named unit is not a disclosed protocol.

1b. **The field's reference baseline does not exist**: "58.0%, Defferrard et al., artist-filtered" appears in no version of the FMA paper, its repository, or any arXiv revision. It is a 16-genre *medium*-subset cell (Table 6) that acquired a small-subset attribution through citation. The only sourceable artist-filtered FMA-small number is 54.8%, from Park et al.'s re-run of FMA's released baseline code — and it is artist-filtered because that code reads the `split` column, not because any author says so.

2. **Hybrid feature fusion is the strongest approach**: Concatenating embeddings + probabilities from diverse backbones and training a joint MLP yields 60.3% — surpassing probability-level ensembles (59.6%).

3. **MERT flat probe is strongest individual model**: Concatenating all 13 layers (9,984-dim) beats learned weighted sum by 2pp. Best individual: **58.3%**.

4. **MERT-330M does not dramatically outperform MERT-95M**: Despite 3.5x more parameters, MERT-330M probe (56.6%) is worse than MERT-95M flat probe (58.3%).

5. **Seed averaging stabilizes results**: Individual runs vary by ~3pp; averaging 10 fusion seeds reduces this to <1pp and consistently hits 60%.

6. **Track-level aggregation is crucial**: Flat probe goes from 53.2% episode to 58.3% track (+5.1pp from mean-pooling episode probabilities).

7. **Temperature calibration and genre-conditional weights provide marginal gains**: Calibration adds +0.25pp; genre-conditional weights overfit with only 100 val tracks per genre.

8. **Probe architecture has diminishing returns**: Simple linear probes and flat concatenation outperform attention-based and multi-scale probes — the frozen embedding quality is the binding constraint.

---

## Full Leaderboard (Track-Level, Artist-Filtered)

| Rank | Model | Track Acc | Track F1 | Type |
|------|-------|-----------|----------|------|
| 1 | **Meta-Ensemble (10 fusion + 4 individual)** | **60.3%** | **60.2%** | Ensemble |
| 2 | Fusion Seed-Ensemble (10 seeds) | 60.0% | 59.9% | Ensemble |
| 3 | Weighted Ensemble (4 models) | 59.6% | 59.2% | Ensemble |
| 4 | Calibrated Ensemble (4 models) | 59.5% | 59.2% | Ensemble |
| 5 | Genre-Conditional Ensemble | 59.3% | 58.9% | Ensemble |
| 6 | ESCF A5 Fixed Fusion | 58.6% | 58.1% | Ensemble |
| 7 | mert_flat_probe_v1 | 58.3% | 57.8% | Individual |
| 4 | mert_flat_probe_v2 | 57.8% | 57.3% | Individual |
| 5 | mert_large_probe_v1 | 56.6% | 56.5% | Individual |
| 6 | mert_large_flat_v1 | 56.5% | 55.9% | Individual |
| 7 | mert_probe_tuned | 56.4% | 55.7% | Individual |
| 8 | mert_large_probe_v3 | 56.3% | 56.2% | Individual |
| 9 | mert_probe_wide | 56.1% | 55.3% | Individual |
| 10 | SVM (RBF) on MERT features | 56.1% | 56.3% | Classical |
| 11 | mert_deep_probe_v2 | 56.0% | 55.9% | Individual |
| 12 | mert_probe_baseline | 55.9% | 54.9% | Individual |
| 13 | mert_deep_probe_v1 | 55.5% | 55.0% | Individual |
| 14 | LogReg on MERT features | 55.5% | 55.0% | Classical |
| 15 | mert_attn_probe_v2 | 55.3% | 54.4% | Individual |
| 16 | cnn_improved | 55.1% | 54.2% | Individual |
| 17 | mert_probe_deep | 55.0% | 54.4% | Individual |
| 18 | mert_large_probe_v2 | 55.0% | 55.0% | Individual |
| 19 | mert_attn_probe_v1 | 54.6% | 53.6% | Individual |
| 20 | mert_attn_probe_v3 | 53.5% | 52.8% | Individual |
| 21 | ExtraTrees on MERT features | 53.4% | 52.5% | Classical |
| 22 | cnn_tuned_lr | 52.9% | 52.2% | Individual |
| 23 | XGBoost on MERT features | 52.5% | 51.4% | Classical |
| 24 | cnn_augmented | 51.4% | 50.4% | Individual |
| 25 | mert_large_deep_probe_v1 | 51.1% | 51.8% | Individual |
| 26 | cnn_baseline | 47.1% | 45.0% | Individual |

---

## Environment & Reproduction

### Prerequisites
```bash
# Python 3.11+ with virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Full Pipeline
```bash
# 1. Download FMA-small dataset
python -m src.data.download_fma

# 2. Process audio into 3-second episodes
python -m src.data.process_fma

# 3. Extract MERT-95M embeddings (~2 hours)
python -m src.data.extract_embeddings --workers 4

# 4. Extract MERT-330M embeddings (~3.5 hours, optional)
python -m src.data.extract_mert_large_embeddings --workers 4

# 5. Train any model
python -m src.training.train --config src/training/configs/<config>.yaml

# 6. Evaluate at track level
python -m src.evaluation.metrics --run <run_name>

# 7. Cache outputs for ensemble
python -m src.evaluation.cache_outputs --run <run_name>

# 8. Run ensemble optimization
python scripts/mega_ensemble.py
python scripts/calibrated_ensemble.py
python scripts/genre_conditional_ensemble.py
python scripts/train_hybrid_fusion.py
```

### Hardware
All experiments were run on a laptop with Apple M-series chip (MPS backend). CNN models train in 30-60 minutes; MERT probes train in 1-3 minutes on cached embeddings. MERT embedding extraction requires ~16GB RAM.
