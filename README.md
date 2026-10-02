# Hybrid Feature Fusion for Music Genre Classification Under Strict Artist-Filtered Evaluation

A systematic study of music genre classification on FMA-small (8 genres, 8,000 tracks) under the official artist-filtered evaluation protocol. We train 48 models — 33 under the official split, 15 more under two leakier protocols — spanning CNN baselines, reimplementations of published architectures, frozen MERT / BYOL-A / PANNs probes, and hybrid ensemble strategies. The best reaches **60.3% track-level accuracy**, offered as a documented baseline rather than a state of the art, because no published FMA-small paper states the protocol it would have to be compared against.

## Highlights

- **A three-rung protocol audit**: eight architectures trained identically under artist-filtered → random-track → random-clip splits show **+14.1 to +41.0pp of inflation (mean +30.5pp)**, decomposing into +12.8pp from dropping the artist filter and a further +20.0pp from leaking clips of the same recording
- **60.3% track-level accuracy** on FMA-small (artist-filtered)
- **The field's reference baseline does not exist**: "58.0% (Defferrard et al.)" is a transposed 16-genre *medium*-subset cell, absent from the FMA paper, its repository, and every arXiv revision
- **48 models** across 11 architecture families, all evaluated under identical, stated conditions
- **Hybrid feature fusion**: concatenates CNN + MERT embeddings with multi-model probabilities for embedding-level ensemble

> **All accuracies in this repository are track-level** — per-episode softmax averaged within a track, then argmax — because every published figure we compare against is one prediction per track. Our episode-level numbers run 5-7pp lower and are not interchangeable.

## Results

| Model | Track Accuracy | Track F1 | Type |
|-------|---------------|----------|------|
| Meta-Ensemble (fusion + individual) | **60.3%** | **60.2%** | Ensemble |
| Fusion Seed-Ensemble (10 seeds) | 60.0% | 59.9% | Ensemble |
| Weighted Probability Ensemble | 59.6% | 59.2% | Ensemble |
| MERT-95M Flat Probe | 58.3% | 57.8% | Individual |
| MERT-330M Probe | 56.6% | 56.5% | Individual |
| CNN Improved | 55.1% | 54.2% | Individual |

See [`docs/FULL_RESULTS.md`](docs/FULL_RESULTS.md) for every model with per-genre breakdown.

## Protocol Leakage

Identical architectures, identical hyperparameters, three splits:

| Model | Artist-filtered | Random-track | Clip-random | Total inflation |
|-------|-----------------|--------------|-------------|-----------------|
| MERT-330M Probe | 56.6% | 70.9% | 96.0% | +39.4pp |
| MERT Attention Probe | 54.6% | 69.8% | 95.7% | +41.0pp |
| MERT Probe (weighted-sum) | 56.1% | 69.2% | 93.6% | +37.5pp |
| MERT Deep Probe | 55.5% | 68.5% | 92.2% | +36.7pp |
| MERT Flat Probe | 58.2% | 70.2% | 90.2% | +32.0pp |
| BYOL-A + DNN (Rai & Bhattacharjee 2025) | 57.4% | 69.9% | 84.5% | +27.2pp |
| PANNs CNN14 (Liu et al. 2022) | 57.4% | 67.2% | 73.4% | +16.0pp |
| CNN (log-mel) | 55.1% | — | 69.2% | +14.1pp |
| **Mean** | | | | **+30.5pp** |

The middle rung matters because the literature is not binary: most papers state a ratio ("random 8:1:1") without naming the *unit* it applies to, leaving undecided which of the two failures they incurred. Full table: [`outputs/protocol_leakage_table.md`](outputs/protocol_leakage_table.md).

## Comparison with Published Baselines

The `Stated protocol` column records what each paper's own text discloses — never what its accuracy implies.

| Method | Accuracy | Stated protocol |
|--------|----------|-----------------|
| Bian et al. (2019) — DenseNet+SVM | 68.9% | none — no split sentence in the paper |
| Ashraf et al. (2025) — Dual CNN | 68.5% | **clip-random, stated** |
| Rai & Bhattacharjee (2025) — BYOL-A+DNN | 64.3% | none |
| Liu et al. (2022) — CNN14 (PANNs) | 60.9% | partitions described; artist-conditioning not |
| **Ours — Meta-Ensemble** | **60.3%** | **official artist-filtered, track-level** |
| Park et al. (2018) — Transfer CNN | 56.9% | none for FMA |
| Features+SVM, re-run by Park et al. (2018) | 54.8% | artist-filtered *by construction* (FMA's own code reads the `split` column) |

Of the nine genuine published FMA-small results, **none states in prose that it used the artist-filtered split**. Our leakage analysis shows the same architecture moving +14 to +41pp across the protocol ladder, so undisclosed-protocol numbers are not comparable to ours in either direction.

We reimplemented two of these architectures and ran them down the ladder ourselves. Neither reproduces its published number under the artist-filtered split, and both overshoot it once the filter is removed:

| Paper | Published | Stated protocol | Ours: artist-filtered | random-track | clip-random |
|-------|-----------|-----------------|----------------------|--------------|-------------|
| Rai & Bhattacharjee 2025 | 64.3% | none | 57.4% | 69.9% | 84.5% |
| Liu et al. 2022 | 60.9% | partitions described, artist-conditioning unstated | 57.4% | 67.2% | 73.4% |

Where a paper never names its split, **no column is the fair comparison** — which is precisely the problem.

> **The widely cited "58.0% (Defferrard et al.)" does not exist.** It is not in the FMA paper (whose only accuracy table is captioned "…on the **medium** subset", 16 genres), not in its repository's executed baselines notebook (which subsets to `<= 'medium'`), and not in any arXiv revision. See [mdeff/fma#69](https://github.com/mdeff/fma/issues/69), unanswered since Dec 2023. Details in [`study/evaluation_protocols.md`](study/evaluation_protocols.md).

## Method

```
                     FMA-small (8,000 tracks, 3s episodes)
                                    |
                 +------------------+------------------+
                 |                                     |
          Log-mel spectrograms                  Raw waveform (16kHz)
                 |                                     |
            Custom CNN                     MERT-v1-95M (frozen)
          (~380K params)                   13 layers x 768 hidden
                 |                                     |
          128-dim embedding              9,984-dim flat concatenation
                 |                                     |
                 +------------------+------------------+
                                    |
                   Hybrid Feature Fusion MLP
            (embeddings + multi-model softmax probs = 704-dim)
                   LayerNorm -> 768 -> 384 -> 8 genres
                                    |
                    Seed-averaged (K=10) + Nelder-Mead
                         meta-ensemble weighting
                                    |
                         60.3% track accuracy
```

## Reproduction

### Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Full Pipeline

```bash
# 1. Download and process FMA-small
python -m src.data.download_fma
python -m src.data.process_fma

# 2. Extract MERT-v1-95M embeddings (~2h on CPU)
python -m src.data.extract_embeddings --workers 4

# 3. Extract MERT-v1-330M embeddings (~3.5h, optional)
python -m src.data.extract_mert_large_embeddings --workers 4

# 4. Train any model (MERT probes train in 1-3 minutes on cached embeddings)
python -m src.training.train --config src/training/configs/mert_flat_probe_v1.yaml

# 5. Evaluate at track level
python -m src.evaluation.metrics --run mert_flat_probe_v1

# 6. Cache outputs for ensemble experiments
python -m src.evaluation.cache_outputs --run mert_flat_probe_v1

# 7. Run ensemble experiments
python scripts/mega_ensemble.py              # weighted probability averaging
python scripts/calibrated_ensemble.py        # temperature-scaled ensemble
python scripts/genre_conditional_ensemble.py # per-genre weight optimization
python scripts/train_hybrid_fusion.py        # embedding-level fusion
python scripts/best_ensemble.py --seeds 10   # full meta-ensemble (best result)
```

### Training Configs

All training configurations are in `src/training/configs/`. Each YAML file specifies the model type, hyperparameters, and random seed. Example:

```yaml
run_name: mert_flat_probe_v1
model: mert_flat_probe
epochs: 50
batch_size: 128
lr: 0.0003
optimizer: adamw
scheduler: cosine
proj_size: 512
dropout: 0.5
label_smoothing: 0.1
mixup_alpha: 0.3
seed: 42
```

## Project Structure

```
src/
  data/           Download, validate, preprocess FMA-small; extract MERT embeddings
  models/         GenreCNN, MERT probes (weighted-sum/flat/deep/attention), EmbeddingProbe
                  (BYOL-A, PANNs), and reimplementations: GenreDenseNet, DualCNN,
                  CNNBiGRU, CapsNet, CNNTransformer
  training/       Config-driven train.py, loss functions (CE, focal), dataset classes
  evaluation/     Track-level metrics, output caching, baseline comparison
  fusion/         ESCF (Epoch-Scheduled Conditional Fusion) experiments
  mood/           Russell circumplex mood tagging
  curator/        Genre/mood-aware playlist agent
  spotify/        Spotify OAuth + playlist creation
  api/            FastAPI inference service
  config.py       Shared paths and constants
scripts/          Ensemble experiments, embedding extraction, analysis utilities
frontend/         Web demo UI
tests/            Unit tests
docs/
  FULL_RESULTS.md   Complete results for every model with reproduction steps
  model_inventory.md  Detailed experiment log with hyperparameters
  research.md       Draft research paper (IEEE format)
```

## Hardware

All experiments were conducted on a laptop with Apple M-series chip (MPS backend). CNN models train in 30–60 minutes; MERT probe models train in 1–3 minutes on pre-cached embeddings. MERT embedding extraction requires approximately 16 GB RAM.

## License

[TBD]
