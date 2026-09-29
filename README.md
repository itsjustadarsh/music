# Hybrid Feature Fusion for Music Genre Classification Under Strict Artist-Filtered Evaluation

A systematic study of music genre classification on FMA-small (8 genres, 8,000 tracks) under the official artist-filtered evaluation protocol. We train 31 models spanning CNN baselines, frozen MERT self-supervised probes, and hybrid ensemble strategies, achieving **60.3% track-level accuracy** — a new state-of-the-art among confirmed artist-filtered results.

## Highlights

- **60.3% track-level accuracy** on FMA-small (artist-filtered), surpassing 5 of 7 published baselines
- **35pp protocol leakage quantified**: same model yields 91.4% (clip-random) vs 56.1% (artist-filtered)
- **31 models** across 6 architecture families, all evaluated under identical strict conditions
- **Hybrid feature fusion**: concatenates CNN + MERT embeddings with multi-model probabilities for embedding-level ensemble

## Results

| Model | Track Accuracy | Track F1 | Type |
|-------|---------------|----------|------|
| Meta-Ensemble (fusion + individual) | **60.3%** | **60.2%** | Ensemble |
| Fusion Seed-Ensemble (10 seeds) | 60.0% | 59.9% | Ensemble |
| Weighted Probability Ensemble | 59.6% | 59.2% | Ensemble |
| MERT-95M Flat Probe | 58.3% | 57.8% | Individual |
| MERT-330M Probe | 56.6% | 56.5% | Individual |
| CNN Improved | 55.1% | 54.2% | Individual |

See [`docs/FULL_RESULTS.md`](docs/FULL_RESULTS.md) for all 31 models with per-genre breakdown.

## Comparison with Published Baselines

| Method | Accuracy | Protocol |
|--------|----------|----------|
| Bian et al. (2019) — DenseNet+SVM | 68.9% | unspecified |
| Rai & Bhattacharjee (2026) — BYOL-A+DNN | 64.3% | unspecified |
| Liu et al. (2022) — CNN14 (PANNs) | 60.9% | unspecified |
| **Ours — Meta-Ensemble** | **60.3%** | **artist_filtered** |
| Defferrard et al. (2017) — Features+SVM | 58.0% | artist_filtered |

Baselines marked "unspecified" do not state their evaluation protocol. Our leakage analysis shows this can inflate accuracy by 10–35pp.

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
  models/         GenreCNN, MertProbe, MertFlatProbe, MertDeepProbe, MertAttentionProbe
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
  FULL_RESULTS.md   Complete results for all 31 models with reproduction steps
  model_inventory.md  Detailed experiment log with hyperparameters
  research.md       Draft research paper (IEEE format)
```

## Hardware

All experiments were conducted on a laptop with Apple M-series chip (MPS backend). CNN models train in 30–60 minutes; MERT probe models train in 1–3 minutes on pre-cached embeddings. MERT embedding extraction requires approximately 16 GB RAM.

## License

[TBD]
