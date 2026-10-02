# Experiment log

<!-- GENERATED FILE — do not edit. `python scripts/update_results.py`. Last run 2026-10-02. -->

> **Generated file — do not edit by hand.** Complete inventory of every evaluated run,
> with the hyperparameters it was trained under. Regenerate with
> `python scripts/update_results.py`.

52 evaluated runs. All accuracies track-level.

## artist-filtered (`official`) — 36 runs

| Run | Architecture | Accuracy | Macro-F1 | Hyperparameters |
|---|---|---|---|---|
| `ensemble_uniform` | Ensemble of per-architecture bests | 58.5% | 58.2% | — |
| `ensemble_weighted` | Ensemble of per-architecture bests | 58.2% | 58.1% | — |
| `mert_flat_probe_v1` | MERT-95M flat probe (13-layer concat) | 58.2% | 57.8% | lr=0.0003, batch_size=128, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.001, dropout=0.5, label_smoothing=0.1, mixup_alpha=0.3, seed=42 |
| `mert_flat_probe_v2` | MERT-95M flat probe (13-layer concat) | 57.8% | 57.3% | lr=0.0001, batch_size=64, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.01, dropout=0.6, label_smoothing=0.15, mixup_alpha=0.4, seed=42 |
| `byola_probe_v1` | BYOL-A + DNN probe | 57.4% | 57.0% | lr=0.0003, batch_size=128, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.001, dropout=0.4, label_smoothing=0.1, mixup_alpha=0.3, seed=42 |
| `mert_flat_diverse_s3` | MERT-95M flat probe (13-layer concat) | 57.4% | 56.5% | lr=0.0002, batch_size=128, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.002, dropout=0.6, label_smoothing=0.15, mixup_alpha=0.4, seed=300 |
| `panns_probe_v1` | PANNs CNN14 probe | 57.4% | 56.6% | lr=0.0003, batch_size=128, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.001, dropout=0.4, label_smoothing=0.1, mixup_alpha=0.3, seed=42 |
| `mert_flat_diverse_s4` | MERT-95M flat probe (13-layer concat) | 56.6% | 56.1% | lr=0.0003, batch_size=128, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.001, dropout=0.5, label_smoothing=0.1, mixup_alpha=0.3, seed=400 |
| `mert_large_probe_v1` | MERT-330M probe (learned layer weights) | 56.6% | 56.5% | lr=0.0005, batch_size=128, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.0001, dropout=0.4, label_smoothing=0.1, mixup_alpha=0.2, seed=42 |
| `mert_flat_diverse_s1` | MERT-95M flat probe (13-layer concat) | 56.5% | 55.8% | lr=0.0003, batch_size=128, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.001, dropout=0.5, label_smoothing=0.1, mixup_alpha=0.3, seed=100 |
| `mert_large_flat_v1` | MERT-330M flat probe (25-layer concat) | 56.5% | 55.9% | lr=0.0001, batch_size=64, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.01, dropout=0.6, label_smoothing=0.15, mixup_alpha=0.4, seed=42 |
| `mert_probe_tuned` | MERT-95M probe (learned layer weights) | 56.4% | 55.7% | lr=0.0002, batch_size=64, epochs=60, optimizer=adamw, scheduler=cosine, weight_decay=0.001, dropout=0.5, label_smoothing=0.1, seed=42 |
| `mert_large_probe_v3` | MERT-330M probe (learned layer weights) | 56.2% | 56.2% | lr=0.0001, batch_size=128, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.0001, dropout=0.3, seed=42 |
| `mert_probe_wide` | MERT-95M probe (learned layer weights) | 56.1% | 55.3% | lr=0.0005, batch_size=128, epochs=40, optimizer=adamw, scheduler=cosine, weight_decay=0.0001, dropout=0.4, seed=42 |
| `mert_deep_probe_v2` | MERT-95M deep probe (multi-scale) | 56.0% | 55.9% | lr=0.0001, batch_size=64, epochs=60, optimizer=adamw, scheduler=cosine, weight_decay=0.01, dropout=0.5, label_smoothing=0.15, mixup_alpha=0.4, seed=42 |
| `mert_flat_diverse_s6` | MERT-95M flat probe (13-layer concat) | 56.0% | 55.4% | lr=0.00025, batch_size=128, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.0015, dropout=0.55, label_smoothing=0.2, mixup_alpha=0.35, seed=600 |
| `mert_probe_baseline` | MERT-95M probe (learned layer weights) | 55.9% | 54.9% | lr=0.001, batch_size=256, epochs=40, optimizer=adamw, scheduler=cosine, weight_decay=1e-05, dropout=0.3, seed=42 |
| `mert_flat_diverse_s5` | MERT-95M flat probe (13-layer concat) | 55.8% | 55.6% | lr=0.0004, batch_size=128, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.0005, dropout=0.4, label_smoothing=0.0, mixup_alpha=0.2, seed=500 |
| `mert_deep_probe_v1` | MERT-95M deep probe (multi-scale) | 55.5% | 55.0% | lr=0.0003, batch_size=128, epochs=60, optimizer=adamw, scheduler=cosine, weight_decay=0.001, dropout=0.4, label_smoothing=0.1, mixup_alpha=0.3, seed=42 |
| `mert_attn_probe_v2` | MERT-95M attention probe | 55.2% | 54.4% | lr=0.0005, batch_size=128, epochs=60, optimizer=adamw, scheduler=cosine, weight_decay=0.001, dropout=0.4, label_smoothing=0.1, mixup_alpha=0.2, seed=42 |
| `mert_probe_seed2` | MERT-95M probe (learned layer weights) | 55.2% | 55.0% | lr=0.0005, batch_size=128, epochs=40, optimizer=adamw, scheduler=cosine, weight_decay=0.0001, dropout=0.4, seed=2 |
| `cnn_improved` | GenreCNN (log-mel, from scratch) | 55.1% | 54.2% | lr=0.0003, batch_size=32, epochs=60, optimizer=adamw, scheduler=cosine, weight_decay=0.0001, label_smoothing=0.1, mixup_alpha=0.3, seed=42 |
| `mert_large_probe_v2` | MERT-330M probe (learned layer weights) | 55.0% | 55.0% | lr=0.0003, batch_size=64, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.001, dropout=0.5, label_smoothing=0.1, mixup_alpha=0.3, seed=42 |
| `mert_probe_deep` | MERT-95M probe (learned layer weights) | 55.0% | 54.4% | lr=0.0001, batch_size=256, epochs=80, optimizer=adamw, scheduler=cosine, weight_decay=0.01, dropout=0.5, label_smoothing=0.15, mixup_alpha=0.4, seed=42 |
| `mert_large_deep_probe_v1` | MERT-330M deep probe (multi-scale) | 54.9% | 55.5% | lr=0.0003, batch_size=64, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.001, dropout=0.4, label_smoothing=0.1, mixup_alpha=0.2, seed=42 |
| `mert_attn_probe_v1` | MERT-95M attention probe | 54.6% | 53.6% | lr=0.0003, batch_size=128, epochs=60, optimizer=adamw, scheduler=cosine, weight_decay=0.0001, dropout=0.3, label_smoothing=0.1, mixup_alpha=0.3, seed=42 |
| `mert_probe_seed3` | MERT-95M probe (learned layer weights) | 54.5% | 54.4% | lr=0.0005, batch_size=128, epochs=40, optimizer=adamw, scheduler=cosine, weight_decay=0.0001, dropout=0.4, seed=3 |
| `mert_flat_diverse_s2` | MERT-95M flat probe (13-layer concat) | 54.4% | 53.7% | lr=0.0003, batch_size=128, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.001, dropout=0.5, label_smoothing=0.1, mixup_alpha=0.3, seed=200 |
| `densenet_v1` | DenseNet121 | 53.9% | 52.2% | lr=0.0003, batch_size=32, epochs=60, optimizer=adamw, scheduler=cosine, weight_decay=0.0001, dropout=0.3, label_smoothing=0.1, mixup_alpha=0.3, seed=42 |
| `mert_attn_probe_v3` | MERT-95M attention probe | 53.5% | 52.8% | lr=0.0001, batch_size=64, epochs=60, optimizer=adamw, scheduler=cosine, weight_decay=0.0001, dropout=0.2, label_smoothing=0.05, mixup_alpha=0.4, seed=42 |
| `mert_probe_seed1` | MERT-95M probe (learned layer weights) | 53.5% | 52.9% | lr=0.0005, batch_size=128, epochs=40, optimizer=adamw, scheduler=cosine, weight_decay=0.0001, dropout=0.4, seed=1 |
| `cnn_tuned_lr` | GenreCNN (log-mel, from scratch) | 52.9% | 52.2% | lr=0.0003, batch_size=32, epochs=35, optimizer=adamw, scheduler=cosine, weight_decay=1e-05, seed=42 |
| `mert_probe_lastlayer` | MERT-95M probe (learned layer weights) | 52.8% | 51.8% | lr=0.001, batch_size=256, epochs=40, optimizer=adamw, scheduler=cosine, weight_decay=1e-05, dropout=0.3, seed=42 |
| `xgboost_mert` | Tree ensemble on MERT embeddings | 52.5% | 51.4% | batch_size=256 |
| `cnn_augmented` | GenreCNN (log-mel, from scratch) | 51.4% | 50.4% | lr=0.001, batch_size=64, epochs=30, optimizer=adam, scheduler=cosine, weight_decay=1e-05, seed=42 |
| `cnn_baseline` | GenreCNN (log-mel, from scratch) | 47.1% | 45.0% | lr=0.001, batch_size=64, epochs=25, optimizer=adam, scheduler=none, weight_decay=0.0, seed=42 |

## random-track (`random_track`) — 8 runs

**Diagnostic only.** These runs deliberately leak and exist to measure the cost
of doing so. No number in this section may be quoted as a result.

| Run | Architecture | Accuracy | Macro-F1 | Hyperparameters |
|---|---|---|---|---|
| `mert_large_flat_v1_randomtrack` | MERT-330M flat probe (25-layer concat) | 71.2% | 71.5% | lr=0.0001, batch_size=64, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.01, dropout=0.6, label_smoothing=0.15, mixup_alpha=0.4, seed=42 |
| `mert_large_probe_v1_randomtrack` | MERT-330M probe (learned layer weights) | 70.9% | 71.2% | lr=0.0005, batch_size=128, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.0001, dropout=0.4, label_smoothing=0.1, mixup_alpha=0.2, seed=42 |
| `mert_flat_probe_v1_randomtrack` | MERT-95M flat probe (13-layer concat) | 70.2% | 70.6% | lr=0.0003, batch_size=128, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.001, dropout=0.5, label_smoothing=0.1, mixup_alpha=0.3, seed=42 |
| `byola_probe_v1_randomtrack` | BYOL-A + DNN probe | 69.9% | 70.3% | lr=0.0003, batch_size=128, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.001, dropout=0.4, label_smoothing=0.1, mixup_alpha=0.3, seed=42 |
| `mert_attn_probe_v1_randomtrack` | MERT-95M attention probe | 69.8% | 70.1% | lr=0.0003, batch_size=128, epochs=60, optimizer=adamw, scheduler=cosine, weight_decay=0.0001, dropout=0.3, label_smoothing=0.1, mixup_alpha=0.3, seed=42 |
| `mert_probe_randomtrack` | MERT-95M probe (learned layer weights) | 69.2% | 69.6% | lr=0.0005, batch_size=128, epochs=40, optimizer=adamw, scheduler=cosine, weight_decay=0.0001, dropout=0.4, label_smoothing=0.1, mixup_alpha=0.2, seed=42 |
| `mert_deep_probe_v1_randomtrack` | MERT-95M deep probe (multi-scale) | 68.5% | 69.0% | lr=0.0003, batch_size=128, epochs=60, optimizer=adamw, scheduler=cosine, weight_decay=0.001, dropout=0.4, label_smoothing=0.1, mixup_alpha=0.3, seed=42 |
| `panns_probe_v1_randomtrack` | PANNs CNN14 probe | 67.2% | 67.2% | lr=0.0003, batch_size=128, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.001, dropout=0.4, label_smoothing=0.1, mixup_alpha=0.3, seed=42 |

## clip-random (`random_clip`) — 8 runs

**Diagnostic only.** These runs deliberately leak and exist to measure the cost
of doing so. No number in this section may be quoted as a result.

| Run | Architecture | Accuracy | Macro-F1 | Hyperparameters |
|---|---|---|---|---|
| `mert_large_probe_v1_cliprandom` | MERT-330M probe (learned layer weights) | 96.0% | 96.0% | lr=0.0005, batch_size=128, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.0001, dropout=0.4, label_smoothing=0.1, mixup_alpha=0.2, seed=42 |
| `mert_attn_probe_v1_cliprandom` | MERT-95M attention probe | 95.7% | 95.7% | lr=0.0003, batch_size=128, epochs=60, optimizer=adamw, scheduler=cosine, weight_decay=0.0001, dropout=0.3, label_smoothing=0.1, mixup_alpha=0.3, seed=42 |
| `mert_probe_cliprandom` | MERT-95M probe (learned layer weights) | 93.6% | 93.6% | lr=0.0005, batch_size=128, epochs=40, optimizer=adamw, scheduler=cosine, weight_decay=0.0001, dropout=0.4, label_smoothing=0.1, mixup_alpha=0.2, seed=42 |
| `mert_deep_probe_v1_cliprandom` | MERT-95M deep probe (multi-scale) | 92.2% | 92.2% | lr=0.0003, batch_size=128, epochs=60, optimizer=adamw, scheduler=cosine, weight_decay=0.001, dropout=0.4, label_smoothing=0.1, mixup_alpha=0.3, seed=42 |
| `mert_flat_probe_v1_cliprandom` | MERT-95M flat probe (13-layer concat) | 90.2% | 90.2% | lr=0.0003, batch_size=128, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.001, dropout=0.5, label_smoothing=0.1, mixup_alpha=0.3, seed=42 |
| `byola_probe_v1_cliprandom` | BYOL-A + DNN probe | 84.5% | 84.5% | lr=0.0003, batch_size=128, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.001, dropout=0.4, label_smoothing=0.1, mixup_alpha=0.3, seed=42 |
| `panns_probe_v1_cliprandom` | PANNs CNN14 probe | 73.4% | 73.0% | lr=0.0003, batch_size=128, epochs=50, optimizer=adamw, scheduler=cosine, weight_decay=0.001, dropout=0.4, label_smoothing=0.1, mixup_alpha=0.3, seed=42 |
| `cnn_improved_cliprandom` | GenreCNN (log-mel, from scratch) | 69.2% | 68.8% | lr=0.0003, batch_size=32, epochs=60, optimizer=adamw, scheduler=cosine, weight_decay=0.0001, label_smoothing=0.1, mixup_alpha=0.3, seed=42 |
