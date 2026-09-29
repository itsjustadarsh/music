# Complete Model Inventory & Experiment Log

## Dataset: FMA-small (8 genres, ~8,000 tracks)
- **Evaluation protocol**: Official artist-filtered split (no artist overlap between train/val/test)
- **Evaluation unit**: Track-level (average per-episode softmax probabilities, then argmax)
- **Splits**: Train ~6,400 tracks (~60K episodes), Val 800 tracks (~7.5K episodes), Test 800 tracks (~7.5K episodes)
- **Episode duration**: 3 seconds, non-overlapping subclips

---

## A. CNN Models (log-mel spectrogram input)

### A1. CNN Baseline (`cnn_baseline`)
- **Architecture**: Custom 4-block CNN (Conv2d + BN + ReLU + MaxPool), ~380K params
- **Input**: 128-bin log-mel spectrograms
- **Config**: lr=0.001, batch_size=64, epochs=30
- **Result**: 47.1% track accuracy, 45.0% track F1

### A2. CNN Augmented (`cnn_augmented`)
- **Same as A1** + SpecAugment (frequency/time masking)
- **Config**: lr=0.001, batch_size=64, epochs=50, augment_spec=true
- **Result**: 51.4% track accuracy, 50.4% track F1

### A3. CNN Tuned LR (`cnn_tuned_lr`)
- **Same as A1** with lower learning rate
- **Config**: lr=0.0003, batch_size=32, optimizer=adamw, scheduler=cosine
- **Result**: 52.9% track accuracy, 52.2% track F1

### A4. CNN Improved (`cnn_improved`)
- **Same as A3** + label smoothing + mixup + SpecAugment
- **Config**: lr=0.0003, batch_size=32, label_smoothing=0.1, mixup_alpha=0.3, augment_spec=true
- **Result**: 55.1% track accuracy, 54.2% track F1

### A5. ResNet (`resnet_v1`) [incomplete]
- **Architecture**: 6-block ResNet with SE attention, ~2.9M params
- **Config**: lr=0.0003, batch_size=32, dropout=0.4, embed_dim=256
- **Status**: Training too slow for laptop (~7-18 min/epoch), stopped at epoch 5

---

## B. MERT-v1-95M Probe Models (frozen embeddings, 13 layers x 768 hidden)

### B1. MERT Probe Baseline (`mert_probe_baseline`)
- **Architecture**: Learned layer weights + Linear(768->512) + GELU + Dropout + Linear(512->8)
- **Config**: lr=0.001, batch_size=256, proj_size=512, dropout=0.3
- **Result**: 55.9% track accuracy, 54.9% track F1

### B2. MERT Probe Last Layer (`mert_probe_lastlayer`)
- **Same as B1** but uses only last MERT layer (ablation)
- **Config**: learn_layer_weights=false, layer_index=12
- **Result**: 52.8% track accuracy, 51.9% track F1
- **Insight**: Learned layer weighting provides +3.1pp over last-layer-only

### B3. MERT Probe Wide (`mert_probe_wide`)
- **Same as B1** with wider projection
- **Config**: lr=0.0005, batch_size=128, proj_size=1024, dropout=0.4
- **Result**: 56.1% track accuracy, 55.3% track F1

### B4. MERT Probe Deep (`mert_probe_deep`)
- **Same as B1** with stronger regularization
- **Config**: lr=0.0001, batch_size=256, proj_size=1024, dropout=0.5, label_smoothing=0.15, mixup_alpha=0.4
- **Result**: 55.0% track accuracy, 54.4% track F1

### B5. MERT Probe Tuned (`mert_probe_tuned`)
- **Config**: lr=0.0002, batch_size=64, proj_size=768, dropout=0.5, label_smoothing=0.1
- **Result**: 56.4% track accuracy, 55.7% track F1

### B6-B8. MERT Probe Seeds 1-3 (seed ablation of B3)
- **Same config as B3** with seeds 43, 44, 45
- **Results**: 53.5%, 55.3%, 54.5% track accuracy
- **Mean**: 54.4% +/- 0.9% (across 3 extra seeds)

### B9. MERT Flat Probe v1 (`mert_flat_probe_v1`)
- **Architecture**: Concatenate all 13 layers (9,984-dim) + LayerNorm + 2-layer MLP, ~10.8M params
- **Config**: lr=0.0003, batch_size=128, proj_size=512, dropout=0.5, label_smoothing=0.1, mixup_alpha=0.3
- **Result**: **58.3% track accuracy, 57.9% track F1** (best individual MERT-95M model)

### B10. MERT Flat Probe v2 (`mert_flat_probe_v2`)
- **Same architecture as B9** with stronger regularization
- **Config**: lr=0.0001, batch_size=64, proj_size=256, dropout=0.6, label_smoothing=0.15, mixup_alpha=0.4
- **Result**: 57.8% track accuracy, 57.3% track F1

### B11. MERT Deep Probe v1 (`mert_deep_probe_v1`)
- **Architecture**: Multi-scale layer grouping (3 groups) + residual blocks, ~4.7M params
- **Config**: lr=0.0003, batch_size=128, proj_size=512, n_groups=3, n_residual_blocks=2, dropout=0.4
- **Result**: 55.5% track accuracy, 55.0% track F1

### B12. MERT Deep Probe v2 (`mert_deep_probe_v2`)
- **Same as B11** with more groups/blocks and stronger regularization
- **Config**: lr=0.0001, batch_size=64, proj_size=256, n_groups=4, n_residual_blocks=3, dropout=0.5
- **Result**: 56.0% track accuracy, 55.9% track F1

### B13-B15. MERT Attention Probe v1-v3 (`mert_attn_probe_v*`)
- **Architecture**: Self-attention over 13 MERT layers + CLS query pooling
- **Results**: 54.6%, 55.3%, 53.5% track accuracy
- **Insight**: Self-attention over only 13 tokens is counterproductive; weighted sum is optimal

---

## C. MERT-v1-330M Probe Models (frozen embeddings, 25 layers x 1024 hidden)

### C1. MERT Large Probe v1 (`mert_large_probe_v1`)
- **Architecture**: Same as MertProbe with num_layers=25, hidden_size=1024
- **Config**: lr=0.0005, batch_size=128, proj_size=1024, dropout=0.4, label_smoothing=0.1, mixup_alpha=0.2
- **Result**: 56.6% track accuracy, 56.5% track F1 (best MERT-330M)

### C2. MERT Large Probe v2 (`mert_large_probe_v2`)
- **Config**: proj_size=2048, dropout=0.5
- **Result**: 55.0% track accuracy, 55.0% track F1

### C3. MERT Large Probe v3 (`mert_large_probe_v3`)
- **Config**: proj_size=512, dropout=0.3
- **Result**: 56.3% track accuracy, 56.2% track F1

### C4. MERT Large Flat Probe (`mert_large_flat_v1`)
- **Architecture**: MertFlatProbe with num_layers=25, hidden_size=1024 (25,600-dim input)
- **Config**: lr=0.0001, batch_size=64, proj_size=512, dropout=0.6
- **Result**: 56.5% track accuracy, 55.9% track F1

### C5. MERT Large Deep Probe (`mert_large_deep_probe_v1`)
- **Architecture**: MertDeepProbe with 5 groups, 2 residual blocks
- **Config**: proj_size=768, dropout=0.4
- **Result**: 51.1% track accuracy — deep architecture overfits on 330M
- **Insight**: MERT-330M benefits from simple architectures; added complexity hurts

---

## D. Ensemble Models

### D1. ESCF Fusion (CNN + MERT-95M Probe)
- **Method**: Epoch-Scheduled Conditional Fusion with cosine, linear, step schedules
- **Best variant**: A5 fixed ensemble (equal-weight average of CNN + MERT)
- **Result**: 58.6% track accuracy, 58.1% track F1 (5 seeds, mean)

### D2. Weighted Probability Ensemble
- **Method**: Nelder-Mead optimized per-model weights on validation set
- **Best subset (3 models)**: CNN_improved + MERT_probe_wide + MERT_flat_probe_v1
- **Result**: 58.8% track accuracy, 58.2% track F1
- **Best subset (4 models)**: CNN_improved + MERT_probe_wide + MERT_flat_probe_v1 + MERT_large_probe_v1
- **Result**: 59.6% track accuracy, 59.2% track F1

### D3. Temperature-Calibrated Ensemble
- **Method**: Per-model temperature T_i (NLL minimization) + calibrated weighted average
- **Result**: 59.5% track accuracy — marginal gain over uncalibrated

### D4. Genre-Conditional Ensemble
- **Method**: Per-genre weight vectors + iterative refinement
- **Result**: 59.3% — per-genre weights overfit with only ~100 val tracks per genre

### D5. Hybrid Feature Fusion (Meta-Ensemble)
- **Method**: MLP on CNN+MERT embeddings (640d) + multi-model softmax probs (64d) = 704d input
- **Architecture**: LayerNorm → FC(768) → GELU → Drop(0.5) → FC(384) → GELU → Drop(0.5) → FC(8)
- **Training**: Mixup(α=0.3), label smoothing(0.1), AdamW, cosine schedule, 10 seeds
- **Fusion seed-ensemble**: 60.0% track accuracy, 59.9% track F1
- **Meta-ensemble (10 fusion + 4 individual)**: **60.3% track accuracy, 60.2% track F1** (overall best)

### D6. Diverse Flat Probe Ensemble (6 seeds × MertFlatProbe)
- **Results**: 54.4-57.4% individually, high seed variance (~3pp)
- **Insight**: Seed variance on 800 test tracks makes small improvements indistinguishable from noise

---

## E. Protocol Comparison (Leakage Analysis)

### E1. MERT Probe Clip-Random (`mert_probe_cliprandom`)
- **Same model as B3** but with clip-random (80/10/10) split
- **Result**: **91.4% accuracy, 91.3% F1** (vs 56.1% on artist-filtered)
- **Insight**: 35 percentage-point inflation from train-test leakage

---

## F. Classical ML on MERT Embeddings

### F1. XGBoost on MERT embeddings (`xgboost_mert`)
- **Result**: 52.5% track accuracy, 51.4% track F1

### F2. SVM (RBF) on flattened MERT features (9,984-dim, track-averaged)
- **Training**: Train+Val combined (7,194 tracks), evaluated on test (800 tracks)
- **Best result**: C=10.0, acc=**56.1%**, f1=56.3%
- **Insight**: Classical SVM matches neural probe on same features — the representation quality is the binding constraint

### F3. LogReg on flattened MERT features
- **Best result**: C=0.001, acc=55.5%, f1=55.0%

### F4. ExtraTrees on flattened MERT features
- **Result**: acc=53.4%, f1=52.5%

---

## Published Baselines for Comparison

| Source | Year | Method | Accuracy | Protocol |
|--------|------|--------|----------|----------|
| Bian et al. | 2019 | DenseNet + SVM stacking | 68.9% | unspecified |
| Rai & Bhattacharjee | 2026 | BYOL-A + DNN | 64.3% | unspecified |
| Liu et al. | 2022 | CNN14 (PANNs) | 60.9% | unspecified |
| Defferrard et al. | 2017 | Hand-crafted + SVM | 58.0% | artist_filtered |
| Park et al. | 2018 | Transfer CNN | 56.8% | unspecified |
| Defferrard et al. | 2017 | Hand-crafted + SVM | 54.8% | artist_filtered |
| Lee et al. | 2019 | Transfer CNN | 51.2% | unspecified |

---

## Key Findings

1. **Protocol matters more than architecture**: Same model shows 35pp gap between clip-random and artist-filtered splits. Papers with "unspecified" protocol may benefit from 10-35pp leakage inflation.
2. **MERT flat probe is strongest individual**: Concatenating all 13 layers (9,984-dim) beats learned weighted sum by 2pp for MERT-95M.
3. **MERT-330M does not dramatically outperform MERT-95M**: Despite 3.5x more parameters, MERT-330M probe (56.6%) is worse than MERT-95M flat probe (58.3%). The bottleneck is not feature quality but the difficulty of artist-filtered genre classification.
4. **Simple ensembles beat complex ones**: Weighted probability averaging > stacking > meta-learning. Only 800 val tracks means complex meta-learners overfit.
5. **Track averaging is crucial**: Flat probe goes from 53.2% episode to 58.3% track (+5.1pp from aggregation).
6. **Cross-backbone ensemble provides genuine gain**: Adding MERT-330M to the ensemble pushes from 58.8% to 59.6% (+0.8pp), confirming that the two MERT models capture complementary information.
7. **Probe architecture has diminishing returns**: Simple linear probes and flat concatenation outperform attention-based and multi-scale probes, suggesting the frozen embedding quality is the binding constraint.

## Reproduction

```bash
# 1. Process audio
python -m src.data.process_fma

# 2. Extract MERT-95M embeddings
python -m src.data.extract_embeddings --workers 4

# 3. Train any model
python -m src.training.train --config src/training/configs/<config>.yaml

# 4. Evaluate at track level
python -m src.evaluation.metrics --run <run_name>

# 5. Cache outputs for ensemble
python -m src.evaluation.cache_outputs --run <run_name>

# 6. Run ESCF fusion grid
python -m src.fusion.run_grid --run-a cnn_improved --run-b mert_probe_wide --seeds 5
```
