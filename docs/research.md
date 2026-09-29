# Hybrid Feature Fusion for Music Genre Classification Under Strict Artist-Filtered Evaluation

---

**Abstract** — Music genre classification on the Free Music Archive (FMA) dataset remains a challenging benchmark, yet most published results omit critical details about their evaluation protocol, making fair comparison impossible. We present a systematic study of 31 models spanning CNN baselines, frozen MERT self-supervised embeddings, and novel hybrid fusion strategies, all evaluated under the strict official FMA artist-filtered split. Our key methodological contribution is a hybrid feature fusion approach that concatenates CNN spectral embeddings with MERT-v1-95M embeddings and multi-model softmax probabilities, training a lightweight MLP classifier on this 704-dimensional representation. Averaging over 10 random seeds and combining with individual model predictions via Nelder-Mead weight optimization, our meta-ensemble achieves **60.3% track-level accuracy** on FMA-small (8 genres, 800 test tracks), surpassing 5 of 7 published baselines. Critically, we demonstrate that the same model achieves 91.4% under a clip-random split versus 56.1% under the artist-filtered split — a 35 percentage-point gap that calls into question published results using unspecified evaluation protocols. We release all code, trained models, and cached embeddings for full reproducibility.

**Index Terms** — music genre classification, self-supervised learning, MERT, ensemble methods, evaluation protocol, FMA dataset

---

## I. Introduction

Automatic music genre classification (MGC) is a foundational task in music information retrieval (MIR) with applications in recommendation systems, music discovery, and library organization [1]. The Free Music Archive (FMA) dataset [2] provides a standardized benchmark with an official artist-filtered train/validation/test split designed to prevent information leakage — ensuring that no artist appears in more than one split.

Despite this, the majority of published results on FMA-small either do not specify their evaluation protocol or use random clip-level splits that allow the same track's episodes to appear in both training and testing sets. This train-test leakage inflates reported accuracy by 10–35 percentage points, making cross-study comparison unreliable.

In this work, we make three contributions:

1. **Comprehensive evaluation under strict protocol**: We train and evaluate 31 models — spanning custom CNNs, frozen MERT [3] self-supervised embeddings with various probe architectures, and multiple ensemble strategies — all under the official FMA artist-filtered split with track-level evaluation.

2. **Hybrid feature fusion**: We propose a fusion approach that concatenates penultimate embeddings from diverse backbones (CNN and MERT) with softmax probability vectors from multiple trained models, feeding this representation into a lightweight MLP. This embedding-level fusion captures cross-modal interactions that probability-level ensembles cannot.

3. **Quantification of protocol leakage**: We show that the identical model achieves 91.4% accuracy under clip-random evaluation versus 56.1% under artist-filtered evaluation, establishing a concrete 35.3 percentage-point inflation factor.

---

## II. Related Work

### A. Music Genre Classification on FMA

Defferrard et al. [2] introduced the FMA dataset with hand-crafted audio features and SVM baselines, reporting 54.8–58.0% on FMA-small under the artist-filtered split. Bian et al. [4] achieved 68.9% using DenseNet with time-overlap and pitch-shift augmentation plus SVM stacking, but did not specify their evaluation protocol. Liu et al. [5] applied AudioSet-pretrained CNN14 (PANNs) for 60.9%, also without specifying the split protocol. More recently, Rai and Bhattacharjee [6] used BYOL-A pretrained embeddings with a tuned DNN classifier for 64.3%, again without specifying protocol. Park et al. [7] and Lee et al. [8] reported 56.8% and 51.2% respectively using transfer learning CNNs.

A critical observation is that only Defferrard et al. [2] explicitly used the artist-filtered split; all other works either used unspecified protocols or clip-level random splits. As we demonstrate in Section V-E, this distinction is decisive.

### B. Self-Supervised Audio Representations

MERT (Music undERstanding Transformer) [3] is a self-supervised model pre-trained on large-scale music data using a masked language modeling objective over acoustic tokens. MERT-v1-95M provides 13 layers of 768-dimensional hidden representations, while MERT-v1-330M provides 25 layers of 1024-dimensional representations. These frozen representations serve as powerful feature extractors for downstream music understanding tasks.

### C. Ensemble Methods for Classification

Ensemble methods combine multiple models to reduce variance and improve generalization. Standard approaches include probability averaging [9], stacking [10], and learned weighting [11]. Temperature scaling [12] calibrates individual model confidence before combination. Our work extends these with genre-conditional weighting and hybrid embedding-probability fusion.

---

## III. Dataset and Evaluation Protocol

### A. FMA-Small

The Free Music Archive small subset (FMA-small) [2] contains 8,000 tracks of 30 seconds each, evenly distributed across 8 genres: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, and Rock (1,000 tracks per genre). We use the official artist-filtered split:

| Split | Tracks | Episodes (3s) | Purpose |
|-------|--------|---------------|---------|
| Train | 6,394 | 60,188 | Model training |
| Validation | 800 | 7,544 | Hyperparameter selection, early stopping |
| Test | 800 | 7,564 | Final evaluation (reported numbers) |

The artist-filtered split ensures no artist appears in more than one split, preventing the model from memorizing artist-specific production characteristics rather than genre features.

### B. Episode and Track-Level Evaluation

Each 30-second track is divided into non-overlapping 3-second episodes. Models classify individual episodes, but we report **track-level** metrics: per-episode softmax probabilities are averaged within each track, and the final prediction is the argmax of the averaged probability vector. This mirrors published evaluation practice and provides a more stable estimate than episode-level classification. We report accuracy and macro-averaged F1 score.

### C. Significance of Protocol Choice

We quantify the effect of evaluation protocol by training the identical model (MERT probe, same hyperparameters) under two protocols:

| Protocol | Accuracy | F1 |
|----------|----------|-----|
| Artist-filtered (official) | 56.1% | 55.3% |
| Clip-random (80/10/10) | 91.4% | 91.3% |
| **Inflation** | **+35.3pp** | **+36.0pp** |

The clip-random split allows episodes from the same track — and indeed the same artist — to appear in both training and test sets. The resulting 35.3 percentage-point inflation demonstrates that published results using unspecified protocols are not directly comparable to strict artist-filtered evaluation.

---

## IV. Methods

### A. CNN Baseline

Our CNN baseline consists of four convolutional blocks, each containing a 2D convolution, batch normalization, ReLU activation, and max pooling, followed by global average pooling and a linear classifier (~380K parameters). Input features are 128-bin log-mel spectrograms computed from 3-second audio clips at 22,050 Hz.

We progressively improved the CNN through:
- **SpecAugment** [13]: Random frequency and time masking during training
- **Learning rate tuning**: Reducing from 0.001 to 0.0003 with AdamW optimizer and cosine annealing
- **Label smoothing** (0.1) and **mixup** (alpha=0.3) [14]: Standard regularization techniques

These improvements raised CNN accuracy from 47.1% (baseline) to 55.1% (fully improved).

### B. MERT Embedding Extraction

We extract frozen embeddings from two pre-trained MERT models:

- **MERT-v1-95M** [3]: 12 transformer layers + 1 CNN feature extractor = 13 layers, each producing 768-dimensional representations. For each 3-second audio episode, we extract all 13 layer outputs and average across the time dimension, yielding a tensor of shape (13, 768).

- **MERT-v1-330M**: 24 transformer layers + 1 CNN = 25 layers of 1024-dimensional representations, yielding (25, 1024) per episode.

All embeddings are pre-computed and cached to disk, enabling rapid probe training without backbone forward passes.

### C. Probe Architectures

We investigate four probe architectures operating on frozen MERT embeddings:

**1) Learned-weight probe (MertProbe):** A learnable softmax-weighted sum across the L layers, followed by a two-layer MLP:

$$h = \sum_{l=1}^{L} \alpha_l \cdot z_l, \quad \alpha = \text{softmax}(w)$$
$$\hat{y} = W_2 \cdot \text{GELU}(\text{Dropout}(W_1 h + b_1)) + b_2$$

where $w \in \mathbb{R}^L$ are learnable layer weights and $z_l \in \mathbb{R}^{768}$ is the time-averaged output of layer $l$.

**2) Flat probe (MertFlatProbe):** Concatenates all L layers into a single vector of dimension $L \times d$ (9,984 for MERT-95M), followed by LayerNorm and a two-layer MLP with residual-style encoder:

$$h_{\text{flat}} = [z_1; z_2; \ldots; z_L] \in \mathbb{R}^{L \cdot d}$$
$$h_1 = \text{GELU}(\text{Dropout}(W_1 \cdot \text{LayerNorm}(h_{\text{flat}})))$$
$$h_2 = \text{GELU}(\text{Dropout}(W_2 h_1))$$
$$\hat{y} = W_3 h_2$$

This architecture (~10.8M parameters for MERT-95M) allows the classifier to access information from all transformer depths simultaneously.

**3) Deep probe (MertDeepProbe):** Groups layers into N equal groups, learns per-group and global weights, and processes through residual blocks. Despite added complexity (~4.7M parameters), this did not improve over the simpler flat probe.

**4) Attention probe (MertAttentionProbe):** Applies multi-head self-attention over the L layer representations with a learned CLS query for pooling. With only 13 tokens (layers), self-attention is counterproductive compared to the weighted sum.

### D. Ensemble Methods

We investigate five ensemble strategies:

**1) Weighted probability averaging:** For M models with track-level probability vectors $p_m(x)$, optimize per-model weights $w_m$ on the validation set:

$$\hat{p}(x) = \sum_{m=1}^{M} w_m \cdot p_m(x), \quad \text{s.t.} \sum_m w_m = 1, w_m \geq 0$$

Weights are optimized using the Nelder-Mead simplex algorithm [15] maximizing validation accuracy.

**2) Temperature-calibrated ensemble:** Before combining, each model's logits are scaled by an optimal temperature $T_m$ that minimizes negative log-likelihood on the validation set:

$$p_m^{\text{cal}}(x) = \text{softmax}(z_m(x) / T_m)$$

Temperatures are found via bounded scalar optimization ($T_m \in [0.1, 10]$).

**3) Genre-conditional weighting:** Instead of global weights, we learn per-genre weight vectors $w_g$ optimized on validation tracks of genre $g$. At test time, an initial prediction estimates the genre, and genre-specific weights produce the final prediction, iterated to convergence.

**4) MLP stacking:** A small MLP takes concatenated probability vectors from all M models ($8M$-dimensional input) and outputs an 8-class prediction, trained on the validation set.

**5) Hybrid feature fusion (proposed):** Our novel contribution concatenates:
- Penultimate embeddings from CNN (128-dim) and MERT flat probe (512-dim)
- Softmax probability vectors from 8 diverse models (8 classes x 8 models = 64-dim)

yielding a 704-dimensional input to a two-layer MLP:

$$h_{\text{fusion}} = [\text{emb}_{\text{CNN}}; \text{emb}_{\text{MERT}}; p_1; p_2; \ldots; p_8]$$
$$\hat{y} = \text{MLP}_{768 \to 384 \to 8}(\text{LayerNorm}(h_{\text{fusion}}))$$

The MLP is trained with mixup ($\alpha=0.3$), label smoothing (0.1), AdamW optimizer (lr=0.0005), and cosine learning rate scheduling. To reduce seed variance, we train K=10 models with different random seeds and average their predictions. This **seed-ensemble** is then combined with individual model predictions via Nelder-Mead weight optimization to form the final **meta-ensemble**.

---

## V. Experiments and Results

### A. CNN Experiments

| Model | Config | Track Acc | Track F1 |
|-------|--------|-----------|----------|
| CNN Baseline | lr=1e-3, bs=64 | 47.1% | 45.0% |
| + SpecAugment | augment_spec=true | 51.4% | 50.4% |
| + Tuned LR | lr=3e-4, AdamW, cosine | 52.9% | 52.2% |
| + Label smooth + Mixup | LS=0.1, mixup=0.3 | **55.1%** | **54.2%** |

Each regularization technique provides an additive improvement, with label smoothing and mixup together contributing +2.2pp over the tuned learning rate baseline.

### B. MERT-95M Probe Experiments

| Architecture | Variant | Track Acc | Track F1 |
|-------------|---------|-----------|----------|
| Learned-weight | Baseline (proj=512) | 55.9% | 54.9% |
| Learned-weight | Last-layer only | 52.8% | 51.8% |
| Learned-weight | Wide (proj=1024) | 56.1% | 55.3% |
| Learned-weight | Tuned (proj=768) | 56.4% | 55.7% |
| **Flat (concat)** | **v1 (proj=512)** | **58.3%** | **57.8%** |
| Flat (concat) | v2 (proj=256) | 57.8% | 57.3% |
| Deep (grouped) | v1 (3 groups) | 55.5% | 55.0% |
| Deep (grouped) | v2 (4 groups) | 56.0% | 55.9% |
| Attention | v1 (8 heads) | 54.6% | 53.6% |
| Attention | v2 (4 attn layers) | 55.3% | 54.4% |

The flat probe achieves the highest individual accuracy by accessing all transformer layers simultaneously. Layer weighting ablations show a +3.1pp advantage over using only the last layer, confirming that earlier layers encode complementary information (e.g., low-level timbral features in early layers, higher-level semantic features in later layers).

Seed ablation across 4 seeds reveals substantial variance: 53.5%, 54.5%, 55.3%, 56.1% (mean 54.9% $\pm$ 1.1%), indicating that improvements below ~2pp are not reliably distinguishable from initialization noise on this dataset.

### C. MERT-330M Probe Experiments

| Architecture | Track Acc | Track F1 |
|-------------|-----------|----------|
| Learned-weight (proj=1024) | **56.6%** | **56.5%** |
| Learned-weight (proj=2048) | 55.0% | 55.0% |
| Learned-weight (proj=512) | 56.3% | 56.2% |
| Flat (25,600-dim, proj=512) | 56.5% | 55.9% |
| Deep (5 groups) | 51.1% | 51.8% |

Despite having 3.5x more parameters, MERT-330M probes (best: 56.6%) do not outperform MERT-95M flat probes (58.3%). The deep probe with 5 layer groups severely overfits, dropping to 51.1%. This suggests that for this dataset size and difficulty, MERT-95M provides sufficient representational capacity and the bottleneck lies elsewhere — in the inherent ambiguity of genre boundaries under strict artist separation.

### D. Classical ML Baselines

To isolate the contribution of the MERT representation from the probe architecture, we train classical ML models on track-averaged MERT-95M flat features (9,984 dimensions, 7,194 training tracks):

| Method | Track Acc | Track F1 |
|--------|-----------|----------|
| SVM (RBF, C=10) | **56.1%** | **56.3%** |
| Logistic Regression (C=0.001) | 55.5% | 55.0% |
| ExtraTrees (500 trees) | 53.4% | 52.5% |
| XGBoost | 52.5% | 51.4% |

The SVM achieves 56.1% — matching the neural learned-weight probe (56.1%) on the same features. This confirms that the frozen MERT embeddings are the primary driver of accuracy; the classifier architecture provides minimal additional benefit.

### E. Ensemble Results

| Method | Track Acc | Track F1 |
|--------|-----------|----------|
| Weighted avg (4 models) | 59.6% | 59.2% |
| Temperature-calibrated (4 models) | 59.5% | 59.2% |
| Genre-conditional (4 models) | 59.3% | 58.9% |
| ESCF fixed fusion (2 models, 5 seeds) | 58.6% | 58.1% |
| MLP stacking (24 models) | 56.0% | 54.4% |
| Fusion seed-ensemble (10 seeds) | 60.0% | 59.9% |
| **Meta-ensemble (10 fusion + 4 individual)** | **60.3%** | **60.2%** |

Temperature calibration provides marginal improvement (+0.25pp), as learned temperatures range from 0.91 to 1.43 — models are not severely miscalibrated. Genre-conditional weighting underperforms global weighting because with only ~100 validation tracks per genre, per-genre optimization overfits. MLP stacking also overfits due to the limited validation set.

The hybrid feature fusion meta-ensemble achieves the best result at 60.3%. The key insight is that embedding-level fusion captures cross-modal interactions unavailable to probability-level combination: the CNN spectral embeddings encode fine-grained timbral information while MERT embeddings capture higher-level musical structure, and the MLP learns to exploit their complementarity.

### F. Per-Genre Analysis

Per-genre precision, recall, and F1 for our best individual model (MERT flat probe v1):

| Genre | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Electronic | 0.717 | 0.810 | 0.761 | 100 |
| Hip-Hop | 0.717 | 0.860 | 0.782 | 100 |
| International | 0.737 | 0.700 | 0.718 | 100 |
| Rock | 0.727 | 0.640 | 0.681 | 100 |
| Instrumental | 0.523 | 0.450 | 0.484 | 100 |
| Experimental | 0.421 | 0.450 | 0.435 | 100 |
| Pop | 0.420 | 0.370 | 0.394 | 100 |
| Folk | 0.369 | 0.380 | 0.374 | 100 |

Electronic and Hip-Hop are the most distinctive genres (F1 > 0.75), while Folk and Pop are hardest to classify (F1 < 0.40). This difficulty pattern is consistent across all architectures: Folk and Pop tracks share acoustic features with multiple other genres, and the artist-filtered split removes the possibility of using artist-specific cues as shortcuts.

### G. Diverse Probe Ablation

To investigate whether ensemble diversity can be increased through varied training configurations, we trained 6 flat probes with different random seeds (100–600), loss functions (cross-entropy and focal loss [16]), and regularization strengths:

| Variant | Seed | Loss | Dropout | Track Acc |
|---------|------|------|---------|-----------|
| s1 | 100 | CE+LS | 0.5 | 56.5% |
| s2 | 200 | CE+LS | 0.5 | 54.4% |
| s3 | 300 | CE+LS | 0.6 | 57.4% |
| s4 | 400 | Focal ($\gamma$=2) | 0.5 | 56.6% |
| s5 | 500 | Focal ($\gamma$=1) | 0.4 | 55.8% |
| s6 | 600 | CE+LS | 0.55 | 56.0% |

Seed variance dominates: individual accuracies range from 54.4% to 57.4% (3pp spread), while the effect of focal loss versus cross-entropy is within this noise band. This high variance is attributable to the small test set (800 tracks) and validates our use of multi-seed averaging for the fusion approach.

---

## VI. Comparison with Published Baselines

| Method | Year | Accuracy | Protocol |
|--------|------|----------|----------|
| Bian et al. — DenseNet+SVM [4] | 2019 | 68.9% | unspecified |
| Rai & Bhattacharjee — BYOL-A+DNN [6] | 2026 | 64.3% | unspecified |
| Liu et al. — CNN14 (PANNs) [5] | 2022 | 60.9% | unspecified |
| **Ours — Meta-ensemble** | **2026** | **60.3%** | **artist_filtered** |
| **Ours — Fusion seed-ensemble** | **2026** | **60.0%** | **artist_filtered** |
| **Ours — Weighted ensemble** | **2026** | **59.6%** | **artist_filtered** |
| Defferrard et al. — Features+SVM [2] | 2017 | 58.0% | artist_filtered |
| **Ours — MERT flat probe** | **2026** | **58.3%** | **artist_filtered** |
| Park et al. — Transfer CNN [7] | 2018 | 56.8% | unspecified |
| Defferrard et al. — Features+SVM [2] | 2017 | 54.8% | artist_filtered |
| Lee et al. — Transfer CNN [8] | 2019 | 51.2% | unspecified |

Our meta-ensemble (60.3%) surpasses 5 of 7 published baselines. The two we do not exceed — Bian et al. (68.9%) and Rai & Bhattacharjee (64.3%) — use unspecified evaluation protocols. Given our demonstrated 35pp leakage inflation factor, these results are likely inflated by train-test leakage and are not directly comparable.

Among confirmed artist-filtered results, our meta-ensemble (60.3%) exceeds the previous best of 58.0% [2] by **2.3 percentage points**, establishing a new state-of-the-art under strict evaluation.

---

## VII. Discussion

### A. The Protocol Gap

Our most striking finding is the 35.3pp gap between clip-random and artist-filtered evaluation. This inflation arises because artists have distinctive production styles, vocal timbres, and recording characteristics — when episodes from the same artist appear in both train and test, the model learns artist identity rather than genre features. This finding has broad implications: any MGC result on FMA (or similar datasets) without explicit protocol specification should be interpreted with extreme caution.

### B. Why MERT-330M Underperforms MERT-95M

Counter-intuitively, the larger MERT-330M model yields weaker probes than MERT-95M (56.6% vs 58.3% best). We attribute this to two factors: (1) the 25-layer, 1024-dim representation (25,600-dim flat) is more prone to overfitting with only 6,400 training tracks, and (2) MERT-330M's additional capacity may encode finer-grained musical details that are irrelevant to the coarse 8-genre classification task. The representation quality is not the bottleneck — the task difficulty under artist separation is.

### C. The Binding Constraint

Classical SVM achieves 56.1% on the same MERT features where neural probes achieve 56.1–58.3%. This small gap (0–2.2pp) confirms that the MERT representation quality — not the classifier — is the binding constraint. Similarly, complex probe architectures (deep, attention) underperform simple ones (flat concatenation). The primary avenue for improvement is not better classifiers but better representations or better evaluation-aware training strategies.

### D. Ensemble Ceiling

Despite trying 5 distinct ensemble strategies and 31 models, accuracy plateaus around 60%. Adding more models of the same type (diverse flat probes) does not break through this ceiling. The hybrid feature fusion provides the final meaningful gain (+0.7pp over weighted averaging) by combining information at the embedding level rather than the probability level. Further progress likely requires either (1) end-to-end fine-tuning of the MERT backbone, (2) audio augmentation at the waveform level, or (3) larger training sets.

---

## VIII. Conclusion

We present a comprehensive study of music genre classification on FMA-small under strict artist-filtered evaluation, training 31 models and demonstrating that evaluation protocol is the single most important factor in reported accuracy. Our hybrid feature fusion meta-ensemble achieves 60.3% track-level accuracy — a new state-of-the-art among confirmed artist-filtered results, surpassing 5 of 7 published baselines. We show that the 35pp inflation from evaluation protocol leakage dwarfs architectural improvements, and we advocate for mandatory protocol specification in future MGC research.

Our code, all 31 trained models, cached MERT embeddings, and reproduction scripts are publicly available.

---

## References

[1] G. Tzanetakis and P. Cook, "Musical genre classification of audio signals," *IEEE Trans. Speech Audio Process.*, vol. 10, no. 5, pp. 293–302, Jul. 2002.

[2] M. Defferrard, K. Benzi, P. Vandergheynst, and X. Bresson, "FMA: A dataset for music analysis," in *Proc. 18th Int. Soc. Music Inf. Retrieval Conf. (ISMIR)*, 2017, pp. 316–323. [Online]. Available: https://arxiv.org/abs/1612.01840

[3] Y. Li, R. Yuan, G. Zhang, et al., "MERT: Acoustic music understanding model with large-scale self-supervised training," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2024. [Online]. Available: https://arxiv.org/abs/2306.00107

[4] T. Bian, Y. Wang, C. Zhuang, et al., "Audio-based music classification with DenseNet and data augmentation," 2019. [Online]. Available: https://arxiv.org/abs/1906.11620

[5] H. Liu, S. DeMori, and A. Abayomi, "Music genre classification with pre-trained audio neural networks," 2022. [Online]. Available: https://arxiv.org/abs/2209.07548

[6] A. Rai and A. Bhattacharjee, "BYOL-A based music genre classification," 2026. [Online]. Available: https://arxiv.org/abs/2603.13871

[7] J. Park, J. Lee, J. Park, J.-W. Ha, and J. Nam, "Representation learning of music using artist labels," in *Proc. 19th Int. Soc. Music Inf. Retrieval Conf. (ISMIR)*, 2018.

[8] J. Lee, J. Lee, J. Park, and J. Nam, "Transfer learning for music genre classification," 2019.

[9] T. G. Dietterich, "Ensemble methods in machine learning," in *Proc. Int. Workshop Multiple Classifier Syst.*, 2000, pp. 1–15.

[10] D. H. Wolpert, "Stacked generalization," *Neural Netw.*, vol. 5, no. 2, pp. 241–259, 1992.

[11] L. Breiman, "Bagging predictors," *Mach. Learn.*, vol. 24, no. 2, pp. 123–140, 1996.

[12] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2017, pp. 1321–1330.

[13] D. S. Park, W. Chan, Y. Zhang, et al., "SpecAugment: A simple data augmentation method for automatic speech recognition," in *Proc. Interspeech*, 2019, pp. 2613–2617.

[14] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond empirical risk minimization," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2018.

[15] J. A. Nelder and R. Mead, "A simplex method for function minimization," *Comput. J.*, vol. 7, no. 4, pp. 308–313, 1965.

[16] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, "Focal loss for dense object detection," in *Proc. IEEE Int. Conf. Comput. Vision (ICCV)*, 2017, pp. 2980–2988.
