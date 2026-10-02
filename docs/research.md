# Hybrid Feature Fusion for Music Genre Classification Under Strict Artist-Filtered Evaluation

---

**Abstract** — Music genre classification on the Free Music Archive (FMA) dataset is reported as a solved-to-90% benchmark, yet of the nine published 8-way FMA-small results we could verify, not one states in prose that it used the dataset's official artist-filtered split, and six disclose nothing about their split beyond a ratio. We quantify what that silence costs. Reimplementing published architectures and running them, alongside our own, down a three-rung protocol ladder — artist-filtered, random-track, random-clip — we measure total inflation of +14.1 to +41.0 percentage points (mean +30.5pp across eight architectures, all track-level), decomposing into +12.8pp from dropping the artist filter and a further +20.0pp from letting clips of one recording span train and test. The reproduction is specific, not analogical: two papers that disclose no artist-conditioning report 64.3% and 60.9%, while reimplementations of their architectures reach only 57.4% artist-filtered and overshoot both figures once the artist filter is removed. We also show that the field's reference baseline, "58.0% (Defferrard et al.)", appears in no version of the FMA paper or its repository and is a transposed 16-genre medium-subset cell. Alongside the audit we present a hybrid feature fusion meta-ensemble that concatenates CNN and MERT-v1-95M embeddings with multi-model softmax probabilities, reaching **60.3% track-level accuracy** under the official split — which we offer as a documented baseline rather than a state of the art. All code, 48 trained models, and cached embeddings are released.

**Index Terms** — music genre classification, self-supervised learning, MERT, ensemble methods, evaluation protocol, FMA dataset

---

## I. Introduction

Automatic music genre classification (MGC) is a foundational task in music information retrieval (MIR) with applications in recommendation systems, music discovery, and library organization [1]. The Free Music Archive (FMA) dataset [2] provides a standardized benchmark with an official artist-filtered train/validation/test split designed to prevent information leakage — ensuring that no artist appears in more than one split.

Despite this, the majority of published results on FMA-small either do not specify their evaluation protocol or split at a unit finer than the track, allowing episodes of one recording to appear in both training and testing. We show this inflates reported accuracy by 17 to 43 percentage points, which is larger than the entire span of accuracies the literature competes over, and makes cross-study comparison unreliable in the strong sense: the ranking is not noisy, it is measuring different quantities.

In this work, we make four contributions:

1. **A protocol audit of the literature**: We read every published 8-way FMA-small result from 2017–2026 and record what each paper's own text discloses about its split — never what its accuracy implies, which would be circular. We find zero prose statements of artist-filtered evaluation, one stated clip-random split, one stated track-level split, and six papers that disclose nothing beyond a ratio. We also exclude three widely circulated "FMA-small records" that do not survive checking, including the field's reference baseline (Section II-A).

2. **A three-rung protocol ladder, and reproduction on it**: Rather than a binary leaking/non-leaking comparison, we evaluate under artist-filtered, random-track, and random-clip splits, because most of the literature states a unit too vaguely to be scored against a binary. We reimplement published architectures from their papers and run them down all three rungs on identical data and scoring.

3. **Comprehensive evaluation under strict protocol**: We train and evaluate 48 models — 33 under the official FMA artist-filtered split and 15 more under the two leakier protocols — spanning custom CNNs, reimplementations of published architectures, frozen MERT [3] / BYOL-A / PANNs embeddings with various probe architectures, and multiple ensemble strategies, all scored track-level.

4. **Hybrid feature fusion**: We propose a fusion approach that concatenates penultimate embeddings from diverse backbones (CNN and MERT) with softmax probability vectors from multiple trained models, feeding this representation into a lightweight MLP. This embedding-level fusion captures cross-modal interactions that probability-level ensembles cannot.

---

## II. Related Work

### A. Music Genre Classification on FMA

Defferrard et al. [2] introduced FMA together with an artist-filtered split shipped in the dataset's own metadata, and with hand-crafted feature + SVM baselines. A figure of "58.0% (Defferrard et al.)" is widely quoted as the artist-filtered reference point on FMA-small; it does not exist. We traced it and report the trace here, because the field's only agreed-upon honest baseline turns out to be a citation artifact.

The FMA paper proposes four tasks, of which problem 1 is "single top genre on the balanced small subset". It is proposed and never reported. Table 6, the only accuracy table, is captioned "Test set accuracies of various features and classifiers for top genre recognition on the **medium** subset" — 25,000 tracks, 16 genres. The value 58 does appear in Table 6, as the MLP/all-features cell, on medium. The repository's executed baseline notebook likewise subsets to `<= 'medium'` and prints 19,922/2,505/2,573 examples over 16 genres; it contains no FMA-small run in any version. An open issue on the FMA repository (mdeff/fma#69, December 2023) asks the authors directly for small-subset baselines and has received no reply in over two years. We conclude that **no FMA-small accuracy was ever published by Defferrard et al.**, and that the circulating 58.0% is a transposition of a medium-subset, 16-genre cell.

The number that *is* real is 54.82%, and it belongs to Park et al. [7], whose Table 4 lists `SVM [7] — 0.5482 †` with the footnote "this result was obtained using the provided code and dataset in [7]". It is a re-run of FMA's released baseline code on fma_small. Because that code reads the official `split` column, the result is artist-filtered **by construction** — though note that even here the property is inherited from the code, not asserted by the authors. Bian et al. [4] later reproduce the figure as "SVM (Defferrard, 2017) 54.8", which is how a third-party re-run became attributed to the dataset authors. We cite it to Park.

Every protocol statement attributed to a paper below was read from that paper's own text; where a paper is silent we say so rather than inferring a protocol from its accuracy, which would make our argument circular.

**CNN-based approaches.** Bian et al. [4] achieved 68.9% using DenseNet with time-overlap and pitch-shift augmentation plus SVM stacking. Their paper contains no train/test split description of any kind. Their augmentation generates extra windows at 50% overlap and they slice each track into ten 2.56s segments, so clip-level leakage is *possible*, but they never state whether augmentation precedes or follows splitting and we do not assume it. Evaluation is track-level: segment predictions are averaged into "a single song-level prediction". Ashraf et al. [17] reported 68.5% with a dual CNN using modified residual learning and hybrid pooling, and are the one paper in this survey that states clip-level splitting outright: "we segmented all music files into 3-second clips and split them into training, validation, and test sets using an 8:1:1 ratio". Segmentation precedes splitting, so clips of one track span train and test. Kostrzewa et al. [21] tested CNN, RNN, and CRNN ensembles, reporting 56.4% accuracy and 54.9% macro F1. They give a ratio — "divided into three independent subsets in an 80:10:10 ratio" — but no selection mechanism and no mention of artists. Their input is one 128×128 mel-spectrogram per song with no segmentation, so same-track leakage is structurally impossible; the open question is only artist overlap. Notably, they themselves criticise a competing result for "data leakage from the training of deep learning model to the test data, which leads to over-optimistic outcomes".

**Hybrid CNN-Transformer architectures.** Ma et al. [18] proposed CT-GateNet, a CNN-Transformer hybrid with gated channel-spatial attention, reporting 89.42% on FMA-small. Their only split sentence is "all datasets were evenly split into training, validation and test sets at an 8:1:1 ratio" — the unit of splitting is never named, so whether clips or tracks were split cannot be determined from the paper. Zhang [19] achieved 90.23% with TS-Resformer (ResNet + Transformer fusion). The paper does slice audio into 12-second segments with 50% overlap, but the split is described over *songs* — "the dataset is sliced into training set, validation set, and test set according to 8:1:1, and the number of songs of different genres in the three sets is guaranteed to be balanced" — which places the overlap *within* each partition rather than across them. We therefore classify it as a track-level random split without artist filtering, not as clip-random. Neither paper mentions artist filtering or FMA's official split files.

**Pretrained audio representations.** Liu et al. [5] applied AudioSet-pretrained CNN14 (PANNs) and report 60.9% on the eight FMA-small genres. Two qualifications are needed. First, their paper is about *open-set* recognition: 60.9% is the closed-set sub-metric $Acc_c$ from Experiment 2, measured on the eight known genres within a test set that also contains unknown classes drawn from fma_medium. Reading it as a plain 8-way fma_small accuracy is defensible but not exactly what the paper measures. Second, and unlike the other works surveyed here, they *do* describe their partitioning — "each dataset is split into a training set, an evaluation set, and a test set for each experiment" — and their Experiment 2 sizes of 6,400/800 reproduce FMA-small's official partition exactly, with the text referring to "the valid split". The strong likelihood is that they used FMA's official files. What they never state is whether those splits are artist-conditional; the word "artist" appears in the paper only in discussion of label quality. Their input is the whole 30s waveform, so clip leakage is structurally impossible in any case. Rai and Bhattacharjee [6] used BYOL-A self-supervised embeddings with a tuned DNN classifier for 64.3%, averaging embeddings over each whole track. Their paper contains no split description of any kind: the strings "split", "artist", "fold", "partition", "random" and "stratif" do not occur in it. Park et al. [7] reported 56.9% using transfer learning from artist labels. Their treatment of artists is instructive: they explicitly use a "fault-filtered" GTZAN "where the dataset was divided to prevent artist repetition in training/validation/test sets", but make no equivalent statement for FMA-small — the awareness is present, the disclosure is not.

**Other approaches.** Two results that circulate as FMA-small records do not survive checking, and we exclude both. The capsule network with UIGMM optimisation [20] is widely listed at 68.87% on FMA, but that paper runs only on ISMIR2004, GTZAN and Extended Ballroom; the 68.87% figure appears in its related-work discussion of Ashraf et al. and is not a result of its own. The DPADMM-G parallel sparse SVM [22] is listed at 97.27% on the FMA-small test set, but that number is a *binary* two-genre task on FMA's precomputed feature vectors: "we randomly selected two categories from the eight categories in fma_small... from each category, we randomly chose 800 samples as the training set and the remaining 200 samples as the test set". It is not an 8-way audio classification result and is not comparable. That a leaderboard has absorbed a binary-subtask number into an 8-way benchmark is itself evidence for the argument of this paper.

A critical observation is that of the nine publications that genuinely report 8-way FMA-small results (2017–2026), **not one states in prose that it used the artist-filtered split**. The closest is the 54.8% SVM row in Park et al. [7], which is artist-filtered only because the FMA baseline code it re-ran reads the official `split` column. Exactly one paper — Ashraf et al. [17] — states a clip-random protocol outright. One [19] states a track-level random split. One [5] describes its partitions without saying whether they are artist-conditional. The remaining six give either a bare ratio or nothing at all. As we demonstrate in Section V-G and Section VI, this silence is decisive: the same model can score 30+ percentage points higher under clip-random evaluation, and a paper reporting only "random 8:1:1" has not told the reader which number it is.

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

Because every number in this paper depends on which split produced it, we define three protocols and use them consistently throughout:

| Mode | Definition | What can leak |
|------|-----------|---------------|
| `official` | FMA's artist-filtered split, shipped in the dataset metadata | nothing |
| `random_track` | random 8:1:1 over whole tracks, seed 42 | artist and album identity |
| `random_clip` | random 8:1:1 over individual 3s clips, seed 42 | artist, album, *and the recording itself* |

`random_track` is constructed so that no `track_id` appears in more than one partition (verified: 60,263 / 7,521 / 7,512 episodes with zero tracks spanning splits). It exists because the literature is not binary — several papers state a random 8:1:1 ratio without naming the unit it applies to, and scoring those against `official` alone would misattribute the entire gap to clip leakage.

Only `official` produces a headline number in this paper. The other two rungs are diagnostic instruments, and we report them as such. Section V-G gives the measured inflation across eight architectures; Section VI-C runs reimplemented published architectures down the same ladder.

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

### G. Multi-Model Protocol Leakage Proof

To test whether protocol inflation is a property of one architecture or of the evaluation itself, we train eight architectures down the ladder defined in Section III-C — five MERT probe variants, a log-mel CNN, and frozen BYOL-A and PANNs embeddings with DNN heads — holding data, preprocessing, and scoring fixed.

All accuracies in this table are **track-level**: per-episode softmax averaged within a track, then argmax. We report track-level throughout because every published number we compare against is one prediction per track — Liu et al. [5] and Rai and Bhattacharjee [6] feed a whole 30s track and never segment it, while Bian et al. [4] and Park et al. [7] both average segment outputs into a single song-level prediction. Quoting our episode-level accuracy beside their track-level accuracy would compare two different measurements, which is the error this paper exists to document. Episode-level figures for every run are in the released `summary.json` files, and the ladder can be regenerated at that level with `LEAKAGE_LEVEL=episode`.

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

Total inflation ranges from +14.1pp to +41.0pp with a mean of +30.5pp. Protocol leakage is not an artifact of a single architecture: it affects log-mel CNNs, self-supervised music transformers, and AudioSet-pretrained audio CNNs alike.

The magnitude, however, is architecture-dependent in an interpretable way. MERT-based models inflate most (+32 to +41pp) because their waveform-level representations capture fine-grained acoustic signatures — timbre, production style, vocal characteristics — which is exactly the capability that makes them good music models and exactly the capability that lets them recognise a recording they have already seen. The log-mel CNN still inflates by 14.1pp despite processing each clip independently with far fewer parameters, because even coarse spectral statistics carry artist identity. Representation quality and leakage susceptibility are the same axis.

Averaged over the seven architectures trained on all three rungs, the two failures are of comparable size rather than one dominating: dropping the artist filter alone accounts for +12.8pp, and leaking clips on top of that adds a further +20.0pp. A paper that reports "random 8:1:1" without naming the unit has not disclosed which of those two it incurred — and the difference between them exceeds the margins this literature competes over.

### H. Diverse Probe Ablation

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

### A. Published Results Registry

We compile every published 8-way FMA-small result from 2017–2026. The *Stated protocol* column records what each paper's own text discloses, not what its accuracy implies — inferring the protocol from the number would make the argument of this paper circular.

| Method | Year | Accuracy | Stated protocol | Disclosed? |
|--------|------|----------|-----------------|------------|
| TS-Resformer — ResNet+Transformer [19] | 2025 | 90.23% | random 8:1:1 over *songs*; 12s segments with 50% overlap inside each partition | partial — track-level, artists unaddressed |
| CT-GateNet — CNN+Transformer [18] | 2026 | 89.42% | "evenly split ... at an 8:1:1 ratio"; unit of splitting never named | **no** |
| Bian et al. — DenseNet+SVM [4] | 2019 | 68.9% | none; track-level scoring stated, split never described | **no** |
| Ashraf et al. — Dual CNN [17] | 2025 | 68.5% | 3s clips segmented *then* split 8:1:1 → clip-random | **yes** |
| Rai & Bhattacharjee — BYOL-A+DNN [6] | 2025 | 64.3% | none; embeddings averaged over whole tracks | **no** |
| Liu et al. — CNN14 (PANNs) [5] | 2022 | 60.9%\* | splits described and 6,400/800 counts match FMA's official partition; whether those splits are artist-conditional is never stated | partial |
| **Ours — Meta-ensemble** | **2026** | **60.3%** | **official artist-filtered, track-level** | **yes** |
| **Ours — Fusion seed-ensemble** | **2026** | **60.0%** | **official artist-filtered, track-level** | **yes** |
| **Ours — Weighted ensemble** | **2026** | **59.6%** | **official artist-filtered, track-level** | **yes** |
| **Ours — MERT flat probe** | **2026** | **58.3%** | **official artist-filtered, track-level** | **yes** |
| Park et al. — Transfer CNN [7] | 2018 | 56.9% | none for FMA; the same paper *does* specify artist-filtering for GTZAN | **no** |
| Features+SVM, re-run by Park et al. [7] of FMA's baseline code | 2018 | 54.8% | artist-filtered *by construction* — the released code reads FMA's `split` column — but not asserted in prose | inherited |
| Kostrzewa et al. — CNN/RNN ensembles [21] | 2021 | 56.4% (54.9% F1) | "80:10:10 ratio", selection mechanism unstated; one spectrogram per song, so clip leakage is structurally impossible | **no** |

\* Liu et al.'s 60.9% is the closed-set sub-metric of an open-set experiment, not a standalone 8-way benchmark result; see Section II-A.

Three results that circulate as FMA-small records are excluded, for reasons given in Section II-A: the CapsNet+UIGMM paper [20] never runs on FMA (its 68.87% is a quotation of Ashraf et al. in related work); the DPADMM-G 97.27% [22] is a *binary* two-genre task on precomputed feature vectors, not 8-way audio classification; and "Defferrard et al., 58.0%" is a transposition of a 16-genre medium-subset cell and has no FMA-small source.

Of the nine external works that genuinely report 8-way FMA-small accuracy, **not one** states in prose that it used FMA's artist-filtered split. The strongest artist-filtered prior is 54.8% — and it holds that property only because the code it ran reads the `split` column. **One** paper states clip-random splitting outright, **one** states a track-level random split, and the remaining **six** disclose nothing beyond a ratio or nothing at all.

### B. The Three-Rung Protocol Ladder

A binary artist-filtered/clip-random comparison is not sufficient to score this literature, because most of the literature is not binary. Only one paper states clip-level splitting; one states song-level splitting; the rest state a ratio and nothing more. We therefore run every architecture down a three-rung ladder, so that each paper can be placed against the rung its own text supports:

| Rung | Definition | What leaks |
|------|-----------|------------|
| `official` | FMA's artist-filtered split, shipped with the dataset | nothing |
| `random_track` | random 8:1:1 over whole tracks (seed 42) | artist and album identity |
| `random_clip` | random 8:1:1 over 3s clips (seed 42) | artist, album, *and the recording itself* |

The middle rung is constructed so that no `track_id` appears in more than one partition (verified: 60,263 / 7,521 / 7,512 episodes, zero tracks spanning splits). It isolates the artist effect from the same-recording effect, which the binary framing conflates.

### C. Reproduction of Published Architectures

Rather than arguing by analogy, we reimplement each published architecture from its paper and run it down the full ladder on identical data, identical preprocessing, and identical track-level scoring. The claim is not that our reimplementation reproduces every engineering detail — it does not, and we drop the SVM stacking in [4] and the gating in [18] — but that the *protocol* accounts for the gap between a paper's headline and the artist-filtered reality.

| Paper | Published | Stated protocol | Our `official` | Our `random_track` | Our `random_clip` |
|-------|-----------|-----------------|----------------|--------------------|-------------------|
| Rai & Bhattacharjee [6] BYOL-A+DNN | 64.3% | none | 57.4% | **69.9%** | 84.5% |
| Liu et al. [5] PANNs CNN14 | 60.9% | partitions described, artist-conditioning not | 57.4% | **67.2%** | 73.4% |
| Bian et al. [4] DenseNet121 | 68.9% | none | 53.9% | — | *not yet trained* |
| Ashraf et al. [17] Dual CNN | 68.5% | clip-random (stated) | *not yet trained* | — | *not yet trained* |
| Kostrzewa et al. [21] CNN+BiGRU | 56.4% | ratio only, track-level input | *not yet trained* | — | *not yet trained* |
| Ma et al. [18] CT-GateNet | 89.42% | ratio only, unit unnamed | *not yet trained* | — | *not yet trained* |

Two rows are complete across all three rungs, and both tell the same story with an important qualification. Rai and Bhattacharjee disclose no split at all and report 64.3%; Liu et al. describe partitions without stating artist-conditioning and report 60.9%. Neither published figure is reachable by our reimplementation under the artist-filtered protocol, where we measure 57.4% for both. Both published figures sit *below* our corresponding random-track results (69.9% and 67.2%) and far below the clip-random ones.

We are deliberate about what this does and does not establish. It does **not** show that we reproduced either paper's number: at the middle rung our reimplementations overshoot both, by +5.6pp and +6.3pp respectively. Reimplementations differ from their originals in optimiser, schedule, head capacity, and augmentation, so an exact match was never the right expectation, and an earlier draft of this paper claimed one only because it compared our episode-level accuracy against their track-level accuracy — the same unit mismatch we criticise elsewhere. What the rows do establish is narrower and sufficient: the published figures lie in the band these architectures occupy *once the artist filter is removed*, and well outside the band they occupy with it. For both papers the honest conclusion is that their protocol is unresolvable from their text, and that the range of values it could denote spans more than the margin separating most published FMA-small results from one another.

### D. Protocol Disclosure Analysis

| Protocol status | Count (of 9 external works) | Share |
|----------------|------|-------|
| Artist-filtered, stated in prose | **0** | 0% |
| Artist-filtered by construction, not asserted | 1 (SVM re-run in Park et al. [7]) | 11% |
| Clip-random, stated | 1 (Ashraf et al. [17]) | 11% |
| Track-level random, stated | 1 (Zhang [19]) | 11% |
| Splits described, artist-conditioning unstated | 1 (Liu et al. [5]) | 11% |
| Ratio given, unit unstated | 2 (Ma et al. [18], Kostrzewa et al. [21]) | 22% |
| No split description at all | 3 (Bian [4], Rai [6], Park [7]) | 33% |

Two-thirds of the literature leaves the reader unable to determine which of our three columns the number belongs beside. This is the finding, and it is independent of whether any individual paper leaked: a benchmark on which six of nine results cannot be placed on a protocol ladder is not a benchmark that supports cross-study comparison, and the leaderboards that rank these numbers against each other are ranking incomparable quantities.

The disclosure failure extends to the baseline itself. The single number the field treats as the honest artist-filtered reference — "58.0%, Defferrard et al." — is not in the FMA paper, not in its repository, and not in any arXiv version; it is a medium-subset, 16-genre cell that acquired a small-subset attribution through repeated citation. A reader who wanted to check it would have had to read Table 6's caption, which is one sentence of work that nobody appears to have done. We make no claim of bad faith on anyone's part; we note only that a benchmark whose reference point survived nine years without that check is a benchmark whose numbers should not be compared without one.

---

## VII. Discussion

### A. The Protocol Gap

Our most striking finding is the systematic inflation caused by clip-random evaluation. Across eight architectures — a log-mel CNN, learned-weight probes, flat concatenation, multi-scale probes, self-attention probes, MERT-330M, and frozen BYOL-A / PANNs embeddings — total track-level inflation ranges from +14.1pp to +41.0pp with a mean of +30.5pp. This is not a quirk of one architecture: it is a property of the evaluation protocol.

The inflation arises because artists have distinctive production styles, vocal timbres, and recording characteristics. When episodes from the same track appear in both train and test, the model learns acoustic fingerprints rather than genre features. MERT-based models inflate more (35–43pp) than log-mel CNNs (17pp) precisely because their richer waveform-level representations are better at recognizing specific recordings — a capability that is an asset for most music tasks and a liability for this evaluation.

The three-rung ladder shows the inflation is not one effect but two, of comparable size. Averaged over the seven architectures run on all three rungs, dropping the artist filter alone costs +12.8pp and leaking clips on top of that adds a further +20.0pp. A paper reporting "random 8:1:1" without naming the unit has not told the reader which of those two numbers applies to it, and the difference is larger than the margin by which most of this literature claims to improve on its predecessors.

The reproduction in Section VI-C is the load-bearing evidence. Rai and Bhattacharjee [6] disclose no split and report 64.3%; Liu et al. [5] describe partitions without stating artist-conditioning and report 60.9%. Our reimplementations of both architectures score 57.4% artist-filtered — below either published figure — and 69.9% and 67.2% respectively once the artist filter is dropped, overshooting both. We do not claim to have reproduced either number; our reimplementations differ in optimiser, head capacity, and augmentation, and they land above rather than on the published values. We claim something weaker and more uncomfortable: each published figure falls inside the range its architecture spans across undisclosed protocol choices, and outside the range it spans under the disclosed one. Nothing in either paper rules out the leaking interpretation, and that is the whole problem.

Of nine published 8-way FMA-small results, not one states in prose that it used the artist-filtered split. We advocate for mandatory protocol specification — specifically, naming the *unit* of splitting and whether artist filtering was applied — in all future music genre classification research.

### B. Why MERT-330M Underperforms MERT-95M

Counter-intuitively, the larger MERT-330M model yields weaker probes than MERT-95M (56.6% vs 58.3% best). We attribute this to two factors: (1) the 25-layer, 1024-dim representation (25,600-dim flat) is more prone to overfitting with only 6,400 training tracks, and (2) MERT-330M's additional capacity may encode finer-grained musical details that are irrelevant to the coarse 8-genre classification task. The representation quality is not the bottleneck — the task difficulty under artist separation is.

### C. The Binding Constraint

Classical SVM achieves 56.1% on the same MERT features where neural probes achieve 56.1–58.3%. This small gap (0–2.2pp) confirms that the MERT representation quality — not the classifier — is the binding constraint. Similarly, complex probe architectures (deep, attention) underperform simple ones (flat concatenation). The primary avenue for improvement is not better classifiers but better representations or better evaluation-aware training strategies.

### D. Ensemble Ceiling

Despite trying 5 distinct ensemble strategies and 33 artist-filtered models, accuracy plateaus around 60%. Adding more models of the same type (diverse flat probes) does not break through this ceiling. The hybrid feature fusion provides the final meaningful gain (+0.7pp over weighted averaging) by combining information at the embedding level rather than the probability level. Further progress likely requires either (1) end-to-end fine-tuning of the MERT backbone, (2) audio augmentation at the waveform level, or (3) larger training sets.

---

## VIII. Conclusion

We present a study of music genre classification on FMA-small under strict artist-filtered evaluation, training 48 models and demonstrating that evaluation protocol is the single largest factor in reported accuracy. Our hybrid feature fusion meta-ensemble achieves 60.3% track-level accuracy. We state the comparison carefully: the only artist-filtered FMA-small result we could source is 54.8%, which is Park et al.'s [7] re-run of FMA's own baseline code and is artist-filtered because that code reads the official split column, not because any author says so. The widely cited "58.0% (Defferrard et al.)" does not exist in the FMA paper, its repository, or any arXiv version — it is a 16-genre medium-subset cell. Against the one sourceable artist-filtered number, we improve by 5.5pp; against a literature that has never stated this protocol, we decline to claim a state of the art and instead supply the missing baseline.

Our protocol analysis spans eight architectures down a three-rung ladder (artist-filtered → random-track → random-clip), revealing total track-level inflation of +14.1 to +41.0pp, mean +30.5pp, decomposing into +12.8pp from dropping the artist filter and a further +20.0pp from leaking clips. Reimplementing published architectures and running them down the same ladder, we find that reported accuracies of 64–90% on FMA-small land on the leaking rungs rather than the honest one: the undisclosed-protocol figures of BYOL-A + DNN (64.3%) and PANNs CNN14 (60.9%) both exceed what their architectures achieve artist-filtered in our hands (57.4% for both) and fall below what they achieve once the artist filter is removed. Of nine published works, none states artist-filtered evaluation in prose, one states clip-random, one states track-level random, and six disclose nothing beyond a ratio. We advocate for mandatory protocol specification — the unit of splitting, and whether artist filtering was applied — in future MGC research.

Our code, all 48 trained models, cached MERT embeddings, and reproduction scripts are publicly available.

---

## References

[1] G. Tzanetakis and P. Cook, "Musical genre classification of audio signals," *IEEE Trans. Speech Audio Process.*, vol. 10, no. 5, pp. 293–302, Jul. 2002.

[2] M. Defferrard, K. Benzi, P. Vandergheynst, and X. Bresson, "FMA: A dataset for music analysis," in *Proc. 18th Int. Soc. Music Inf. Retrieval Conf. (ISMIR)*, 2017, pp. 316–323. [Online]. Available: https://arxiv.org/abs/1612.01840

[3] Y. Li, R. Yuan, G. Zhang, et al., "MERT: Acoustic music understanding model with large-scale self-supervised training," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2024. [Online]. Available: https://arxiv.org/abs/2306.00107

[4] T. Bian, Y. Wang, C. Zhuang, et al., "Audio-based music classification with DenseNet and data augmentation," 2019. [Online]. Available: https://arxiv.org/abs/1906.11620

[5] K. Liu, J. DeMori, and K. Abayomi, "Open set recognition for music genre classification," arXiv preprint arXiv:2209.07548, 2022 (rev. 2024). [Online]. Available: https://arxiv.org/abs/2209.07548

[6] K. Rai and M. Bhattacharjee, "Evaluating pretrained general-purpose audio representations for music genre classification," in *Proc. Int. Conf. Pattern Recognition and Machine Intelligence (PReMI)*, 2025. [Online]. Available: https://arxiv.org/abs/2603.13871

[7] J. Park, J. Lee, J. Park, J.-W. Ha, and J. Nam, "Representation learning of music using artist labels," in *Proc. 19th Int. Soc. Music Inf. Retrieval Conf. (ISMIR)*, 2018. [Online]. Available: https://arxiv.org/abs/1710.06648 (Table 4 is the source of the 54.82% FMA-small SVM figure commonly re-attributed to [2].)

[8] "Baseline performances for 'Single top genre on the balanced small subset'," GitHub issue mdeff/fma#69, Dec. 2023. Unanswered as of this writing; cited as evidence that no FMA-small baselines were ever released. [Online]. Available: https://github.com/mdeff/fma/issues/69. See also the executed baselines notebook on the repository's `outputs` branch, which subsets to `<= 'medium'`.

[9] T. G. Dietterich, "Ensemble methods in machine learning," in *Proc. Int. Workshop Multiple Classifier Syst.*, 2000, pp. 1–15.

[10] D. H. Wolpert, "Stacked generalization," *Neural Netw.*, vol. 5, no. 2, pp. 241–259, 1992.

[11] L. Breiman, "Bagging predictors," *Mach. Learn.*, vol. 24, no. 2, pp. 123–140, 1996.

[12] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2017, pp. 1321–1330.

[13] D. S. Park, W. Chan, Y. Zhang, et al., "SpecAugment: A simple data augmentation method for automatic speech recognition," in *Proc. Interspeech*, 2019, pp. 2613–2617.

[14] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond empirical risk minimization," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2018.

[15] J. A. Nelder and R. Mead, "A simplex method for function minimization," *Comput. J.*, vol. 7, no. 4, pp. 308–313, 1965.

[16] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, "Focal loss for dense object detection," in *Proc. IEEE Int. Conf. Comput. Vision (ICCV)*, 2017, pp. 2980–2988.

[17] M. Ashraf, S. Abid, M. A. Raza, et al., "Music genre classification with modified residual learning and dual neural network," *PLoS ONE*, vol. 20, no. 10, e0333808, Oct. 2025. [Online]. Available: https://doi.org/10.1371/journal.pone.0333808

[18] J. Ma et al., "Adaptive feature fusion gate and gated channel-spatial attention for music genre classification (CT-GateNet)," *PLoS ONE*, Apr. 2026. [Online]. Available: https://doi.org/10.1371/journal.pone.0344606

[19] Y. Zhang, "TS-Resformer: A model based on multimodal fusion for the classification of music signals," *Frontiers in Neurorobotics*, vol. 19, art. 1568811, 2025. [Online]. Available: https://doi.org/10.3389/fnbot.2025.1568811

[20] Advanced music classification using a combination of capsule neural network by upgraded ideal gas molecular movement algorithm, *Scientific Reports*, vol. 14, Dec. 2024. [Online]. Available: https://doi.org/10.1038/s41598-024-81700-8

[21] D. Kostrzewa, P. Kaminski, and R. Brzeski, "Music genre classification: Looking for the perfect network," in *Proc. 21st Int. Conf. Computational Science (ICCS)*, 2021, pp. 55–67. [Online]. Available: https://doi.org/10.1007/978-3-030-77961-0_6

[22] DPADMM-G, "Parallel algorithms for structured sparse support vector machines: Application in music genre classification," 2025. [Online]. Available: https://www.sota2.com/research/sota/music-genre-classification-on-fma-small-test
