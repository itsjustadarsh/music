---
title: "Evaluation Protocol Leakage in FMA-Small Music Genre Classification: A Systematic Study"
author: "Adarsh"
geometry: margin=1in
fontsize: 12pt
mainfont: "Times New Roman"
header-includes:
  - \usepackage{titlesec}
  - \titleformat{\section}{\normalfont\fontsize{14}{16}\bfseries}{\thesection.}{1em}{}
  - \titleformat{\subsection}{\normalfont\fontsize{13}{15}\bfseries}{\thesubsection}{1em}{}
  - \setlength{\parskip}{6pt}
  - \setlength{\parindent}{0pt}
---

# 1. Problem Domain Identification and Analysis

## 1.1 Problem Domain

This project falls within the domain of **Music Information Retrieval (MIR)**, specifically the sub-problem of automatic music genre classification. Genre classification is a foundational MIR task with applications in music recommendation systems (Spotify, Apple Music), digital library organisation, content-based music search, and royalty management. The problem is inherently challenging because genre is a subjective, culturally constructed label --- a single track may legitimately belong to multiple genres, and genre boundaries shift over time.

## 1.2 Problem Identification

While surveying the literature on FMA-small music genre classification, we identified a critical but unreported problem: **published results on the same benchmark are not mutually comparable** because authors do not disclose which evaluation protocol they use. The FMA dataset ships with an artist-filtered train/test split (where no artist appears in more than one partition), but using it is optional and disclosing whether it was used is rarer still.

We hypothesised that this undisclosed variable --- the evaluation protocol --- dominates the accuracy axis, making it impossible to determine whether one model genuinely outperforms another or merely benefits from a more permissive splitting strategy.

## 1.3 Problem Analysis

We decomposed the evaluation protocol into three independent choices, each introducing a different type of data leakage:

1. **Artist leakage**: When artists appear in both train and test sets, the model can exploit artist-specific production signatures (mixing style, vocal timbre, preferred instrumentation) rather than learning genre features.

2. **Track leakage**: When different clips from the same 30-second track appear in both train and test sets, the model can recognise the specific recording rather than its genre.

3. **Measurement-level mismatch**: Reporting accuracy at the episode level (3-second clips) versus the track level (averaging predictions across all clips of a track) produces systematically different numbers.

The compounding effect of these choices is not a theoretical concern --- we measure it empirically at +14 to +41 percentage points across 8 architectures, with a mean inflation of +30.5 pp. This range exceeds the entire spread of accuracies that the literature competes over.

## 1.4 Dataset Description

**FMA-small** (Defferrard et al., 2017) contains 8,000 tracks (7,996 readable) balanced across eight genres: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, and Rock. Each track is approximately 30 seconds of Creative Commons-licensed music from the Free Music Archive. The dataset includes metadata on artist, album, and genre, plus a pre-computed artist-filtered train/validation/test split (80/10/10).

For our experiments, each track is segmented into ten non-overlapping 3-second episodes, yielding 75,296 total episodes (60,188 train / 7,544 validation / 7,564 test).

# 2. Review of Existing Systems and Project Feasibility

## 2.1 Literature Survey

We conducted a systematic audit of nine published FMA-small genre classification results, recording what each paper actually states about its evaluation protocol. The findings are summarised below:

| Method | Year | Accuracy | Protocol Stated? |
|--------|------|----------|------------------|
| TS-Resformer (Zhang, 2025) | 2025 | 90.2% | Random 8:1:1 over songs; artists unaddressed |
| CT-GateNet (Ma, 2026) | 2026 | 89.4% | "8:1:1 ratio" --- unit unnamed |
| DenseNet + augmentation (Bian et al., 2019) | 2019 | 68.9% | Nothing |
| Dual residual CNN (Ashraf et al., 2025) | 2025 | 68.5% | Clip-random, stated outright |
| BYOL-A + DNN (Rai & Bhattacharjee, 2025) | 2025 | 64.3% | Nothing |
| PANNs CNN14 (Liu et al., 2022) | 2022 | 60.9% | Partitions described; artist-conditioning unstated |
| Transfer CNN (Park et al., 2018) | 2018 | 56.9% | Nothing for FMA |
| CNN/RNN ensemble (Kostrzewa et al., 2021) | 2021 | 56.4% | "80:10:10 ratio"; mechanism unstated |
| Features + SVM baseline re-run (Park et al., 2018) | 2018 | 54.8% | Artist-filtered by construction |

**Key finding: Zero of nine papers state that they used artist-filtered evaluation.**

## 2.2 Existing Approaches

Existing approaches to FMA-small genre classification fall into three categories:

- **From-scratch spectrogram models**: CNNs (vanilla, DenseNet, ResNet) trained on log-mel spectrograms. These learn audio representations from scratch but require careful data augmentation to avoid overfitting on 8,000 tracks.

- **Self-supervised pretrained models**: MERT (music-specific transformer, pretrained on 20K hours), BYOL-A (general audio, pretrained on AudioSet), PANNs CNN14 (supervised on AudioSet). These provide powerful fixed representations but require only a lightweight probe head, making them less prone to overfitting.

- **Hybrid architectures**: CNN-Transformer hybrids (CT-GateNet), capsule networks, CNN-BiGRU temporal models. These combine multiple processing paradigms but add complexity and parameters.

## 2.3 Gaps in Existing Systems

1. **No protocol reporting convention**: The field lacks a standard for disclosing evaluation protocol, making published results incomparable.
2. **No systematic measurement of leakage**: While source-level leakage is well-understood in medical imaging (patient-level splits) and speaker verification (speaker-level splits), music genre classification has not quantified the effect.
3. **No genre-specific leakage analysis**: It was unknown whether all genres inflate equally or whether some are more "fingerprintable" than others.

## 2.4 Project Feasibility

The project is feasible because:

- **Data availability**: FMA-small is freely available under Creative Commons, with pre-computed splits and metadata.
- **Compute requirements**: All training runs on a single Apple M-series GPU via MPS backend. Probe models train in under 30 minutes; spectrogram models in 2--4 hours.
- **Pretrained models**: MERT, BYOL-A, and PANNs CNN14 checkpoints are publicly available. Feature extraction is a one-time cost (approximately 2 hours for MERT-95M).
- **Reproducibility**: All code, configs, and evaluation scripts are version-controlled. Results are mechanically generated from run artifacts --- no hand-edited numbers.

# 3. Objectives and Methodology of the Proposed Work

## 3.1 Objectives

1. **Quantify protocol leakage**: Measure the accuracy inflation caused by each evaluation protocol choice (artist-filtered vs. random-track vs. clip-random) across multiple architectures.

2. **Audit published literature**: Record what each published FMA-small result states about its evaluation protocol, without inferring protocols from accuracy values.

3. **Build a reproducible benchmark pipeline**: Train 13 architectures under a named, honest protocol (artist-filtered, track-level) to establish placeable reference points.

4. **Genre-specific leakage analysis**: Determine which genres are most affected by protocol leakage and explain why.

5. **Frame the research gap**: Present the undisclosed protocol variable as a reporting gap, not an accusation.

## 3.2 Methodology

### Phase 1: Feature Extraction and Caching

All audio representations are extracted once and cached to disk:

- **Log-mel spectrograms**: 128-bin, 22,050 Hz, producing 128 x 130 feature maps per 3-second episode.
- **MERT-v1-95M**: All 13 layer outputs (12 transformer + 1 convolutional), mean-pooled over time, producing (13, 768) tensors per episode at 24 kHz.
- **MERT-v1-330M**: 25 layers, 1024-dimensional, same extraction procedure.
- **BYOL-A**: Single 2048-dimensional embedding per episode at 16 kHz.
- **PANNs CNN14**: 2048-dimensional penultimate-layer embedding at 32 kHz.

### Phase 2: Model Training

We train 13 architectures across three families:

**From-scratch models**: GenreCNN (4-block CNN, ~260K params), DenseNet121 (ImageNet-pretrained backbone).

**Frozen-backbone probes**: MERT learned weighted-sum probe, flat probe (all-layer concatenation), deep probe (multi-scale aggregation), attention probe (self-attention over layers), BYOL-A + DNN probe, PANNs CNN14 probe. All MERT variants tested with both 95M and 330M backbones.

**Non-neural baseline**: XGBoost on flattened MERT embeddings.

Training configuration: AdamW optimiser, cosine annealing, SpecAugment (for spectrogram models), label smoothing (0.1), mixup (alpha 0.3), early stopping (patience 6--15 epochs).

### Phase 3: Three-Rung Protocol Ladder

Each architecture is trained under three split modes to isolate leakage sources:

1. **Official (artist-filtered)**: FMA's shipped split. No artist in multiple partitions.
2. **Random-track**: Random 80/10/10 over whole tracks. Artists may appear in multiple partitions.
3. **Clip-random**: Random 80/10/10 over individual 3-second episodes. Same recording in train and test.

### Phase 4: Evaluation

All results are track-level: per-episode softmax probabilities are averaged within a track, then argmax selects the genre. This is the only unit reported. Per-genre precision, recall, and F1 are computed using scikit-learn. Confusion matrices are generated for all 52 runs.

### Phase 5: Genre-Specific Leakage Analysis

Per-genre F1 is compared across protocols for all 8 architectures tested on the full ladder. Genres are ranked by mean inflation to identify which are most "fingerprintable."

## 3.3 Results Summary

- **52 training runs** across 13 architectures (36 artist-filtered, 8 random-track, 8 clip-random).
- **Best model**: Uniform ensemble at 58.5% (artist-filtered, track-level), 3.7 pp above the only prior sourceable artist-filtered number (54.8%).
- **Mean inflation**: +30.5 pp from clip-random to artist-filtered, ranging from +14.1 pp (GenreCNN) to +41.0 pp (MERT attention probe).
- **Genre leakage ranking**: Folk (+49.8 pp), Instrumental (+45.2 pp), Experimental (+42.8 pp) inflate most; Hip-Hop (+14.0 pp) and Electronic (+16.3 pp) inflate least.
- **Literature audit**: 0 of 9 published papers state artist-filtered evaluation.

# 4. Relevance of Algorithms and Techniques

## 4.1 Self-Supervised Audio Representations (MERT)

MERT (Music Understanding Model with Large-Scale Self-supervised Training; Li et al., 2023) is directly relevant because it was pretrained specifically on music data using masked language modelling over audio tokens. Its 12 transformer layers capture hierarchical features --- lower layers encode acoustic properties (timbre, pitch), while higher layers capture semantic content (genre, mood). Our layer-wise probing strategy (inspired by Pasad et al., 2021 for speech) reveals that genre-discriminative information is distributed across all depths, justifying the use of learned layer-weighting rather than using only the last layer.

**Relevance**: MERT represents the state of the art in music-specific self-supervised learning. Probing its frozen representations isolates the contribution of the pretrained features from the probe head, making it ideal for controlled protocol leakage measurements.

## 4.2 General-Purpose Audio Models (BYOL-A, PANNs)

**BYOL-A** (Bootstrap Your Own Latent for Audio; Niizumi et al., 2023) uses self-supervised contrastive learning on AudioSet, producing general audio representations without genre-specific pretraining. **PANNs CNN14** (Kong et al., 2020) is supervised on AudioSet's 527 sound event categories.

**Relevance**: Comparing music-specific (MERT) vs. general-purpose (BYOL-A, PANNs) representations tests whether genre classification benefits from music-specific pretraining. Our results show BYOL-A (57.4%) matches PANNs (57.4%) and approaches MERT (58.2%), suggesting that general audio understanding transfers well to genre.

## 4.3 Convolutional Neural Networks (GenreCNN, DenseNet)

GenreCNN is a lightweight 4-block CNN inspired by Choi et al. (2017), processing 128-bin log-mel spectrograms. DenseNet121 (Huang et al., 2017) uses dense connectivity for feature reuse, applied to music by Bian et al. (2019).

**Relevance**: From-scratch CNNs serve as a baseline to measure how much pretrained representations add. GenreCNN (55.1%) vs. MERT flat probe (58.2%) shows that pretraining provides a +3.1 pp advantage. The GenreCNN's lower protocol inflation (+14.1 pp vs. +32.0 pp for MERT flat) reveals that pretrained models are better acoustic fingerprinters --- an important finding for the leakage analysis.

## 4.4 Ensemble Methods

Our uniform ensemble (Kittler et al., 1998) averages the softmax outputs of the best run from each architecture. Learned ensemble weights were also tested but underperformed fixed uniform averaging, consistent with the project's negative result that learned combiners overfit at this data scale (6,400 training tracks).

**Relevance**: The ensemble reaching 58.5% (only +0.3 pp over the best single model) demonstrates diminishing returns from model diversity when features are frozen and data is limited. This result is relevant to practitioners who might otherwise invest in complex fusion strategies.

## 4.5 Gradient-Boosted Trees (XGBoost)

XGBoost (Chen & Guestrin, 2016) on flattened MERT embeddings provides a non-neural baseline that isolates the pretrained features from the probe architecture. At 52.5%, it trails the best neural probe by 5.7 pp, showing that the MLP head contributes meaningful non-linear processing beyond what the pretrained features alone provide.

**Relevance**: A tree-based model on the same features helps separate "how good are the features" from "how good is the probe." The 5.7 pp gap justifies using neural probes over simpler classifiers.

## 4.6 Data Augmentation (SpecAugment)

SpecAugment (Park et al., 2019), originally designed for speech recognition, masks random frequency bands and time steps in spectrograms. We apply it to spectrogram-based models with 2 frequency masks (up to 16 bins) and 2 time masks (up to 20 frames).

**Relevance**: SpecAugment contributed to the GenreCNN's improvement from 47.1% (baseline) to 55.1% (improved), an +8.0 pp gain. This demonstrates that speech augmentation techniques transfer effectively to music genre classification.

## 4.7 Protocol Measurement Methodology

The three-rung ladder design is inspired by leakage measurement practices in medical imaging (patient-level splits), speaker verification (speaker-level splits), and satellite imagery (spatial hold-outs). By holding the model fixed and varying only the split, we isolate the protocol variable with no confounds from architecture differences.

**Relevance**: This methodology is the project's primary contribution. By quantifying the inflation at +14 to +41 pp, we establish that the undisclosed protocol variable is larger than the accuracy differences that the literature competes over, rendering published rankings meaningless without protocol disclosure.
