---
title: "Evaluation Protocol Leakage in FMA-Small Music Genre Classification: A Systematic Study"
author:
  - "Adarsh"
abstract: |
  Music genre classification on FMA-small is a widely used benchmark,
  yet published results rarely disclose the evaluation protocol that
  produced them. We show that three independent protocol choices ---
  whether the train/test boundary falls between artists, between tracks,
  or between 3-second clips --- move a single model's accuracy by 14 to
  41 percentage points. Across 52 training runs spanning 13 architectures,
  the mean inflation from the most permissive protocol (clip-random) over
  the honest one (artist-filtered) is +30.5 pp. A literature audit of
  nine published FMA-small results finds that zero state artist-filtered
  evaluation in prose. We frame this as a research gap in reporting
  conventions: the field benchmarks on a variable it does not name, and
  the resulting numbers are not mutually comparable. Our best model, a
  uniform ensemble of per-architecture bests, reaches 58.5\% under the
  artist-filtered, track-level protocol --- 3.7 pp above the only prior
  sourceable artist-filtered number (54.8\%).
---

# Introduction

Music genre classification is one of the longest-standing tasks in music information retrieval (MIR). The Free Music Archive small subset (FMA-small) [@defferrard2017fma] --- 8,000 tracks balanced across eight genres --- has become a standard benchmark for comparing architectures, from convolutional networks to self-supervised transformer probes.

Yet a reader attempting to rank published results on FMA-small faces a quiet problem: most papers do not say how they split the data. FMA ships an artist-filtered split in which no artist appears in more than one partition, but using it is optional and disclosing whether it was used is rarer still. A paper that reports "random 8:1:1" without naming the unit --- songs or 3-second clips --- leaves open a question whose answer, as we measure here, is worth tens of accuracy points.

This is not an accusation. It is a measurement. The field has no reporting convention for a variable that dominates the accuracy axis, and so published numbers are not mutually comparable even in principle. We document the size of the effect, audit what the literature actually states, and report our own results under a named protocol so that future work has at least one placeable reference point.

**Contributions.** (1) A three-rung protocol ladder quantifying the accuracy cost of each leakage type across 8 architectures. (2) A literature audit of 9 published FMA-small results, recording what each paper states about its split. (3) A reproducible training pipeline covering 13 architectures and 52 evaluated runs, all under the artist-filtered, track-level protocol.

# Related Work

## A. Published FMA-small Results

We traced every FMA-small genre-classification result we could find to its primary source, recording what each paper says about its evaluation protocol --- not what its accuracy implies. The distinction matters: inferring a protocol from an accuracy would assume the conclusion this study sets out to test.

Table I summarises the registry. Of the nine external results, zero state artist-filtered evaluation in prose. One states clip-random splitting outright [@ashraf2025music]. One states a track-level random split [@zhang2025tsresformer]. One describes its partitions without stating artist-conditioning [@liu2022openset]. Two give a ratio with no unit [@ma2026ctgatenet; @kostrzewa2021music]. Four say nothing at all [@bian2019audio; @rai2025evaluating; @park2018representation].

```{=latex}
\begin{table*}[t]
\centering
\caption{Registry of Published FMA-small Results with Stated Evaluation Protocols}
\label{tab:literature}
\footnotesize
\begin{tabular}{clccl}
\toprule
\textbf{\#} & \textbf{Method} & \textbf{Year} & \textbf{Acc.\ (\%)} & \textbf{What the paper states about its split} \\
\midrule
1 & TS-Resformer \cite{zhang2025tsresformer} & 2025 & 90.2 & Random 8:1:1 over \emph{songs}; artists unaddressed \\
2 & CT-GateNet \cite{ma2026ctgatenet} & 2026 & 89.4 & ``8:1:1 ratio'' --- unit unnamed \\
3 & DenseNet + augmentation \cite{bian2019audio} & 2019 & 68.9 & Nothing \\
4 & Dual residual CNN \cite{ashraf2025music} & 2025 & 68.5 & Clip-random, stated outright \\
5 & BYOL-A + DNN \cite{rai2025evaluating} & 2025 & 64.3 & Nothing \\
6 & PANNs CNN14 \cite{liu2022openset} & 2022 & 60.9* & Partitions described; artist-conditioning unstated \\
7 & Transfer CNN \cite{park2018representation} & 2018 & 56.9 & Nothing for FMA \\
8 & CNN/RNN ensemble \cite{kostrzewa2021music} & 2021 & 56.4 & ``80:10:10 ratio''; mechanism unstated \\
9 & Features + SVM (FMA baseline re-run) \cite{park2018representation} & 2018 & 54.8 & Artist-filtered by construction, not by claim \\
\midrule
  & \textbf{Ours --- Ensemble} & 2026 & \textbf{58.5} & \textbf{Official artist-filtered, track-level} \\
\bottomrule
\multicolumn{5}{l}{\footnotesize *60.9\% is the closed-set sub-metric of an open-set experiment; not a standalone 8-way result.}
\end{tabular}
\end{table*}
```

One widely cited baseline --- "Defferrard et al., 58.0\%" --- does not exist. The FMA paper's only accuracy table covers the 16-genre medium subset; 58 is the MLP cell there. The 54.8\% attributed to FMA's authors is actually Park et al.'s re-run of FMA's released baseline code [@park2018representation], artist-filtered by code inheritance rather than authorial claim.

## B. Precedent in Other Fields

Source-level leakage is well understood in adjacent domains. In medical imaging, patient-level splits prevent same-patient scans from appearing in both train and test. In speaker verification, speaker-level splits are standard. In satellite imagery, spatial hold-outs avoid adjacent-tile leakage. Music genre classification has all three forms at once --- the same recording (clip-level), the same performance (track-level), and the same production signature (artist-level) --- yet lacks a reporting convention for which boundary is enforced.

# Methodology

## A. Dataset

FMA-small [@defferrard2017fma] contains 8,000 tracks (7,996 readable after excluding 4 corrupt files) balanced across eight genres: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, and Rock. Each track is approximately 30 seconds long. Following standard practice in the FMA/GTZAN literature, each track is segmented into ten non-overlapping 3-second episodes, yielding 75,296 total episodes (60,188 train / 7,544 validation / 7,564 test).

## B. The Three-Rung Protocol Ladder

We define three split modes to isolate each source of leakage:

1. **Official (artist-filtered).** FMA's shipped split. No artist appears in more than one partition. This is the only protocol used for headline results.

2. **Random-track.** Random 80/10/10 over whole tracks. Artists may appear in multiple partitions, but clips from the same track stay together. This rung exists because most papers state "random 8:1:1" without naming the unit.

3. **Clip-random.** Random 80/10/10 over individual 3-second episodes. The same recording appears in both train and test. This is the most permissive protocol and produces the highest reported accuracies.

## C. Track-Level Evaluation

All accuracies are track-level: per-episode softmax probabilities are averaged within a track, then argmax selects the genre. This is stricter than episode-level scoring and is the only unit we report, following the principle that a 30-second track is the natural prediction target.

## D. Feature Extraction Pipeline

All representations are cached once and reused across experiments, ensuring that backbone computation is not a variable between runs.

**Log-mel spectrograms.** For from-scratch CNN models, each 3-second episode is converted to a 128-bin log-mel spectrogram at 22,050 Hz (n\_fft=2048, hop\_length=512), yielding a 128 x 130 feature map. These are stored as NumPy arrays, one per episode.

**MERT embeddings.** MERT-v1-95M [@li2023mert] is a 12-layer transformer pretrained on 20,000 hours of music via self-supervised learning. We extract all 13 layer outputs (12 transformer + 1 convolutional projection), mean-pooled over the time dimension, producing a (13, 768) tensor per episode. Crucially, we cache *all* layers rather than just the last, since genre-discriminative information is distributed across depths; the probe learns its own layer weighting. MERT-v1-330M (25 layers, 1024-dim) is extracted analogously. Audio is decoded at MERT's native 24 kHz via librosa. Extraction takes approximately 2 hours for the 95M variant (4 workers).

**BYOL-A embeddings.** BYOL-A [@niizumi2023byola] is a general-purpose audio representation pretrained via self-supervised Bootstrap Your Own Latent on AudioSet. We extract a single 2048-dimensional embedding per episode at 16 kHz. The model is loaded from a local clone (not available on PyPI).

**PANNs CNN14 embeddings.** PANNs CNN14 [@kong2020panns] is an AudioSet-supervised convolutional network. We extract 2048-dimensional penultimate-layer embeddings per episode at 32 kHz from the pretrained checkpoint (mAP=0.431 on AudioSet).

## E. Model Families

We train 13 architectures spanning three families:

**From-scratch spectrogram models** consume 128-bin log-mel spectrograms and train all parameters from scratch: GenreCNN (4-block CNN, ~260K params) [@choi2017transfer], DenseNet121 (ImageNet-pretrained, adapted to single-channel input) [@huang2017densenet; @bian2019audio].

**Frozen-backbone probes** train only a lightweight head on cached embeddings from pretrained models:

- *MERT learned weighted-sum probe* [@pasad2021layer]: softmax-weighted sum over all layers, followed by a two-layer MLP. Reports layer weights as evidence of which transformer depths carry genre information.
- *MERT flat probe*: concatenates all 13 layers into a 9,984-dim vector, followed by a two-layer MLP with LayerNorm. Higher dropout (0.5) manages the larger feature space.
- *MERT deep probe* [@yang2021superb]: groups layers into multi-scale aggregations with residual blocks, capturing both low-level acoustic and high-level semantic features.
- *MERT attention probe* [@safari2020self]: multi-head self-attention over the 13 layers with learnable positional embeddings and a cross-attention pooling query.
- *BYOL-A + DNN probe* [@niizumi2023byola]: LayerNorm followed by a variable-depth MLP on the 2048-dim BYOL-A embedding.
- *PANNs CNN14 probe* [@kong2020panns]: same architecture as BYOL-A probe, on PANNs 2048-dim embeddings.

All MERT probe variants are also trained with the 330M backbone where applicable.

**Non-neural baseline.** XGBoost [@chen2016xgboost] on flattened MERT embeddings, providing a reference that isolates the contribution of the learned probe head from the pretrained features themselves.

## F. Training Configuration

All neural models use AdamW optimisation with cosine annealing. Spectrogram models apply SpecAugment [@park2019specaugment] (2 frequency masks up to 16 bins, 2 time masks up to 20 frames). Label smoothing (0.1) and mixup augmentation (alpha 0.3) are used for most runs. Early stopping with patience 6--15 epochs prevents overfitting. All training is sequential on a single Apple M-series GPU (MPS backend); concurrent jobs cause severe slowdowns (350s/epoch to 1800s/epoch for DenseNet).

Loss functions include cross-entropy with label smoothing (default), focal loss (for class-imbalance experiments), and margin loss (CapsNet-specific, from [@sabour2017dynamic]).

# Experiments

We conducted 52 evaluated training runs: 36 under the artist-filtered split, 8 under random-track, and 8 under clip-random. Every result is mechanically generated from run artifacts by dedicated scripts; no number in the results tables is hand-edited. Each run produces an `eval\_metrics.json` with track-level and clip-level accuracy, per-genre precision/recall/F1, and a confusion matrix visualisation.

# Results

## A. Best Model Per Architecture

Table II presents the best run per architecture (13 unique architectures), ranked by accuracy under the artist-filtered, track-level protocol.

```{=latex}
\begin{table}[t]
\centering
\caption{Best Run Per Architecture (Artist-Filtered, Track-Level)}
\label{tab:best}
\footnotesize
\begin{tabular}{clcc}
\toprule
\textbf{\#} & \textbf{Architecture} & \textbf{Acc.} & \textbf{F1} \\
\midrule
1 & Ensemble (uniform) \cite{kittler1998combining} & 58.5 & 58.2 \\
2 & MERT-95M flat probe \cite{li2023mert} & 58.2 & 57.8 \\
3 & BYOL-A + DNN probe \cite{niizumi2023byola} & 57.4 & 57.0 \\
4 & PANNs CNN14 probe \cite{kong2020panns} & 57.4 & 56.6 \\
5 & MERT-330M probe \cite{li2023mert} & 56.6 & 56.5 \\
6 & MERT-330M flat probe \cite{li2023mert} & 56.5 & 55.9 \\
7 & MERT-95M probe \cite{pasad2021layer} & 56.4 & 55.7 \\
8 & MERT-95M deep probe \cite{yang2021superb} & 56.0 & 55.9 \\
9 & MERT-95M attn probe \cite{safari2020self} & 55.2 & 54.4 \\
10 & GenreCNN (log-mel) \cite{choi2017transfer} & 55.1 & 54.2 \\
11 & MERT-330M deep probe \cite{yang2021superb} & 54.9 & 55.5 \\
12 & DenseNet121 \cite{huang2017densenet} & 53.9 & 52.2 \\
13 & XGBoost on MERT \cite{chen2016xgboost} & 52.5 & 51.4 \\
\bottomrule
\end{tabular}
\end{table}
```

The best single model is the MERT-95M flat probe (58.2\%), which concatenates all 13 transformer layers into a 9,984-dimensional feature vector and trains a two-layer MLP. A uniform-weight ensemble of per-architecture bests reaches 58.5\%. Notably, MERT-330M does not outperform MERT-95M, and learned ensemble weights lose to fixed uniform averaging --- both symptoms of overfitting at this data scale (6,400 training tracks).

## B. Complete Run Inventory

Table III presents all 36 artist-filtered runs, including hyperparameter variants and seed ablations. This complete inventory demonstrates the sensitivity of each architecture to training configuration and provides the full evidence base behind the best-per-architecture summary above.

```{=latex}
\begin{table*}[t]
\centering
\caption{Complete Inventory: All 36 Artist-Filtered Runs (Track-Level Accuracy and Macro-F1, \%)}
\label{tab:allruns}
\scriptsize
\begin{tabular}{rllccccccl}
\toprule
\textbf{\#} & \textbf{Run} & \textbf{Architecture} & \textbf{Acc.} & \textbf{F1} & \textbf{LR} & \textbf{BS} & \textbf{Ep.} & \textbf{Drop.} & \textbf{Ref.} \\
\midrule
1 & ensemble\_uniform & Ensemble (uniform avg) & 58.5 & 58.2 & --- & --- & --- & --- & \cite{kittler1998combining} \\
2 & ensemble\_weighted & Ensemble (weighted avg) & 58.2 & 58.1 & --- & --- & --- & --- & \cite{kittler1998combining} \\
3 & mert\_flat\_probe\_v1 & MERT-95M flat probe & 58.2 & 57.8 & 3e-4 & 128 & 50 & 0.5 & \cite{li2023mert} \\
4 & mert\_flat\_probe\_v2 & MERT-95M flat probe & 57.8 & 57.3 & 1e-4 & 64 & 50 & 0.6 & \cite{li2023mert} \\
5 & byola\_probe\_v1 & BYOL-A + DNN probe & 57.4 & 57.0 & 3e-4 & 128 & 50 & 0.4 & \cite{niizumi2023byola} \\
6 & mert\_flat\_diverse\_s3 & MERT-95M flat probe & 57.4 & 56.5 & 2e-4 & 128 & 50 & 0.6 & \cite{li2023mert} \\
7 & panns\_probe\_v1 & PANNs CNN14 probe & 57.4 & 56.6 & 3e-4 & 128 & 50 & 0.4 & \cite{kong2020panns} \\
8 & mert\_flat\_diverse\_s4 & MERT-95M flat probe & 56.6 & 56.1 & 3e-4 & 128 & 50 & 0.5 & \cite{li2023mert} \\
9 & mert\_large\_probe\_v1 & MERT-330M probe & 56.6 & 56.5 & 5e-4 & 128 & 50 & 0.4 & \cite{li2023mert} \\
10 & mert\_flat\_diverse\_s1 & MERT-95M flat probe & 56.5 & 55.8 & 3e-4 & 128 & 50 & 0.5 & \cite{li2023mert} \\
11 & mert\_large\_flat\_v1 & MERT-330M flat probe & 56.5 & 55.9 & 1e-4 & 64 & 50 & 0.6 & \cite{li2023mert} \\
12 & mert\_probe\_tuned & MERT-95M probe (learned wt.) & 56.4 & 55.7 & 2e-4 & 64 & 60 & 0.5 & \cite{pasad2021layer} \\
13 & mert\_large\_probe\_v3 & MERT-330M probe & 56.2 & 56.2 & 1e-4 & 128 & 50 & 0.3 & \cite{li2023mert} \\
14 & mert\_probe\_wide & MERT-95M probe (learned wt.) & 56.1 & 55.3 & 5e-4 & 128 & 40 & 0.4 & \cite{pasad2021layer} \\
15 & mert\_deep\_probe\_v2 & MERT-95M deep probe & 56.0 & 55.9 & 1e-4 & 64 & 60 & 0.5 & \cite{yang2021superb} \\
16 & mert\_flat\_diverse\_s6 & MERT-95M flat probe & 56.0 & 55.4 & 2.5e-4 & 128 & 50 & 0.55 & \cite{li2023mert} \\
17 & mert\_probe\_baseline & MERT-95M probe (learned wt.) & 55.9 & 54.9 & 1e-3 & 256 & 40 & 0.3 & \cite{pasad2021layer} \\
18 & mert\_flat\_diverse\_s5 & MERT-95M flat probe & 55.8 & 55.6 & 4e-4 & 128 & 50 & 0.4 & \cite{li2023mert} \\
19 & mert\_deep\_probe\_v1 & MERT-95M deep probe & 55.5 & 55.0 & 3e-4 & 128 & 60 & 0.4 & \cite{yang2021superb} \\
20 & mert\_attn\_probe\_v2 & MERT-95M attention probe & 55.2 & 54.4 & 5e-4 & 128 & 60 & 0.4 & \cite{safari2020self} \\
21 & mert\_probe\_seed2 & MERT-95M probe (seed 2) & 55.2 & 55.0 & 5e-4 & 128 & 40 & 0.4 & \cite{pasad2021layer} \\
22 & cnn\_improved & GenreCNN (log-mel) & 55.1 & 54.2 & 3e-4 & 32 & 60 & --- & \cite{choi2017transfer} \\
23 & mert\_large\_probe\_v2 & MERT-330M probe & 55.0 & 55.0 & 3e-4 & 64 & 50 & 0.5 & \cite{li2023mert} \\
24 & mert\_probe\_deep & MERT-95M probe (deep head) & 55.0 & 54.4 & 1e-4 & 256 & 80 & 0.5 & \cite{pasad2021layer} \\
25 & mert\_large\_deep\_v1 & MERT-330M deep probe & 54.9 & 55.5 & 3e-4 & 64 & 50 & 0.4 & \cite{yang2021superb} \\
26 & mert\_attn\_probe\_v1 & MERT-95M attention probe & 54.6 & 53.6 & 3e-4 & 128 & 60 & 0.3 & \cite{safari2020self} \\
27 & mert\_probe\_seed3 & MERT-95M probe (seed 3) & 54.5 & 54.4 & 5e-4 & 128 & 40 & 0.4 & \cite{pasad2021layer} \\
28 & mert\_flat\_diverse\_s2 & MERT-95M flat probe & 54.4 & 53.7 & 3e-4 & 128 & 50 & 0.5 & \cite{li2023mert} \\
29 & densenet\_v1 & DenseNet121 & 53.9 & 52.2 & 3e-4 & 32 & 60 & 0.3 & \cite{huang2017densenet} \\
30 & mert\_attn\_probe\_v3 & MERT-95M attention probe & 53.5 & 52.8 & 1e-4 & 64 & 60 & 0.2 & \cite{safari2020self} \\
31 & mert\_probe\_seed1 & MERT-95M probe (seed 1) & 53.5 & 52.9 & 5e-4 & 128 & 40 & 0.4 & \cite{pasad2021layer} \\
32 & cnn\_tuned\_lr & GenreCNN (log-mel) & 52.9 & 52.2 & 3e-4 & 32 & 35 & --- & \cite{choi2017transfer} \\
33 & mert\_probe\_lastlayer & MERT-95M probe (last layer) & 52.8 & 51.8 & 1e-3 & 256 & 40 & 0.3 & \cite{pasad2021layer} \\
34 & xgboost\_mert & XGBoost on MERT & 52.5 & 51.4 & --- & 256 & --- & --- & \cite{chen2016xgboost} \\
35 & cnn\_augmented & GenreCNN (log-mel) & 51.4 & 50.4 & 1e-3 & 64 & 30 & --- & \cite{choi2017transfer} \\
36 & cnn\_baseline & GenreCNN (log-mel) & 47.1 & 45.0 & 1e-3 & 64 & 25 & --- & \cite{choi2017transfer} \\
\bottomrule
\end{tabular}
\end{table*}
```

Key observations from the full inventory: (a) The MERT-95M flat probe spans 54.4--58.2\% across 7 runs differing only in seed and regularisation, a 3.8 pp range from hyperparameters alone. (b) Seed ablations for the MERT-95M probe (seeds 1--3, 42) span 53.5--56.4\%, showing that a 3 pp difference between two papers could be seed noise. (c) The CNN baseline improves from 47.1\% to 55.1\% (+8.0 pp) through augmentation and hyperparameter tuning alone, without any architectural change.

## C. Diagnostic Runs: Protocol Leakage

Tables IV and V present the 16 diagnostic runs under relaxed protocols, run solely to quantify inflation. No number in these tables may be quoted as a result.

```{=latex}
\begin{table}[t]
\centering
\caption{Random-Track Diagnostic Runs (\textbf{not results} --- artist leakage only)}
\label{tab:randomtrack}
\footnotesize
\begin{tabular}{lcc}
\toprule
\textbf{Architecture} & \textbf{Acc.} & \textbf{F1} \\
\midrule
MERT-330M flat probe \cite{li2023mert} & 71.2 & 71.5 \\
MERT-330M probe \cite{li2023mert} & 70.9 & 71.2 \\
MERT-95M flat probe \cite{li2023mert} & 70.2 & 70.6 \\
BYOL-A + DNN \cite{niizumi2023byola} & 69.9 & 70.3 \\
MERT-95M attn probe \cite{safari2020self} & 69.8 & 70.1 \\
MERT-95M probe \cite{pasad2021layer} & 69.2 & 69.6 \\
MERT-95M deep probe \cite{yang2021superb} & 68.5 & 69.0 \\
PANNs CNN14 \cite{kong2020panns} & 67.2 & 67.2 \\
\bottomrule
\end{tabular}
\end{table}
```

```{=latex}
\begin{table}[t]
\centering
\caption{Clip-Random Diagnostic Runs (\textbf{not results} --- full leakage)}
\label{tab:cliprandom}
\footnotesize
\begin{tabular}{lcc}
\toprule
\textbf{Architecture} & \textbf{Acc.} & \textbf{F1} \\
\midrule
MERT-330M probe \cite{li2023mert} & 96.0 & 96.0 \\
MERT-95M attn probe \cite{safari2020self} & 95.7 & 95.7 \\
MERT-95M probe \cite{pasad2021layer} & 93.6 & 93.6 \\
MERT-95M deep probe \cite{yang2021superb} & 92.2 & 92.2 \\
MERT-95M flat probe \cite{li2023mert} & 90.2 & 90.2 \\
BYOL-A + DNN \cite{niizumi2023byola} & 84.5 & 84.5 \\
PANNs CNN14 \cite{kong2020panns} & 73.4 & 73.0 \\
GenreCNN (log-mel) \cite{choi2017transfer} & 69.2 & 68.8 \\
\bottomrule
\end{tabular}
\end{table}
```

## D. Protocol Leakage Ladder

Table VI combines the three protocols for the 8 architectures tested on all rungs. This is the project's core measurement.

```{=latex}
\begin{table*}[t]
\centering
\caption{Protocol Leakage Ladder: Identical Architectures Under Three Split Protocols (Track-Level Accuracy, \%)}
\label{tab:leakage}
\footnotesize
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Artist-filtered} & \textbf{Random-track} & \textbf{Clip-random} & \textbf{Inflation (pp)} \\
\midrule
MERT Probe (weighted-sum) \cite{pasad2021layer} & 56.1 & 69.2 & 93.6 & +37.5 \\
MERT Flat Probe (13-layer) \cite{li2023mert} & 58.2 & 70.2 & 90.2 & +32.0 \\
MERT Deep Probe (multi-scale) \cite{yang2021superb} & 55.5 & 68.5 & 92.2 & +36.7 \\
MERT Attention Probe \cite{safari2020self} & 54.6 & 69.8 & 95.7 & +41.0 \\
MERT-330M Probe \cite{li2023mert} & 56.6 & 70.9 & 96.0 & +39.4 \\
CNN (log-mel spectrogram) \cite{choi2017transfer} & 55.1 & --- & 69.2 & +14.1 \\
BYOL-A + DNN \cite{niizumi2023byola} & 57.4 & 69.9 & 84.5 & +27.2 \\
PANNs CNN14 \cite{kong2020panns} & 57.4 & 67.2 & 73.4 & +16.0 \\
\midrule
\textbf{Mean} & \textbf{56.4} & \textbf{69.4} & \textbf{86.9} & \textbf{+30.5} \\
\bottomrule
\end{tabular}
\end{table*}
```

Decomposed over the seven architectures with all three rungs: dropping the artist filter alone accounts for +12.8 pp, and leaking clips on top of that adds a further +20.0 pp. The effect is systematic and architecture-independent, spanning from-scratch CNNs (+14.1 pp) to attention-based MERT probes (+41.0 pp).

The MERT attention probe illustrates the extreme: 54.6\% under honest evaluation, 95.7\% under clip-random. The attention mechanism excels at acoustic fingerprinting --- recognising *which recording* it has heard --- which is rewarded under clip-random but irrelevant under artist-filtered evaluation.

## E. Against Published Literature

Our best artist-filtered model reaches 58.5\%. The only published number carrying an artist-filtered guarantee is 54.8\% (Park et al.'s re-run of FMA's baseline code [@park2018representation]), a margin of +3.7 pp. We deliberately do not report a "beats N of M" count: ranking against numbers whose protocol is unknown would treat incomparable quantities as comparable, which is the failure this project documents.

For two architectures we reimplemented from published work, the published numbers land between our artist-filtered and random-track results:

- **BYOL-A + DNN** [@rai2025evaluating]: published 64.3\%, ours 57.4\% (artist-filtered) / 69.9\% (random-track)
- **PANNs CNN14** [@liu2022openset]: published 60.9\%, ours 57.4\% (artist-filtered) / 67.2\% (random-track)

This is consistent with several explanations, including protocol differences and probe-head fidelity. The architecture-fidelity confound is as large as the effect, so we do not adjudicate.

# Discussion

## A. The Research Gap

The finding is not that the field is cheating --- it is that the field has no reporting convention for a variable we measure at tens of points. When a paper does not state its protocol:

1. The number becomes *unplaceable*. Not "probably leaked" --- unplaceable.
2. It is incomparable in both directions. An undisclosed 64.3\% is neither above nor below an artist-filtered 58.5\% in any meaningful sense.
3. The honest error bar is the full ladder span, which for most architectures exceeds the entire range of accuracies the literature competes over.

We propose that future FMA-small publications state, at minimum: (a) whether the artist-filtered split was used, (b) whether evaluation is track-level or episode-level, and (c) the exact split mechanism if not FMA's shipped metadata.

## B. A Self-Inflicted Illustration

An earlier revision of this project compared our episode-level accuracy against published track-level numbers and concluded our reimplementations "reproduced within 1.1 pp." The claim was wrong: episode-level figures run 5--7 pp below track-level ones, and that gap did all the work. This is a better illustration of the thesis than any borrowed example --- an unstated measurement choice produced a confident false conclusion, against ourselves.

## C. Negative Results

Two lines of investigation were pursued and closed, recorded here because a project about honest reporting that silently deletes its failures would be self-refuting.

**Epoch-Scheduled Conditional Fusion (ESCF).** A learned fusion of CNN and MERT backbones using per-episode sigmoid gates, per-track attention, and a curriculum interpolating from fixed weights toward learned ones. Over 5 seeds and 4 curriculum shapes (cosine, linear, step, learned supervisor), every ESCF variant lost to a fixed 50/50 uniform ensemble (58.6\% vs. best ESCF at 57.9\%). The McNemar test was significant *against* the learned method (p = 0.011).

**Learned attention over episodes.** Pooling a track's ten episodes with learned attention is consistently worse than averaging: CNN attention (54.7\%) < CNN mean-pool (54.8\%), MERT attention (56.4\%) < MERT mean-pool (57.1\%). The weakest MERT probe in our inventory is the attention probe.

**The lesson.** Anything learned on top of frozen features overfits at this data scale (6,400 training tracks, 800 validation). Accuracy gains must come from better features or better-conditioned inputs, not from new learned combiners. Where this project ensembles, it uses fixed uniform weights for this reason.

## D. Limitations

This study covers one dataset (FMA-small), one segmentation length (3 seconds), and reimplements rather than runs original code. The 3-second window is a ceiling on what any model here can see; genre is partly a long-timescale property (form, instrumentation turnover, section changes). The reimplementation fidelity confound means our protocol-ladder measurements are exact only for our own models, though the direction and magnitude are architecture-independent. Additionally, FMA-small's 8 coarse genres include tracks that are genuinely multi-genre; the single `genre\_top` label caps what any classifier can achieve.

# Ongoing and Future Work

## A. Models Remaining to Train

Four published architectures are fully implemented and configured but have not yet been trained, pending compute time:

1. **CapsNet** [@sabour2017dynamic]: dynamic routing capsule network for spectrograms.
2. **CNN-BiGRU** [@kostrzewa2021music]: CNN feature extraction with bidirectional GRU temporal modelling.
3. **CNN-Transformer** [@ma2026ctgatenet]: CNN backbone with transformer encoder and CLS-token pooling (CT-GateNet reimplementation).
4. **Dual residual CNN** [@ashraf2025music]: dual-path CNN with max+avg pooling concatenation.

Each architecture has configs for all three protocol rungs (artist-filtered, random-track, clip-random), totalling 12 pending runs. Completing these will extend the leakage ladder from 8 to 12 architectures and strengthen the claim of architecture-independence.

Additionally, a **ResNet with squeeze-excitation attention** (~4M params) is implemented but never trained, providing a deeper spectrogram baseline than GenreCNN.

## B. Potential Novel Contributions Beyond Protocol Audit

**1. End-to-end MERT fine-tuning.** All current MERT results freeze the backbone and train only a probe head. A `MertFinetune` model is implemented that selectively unfreezes the last N transformer layers with gradient checkpointing, processing raw 24 kHz audio. This has never been evaluated on FMA-small under artist-filtered conditions. The question: does adapting MERT's representations to genre beat frozen probing, given that fine-tuning risks overfitting on 6,400 tracks?

**2. Context window ablation.** Every embedding in this project is computed from 3 seconds of audio. A 10-second extraction grid is implemented and ready to run, training the flat probe with every other setting held fixed. This single-variable ablation isolates whether the 3-second window is a ceiling on achievable accuracy. If 10s yields a measurable gain, the implication is that the entire FMA-small literature has been evaluating under a suboptimal but unreported context length.

**3. Cross-dataset protocol audit.** The protocol inflation phenomenon should transfer to other music benchmarks (GTZAN, MagnaTagATune). Measuring the same ladder on a second dataset would elevate the finding from dataset-specific to field-wide.

**4. Genre-specific leakage analysis.** Per-genre confusion matrices are already generated for all 52 runs. A systematic analysis of which genres are most inflated by each leakage type (e.g., does Hip-Hop inflate more than Experimental?) could reveal whether some genres are more "fingerprintable" than others.

**5. Seed stability measurement.** Our seed ablations (seeds 1--3, 42 for MERT probe) show a 3 pp range. Formalising this as a confidence interval for each architecture would provide the first published seed-stability measurement for FMA-small, establishing a minimum detectable effect size for future comparisons.

# Conclusion

We measured the cost of three evaluation protocol choices on FMA-small music genre classification across 13 architectures and 52 training runs. The effect is large (+30.5 pp mean inflation), systematic (architecture-independent), and undisclosed (0 of 9 published papers state artist-filtered evaluation). Our best model reaches 58.5\% under the artist-filtered, track-level protocol, establishing a placeable reference point for future comparison. The gap is not in the models --- it is in the reporting convention.

Four additional architectures and two novel experimental directions (MERT fine-tuning and context window ablation) are implemented and ready to run. We call on future work to state its split protocol explicitly, so that published numbers can be placed on a common axis.

