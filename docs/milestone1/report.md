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
bibliography: references.bib
csl: ieee.csl
---

# I. Introduction

Music genre classification is one of the longest-standing tasks in music information retrieval (MIR). The Free Music Archive small subset (FMA-small) [@defferrard2017fma] --- 8,000 tracks balanced across eight genres --- has become a standard benchmark for comparing architectures, from convolutional networks to self-supervised transformer probes.

Yet a reader attempting to rank published results on FMA-small faces a quiet problem: most papers do not say how they split the data. FMA ships an artist-filtered split in which no artist appears in more than one partition, but using it is optional and disclosing whether it was used is rarer still. A paper that reports "random 8:1:1" without naming the unit --- songs or 3-second clips --- leaves open a question whose answer, as we measure here, is worth tens of accuracy points.

This is not an accusation. It is a measurement. The field has no reporting convention for a variable that dominates the accuracy axis, and so published numbers are not mutually comparable even in principle. We document the size of the effect, audit what the literature actually states, and report our own results under a named protocol so that future work has at least one placeable reference point.

**Contributions.** (1) A three-rung protocol ladder quantifying the accuracy cost of each leakage type across 8 architectures. (2) A literature audit of 9 published FMA-small results, recording what each paper states about its split. (3) A reproducible training pipeline covering 13 architectures and 52 evaluated runs, all under the artist-filtered, track-level protocol.

# II. Related Work

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

One widely cited baseline --- "Defferrard et al., 58.0%" --- does not exist. The FMA paper's only accuracy table covers the 16-genre medium subset; 58 is the MLP cell there. The 54.8% attributed to FMA's authors is actually Park et al.'s re-run of FMA's released baseline code [@park2018representation], artist-filtered by code inheritance rather than authorial claim.

## B. Precedent in Other Fields

Source-level leakage is well understood in adjacent domains. In medical imaging, patient-level splits prevent same-patient scans from appearing in both train and test. In speaker verification, speaker-level splits are standard. In satellite imagery, spatial hold-outs avoid adjacent-tile leakage. Music genre classification has all three forms at once --- the same recording (clip-level), the same performance (track-level), and the same production signature (artist-level) --- yet lacks a reporting convention for which boundary is enforced.

# III. Methodology

## A. Dataset

FMA-small [@defferrard2017fma] contains 8,000 tracks (7,996 readable after excluding 4 corrupt files) balanced across eight genres: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, and Rock. Each track is approximately 30 seconds long. Following standard practice in the FMA/GTZAN literature, each track is segmented into ten non-overlapping 3-second episodes, yielding 75,296 total episodes (60,188 train / 7,544 validation / 7,564 test).

## B. The Three-Rung Protocol Ladder

We define three split modes to isolate each source of leakage:

1. **Official (artist-filtered).** FMA's shipped split. No artist appears in more than one partition. This is the only protocol used for headline results.

2. **Random-track.** Random 80/10/10 over whole tracks. Artists may appear in multiple partitions, but clips from the same track stay together. This rung exists because most papers state "random 8:1:1" without naming the unit.

3. **Clip-random.** Random 80/10/10 over individual 3-second episodes. The same recording appears in both train and test. This is the most permissive protocol and produces the highest reported accuracies.

## C. Track-Level Evaluation

All accuracies are track-level: per-episode softmax probabilities are averaged within a track, then argmax selects the genre. This is stricter than episode-level scoring and is the only unit we report, following the principle that a 30-second track is the natural prediction target.

## D. Model Families

We train 13 architectures spanning three families:

**From-scratch spectrogram models** consume 128-bin log-mel spectrograms and train all parameters from scratch: GenreCNN (4-block CNN), DenseNet121 [@huang2017densenet; @bian2019audio], and others.

**Frozen-backbone probes** train only a lightweight head on cached embeddings from pretrained models: MERT-v1-95M and MERT-v1-330M [@li2023mert] (multiple probe architectures), BYOL-A [@niizumi2023byola], and PANNs CNN14 [@kong2020panns].

**Non-neural baseline.** XGBoost [@chen2016xgboost] on flattened MERT embeddings.

# IV. Experiments

We conducted 52 evaluated training runs: 36 under the artist-filtered split, 8 under random-track, and 8 under clip-random. All runs use AdamW optimisation with cosine annealing, SpecAugment [@park2019specaugment] for spectrogram models, label smoothing (0.1), mixup augmentation (alpha 0.3), and early stopping with patience 6--15 epochs. Training is sequential on a single Apple M-series GPU (MPS backend). Every result is mechanically generated from run artifacts; no number is hand-edited.

# V. Results

## A. Model Comparison

Table II presents all 13 architectures ranked by accuracy under the artist-filtered, track-level protocol. Each architecture is cited to the foundational work it builds upon.

```{=latex}
\begin{table}[t]
\centering
\caption{All Architectures Under Artist-Filtered, Track-Level Evaluation}
\label{tab:models}
\footnotesize
\begin{tabular}{clcc}
\toprule
\textbf{\#} & \textbf{Architecture} & \textbf{Acc.} & \textbf{F1} \\
\midrule
1 & Ensemble (uniform avg) \cite{kittler1998combining} & 58.5 & 58.2 \\
2 & MERT-95M flat probe \cite{li2023mert} & 58.2 & 57.8 \\
3 & BYOL-A + DNN probe \cite{niizumi2023byola} & 57.4 & 57.0 \\
4 & PANNs CNN14 probe \cite{kong2020panns} & 57.4 & 56.6 \\
5 & MERT-330M probe (learned wt.) \cite{li2023mert} & 56.6 & 56.5 \\
6 & MERT-330M flat probe \cite{li2023mert} & 56.5 & 55.9 \\
7 & MERT-95M probe (learned wt.) \cite{pasad2021layer} & 56.4 & 55.7 \\
8 & MERT-95M deep probe \cite{yang2021superb} & 56.0 & 55.9 \\
9 & MERT-95M attention probe \cite{safari2020self} & 55.2 & 54.4 \\
10 & GenreCNN (log-mel) \cite{choi2017transfer} & 55.1 & 54.2 \\
11 & MERT-330M deep probe \cite{yang2021superb} & 54.9 & 55.5 \\
12 & DenseNet121 \cite{huang2017densenet,bian2019audio} & 53.9 & 52.2 \\
13 & XGBoost on MERT \cite{chen2016xgboost} & 52.5 & 51.4 \\
\bottomrule
\end{tabular}
\end{table}
```

The best single model is the MERT-95M flat probe (58.2%), which concatenates all 13 transformer layers into a 9,984-dimensional feature vector and trains a two-layer MLP. A uniform-weight ensemble of per-architecture bests reaches 58.5%. Notably, MERT-330M does not outperform MERT-95M, and learned ensemble weights lose to fixed uniform averaging --- both symptoms of overfitting at this data scale (6,400 training tracks).

## B. Protocol Leakage

Table III presents the core finding: identical architectures trained under all three protocols. The same model moves by 14 to 41 percentage points depending solely on the evaluation protocol.

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
MERT Probe (weighted-sum) & 56.1 & 69.2 & 93.6 & +37.5 \\
MERT Flat Probe (13-layer) & 58.2 & 70.2 & 90.2 & +32.0 \\
MERT Deep Probe (multi-scale) & 55.5 & 68.5 & 92.2 & +36.7 \\
MERT Attention Probe & 54.6 & 69.8 & 95.7 & +41.0 \\
MERT-330M Probe & 56.6 & 70.9 & 96.0 & +39.4 \\
CNN (log-mel spectrogram) & 55.1 & --- & 69.2 & +14.1 \\
BYOL-A + DNN \cite{rai2025evaluating} & 57.4 & 69.9 & 84.5 & +27.2 \\
PANNs CNN14 \cite{liu2022openset} & 57.4 & 67.2 & 73.4 & +16.0 \\
\midrule
\textbf{Mean} & \textbf{56.4} & \textbf{69.4} & \textbf{86.9} & \textbf{+30.5} \\
\bottomrule
\end{tabular}
\end{table*}
```

Decomposed over the seven architectures with all three rungs: dropping the artist filter alone accounts for +12.8 pp, and leaking clips on top of that adds a further +20.0 pp. The effect is systematic and architecture-independent, spanning from-scratch CNNs (+14.1 pp) to attention-based MERT probes (+41.0 pp).

The MERT attention probe illustrates the extreme: 54.6% under honest evaluation, 95.7% under clip-random. The attention mechanism excels at acoustic fingerprinting --- recognising *which recording* it has heard --- which is rewarded under clip-random but irrelevant under artist-filtered evaluation.

## C. Against Published Literature

Our best artist-filtered model reaches 58.5%. The only published number carrying an artist-filtered guarantee is 54.8% (Park et al.'s re-run of FMA's baseline code [@park2018representation]), a margin of +3.7 pp. We deliberately do not report a "beats N of M" count: ranking against numbers whose protocol is unknown would treat incomparable quantities as comparable, which is the failure this project documents.

For two architectures we reimplemented from published work, the published numbers land between our artist-filtered and random-track results:

- **BYOL-A + DNN** [@rai2025evaluating]: published 64.3%, ours 57.4% (artist-filtered) / 69.9% (random-track)
- **PANNs CNN14** [@liu2022openset]: published 60.9%, ours 57.4% (artist-filtered) / 67.2% (random-track)

This is consistent with several explanations, including protocol differences and probe-head fidelity. The architecture-fidelity confound is as large as the effect, so we do not adjudicate.

# VI. Discussion

## A. The Research Gap

The finding is not that the field is cheating --- it is that the field has no reporting convention for a variable we measure at tens of points. When a paper does not state its protocol:

1. The number becomes *unplaceable*. Not "probably leaked" --- unplaceable.
2. It is incomparable in both directions. An undisclosed 64.3% is neither above nor below an artist-filtered 58.5% in any meaningful sense.
3. The honest error bar is the full ladder span, which for most architectures exceeds the entire range of accuracies the literature competes over.

We propose that future FMA-small publications state, at minimum: (a) whether the artist-filtered split was used, (b) whether evaluation is track-level or episode-level, and (c) the exact split mechanism if not FMA's shipped metadata.

## B. A Self-Inflicted Illustration

An earlier revision of this project compared our episode-level accuracy against published track-level numbers and concluded our reimplementations "reproduced within 1.1 pp." The claim was wrong: episode-level figures run 5--7 pp below track-level ones, and that gap did all the work. This is a better illustration of the thesis than any borrowed example --- an unstated measurement choice produced a confident false conclusion, against ourselves.

## C. Limitations

This study covers one dataset (FMA-small), one segmentation length (3 seconds), and reimplements rather than runs original code. The 3-second window is a ceiling on what any model here can see; genre is partly a long-timescale property. The reimplementation fidelity confound means our protocol-ladder measurements are exact only for our own models, though the direction and magnitude are architecture-independent.

# VII. Conclusion

We measured the cost of three evaluation protocol choices on FMA-small music genre classification across 13 architectures and 52 training runs. The effect is large (+30.5 pp mean inflation), systematic (architecture-independent), and undisclosed (0 of 9 published papers state artist-filtered evaluation). Our best model reaches 58.5% under the artist-filtered, track-level protocol, establishing a placeable reference point for future comparison. The gap is not in the models --- it is in the reporting convention. We call on future work to state its split protocol explicitly, so that published numbers can be placed on a common axis.

# References
