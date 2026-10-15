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

Music genre classification is one of the longest-standing tasks in music information retrieval (MIR). Since Tzanetakis and Cook's foundational work on the GTZAN dataset [@tzanetakis2002musical], the problem has served as a proving ground for successive waves of audio representation: hand-engineered features, convolutional networks, recurrent architectures, and most recently self-supervised transformer models. The Free Music Archive small subset (FMA-small) [@defferrard2017fma] --- 8,000 tracks balanced across eight genres --- has become a standard benchmark for comparing these architectures, offering balanced classes, open licensing, and pre-computed metadata including an artist-filtered train/validation/test split.

Yet a reader attempting to rank published results on FMA-small faces a quiet problem: most papers do not say how they split the data. FMA ships an artist-filtered split in which no artist appears in more than one partition, but using it is optional and disclosing whether it was used is rarer still. A paper that reports "random 8:1:1" without naming the unit --- songs or 3-second clips --- leaves open a question whose answer, as we measure here, is worth tens of accuracy points.

This is not an accusation. It is a measurement. The field has no reporting convention for a variable that dominates the accuracy axis, and so published numbers are not mutually comparable even in principle. We document the size of the effect, audit what the literature actually states, and report our own results under a named protocol so that future work has at least one placeable reference point.

**Contributions.** (1) A three-rung protocol ladder quantifying the accuracy cost of each leakage type across 8 architectures. (2) A literature audit of 9 published FMA-small results, recording what each paper states about its split. (3) A reproducible training pipeline covering 13 architectures and 52 evaluated runs, all under the artist-filtered, track-level protocol.

# Related Work

## A. Genre Classification: From Hand-Crafted Features to Deep Learning

Automatic genre classification originated with Tzanetakis and Cook [@tzanetakis2002musical], who introduced the GTZAN dataset and proposed a feature-based classification pipeline using timbral, rhythmic, and pitch features fed to Gaussian mixture models. This paper established both the standard feature vocabulary (MFCCs, spectral centroid, zero-crossing rate) and the evaluation paradigm (10-fold cross-validation on 1,000 clips) that persisted for over a decade. Sturm [@sturm2013gtzan] later demonstrated that GTZAN contained duplicate recordings, mislabellings, and distortions that allowed models to achieve inflated results --- an early warning about evaluation integrity that the field absorbed for GTZAN but did not generalise to subsequent benchmarks.

The transition to deep learning for audio began with convolutional approaches applied to mel spectrograms. Choi et al. [@choi2017transfer] proposed transfer learning from large music datasets, demonstrating that features learned for music auto-tagging transferred effectively to genre classification and other MIR tasks. Their work showed that the representational gap between tasks could be bridged by fine-tuning, a principle that underlies the pretrained backbone approaches in our pipeline. DenseNet [@huang2017densenet], originally developed for image recognition, was adapted for audio by treating mel spectrograms as single-channel images. Bian et al. [@bian2019audio] applied DenseNet with SVM stacking to FMA-small, reporting 68.9% accuracy with data augmentation --- though without specifying their evaluation split.

Capsule networks [@sabour2017dynamic] offered an alternative to pooling-based architectures by encoding part-whole relationships through dynamic routing. Their application to audio aimed to capture hierarchical timbral structure, though our experiments suggest they do not outperform simpler CNN architectures on FMA-small under controlled conditions.

The recent trend combines convolutional feature extraction with transformer-based temporal modelling. Zhang [@zhang2025tsresformer] proposed TS-Resformer, using a ResNet backbone with a transformer encoder for temporal aggregation, achieving 90.2% on FMA-small --- the highest published number, though under a random track split without artist filtering. Ma et al. [@ma2026ctgatenet] introduced CT-GateNet, combining CNN and transformer with a learned gating mechanism, reporting 89.4% with an unspecified "8:1:1 ratio" split. Kostrzewa et al. [@kostrzewa2021music] conducted a systematic comparison of CNN, RNN, and CRNN architectures for genre classification, reporting 56.4% on FMA-small with an "80:10:10 ratio" of unstated mechanism. Notably, the same paper criticises a competing result for "data leakage... which leads to over-optimistic outcomes" while leaving its own split mechanism undisclosed.

## B. Self-Supervised Audio Representations

The dominant paradigm shift in audio understanding has been the adoption of self-supervised pretraining, paralleling developments in NLP and computer vision. The key insight is that large unlabelled audio corpora contain sufficient structure for models to learn rich representations without human annotation, which can then be transferred to downstream tasks via probing or fine-tuning.

**wav2vec 2.0** [@baevski2020wav2vec] pioneered contrastive self-supervised learning for speech, learning representations by predicting quantised speech units from masked input. The model's layered architecture --- a CNN feature encoder feeding a transformer --- established the blueprint adopted by subsequent audio foundation models. **HuBERT** [@hsu2021hubert] extended this with offline clustering of learned representations to generate pseudo-labels, iteratively refining both the discrete targets and the learned features. The approach proved particularly effective for speech tasks and forms the architectural basis for MERT.

**MERT** [@li2023mert] adapts the HuBERT framework specifically for music, pretraining on 20,000 hours of music audio with both acoustic and music-specific pseudo-labels (from a CQT-based teacher). MERT-v1-95M (12 transformer layers, 768-dim) and MERT-v1-330M (24 layers, 1024-dim) are architecturally stock HuBERT models loaded via `transformers.HubertModel`, verified bit-identical to the original weights while avoiding the `trust_remote_code` dependency. A critical property for our work: genre-discriminative information is distributed across transformer depths, not concentrated in the final layer. Pasad et al. [@pasad2021layer] showed this layer-distribution effect in speech models, and we observe it directly: probes with learned layer weights or full-layer concatenation consistently outperform last-layer-only probes. The SUPERB benchmark [@yang2021superb] formalised the probing protocol --- freeze the backbone, train only a lightweight head --- that we adopt as our default.

**BYOL-A** [@niizumi2023byola] takes a different approach, applying Bootstrap Your Own Latent to general audio. Unlike contrastive methods that require negative pairs, BYOL-A learns by predicting one augmented view of audio from another through a student-teacher architecture. The result is a 2,048-dimensional general-purpose audio embedding that transfers to a wide range of tasks including environmental sound classification, speaker identification, and music tagging. Its general-purpose pretraining provides complementary features to MERT's music-specific representations.

**PANNs** [@kong2020panns] takes the supervised route, training CNN14 on the full AudioSet [@gemmeke2017audioset] taxonomy of 527 sound classes. With a mAP of 0.431 on AudioSet, PANNs embeddings encode rich acoustic semantics that, while not music-specific, capture timbral and spectral patterns relevant to genre. The 2,048-dimensional penultimate-layer embeddings we extract are among the most information-dense fixed representations available for audio.

More recent self-supervised models include **Audio-MAE** [@huang2022masked], which applies masked autoencoding to audio spectrograms, and **BEATs** [@chen2023beats], which combines audio tokenisation with iterative self-supervised training. These models achieve state-of-the-art results on AudioSet but have not yet been widely evaluated on FMA-small, representing potential future baselines.

## C. Audio Spectrogram Transformers

The success of Vision Transformers (ViT) [@dosovitskiy2021vit] in image recognition prompted direct adaptation to audio. The **Audio Spectrogram Transformer (AST)** [@gong2021ast] treats mel spectrograms as images, splitting them into 16x16 patches and processing them with a ViT pretrained on ImageNet and fine-tuned on AudioSet. AST achieves 45.9 mAP on AudioSet, demonstrating that the patch-based attention mechanism captures both local spectral patterns and long-range temporal dependencies. Its different inductive bias --- supervised discriminative pretraining on general audio events --- makes it complementary to self-supervised music models like MERT for ensemble diversity.

**HTS-AT** [@chen2022htsat] adapts the Swin Transformer [@liu2021swin] for audio, introducing hierarchical patch merging that progressively reduces spatial resolution while increasing channel depth. The shifted-window attention mechanism provides an efficient alternative to global self-attention, and the hierarchical structure naturally captures multi-scale audio features --- from fine-grained spectral details in early layers to coarse temporal patterns in later ones. HTS-AT achieves 47.1 mAP on AudioSet, among the strongest single-model results.

**PaSST** [@koutini2022passt] introduced Patchout training for audio transformers, randomly dropping patches during training to regularise and accelerate training. This approach proved particularly effective for AudioSet where overfitting is common despite the large dataset size.

**EfficientAT** [@schmid2023efficientat] demonstrated that knowledge distillation from large transformers into lightweight CNN architectures can retain most of the accuracy at a fraction of the compute cost. This finding is relevant for practical deployment but also for ensemble construction: architecturally diverse models (transformers + CNNs) tend to make different errors, improving ensemble performance.

## D. Probing vs. Fine-Tuning Pretrained Models

The choice between frozen probing and fine-tuning a pretrained backbone involves a fundamental tradeoff: probes measure what the representation already encodes, while fine-tuning adapts the representation to the target task at the risk of overfitting on small datasets.

The SUPERB benchmark [@yang2021superb] standardised the probing protocol for speech models: freeze the backbone, train only a lightweight classifier head. This protocol isolates the quality of the representation from the capacity of the classifier, making comparisons between backbones more meaningful. Our laptop pipeline follows this protocol exactly, and its 58.5% ceiling reflects the information content of frozen MERT/BYOL-A/PANNs representations for genre classification.

Fine-tuning breaks this ceiling by adapting the backbone's representations to the target task. However, with only 6,400 training tracks in FMA-small, full fine-tuning of a 330M-parameter model risks catastrophic overfitting. **LoRA** (Low-Rank Adaptation) [@hu2021lora] addresses this by injecting trainable low-rank matrices into the attention layers while keeping the original weights frozen. With rank 16, this adds only ~2M trainable parameters to a 330M model, providing a favourable tradeoff between adaptation capacity and overfitting risk. Layer-wise learning rate decay further stabilises fine-tuning by limiting the magnitude of changes to lower layers, which encode more general features.

## E. Published FMA-small Results

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

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{literature_scatter.pdf}
\caption{Published FMA-small accuracies over time. Red circles denote results whose evaluation protocol is not stated; the blue star marks our artist-filtered ensemble. The dashed line is our best result (58.5\%). Results above the line are not necessarily better --- they may reflect a more permissive protocol.}
\label{fig:literature}
\end{figure}
```

One widely cited baseline --- "Defferrard et al., 58.0\%" --- does not exist. The FMA paper's only accuracy table covers the 16-genre medium subset; 58 is the MLP cell there. The 54.8\% attributed to FMA's authors is actually Park et al.'s re-run of FMA's released baseline code [@park2018representation], artist-filtered by code inheritance rather than authorial claim.

## F. Data Leakage in Machine Learning

Data leakage --- the use of information during training that would not be available at prediction time --- is a pervasive concern across machine learning, not unique to audio. Kaufman et al. [@kaufman2012leakage] formalised leakage types and proposed detection methods, distinguishing between leakage through features (training on test-set information) and leakage through examples (inadequate train/test separation). Kapoor and Narayanan [@kapoor2023leakage] surveyed 329 papers across 17 scientific domains and found that data leakage affected a substantial fraction, contributing to a broader reproducibility crisis in ML-based science.

Source-level leakage is well understood in adjacent domains. Roberts et al. [@roberts2017crossvalidation] demonstrated that standard random cross-validation produces severely inflated performance estimates when data has spatial, temporal, or hierarchical structure, and proposed structured cross-validation strategies as remedies. In medical imaging, Whalen et al. [@whalen2022navigating] documented how patient-level splits are essential when multiple images come from the same patient; failure to enforce this leads to models that recognise patients rather than pathologies.

The pattern is consistent across fields: wherever data points share a common origin, evaluation must split at the source level. Music genre classification exhibits all three forms simultaneously --- the same recording (clip-level), the same performance (track-level), and the same production signature (artist-level). FMA ships the artist-level fix in its metadata, but using it is optional and disclosing whether it was used is rarer still.

Flexer and Schnitzer [@flexer2006closer] provided early evidence of artist-filter effects in music classification, showing that removing the artist confound significantly reduced classification accuracy on GTZAN. Sturm [@sturm2014simple] went further, demonstrating that some MIR systems functioned as "horses" --- exploiting dataset-specific artifacts rather than learning musically meaningful features. Our three-rung ladder provides the first systematic quantification of this effect on FMA-small across multiple architectures.

## G. Training Techniques and Regularisation

Several training techniques are relevant to our pipeline and merit brief discussion for reproducibility.

**SpecAugment** [@park2019specaugment] applies random frequency and time masks to mel spectrograms during training, functioning as a regulariser that prevents the model from relying on specific spectral regions or time intervals. Originally developed for automatic speech recognition, it has become standard in audio classification.

**Mixup** [@zhang2018mixup] interpolates between pairs of training examples and their labels, encouraging models to behave linearly between training samples. In our experiments, mixup with alpha=0.3 consistently improves generalisation for probe models on FMA-small.

**Label smoothing** [@szegedy2016rethinking] softens the hard 0/1 targets by redistributing a small fraction of probability mass to non-target classes. With 8 balanced genres, label smoothing of 0.1 allocates approximately 0.014 to each non-target class, penalising overconfident predictions.

**AdamW** [@loshchilov2019adamw] decouples weight decay from the gradient update, providing more effective regularisation than L2 regularisation in the Adam optimiser. Combined with cosine annealing of the learning rate, this forms our default optimisation setup.

**Focal loss** [@lin2017focal] down-weights well-classified examples to concentrate training on hard cases. While designed for class-imbalanced detection tasks, we evaluate it as an alternative to cross-entropy for genres that are systematically hard to classify (Folk, Experimental).

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

**Log-mel spectrograms.** For from-scratch CNN models, each 3-second episode is converted to a 128-bin log-mel spectrogram at 22,050 Hz (n\_fft=2048, hop\_length=512), yielding a 128 x 130 feature map.

**MERT embeddings.** MERT-v1-95M [@li2023mert] is a 12-layer transformer pretrained on 20,000 hours of music via self-supervised learning. We extract all 13 layer outputs (12 transformer + 1 convolutional projection), mean-pooled over the time dimension, producing a (13, 768) tensor per episode. MERT-v1-330M (25 layers, 1024-dim) is extracted analogously. Audio is decoded at MERT's native 24 kHz via librosa [@mcfee2015librosa].

**BYOL-A embeddings.** BYOL-A [@niizumi2023byola] produces a single 2048-dimensional embedding per episode at 16 kHz.

**PANNs CNN14 embeddings.** PANNs CNN14 [@kong2020panns] produces 2048-dimensional penultimate-layer embeddings per episode at 32 kHz.

## E. Model Families

We train 13 architectures spanning three families:

**From-scratch spectrogram models** consume 128-bin log-mel spectrograms and train all parameters from scratch: GenreCNN (4-block CNN, ~260K params) [@choi2017transfer], DenseNet121 (ImageNet-pretrained, adapted to single-channel input) [@huang2017densenet; @bian2019audio].

**Frozen-backbone probes** train only a lightweight head on cached embeddings:

- *MERT learned weighted-sum probe* [@pasad2021layer]: softmax-weighted sum over all layers, followed by a two-layer MLP.
- *MERT flat probe*: concatenates all 13 layers into a 9,984-dim vector, followed by a two-layer MLP with LayerNorm.
- *MERT deep probe* [@yang2021superb]: groups layers into multi-scale aggregations with residual blocks.
- *MERT attention probe* [@safari2020self]: multi-head self-attention over the 13 layers with learnable positional embeddings.
- *BYOL-A + DNN probe* [@niizumi2023byola]: LayerNorm followed by a variable-depth MLP on the 2048-dim BYOL-A embedding.
- *PANNs CNN14 probe* [@kong2020panns]: same architecture as BYOL-A probe, on PANNs 2048-dim embeddings.

All MERT probe variants are also trained with the 330M backbone where applicable.

**Non-neural baseline.** XGBoost [@chen2016xgboost] on flattened MERT embeddings.

## F. Training Configuration

All neural models use AdamW [@loshchilov2019adamw] with cosine annealing. Spectrogram models apply SpecAugment [@park2019specaugment] (2 frequency masks up to 16 bins, 2 time masks up to 20 frames). Label smoothing (0.1) [@szegedy2016rethinking] and mixup augmentation (alpha 0.3) [@zhang2018mixup] are used for most runs. Early stopping with patience 6--15 epochs prevents overfitting. Focal loss [@lin2017focal] is evaluated as an alternative criterion.

# Experiments

We conducted 52 evaluated training runs: 36 under the artist-filtered split, 8 under random-track, and 8 under clip-random. Every result is mechanically generated from run artifacts by dedicated scripts; no number in the results tables is hand-edited.

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

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{architecture_bars.pdf}
\caption{Best track-level accuracy per architecture under the artist-filtered protocol. The top four models (58.2--58.5\%) are within seed noise of each other.}
\label{fig:archbars}
\end{figure}
```

The best single model is the MERT-95M flat probe (58.2\%), which concatenates all 13 transformer layers into a 9,984-dimensional feature vector and trains a two-layer MLP. A uniform-weight ensemble of per-architecture bests reaches 58.5\%. Notably, MERT-330M does not outperform MERT-95M, and learned ensemble weights lose to fixed uniform averaging --- both symptoms of overfitting at this data scale (6,400 training tracks).

## B. Complete Run Inventory

Table III presents all 36 artist-filtered runs, including hyperparameter variants and seed ablations.

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{seed_stability.pdf}
\caption{Run-to-run variability for architectures with multiple training runs. Box plots show median and quartiles; red diamonds mark means. The MERT Flat probe spans 3.8~pp across 7 runs differing only in seed and regularisation.}
\label{fig:seeds}
\end{figure}
```

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{training_curves.pdf}
\caption{Training dynamics for five representative architectures. (a)~Validation macro-F1: probe models (MERT, BYOL-A, PANNs) converge in under 10 epochs; from-scratch models (GenreCNN, DenseNet) need 30--40. (b)~Training loss: probe losses plateau higher because the frozen backbone limits representational flexibility.}
\label{fig:curves}
\end{figure*}
```

Key observations: (a) The MERT-95M flat probe spans 54.4--58.2\% across 7 runs differing only in seed and regularisation, a 3.8 pp range. (b) Seed ablations for the MERT-95M probe (seeds 1--3, 42) span 53.5--56.4\%, showing that a 3 pp difference between two papers could be seed noise. (c) The CNN baseline improves from 47.1\% to 55.1\% (+8.0 pp) through augmentation and hyperparameter tuning alone.

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

## C. Protocol Leakage Ladder

Table IV combines the three protocols for the 8 architectures tested on all rungs.

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

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{leakage_bars.pdf}
\caption{Protocol leakage across all eight architectures. Blue bars show artist-filtered accuracy; orange bars show random-track; red bars show clip-random. Red annotations above each bar show the total inflation in percentage points. The MERT attention probe inflates by +41~pp --- from 54.6\% to 95.7\%.}
\label{fig:leakage}
\end{figure*}
```

Decomposed: dropping the artist filter alone accounts for +12.8 pp, and leaking clips on top adds +20.0 pp. The effect is systematic and architecture-independent.

## D. Genre-Specific Leakage Analysis

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{genre_leakage_decomposition.pdf}
\caption{Per-genre leakage decomposition, averaged across 8 architectures. Genres are sorted by total inflation (clip-random minus artist-filtered). Folk inflates by +49.8~pp; Hip-Hop by only +14.0~pp. The orange segment isolates artist leakage; the red segment isolates clip leakage on top of that.}
\label{fig:genre_decomp}
\end{figure}
```

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{genre_inflation_heatmap.pdf}
\caption{Genre-specific inflation (pp) for each architecture. Dark cells indicate high inflation. Folk and Instrumental are consistently the most inflated genres; Hip-Hop and Electronic are the least. The MERT attention probe on Instrumental inflates by +63~pp.}
\label{fig:genre_heatmap_infl}
\end{figure}
```

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{genre_inflation_bars.pdf}
\caption{Per-genre inflation (mean across architectures). Blue: artist-filtered F1; red: clip-random F1. The gap measures how much each genre benefits from leaky evaluation. Folk (+50~pp) and Instrumental (+45~pp) are most inflated.}
\label{fig:genre_bars}
\end{figure}
```

Not all genres inflate equally. Folk (+49.8 pp mean inflation), Instrumental (+45.2 pp), and Experimental (+42.8 pp) inflate the most. These are also the three hardest genres under honest evaluation (36.1\%, 40.9\%, and 41.4\% mean F1 respectively). The explanation: these genres are stylistically diffuse, so models learn to recognise *which recording* they have heard rather than genre features. Hip-Hop (+14.0 pp) and Electronic (+16.3 pp) inflate the least, having strong acoustic signatures that transfer across artists.

## E. Against Published Literature

Our best artist-filtered model reaches 58.5\%. The only published number carrying an artist-filtered guarantee is 54.8\% (Park et al.'s re-run of FMA's baseline code [@park2018representation]), a margin of +3.7 pp. We deliberately do not report a "beats N of M" count: ranking against numbers whose protocol is unknown would treat incomparable quantities as comparable.

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{genre_heatmap.pdf}
\caption{Per-genre F1 (\%) across the ten best architectures. Hip-Hop and Electronic are consistently well-classified (76--82\%); Folk and Pop are hardest (26--44\%), likely due to stylistic overlap with other genres.}
\label{fig:heatmap}
\end{figure}
```

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{confusion_best_artist_filtered.png}
\caption{Confusion matrix for the best single model (MERT-95M flat probe, 58.2\%) under the artist-filtered protocol. Hip-Hop (86/100) and Electronic (81/100) are well-separated; Folk (38/100) scatters into Experimental, Pop, and Instrumental.}
\label{fig:confusion_af}
\end{figure}
```

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{confusion_best_cliprandom.png}
\caption{Confusion matrix for the \emph{same} model under clip-random evaluation (90.2\%). The diagonal dominates: every genre exceeds 530/630 correct. This is not improved classification --- it is the model recognising \emph{which recording} it has heard.}
\label{fig:confusion_cr}
\end{figure}
```

# Discussion

## A. The Research Gap

The finding is not that the field is cheating --- it is that the field has no reporting convention for a variable we measure at tens of points. When a paper does not state its protocol, the number becomes unplaceable, incomparable in both directions, and the honest error bar spans the full ladder range.

We propose that future FMA-small publications state: (a) whether the artist-filtered split was used, (b) whether evaluation is track-level or episode-level, and (c) the exact split mechanism if not FMA's shipped metadata.

## B. Negative Results

**Epoch-Scheduled Conditional Fusion (ESCF).** A learned fusion of CNN and MERT backbones using per-episode sigmoid gates, per-track attention, and a curriculum. Every ESCF variant lost to a fixed 50/50 uniform ensemble (McNemar p = 0.011).

**Learned attention over episodes.** Pooling with learned attention is consistently worse than averaging. The weakest MERT probe in our inventory is the attention probe.

**The lesson.** Anything learned on top of frozen features overfits at this data scale. Accuracy gains must come from better features, not new learned combiners.

## C. Limitations

This study covers one dataset (FMA-small), one segmentation length (3 seconds), and reimplements rather than runs original code. The 3-second window is a ceiling on what any model here can see; genre is partly a long-timescale property. FMA-small's 8 coarse genres include genuinely multi-genre tracks; the single `genre_top` label caps classifier performance.

# Ongoing and Future Work

## A. GPU-Scale Fine-Tuning

All current MERT results freeze the backbone. We have implemented end-to-end MERT fine-tuning with LoRA adapters [@hu2021lora] (~2M trainable parameters on a 330M backbone), partial layer unfreezing with layer-wise learning rate decay, and multi-crop processing of full 30-second tracks. Additionally, Audio Spectrogram Transformer [@gong2021ast] and HTS-AT [@chen2022htsat] fine-tuning provides ensemble diversity through different pretrained backbones. These experiments require multi-GPU infrastructure and are pending compute allocation.

## B. Context Window Ablation

Every embedding in this project is computed from 3 seconds of audio. A 10-second extraction grid is implemented, as is a multi-crop approach that processes multiple 5-second windows from the full 30-second track. If longer context yields measurable gains, the implication is that the entire FMA-small literature has been evaluating under a suboptimal but unreported context length.

## C. Cross-Dataset Protocol Audit

The protocol inflation phenomenon should transfer to other music benchmarks (GTZAN, MagnaTagATune [@law2009magnatag]). Measuring the same ladder on a second dataset would elevate the finding from dataset-specific to field-wide.

# Conclusion

We measured the cost of three evaluation protocol choices on FMA-small music genre classification across 13 architectures and 52 training runs. The effect is large (+30.5 pp mean inflation), systematic (architecture-independent), and undisclosed (0 of 9 published papers state artist-filtered evaluation). Our best model reaches 58.5\% under the artist-filtered, track-level protocol. The gap is not in the models --- it is in the reporting convention.
