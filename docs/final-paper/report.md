---
title: "Evaluation Protocol Leakage in Music Genre Classification: A Systematic Study on FMA-Small"
author:
  - "Adarsh, Bennett University, e23cseu1189@bennett.edu.in"
abstract: |
  Music genre classification on FMA-small is a widely used benchmark,
  yet published results rarely disclose the evaluation protocol that
  produced them. We show that three independent protocol choices ---
  whether the train/test boundary falls between artists, between tracks,
  or between 3-second clips --- move a single model's accuracy by 14 to
  41 percentage points. Across 62 training runs spanning 14
  architectures and four model families --- from-scratch CNNs, music-specific
  self-supervised models (MERT-95M, MERT-330M), general audio
  and audio-language models (BYOL-A, PANNs, CLAP), and non-neural baselines --- the mean
  inflation from the most permissive protocol (clip-random) over the
  honest one (artist-filtered) is +30.8 pp. A literature audit of nine
  published FMA-small results finds that zero papers state artist-filtered
  evaluation in prose. The leakage pattern holds universally: it is not
  an artifact of any particular architecture, pretraining strategy, or
  model family. Beyond accuracy, we provide three independent diagnostics
  for protocol integrity: (i) leakage distorts calibration, creating
  underconfidence ($T^* < 1$) rather than overconfidence; (ii) leakage
  raises inter-episode temporal consistency from 80\% to 96\%, far
  beyond what genre classification alone can explain; and (iii) leakage
  disproportionately benefits hard tracks (+61.5 pp for tracks no model
  classifies correctly under honest evaluation, vs --1.1 pp for easy
  tracks). Confusion matrix analysis shows that leakage eliminates
  genuine genre ambiguity --- e.g., Folk--Instrumental confusion drops
  by 26 pp --- by substituting recording recognition for genre
  recognition. At the embedding level, 57--90\% of test clips under
  clip-random evaluation have a same-track nearest neighbor in training,
  and the inflation is probe-invariant: linear to multi-layer
  probes all show identical protocol gaps. We frame this as a research gap in reporting
  conventions: the field benchmarks on a variable it does not name, and
  the resulting numbers are not mutually comparable. Our best model, a
  frozen CLAP probe, reaches 59.9\% under the artist-filtered,
  track-level protocol --- 5.1 pp above the only prior sourceable
  artist-filtered number (54.8\%).
---

# 1. Introduction

Music genre classification is one of the longest-standing tasks in music information retrieval (MIR). Since Tzanetakis and Cook's foundational work on the GTZAN dataset [@tzanetakis2002musical], the problem has served as a proving ground for successive waves of audio representation: hand-engineered features, convolutional networks, recurrent architectures, and most recently self-supervised transformer models. The Free Music Archive small subset (FMA-small) [@defferrard2017fma] --- 8,000 tracks balanced across eight genres --- has become a standard benchmark for comparing these architectures, offering balanced classes, open licensing, and pre-computed metadata including an artist-filtered train/validation/test split.

Yet a reader attempting to rank published results on FMA-small faces a quiet problem: most papers do not say how they split the data. FMA ships an artist-filtered split in which no artist appears in more than one partition, but using it is optional and disclosing whether it was used is rarer still. A paper that reports "random 8:1:1" without naming the unit --- songs or 3-second clips --- leaves open a question whose answer, as we measure here, is worth tens of accuracy points.

The emergence of audio foundation models --- self-supervised transformers pretrained on thousands of hours of unlabelled audio --- has shifted the field from training-from-scratch to probing frozen representations. Models like MERT [@li2023mert], Wav2Vec 2.0 [@baevski2020wav2vec], HuBERT [@hsu2021hubert], BEATs [@chen2023beats], and CLAP [@wu2023clap] now define the state of the art across audio tasks. Yet this paradigm shift has not been accompanied by a corresponding shift in evaluation discipline. If anything, the problem is getting worse: foundation models have orders of magnitude more capacity than the shallow classifiers of a decade ago, and that capacity makes it easier --- not harder --- to memorise recording identities when clips from the same track appear in both train and test. A model that can reconstruct audio from its latent representation can trivially identify which recording it has heard.

This is not an accusation. It is a measurement. The field has no reporting convention for a variable that dominates the accuracy axis, and so published numbers are not mutually comparable even in principle. We document the size of the effect, audit what the literature actually states, and report our own results under a named protocol so that future work has at least one placeable reference point.

**Contributions.** We organise our contributions into four themes:

*Protocol quantification.* (1) A three-rung protocol ladder quantifying the accuracy cost of each leakage type across 14 architectures spanning four model families. (2) A literature audit of 9 published FMA-small results, recording what each paper states --- not implies --- about its split. (3) A reproducible training pipeline covering 14 architectures and 62 evaluated runs, all under the artist-filtered, track-level protocol.

*Diagnostic suite for protocol integrity.* (4) Six independent diagnostics, each requiring only standard model outputs: calibration reversal ($T^* < 1$ under leakage), temporal consistency inflation (80\% to 96\%), difficulty--improvement monotonicity (+61.5 pp for the hardest tracks vs --1.1 pp for the easiest), representation collapse (within-track cosine similarity 0.83--0.93 vs 0.30--0.77 between tracks), nearest-neighbor identity recovery (57--90\% of test clips recover a same-track training neighbor under clip-random), and probe architecture invariance (inflation persists across linear to multi-layer probes).

*Representation-level analysis.* (5) Confusion matrix decomposition showing leakage eliminates genuine genre confusions by substituting recording recognition for genre recognition. (6) Layer-wise leakage fingerprinting revealing uniform recording-identity encoding across all 13 MERT transformer layers. (7) Data efficiency analysis showing inflation scales with training set size. (8) t-SNE embedding geometry visualisations making the leakage mechanism directly visible.

*Ensemble and model analysis.* (9) CKA representation similarity and error diversity analyses explaining why architectural diversity, not parameter scaling, drives ensemble performance.

# 2. Related Work

## A. Genre Classification: From Hand-Crafted Features to Deep Learning

Automatic genre classification originated with Tzanetakis and Cook [@tzanetakis2002musical], who introduced the GTZAN dataset and proposed a feature-based classification pipeline using timbral, rhythmic, and pitch features fed to Gaussian mixture models. This paper established both the standard feature vocabulary (MFCCs, spectral centroid, zero-crossing rate) and the evaluation paradigm (10-fold cross-validation on 1,000 clips) that persisted for over a decade. Sturm [@sturm2013gtzan] later demonstrated that GTZAN contained duplicate recordings, mislabellings, and distortions that allowed models to achieve inflated results --- an early warning about evaluation integrity that the field absorbed for GTZAN but did not generalise to subsequent benchmarks.

The transition to deep learning for audio began with convolutional approaches applied to mel spectrograms. Choi et al. [@choi2017transfer] proposed transfer learning from large music datasets, demonstrating that features learned for music auto-tagging transferred effectively to genre classification and other MIR tasks. Their work showed that the representational gap between tasks could be bridged by fine-tuning, a principle that underlies the pretrained backbone approaches in our pipeline. DenseNet [@huang2017densenet], originally developed for image recognition, was adapted for audio by treating mel spectrograms as single-channel images. Bian et al. [@bian2019audio] applied DenseNet with SVM stacking to FMA-small, reporting 68.9% accuracy with data augmentation --- though without specifying their evaluation split.

Capsule networks [@sabour2017dynamic] offered an alternative to pooling-based architectures by encoding part-whole relationships through dynamic routing. Their application to audio aimed to capture hierarchical timbral structure, though our experiments suggest they do not outperform simpler CNN architectures on FMA-small under controlled conditions.

The recent trend combines convolutional feature extraction with transformer-based temporal modelling. Zhang [@zhang2025tsresformer] proposed TS-Resformer, using a ResNet backbone with a transformer encoder for temporal aggregation, achieving 90.2% on FMA-small --- the highest published number, though under a random track split without artist filtering. Ma et al. [@ma2026ctgatenet] introduced CT-GateNet, combining CNN and transformer with a learned gating mechanism, reporting 89.4% with an unspecified "8:1:1 ratio" split. Kostrzewa et al. [@kostrzewa2021music] conducted a systematic comparison of CNN, RNN, and CRNN architectures for genre classification, reporting 56.4% on FMA-small with an "80:10:10 ratio" of unstated mechanism. Notably, the same paper criticises a competing result for "data leakage... which leads to over-optimistic outcomes" while leaving its own split mechanism undisclosed.

The trajectory of these CNN-Transformer hybrids is instructive. TS-Resformer (90.2%), CT-GateNet (89.4%), and similar architectures report ever-higher accuracies, approaching levels that would imply near-perfect genre recognition. Yet our protocol ladder shows that even a simple MLP probe on frozen features can reach 95.7% under clip-random evaluation. The question is not whether these models are powerful --- they are --- but whether the reported numbers reflect genre recognition or recording recognition. Without protocol disclosure, the question is unanswerable.

## B. Self-Supervised Audio Representations

The dominant paradigm shift in audio understanding has been the adoption of self-supervised pretraining, paralleling developments in NLP and computer vision. The key insight is that large unlabelled audio corpora contain sufficient structure for models to learn rich representations without human annotation, which can then be transferred to downstream tasks via probing or fine-tuning.

**wav2vec 2.0** [@baevski2020wav2vec] pioneered contrastive self-supervised learning for speech, learning representations by predicting quantised speech units from masked input. The model's layered architecture --- a CNN feature encoder feeding a transformer --- established the blueprint adopted by subsequent audio foundation models. **HuBERT** [@hsu2021hubert] extended this with offline clustering of learned representations to generate pseudo-labels, iteratively refining both the discrete targets and the learned features. The approach proved particularly effective for speech tasks and forms the architectural basis for MERT.

**MERT** [@li2023mert] adapts the HuBERT framework specifically for music, pretraining on 20,000 hours of music audio with both acoustic and music-specific pseudo-labels (from a CQT-based teacher). MERT-v1-95M (12 transformer layers, 768-dim) and MERT-v1-330M (24 layers, 1024-dim) are architecturally stock HuBERT models loaded via `transformers.HubertModel`, verified bit-identical to the original weights while avoiding the `trust_remote_code` dependency. A critical property for our work: genre-discriminative information is distributed across transformer depths, not concentrated in the final layer. Pasad et al. [@pasad2021layer] showed this layer-distribution effect in speech models, and we observe it directly: probes with learned layer weights or full-layer concatenation consistently outperform last-layer-only probes. The SUPERB benchmark [@yang2021superb] formalised the probing protocol --- freeze the backbone, train only a lightweight head --- that we adopt as our default.

**MusicFM** [@li2024musicfm] extends the wav2vec 2.0 framework with music-aware pretraining objectives, targeting music understanding tasks including auto-tagging, key detection, and genre classification. Like MERT, it is a music-specific foundation model, but its pretraining strategy differs: where MERT uses HuBERT-style pseudo-labels from a CQT teacher, MusicFM uses a contrastive objective more closely aligned with wav2vec 2.0. Including both MERT and MusicFM in our study allows us to test whether the choice of self-supervised objective within the music domain affects leakage vulnerability.

**BYOL-A** [@niizumi2023byola] takes a different approach, applying Bootstrap Your Own Latent to general audio. Unlike contrastive methods that require negative pairs, BYOL-A learns by predicting one augmented view of audio from another through a student-teacher architecture. The result is a 2,048-dimensional general-purpose audio embedding that transfers to a wide range of tasks including environmental sound classification, speaker identification, and music tagging. Its general-purpose pretraining provides complementary features to MERT's music-specific representations.

**PANNs** [@kong2020panns] takes the supervised route, training CNN14 on the full AudioSet [@gemmeke2017audioset] taxonomy of 527 sound classes. With a mAP of 0.431 on AudioSet, PANNs embeddings encode rich acoustic semantics that, while not music-specific, capture timbral and spectral patterns relevant to genre. The 2,048-dimensional penultimate-layer embeddings we extract are among the most information-dense fixed representations available for audio.

**ATST** (Audio Teacher-Student Transformer) [@li2024atst] applies self-distillation to audio transformers, where a student network learns from a teacher's representations across differently augmented views of the same audio. The approach achieves strong results on AudioSet and environmental sound benchmarks without requiring negative pairs or discrete pseudo-labels, offering yet another pretraining strategy in the growing SSL landscape.

**EAT** (Efficient Audio Transformer) [@chen2024eat] combines masked autoencoding with a novel audio tokenizer for efficient self-supervised pretraining. By quantizing audio representations into discrete tokens and training a masked prediction objective, EAT achieves state-of-the-art results on AudioSet with substantially reduced compute compared to prior masked autoencoders.

**Audio-MAE** [@huang2022masked] applies masked autoencoding to audio spectrograms, reconstructing randomly masked patches of mel spectrograms. The approach demonstrates that the masked autoencoding paradigm, successful in vision (MAE) and language (BERT), transfers effectively to audio.

**BEATs** [@chen2023beats] combines audio tokenisation with iterative self-supervised training, alternating between learning an audio tokenizer and training a transformer to predict the resulting tokens. This iterative refinement produces representations that achieve state-of-the-art on AudioSet classification.

**MULE** (Music Understanding with Large-scale Embeddings) [@mccallum2022mule] demonstrated that supervised pretraining on large music datasets produces representations competitive with self-supervised ones for music understanding tasks. This finding complicates the narrative that self-supervised pretraining is strictly necessary, and suggests that the pretraining data distribution may matter more than the pretraining objective.

The diversity of pretraining strategies represented in our study --- contrastive (wav2vec 2.0), predictive (HuBERT, MERT), bootstrap (BYOL-A), supervised (PANNs), multimodal (CLAP), and generative (Audio-MAE) --- makes SSL representations an ideal test bed for protocol leakage universality. If the inflation pattern holds across all of these fundamentally different learning objectives, it cannot be attributed to any particular pretraining mechanism.

## C. Audio-Language Models

**CLAP** (Contrastive Language-Audio Pretraining) [@wu2023clap; @elizalde2023clap] learns aligned audio-text embeddings via contrastive learning on audio-caption pairs. The model maps audio and natural language descriptions into a shared embedding space, enabling zero-shot audio classification by comparing audio embeddings to text embeddings of class names (e.g., "electronic music," "hip-hop music"). CLAP's audio encoder produces fixed-size embeddings (typically 512 dimensions) that can be probed with a lightweight classifier, just like BYOL-A or PANNs embeddings.

CLAP's relevance to this study is twofold. First, its pretraining objective --- audio-text alignment --- is fundamentally different from classification, reconstruction, or within-modality contrastive learning. If a CLAP probe exhibits the same leakage inflation as probes on MERT or PANNs features, it would confirm that the inflation is a property of the evaluation protocol, not the learned representation. Second, CLAP represents the emerging trend toward multimodal audio understanding, and establishing its behaviour under different evaluation protocols contributes baseline knowledge for this growing model family.

The protocol leakage question applies equally to CLAP: a probe trained on clip-random splits can memorise recording identities through the audio encoder's representations just as easily as a probe on MERT or HuBERT features. The text encoder plays no role during probing --- only the audio encoder's embeddings are used.

## D. Music Understanding Benchmarks

The proliferation of audio foundation models has motivated standardised benchmarks for comparing their representations across tasks.

**MARBLE** [@yuan2024marble] standardises evaluation across 14 music understanding tasks, including genre classification, auto-tagging, key detection, beat tracking, and instrument classification. Critically, MARBLE specifies train/test splits for each task, addressing half of the comparability problem. However, not all of MARBLE's splits control for artist identity --- a limitation the authors acknowledge. The benchmark's contribution is in task standardisation; split standardisation remains incomplete.

**HEAR** (Holistic Evaluation of Audio Representations) [@turian2022hear] evaluates audio representations across 19 tasks spanning music, speech, and environmental sound. HEAR provides standardised evaluation code and metrics but does not enforce artist-level splitting for music tasks. Its primary contribution is cross-domain comparison --- showing how a single set of features performs on musically and non-musically relevant tasks.

**SUPERB** [@yang2021superb] is primarily speech-focused but established the probing methodology we adopt: freeze the backbone, train a lightweight head, and report downstream accuracy. SUPERB's influence on our work is methodological rather than topical: we apply its probing protocol to music tasks.

The key insight from this benchmark landscape is that standardising the task definition is necessary but not sufficient. Benchmarks that fix the dataset, the metric, and even the model architecture still leave the splitting methodology as a free variable. MARBLE specifies splits but does not always artist-filter them; HEAR provides evaluation code but leaves splitting to the user; SUPERB defines the probing protocol but targets speech tasks where the source-level confound is less severe. The result is that half of the comparability problem is solved and half is not --- and the unsolved half, as we demonstrate, accounts for tens of accuracy points.

## E. Audio Spectrogram Transformers

The success of Vision Transformers (ViT) [@dosovitskiy2021vit] in image recognition prompted direct adaptation to audio. The **Audio Spectrogram Transformer (AST)** [@gong2021ast] treats mel spectrograms as images, splitting them into 16x16 patches and processing them with a ViT pretrained on ImageNet and fine-tuned on AudioSet. AST achieves 45.9 mAP on AudioSet, demonstrating that the patch-based attention mechanism captures both local spectral patterns and long-range temporal dependencies. Its different inductive bias --- supervised discriminative pretraining on general audio events --- makes it complementary to self-supervised music models like MERT for ensemble diversity.

**HTS-AT** [@chen2022htsat] adapts the Swin Transformer [@liu2021swin] for audio, introducing hierarchical patch merging that progressively reduces spatial resolution while increasing channel depth. The shifted-window attention mechanism provides an efficient alternative to global self-attention, and the hierarchical structure naturally captures multi-scale audio features --- from fine-grained spectral details in early layers to coarse temporal patterns in later ones. HTS-AT achieves 47.1 mAP on AudioSet, among the strongest single-model results.

**PaSST** [@koutini2022passt] introduced Patchout training for audio transformers, randomly dropping patches during training to regularise and accelerate training. This approach proved particularly effective for AudioSet where overfitting is common despite the large dataset size.

**EfficientAT** [@schmid2023efficientat] demonstrated that knowledge distillation from large transformers into lightweight CNN architectures can retain most of the accuracy at a fraction of the compute cost. The MN10 variant, distilled from a large audio transformer into a MobileNetV2-style [@sandler2018mobilenetv2] architecture, achieves competitive AudioSet performance with only 5M parameters. This finding is relevant both for practical deployment and for our study: EfficientAT's CNN architecture provides diversity against the transformer-dominated SSL probes, testing whether the leakage pattern is specific to attention-based architectures.

## F. Probing vs. Fine-Tuning Pretrained Models

The choice between frozen probing and fine-tuning a pretrained backbone involves a fundamental tradeoff: probes measure what the representation already encodes, while fine-tuning adapts the representation to the target task at the risk of overfitting on small datasets.

The SUPERB benchmark [@yang2021superb] standardised the probing protocol for speech models: freeze the backbone, train only a lightweight classifier head. This protocol isolates the quality of the representation from the capacity of the classifier, making comparisons between backbones more meaningful. Our laptop pipeline follows this protocol exactly, and its 59.9% ceiling reflects the information content of frozen MERT/BYOL-A/PANNs/CLAP representations for genre classification.

Fine-tuning breaks this ceiling by adapting the backbone's representations to the target task. However, with only 6,400 training tracks in FMA-small, full fine-tuning of a 330M-parameter model risks catastrophic overfitting. **LoRA** (Low-Rank Adaptation) [@hu2021lora] addresses this by injecting trainable low-rank matrices into the attention layers while keeping the original weights frozen. With rank 16, this adds only ~2M trainable parameters to a 330M model, providing a favourable tradeoff between adaptation capacity and overfitting risk. Layer-wise learning rate decay further stabilises fine-tuning by limiting the magnitude of changes to lower layers, which encode more general features.

The growing LoRA literature suggests that even on small datasets, parameter-efficient fine-tuning can yield meaningful gains over frozen probing. Our current study deliberately restricts itself to frozen probing to isolate the protocol leakage question --- fine-tuning introduces additional degrees of freedom (learning rate schedule, adapter rank, unfrozen layers) that could confound the comparison. However, fine-tuning experiments are a natural extension, and we discuss them in Section 7.

## G. Published FMA-small Results

We traced every FMA-small genre-classification result we could find to its primary source, recording what each paper says about its evaluation protocol --- not what its accuracy implies. The distinction matters: inferring a protocol from an accuracy would assume the conclusion this study sets out to test.

Table I summarises the registry. Of the nine external results, zero state artist-filtered evaluation in prose. One states clip-random splitting outright [@ashraf2025music]. One states a track-level random split [@zhang2025tsresformer]. One describes its partitions without stating artist-conditioning [@liu2022openset]. Two give a ratio with no unit [@ma2026ctgatenet; @kostrzewa2021music]. Three say nothing at all [@bian2019audio; @rai2025evaluating; @park2018representation].

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
  & \textbf{Ours --- CLAP probe} & 2026 & \textbf{59.9} & \textbf{Official artist-filtered, track-level} \\
\bottomrule
\multicolumn{5}{l}{\footnotesize *60.9\% is the closed-set sub-metric of an open-set experiment; not a standalone 8-way result.}
\end{tabular}
\end{table*}
```

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{literature_scatter.pdf}
\caption{Published FMA-small accuracies over time. Red circles denote results whose evaluation protocol is not stated; the blue star marks our artist-filtered CLAP probe. The dashed line is our best result (59.9\%). Results above the line are not necessarily better --- they may reflect a more permissive protocol.}
\label{fig:literature}
\end{figure}
```

One widely cited baseline --- "Defferrard et al., 58.0\%" --- does not exist. The FMA paper's only accuracy table covers the 16-genre medium subset; 58 is the MLP cell there. The 54.8\% attributed to FMA's authors is actually Park et al.'s re-run of FMA's released baseline code [@park2018representation], artist-filtered by code inheritance rather than authorial claim.

Since our original audit, we are not aware of any new FMA-small publications that state their evaluation protocol. The reporting gap persists.

## H. Data Leakage in Machine Learning

Data leakage --- the use of information during training that would not be available at prediction time --- is a pervasive concern across machine learning, not unique to audio. Kaufman et al. [@kaufman2012leakage] formalised leakage types and proposed detection methods, distinguishing between leakage through features (training on test-set information) and leakage through examples (inadequate train/test separation). Kapoor and Narayanan [@kapoor2023leakage] surveyed 329 papers across 17 scientific domains and found that data leakage affected a substantial fraction, contributing to a broader reproducibility crisis in ML-based science. Their finding confirms that the problem we document in MIR is not field-specific but field-wide.

Source-level leakage is well understood in adjacent domains. Roberts et al. [@roberts2017crossvalidation] demonstrated that standard random cross-validation produces severely inflated performance estimates when data has spatial, temporal, or hierarchical structure, and proposed structured cross-validation strategies as remedies. In medical imaging, Whalen et al. [@whalen2022navigating] documented how patient-level splits are essential when multiple images come from the same patient; failure to enforce this leads to models that recognise patients rather than pathologies. The parallel to our setting is exact: replacing "patient" with "artist" and "images" with "clips" yields an identical formulation.

The pattern is consistent across fields: wherever data points share a common origin, evaluation must split at the source level. Music genre classification exhibits all three forms simultaneously --- the same recording (clip-level), the same performance (track-level), and the same production signature (artist-level). FMA ships the artist-level fix in its metadata, but using it is optional and disclosing whether it was used is rarer still.

Flexer and Schnitzer [@flexer2006closer] provided early evidence of artist-filter effects in music classification, showing that removing the artist confound significantly reduced classification accuracy on GTZAN. Sturm [@sturm2014simple] went further, demonstrating that some MIR systems functioned as "horses" --- exploiting dataset-specific artifacts rather than learning musically meaningful features. Our three-rung ladder provides the first systematic quantification of this effect on FMA-small across multiple architectures.

The NeurIPS reproducibility checklist [@pineau2021neurips] now asks about train/test contamination, representing an institutional attempt to address leakage. However, the checklist does not specifically ask about hierarchical source-level splitting --- the distinction between "are your train and test sets disjoint?" (yes, by construction) and "are they disjoint at the source level?" (often not). This gap means that a paper can truthfully complete the checklist while still suffering from the exact form of leakage we measure.

## I. Evaluation Protocol Reporting Standards

Institutional efforts to improve evaluation transparency have made progress but leave significant gaps.

The **NeurIPS reproducibility checklist** [@pineau2021neurips] asks authors to document dataset splits, specify whether existing splits were used, and describe any preprocessing. These requirements are a step forward, but they do not enforce hierarchical splitting. A paper can truthfully report "we used an 80/10/10 random split" without mentioning whether the randomisation operated at the clip, track, or artist level --- and without triggering a checklist violation.

**ISMIR community practices** have historically been more attentive to artist filtering than general ML venues, partly because the MIREX evaluation campaigns standardise both tasks and data. However, MIREX standardises by providing fixed splits, not by requiring papers to disclose their split methodology when using their own data partitioning. Papers published at ISMIR that use FMA-small with custom splits are not required to state whether they artist-filter.

**MARBLE** [@yuan2024marble] and **HEAR** [@turian2022hear] represent the benchmark approach: fix the split so that all papers using the benchmark are comparable by construction. This works within the benchmark but does not help when a paper uses a standard dataset (like FMA-small) outside the benchmark framework.

The gap can be stated precisely: no major venue's submission checklist asks, "Does your train/test split control for source-level identity (artist, patient, location)?" Until it does, the protocol variable we measure --- worth 30+ points on FMA-small --- will remain undisclosed by default.

## J. Training Techniques and Regularisation

Several training techniques are relevant to our pipeline and merit brief discussion for reproducibility.

**SpecAugment** [@park2019specaugment] applies random frequency and time masks to mel spectrograms during training, functioning as a regulariser that prevents the model from relying on specific spectral regions or time intervals. Originally developed for automatic speech recognition, it has become standard in audio classification.

**Mixup** [@zhang2018mixup] interpolates between pairs of training examples and their labels, encouraging models to behave linearly between training samples. In our experiments, mixup with alpha=0.3 consistently improves generalisation for probe models on FMA-small.

**Label smoothing** [@szegedy2016rethinking] softens the hard 0/1 targets by redistributing a small fraction of probability mass to non-target classes. With 8 balanced genres, label smoothing of 0.1 allocates approximately 0.014 to each non-target class, penalising overconfident predictions.

**AdamW** [@loshchilov2019adamw] decouples weight decay from the gradient update, providing more effective regularisation than L2 regularisation in the Adam optimiser. Combined with cosine annealing of the learning rate, this forms our default optimisation setup.

**Focal loss** [@lin2017focal] down-weights well-classified examples to concentrate training on hard cases. While designed for class-imbalanced detection tasks, we evaluate it as an alternative to cross-entropy for genres that are systematically hard to classify (Folk, Experimental).

# 3. Methodology

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

**MusicFM embeddings.** MusicFM [@li2024musicfm] extends the wav2vec 2.0 framework with music-aware pretraining objectives. We extract all layer outputs, mean-pooled over the time dimension, producing a (13, 768) tensor per episode. Audio is decoded at 16 kHz.

**Wav2Vec 2.0 embeddings.** Wav2Vec 2.0-base [@baevski2020wav2vec] is a 12-layer transformer pretrained via contrastive learning on 960 hours of LibriSpeech. We extract all 13 layer outputs (12 transformer + 1 convolutional), mean-pooled over time, producing a (13, 768) tensor per episode. Audio is decoded at 16 kHz.

**HuBERT embeddings.** HuBERT-base [@hsu2021hubert] uses the same architecture as Wav2Vec 2.0 but is pretrained with offline-clustered pseudo-labels on 960 hours of LibriSpeech. Extraction is identical: (13, 768) per episode at 16 kHz.

**BEATs embeddings.** BEATs [@chen2023beats] combines audio tokenisation with iterative self-supervised training, achieving state-of-the-art on AudioSet. We extract all layer outputs, mean-pooled over time, producing (13, 768) per episode at 16 kHz.

**Whisper encoder embeddings.** Whisper-small [@radford2023whisper] is trained on 680,000 hours of web-scraped speech for automatic speech recognition. We extract only the encoder's hidden states (discarding the decoder), mean-pooled over time, producing (13, 768) per episode at 16 kHz. Whisper has zero music-specific pretraining, serving as a control for whether leakage is architecture-dependent or data-dependent.

**BYOL-A embeddings.** BYOL-A [@niizumi2023byola] produces a single 2048-dimensional embedding per episode at 16 kHz.

**PANNs CNN14 embeddings.** PANNs CNN14 [@kong2020panns] produces 2048-dimensional penultimate-layer embeddings per episode at 32 kHz.

**CLAP embeddings.** CLAP [@wu2023clap] learns aligned audio-text representations via contrastive pretraining on audio-caption pairs. We extract the audio encoder's output, a single 512-dimensional vector per episode at 48 kHz. Only the audio encoder is used during probing; the text encoder plays no role.

**EfficientAT embeddings.** EfficientAT MN10 [@schmid2023efficientat] is a lightweight CNN distilled from a large audio transformer. We extract penultimate-layer features, producing a 960-dimensional vector per episode at 32 kHz.

Table II summarises all backbones.

```{=latex}
\begin{table*}[t]
\centering
\caption{Feature Extraction Summary: All Backbones}
\label{tab:backbones}
\footnotesize
\begin{tabular}{lccccc}
\toprule
\textbf{Backbone} & \textbf{Type} & \textbf{Params} & \textbf{Embed Shape} & \textbf{SR} & \textbf{Pretraining} \\
\midrule
Log-mel spectrogram & Hand-crafted & --- & (128, 130) & 22k & --- \\
MERT-v1-95M & Music SSL & 95M & (13, 768) & 24k & 20k hr music \\
MERT-v1-330M & Music SSL & 330M & (25, 1024) & 24k & 20k hr music \\
MusicFM & Music SSL & $\sim$95M & (13, 768) & 16k & Music \\
Wav2Vec 2.0-base & Speech SSL & 95M & (13, 768) & 16k & 960 hr speech \\
HuBERT-base & Speech SSL & 95M & (13, 768) & 16k & 960 hr speech \\
BEATs & Audio SSL & 90M & (13, 768) & 16k & AudioSet \\
Whisper-small enc. & Supervised ASR & 242M & (13, 768) & 16k & 680k hr speech \\
BYOL-A & Audio SSL & 5M & (2048,) & 16k & AudioSet \\
PANNs CNN14 & Supervised & 80M & (2048,) & 32k & AudioSet \\
CLAP & Audio-Language & 150M & (512,) & 48k & Audio+text \\
EfficientAT MN10 & Distilled CNN & 5M & (960,) & 32k & AudioSet (distill) \\
\bottomrule
\end{tabular}
\end{table*}
```

## E. Model Families

We train 14 architectures spanning four families:

**From-scratch spectrogram models** consume 128-bin log-mel spectrograms and train all parameters from scratch: GenreCNN (4-block CNN, ~260K params) [@choi2017transfer], DenseNet121 (ImageNet-pretrained, adapted to single-channel input) [@huang2017densenet; @bian2019audio].

**Music-specific SSL probes** train only a lightweight head on cached embeddings from backbones pretrained specifically on music audio:

- *MERT-95M learned weighted-sum probe* [@pasad2021layer]: softmax-weighted sum over all layers, followed by a two-layer MLP.
- *MERT-95M flat probe*: concatenates all 13 layers into a 9,984-dim vector, followed by a two-layer MLP with LayerNorm.
- *MERT-95M deep probe* [@yang2021superb]: groups layers into multi-scale aggregations with residual blocks.
- *MERT-95M attention probe* [@safari2020self]: multi-head self-attention over the 13 layers with learnable positional embeddings.
- *MERT-330M probe*: learned weighted-sum with the 330M backbone.
- *MERT-330M flat probe*: 13-layer concatenation with the 330M backbone.
- *MERT-330M deep probe*: multi-scale aggregation with the 330M backbone.
- *MusicFM probe* [@li2024musicfm]: learned weighted-sum over MusicFM's layer outputs, followed by a two-layer MLP.

**General audio and speech SSL probes** train only a lightweight head on cached embeddings from backbones with no music-specific pretraining objective:

- *Wav2Vec 2.0 probe* [@baevski2020wav2vec]: learned weighted-sum over 13 layers, followed by a two-layer MLP.
- *HuBERT probe* [@hsu2021hubert]: same architecture as Wav2Vec 2.0 probe, on HuBERT embeddings.
- *BEATs probe* [@chen2023beats]: learned weighted-sum over layer outputs, followed by a two-layer MLP.
- *Whisper probe* [@radford2023whisper]: learned weighted-sum over encoder hidden states, followed by a two-layer MLP.
- *BYOL-A + DNN probe* [@niizumi2023byola]: LayerNorm followed by a variable-depth MLP on the 2048-dim BYOL-A embedding.
- *PANNs CNN14 probe* [@kong2020panns]: same architecture as BYOL-A probe, on PANNs 2048-dim embeddings.
- *CLAP probe* [@wu2023clap]: LayerNorm followed by a two-layer MLP on the 512-dim CLAP audio embedding.
- *EfficientAT probe* [@schmid2023efficientat]: LayerNorm followed by a two-layer MLP on the 960-dim EfficientAT embedding.

**Non-neural baseline.** XGBoost [@chen2016xgboost] on flattened MERT embeddings.

Separating music-specific from general-purpose backbones lets us test whether music-aware pretraining provides any advantage under honest evaluation --- or whether the apparent advantage in published results reflects protocol choice rather than representation quality.

## F. Training Configuration

All neural models use AdamW [@loshchilov2019adamw] with cosine annealing. Spectrogram models apply SpecAugment [@park2019specaugment] (2 frequency masks up to 16 bins, 2 time masks up to 20 frames). Label smoothing (0.1) [@szegedy2016rethinking] and mixup augmentation (alpha 0.3) [@zhang2018mixup] are used for most runs. Early stopping with patience 6--15 epochs prevents overfitting. Focal loss [@lin2017focal] is evaluated as an alternative criterion.

## G. Reproducibility Statement

All experiments run on a single Apple M-series laptop (MPS backend, 16 GB RAM). Embedding extraction takes 30--60 minutes per backbone; probe training takes 1--3 minutes per run. Total compute for the full study is under 40 GPU-hours equivalent. All code, configurations, cached embeddings, and evaluation scripts will be released upon publication. Every number in the results section is mechanically generated from run artifacts by `scripts/update_results.py`; no number is hand-edited. The embedding cache ensures exact reproducibility of downstream experiments: given the same cached features and random seed, training produces identical weights on the same hardware.

# 4. Experiments

We conducted 62 evaluated training runs: 37 under the artist-filtered split, 12 under random-track, and 13 under clip-random, spanning 14 unique architectures across 4 model families. Every result is mechanically generated from run artifacts by dedicated scripts; no number in the results tables is hand-edited.

# 5. Results

## A. Best Model Per Architecture

Table III presents the best run per architecture (14 unique architectures), ranked by accuracy under the artist-filtered, track-level protocol.

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
1 & CLAP probe \cite{wu2023clap} & 59.9 & 59.4 \\
2 & Ensemble (uniform) \cite{kittler1998combining} & 58.5 & 58.2 \\
3 & MERT-95M flat probe \cite{li2023mert} & 58.2 & 57.8 \\
4 & BYOL-A + DNN probe \cite{niizumi2023byola} & 57.4 & 57.0 \\
5 & PANNs CNN14 probe \cite{kong2020panns} & 57.4 & 56.6 \\
6 & MERT-330M probe \cite{li2023mert} & 56.6 & 56.5 \\
7 & MERT-330M flat probe \cite{li2023mert} & 56.5 & 55.9 \\
8 & MERT-95M probe \cite{pasad2021layer} & 56.4 & 55.7 \\
9 & MERT-95M deep probe \cite{yang2021superb} & 56.0 & 55.9 \\
10 & MERT-95M attn probe \cite{safari2020self} & 55.2 & 54.4 \\
11 & GenreCNN (log-mel) \cite{choi2017transfer} & 55.1 & 54.2 \\
12 & MERT-330M deep probe \cite{yang2021superb} & 54.9 & 55.5 \\
13 & DenseNet121 \cite{huang2017densenet} & 53.9 & 52.2 \\
14 & XGBoost on MERT \cite{chen2016xgboost} & 52.5 & 51.4 \\
\bottomrule
\end{tabular}
\end{table}
```

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{architecture_bars.pdf}
\caption{Best track-level accuracy per architecture under the artist-filtered protocol. The CLAP probe (59.9\%) leads, followed by a cluster of MERT/BYOL-A/PANNs models at 57--58\%.}
\label{fig:archbars}
\end{figure}
```

The best single model is the CLAP probe (59.9\%), whose contrastive audio-language pretraining on diverse audio-text pairs produces representations that transfer effectively to genre classification even through a frozen probe. The previous best, a MERT-95M flat probe (58.2\%), concatenates all 13 transformer layers into a 9,984-dimensional feature vector. A uniform-weight ensemble of per-architecture bests (excluding CLAP) reaches 58.5\%; including CLAP would likely improve it further. Notably, MERT-330M does not outperform MERT-95M, and learned ensemble weights lose to fixed uniform averaging --- both symptoms of overfitting at this data scale (6,400 training tracks).

## B. Complete Run Inventory

Table IV presents all 37 artist-filtered runs across 14 architectures, including hyperparameter variants and seed ablations.

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
\caption{Complete Inventory: All 37 Artist-Filtered Runs (Track-Level Accuracy and Macro-F1, \%)}
\label{tab:allruns}
\scriptsize
\begin{tabular}{rllccccccl}
\toprule
\textbf{\#} & \textbf{Run} & \textbf{Architecture} & \textbf{Acc.} & \textbf{F1} & \textbf{LR} & \textbf{BS} & \textbf{Ep.} & \textbf{Drop.} & \textbf{Ref.} \\
\midrule
1 & clap\_probe\_v1 & CLAP probe & 59.9 & 59.4 & 3e-4 & 128 & 50 & 0.4 & \cite{wu2023clap} \\
2 & ensemble\_uniform & Ensemble (uniform avg) & 58.5 & 58.2 & --- & --- & --- & --- & \cite{kittler1998combining} \\
3 & ensemble\_weighted & Ensemble (weighted avg) & 58.2 & 58.1 & --- & --- & --- & --- & \cite{kittler1998combining} \\
4 & mert\_flat\_probe\_v1 & MERT-95M flat probe & 58.2 & 57.8 & 3e-4 & 128 & 50 & 0.5 & \cite{li2023mert} \\
5 & mert\_flat\_probe\_v2 & MERT-95M flat probe & 57.8 & 57.3 & 1e-4 & 64 & 50 & 0.6 & \cite{li2023mert} \\
6 & byola\_probe\_v1 & BYOL-A + DNN probe & 57.4 & 57.0 & 3e-4 & 128 & 50 & 0.4 & \cite{niizumi2023byola} \\
7 & mert\_flat\_diverse\_s3 & MERT-95M flat probe & 57.4 & 56.5 & 2e-4 & 128 & 50 & 0.6 & \cite{li2023mert} \\
8 & panns\_probe\_v1 & PANNs CNN14 probe & 57.4 & 56.6 & 3e-4 & 128 & 50 & 0.4 & \cite{kong2020panns} \\
9 & mert\_flat\_diverse\_s4 & MERT-95M flat probe & 56.6 & 56.1 & 2e-4 & 128 & 50 & 0.5 & \cite{li2023mert} \\
10 & mert\_large\_probe\_v1 & MERT-330M probe & 56.6 & 56.5 & 5e-4 & 128 & 50 & 0.4 & \cite{li2023mert} \\
11 & mert\_flat\_diverse\_s1 & MERT-95M flat probe & 56.5 & 55.8 & 3e-4 & 128 & 50 & 0.5 & \cite{li2023mert} \\
12 & mert\_large\_flat\_v1 & MERT-330M flat probe & 56.5 & 55.9 & 1e-4 & 64 & 50 & 0.6 & \cite{li2023mert} \\
13 & mert\_probe\_tuned & MERT-95M probe (learned wt.) & 56.4 & 55.7 & 2e-4 & 64 & 60 & 0.5 & \cite{pasad2021layer} \\
14 & mert\_large\_probe\_v3 & MERT-330M probe & 56.2 & 56.2 & 1e-4 & 128 & 50 & 0.3 & \cite{li2023mert} \\
15 & mert\_probe\_wide & MERT-95M probe (learned wt.) & 56.1 & 55.3 & 5e-4 & 128 & 40 & 0.4 & \cite{pasad2021layer} \\
16 & mert\_deep\_probe\_v2 & MERT-95M deep probe & 56.0 & 55.9 & 1e-4 & 64 & 60 & 0.5 & \cite{yang2021superb} \\
17 & mert\_flat\_diverse\_s6 & MERT-95M flat probe & 56.0 & 55.4 & 2.5e-4 & 128 & 50 & 0.55 & \cite{li2023mert} \\
18 & mert\_probe\_baseline & MERT-95M probe (learned wt.) & 55.9 & 54.9 & 1e-3 & 256 & 40 & 0.3 & \cite{pasad2021layer} \\
19 & mert\_flat\_diverse\_s5 & MERT-95M flat probe & 55.8 & 55.6 & 4e-4 & 128 & 50 & 0.4 & \cite{li2023mert} \\
20 & mert\_deep\_probe\_v1 & MERT-95M deep probe & 55.5 & 55.0 & 3e-4 & 128 & 60 & 0.4 & \cite{yang2021superb} \\
21 & mert\_attn\_probe\_v2 & MERT-95M attention probe & 55.2 & 54.4 & 5e-4 & 128 & 60 & 0.4 & \cite{safari2020self} \\
22 & mert\_probe\_seed2 & MERT-95M probe (seed 2) & 55.2 & 55.0 & 5e-4 & 128 & 40 & 0.4 & \cite{pasad2021layer} \\
23 & cnn\_improved & GenreCNN (log-mel) & 55.1 & 54.2 & 3e-4 & 32 & 60 & --- & \cite{choi2017transfer} \\
24 & mert\_large\_probe\_v2 & MERT-330M probe & 55.0 & 55.0 & 3e-4 & 64 & 50 & 0.5 & \cite{li2023mert} \\
25 & mert\_probe\_deep & MERT-95M probe (deep head) & 55.0 & 54.4 & 1e-4 & 256 & 80 & 0.5 & \cite{pasad2021layer} \\
26 & mert\_large\_deep\_v1 & MERT-330M deep probe & 54.9 & 55.5 & 3e-4 & 64 & 50 & 0.4 & \cite{yang2021superb} \\
27 & mert\_attn\_probe\_v1 & MERT-95M attention probe & 54.6 & 53.6 & 3e-4 & 128 & 60 & 0.3 & \cite{safari2020self} \\
28 & mert\_probe\_seed3 & MERT-95M probe (seed 3) & 54.5 & 54.4 & 5e-4 & 128 & 40 & 0.4 & \cite{pasad2021layer} \\
29 & mert\_flat\_diverse\_s2 & MERT-95M flat probe & 54.4 & 53.7 & 3e-4 & 128 & 50 & 0.5 & \cite{li2023mert} \\
30 & densenet\_v1 & DenseNet121 & 53.9 & 52.2 & 3e-4 & 32 & 60 & 0.3 & \cite{huang2017densenet} \\
31 & mert\_attn\_probe\_v3 & MERT-95M attention probe & 53.5 & 52.8 & 1e-4 & 64 & 60 & 0.2 & \cite{safari2020self} \\
32 & mert\_probe\_seed1 & MERT-95M probe (seed 1) & 53.5 & 52.9 & 5e-4 & 128 & 40 & 0.4 & \cite{pasad2021layer} \\
33 & cnn\_tuned\_lr & GenreCNN (log-mel) & 52.9 & 52.2 & 3e-4 & 32 & 35 & --- & \cite{choi2017transfer} \\
34 & mert\_probe\_lastlayer & MERT-95M probe (last layer) & 52.8 & 51.8 & 1e-3 & 256 & 40 & 0.3 & \cite{pasad2021layer} \\
35 & xgboost\_mert & XGBoost on MERT & 52.5 & 51.4 & --- & 256 & --- & --- & \cite{chen2016xgboost} \\
36 & cnn\_augmented & GenreCNN (log-mel) & 51.4 & 50.4 & 1e-3 & 64 & 30 & --- & \cite{choi2017transfer} \\
37 & cnn\_baseline & GenreCNN (log-mel) & 47.1 & 45.0 & 1e-3 & 64 & 25 & --- & \cite{choi2017transfer} \\
\bottomrule
\end{tabular}
\end{table*}
```

## C. Protocol Leakage Ladder

Table V combines the three protocols for the 11 architectures tested on all three rungs.

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
MERT-330M Flat Probe \cite{li2023mert} & 56.5 & 71.2 & 91.9 & +35.4 \\
MERT-330M Deep Probe \cite{yang2021superb} & 54.9 & 69.0 & 96.1 & +41.2 \\
CNN (log-mel spectrogram) \cite{choi2017transfer} & 55.1 & 62.9 & 69.2 & +14.1 \\
BYOL-A + DNN \cite{niizumi2023byola} & 57.4 & 69.9 & 84.5 & +27.2 \\
PANNs CNN14 \cite{kong2020panns} & 57.4 & 67.2 & 73.4 & +16.0 \\
CLAP \cite{wu2023clap} & 59.9 & 71.8 & 78.6 & +18.7 \\
\midrule
\textbf{Mean (11 verified)} & \textbf{56.6} & \textbf{69.1} & \textbf{87.4} & \textbf{+30.8} \\
\bottomrule
\end{tabular}
\end{table*}
```

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{leakage_bars.pdf}
\caption{Protocol leakage across all 11 verified architectures. Blue bars show artist-filtered accuracy; orange bars show random-track; red bars show clip-random. Red annotations above each bar show the total inflation in percentage points. The MERT-330M deep probe inflates by +41~pp --- from 54.9\% to 96.1\%.}
\label{fig:leakage}
\end{figure*}
```

Decomposed: dropping the artist filter alone accounts for +12.7 pp, and leaking clips on top adds +19.4 pp. The effect is systematic and architecture-independent.

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

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{confusion_diff.pdf}
\caption{Confusion matrix shift under protocol leakage, averaged across 8 architectures. (a) Artist-filtered: off-diagonal mass in Folk$\rightarrow$Experimental (22.8\%), Instrumental$\rightarrow$Folk (28.6\%), and Pop$\rightarrow$Hip-Hop (16.5\%) reflects genuine genre ambiguity. (b) Clip-random: confusion collapses as models recognise recordings. (c) Difference: diagonal gains (recording recognition) come at the expense of off-diagonal confusion (genre ambiguity). Folk gains +49.4~pp on-diagonal; the largest off-diagonal reduction is Instrumental$\rightarrow$Folk ($-25.9$~pp).}
\label{fig:confusion_diff}
\end{figure*}
```

The confusion matrix shift (Figure \ref{fig:confusion_diff}) reveals *what* leakage fixes. Under honest evaluation, the dominant confusions are between acoustically similar genres: Instrumental$\rightarrow$Folk (28.6\%), Folk$\rightarrow$Experimental (22.8\%), and Folk$\rightarrow$Pop (23.1\%). Under clip-random evaluation, these confusions nearly vanish --- not because the model learned genre boundaries, but because it learned recording identities. The off-diagonal reductions are not a sign of better generalisation; they are a sign of memorisation replacing classification.

## E. Cross-Architecture Leakage Consistency

To test whether the leakage pattern is architecture-independent, we compute the Spearman rank correlation between per-genre inflation vectors across all architecture pairs. Each architecture produces an 8-element vector of per-genre inflation values (clip-random F1 minus artist-filtered F1). If the genre vulnerability ordering is consistent across architectures, the pairwise rank correlations should be high.

Across the 11 verified architectures, the genre vulnerability ordering is remarkably consistent: Folk, Instrumental, and Experimental inflate the most, while Hip-Hop and Electronic inflate the least. This stability confirms that the inflation pattern is a property of the dataset's genre structure and the evaluation protocol, not of any particular learned representation.

The stability of the genre vulnerability ordering has a practical implication: if Folk and Instrumental are always the most inflated genres, then a paper's reported per-genre accuracy profile can serve as a soft diagnostic for protocol choice. A model that achieves 85% on Folk under an unstated protocol is more likely using a permissive split than one that achieves 85% on Hip-Hop.

## F. Audio-Language Models: CLAP

CLAP [@wu2023clap] approaches audio understanding through contrastive language-audio pretraining, learning a joint embedding space where audio and text descriptions are aligned. Its 512-dimensional audio embeddings encode semantic content at a higher level of abstraction than purely acoustic models. Under the artist-filtered protocol, CLAP achieves 59.9\% track-level accuracy --- the best single model, surpassing MERT-95M (58.2\%) and even the prior uniform ensemble (58.5\%). This demonstrates that audio-language pretraining on diverse audio-text pairs produces representations that transfer remarkably well to music genre classification.

The CLAP leakage pattern (+18.7 pp inflation) is notably lower than the MERT family (+32--41 pp) but higher than the from-scratch CNN (+14.1 pp). This intermediate position is consistent with CLAP's pretraining: its contrastive objective encourages semantic invariance across acoustic variations, which may reduce recording-specific memorisation compared to reconstruction-based objectives (MERT) while still capturing more fine-grained features than a shallow CNN.

A natural extension is to test speech-only models (Wav2Vec 2.0, HuBERT, Whisper) --- which have never encountered music during pretraining --- to further characterise how pretraining domain affects leakage magnitude.

## G. Against Published Literature

Our best artist-filtered model reaches 59.9\%. The only published number carrying an artist-filtered guarantee is 54.8\% (Park et al.'s re-run of FMA's baseline code [@park2018representation]), a margin of +5.1 pp. We deliberately do not report a "beats N of M" count: ranking against numbers whose protocol is unknown would treat incomparable quantities as comparable.

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
\caption{Confusion matrix for MERT-95M flat probe (58.2\%) under the artist-filtered protocol. Hip-Hop (86/100) and Electronic (81/100) are well-separated; Folk (38/100) scatters into Experimental, Pop, and Instrumental.}
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

The architecture set confirms the fundamental finding: published FMA-small results remain incomparable without protocol disclosure. The inflation pattern holds across from-scratch CNNs, music-specific SSL (MERT-95M, MERT-330M), general audio SSL (BYOL-A), supervised pretraining (PANNs), audio-language models (CLAP), and non-neural baselines (XGBoost). Future work extending this to speech-only SSL (Wav2Vec 2.0, HuBERT) would further characterise how pretraining domain affects leakage magnitude.

## H. Calibration Across Protocols

A model may achieve high accuracy under a permissive protocol, but does it *know* it is right? We measure Expected Calibration Error (ECE, 15 bins), Brier score, and optimal temperature across all 46 runs with cached episode outputs, grouped by protocol.

```{=latex}
\begin{table}[t]
\centering
\caption{Calibration Metrics by Evaluation Protocol (Mean $\pm$ SD)}
\label{tab:calibration}
\footnotesize
\begin{tabular}{lcccc}
\toprule
\textbf{Protocol} & \textbf{ECE} & \textbf{Brier} & \textbf{Overconf.} & \textbf{$T^*$} \\
\midrule
Artist-filtered (26) & .060$\pm$.016 & .587$\pm$.012 & +0.9\,pp & 1.20$\pm$0.16 \\
Random-track (10)    & .067$\pm$.029 & .431$\pm$.030 & $-$4.1\,pp & 1.18$\pm$0.24 \\
Clip-random (10)     & .085$\pm$.045 & .186$\pm$.120 & $-$8.4\,pp & 0.76$\pm$0.14 \\
\bottomrule
\end{tabular}
\end{table}
```

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{calibration_by_protocol.pdf}
\caption{Calibration comparison across evaluation protocols. Under the artist-filtered protocol, models are roughly calibrated (overconfidence $\approx$ +0.9~pp). Under clip-random evaluation, all 10 models are \emph{underconfident}: their inflated accuracy outpaces the softmax probabilities. Temperature scaling with $T^* < 1$ is needed to sharpen predictions to match the inflated accuracy.}
\label{fig:calibration}
\end{figure}
```

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{reliability_diagrams.pdf}
\caption{Reliability diagrams for representative models under each protocol. Under honest evaluation, the calibration curve tracks the diagonal. Under clip-random evaluation, the curve falls below the diagonal across all confidence bins --- the model is less confident than it is accurate.}
\label{fig:reliability}
\end{figure}
```

The key finding is counterintuitive: **leakage creates underconfidence, not overconfidence.** Under the artist-filtered protocol, models are approximately calibrated (mean overconfidence +0.9 pp). Under clip-random evaluation, all 10 models are underconfident (mean $-$8.4 pp): the inflated accuracy exceeds the model's own confidence. This happens because the softmax probabilities are calibrated to the difficulty of the genre classification task, but the permissive protocol converts the task from genre classification to recording recognition --- which the model solves correctly more often than its softmax entropy suggests.

Temperature scaling confirms this: artist-filtered models need $T^* > 1$ (softening, reducing overconfidence), while clip-random models need $T^* < 1$ (sharpening, increasing confidence to match inflated accuracy). The optimal temperature is itself a diagnostic for protocol leakage: $T^* \ll 1$ signals that the model achieves accuracy it did not learn to expect.

## I. Representation Similarity (CKA)

To understand whether backbone diversity drives ensemble gain, we compute linear Centered Kernel Alignment (CKA) [@kornblith2019similarity] between the mean-pooled test-set representations of all five backbone families (800 tracks, aligned by track ID).

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{cka_combined.pdf}
\caption{(a) Inter-model CKA across 5 backbones: MERT-95M and MERT-330M share a high CKA (0.848); CLAP and PANNs show surprising similarity (0.724), likely reflecting their shared AudioSet-scale pretraining. (b) MERT-95M layer-wise CKA: adjacent layers are highly similar ($>$ 0.95), with a gradual phase transition around layer 8--9 where representations shift from acoustic to semantic.}
\label{fig:cka}
\end{figure*}
```

The inter-model CKA matrix reveals that MERT-95M and MERT-330M encode nearly identical information (CKA = 0.848), explaining why MERT-330M does not outperform MERT-95M under honest evaluation: the larger model's additional capacity does not add complementary features. CLAP and PANNs show surprisingly high similarity (CKA = 0.724), likely because both encode broad acoustic categories --- PANNs via supervised classification on AudioSet's 527 classes, CLAP via contrastive alignment with audio-text pairs spanning a similar domain. BYOL-A occupies an intermediate position, moderately similar to all backbones (0.505--0.576).

The most diverse pair is PANNs $\leftrightarrow$ MERT-330M (CKA = 0.469), followed by MERT-330M $\leftrightarrow$ CLAP (0.507). This diversity explains why a uniform ensemble over architecturally diverse backbones outperforms any single model: the component models make different errors because they encode different information.

Within MERT-95M, the $13\times13$ layer-wise CKA heatmap shows a smooth gradient from early to late layers, with adjacent layers highly redundant (CKA $> 0.95$). A phase transition occurs around layers 8--9, where the CKA between early layers (0--4) and late layers (10--12) drops below 0.80. This is consistent with the speech SSL literature's finding that early transformer layers encode acoustic features while later layers encode more abstract, task-relevant representations [@pasad2021layer].

## J. Error Diversity and Ensemble Complementarity

We compute pairwise Cohen's kappa and Yule's Q between the correctness vectors of 8 representative models (one best-per-architecture) to quantify error diversity.

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{error_diversity_heatmap.pdf}
\caption{(a) Cohen's kappa measures agreement in correct/incorrect predictions. Within-MERT pairs agree strongly ($\kappa = 0.775 \pm 0.035$); cross-family pairs are more diverse ($\kappa = 0.630 \pm 0.034$). (b) Yule's Q confirms the same pattern with the error correlation measure.}
\label{fig:error_diversity}
\end{figure*}
```

Within-MERT variants show high error agreement ($\kappa = 0.775 \pm 0.035$): they share the same backbone, so the same tracks tend to confuse them. Cross-family pairs are substantially more diverse ($\kappa = 0.630 \pm 0.034$). The most diverse pairs involve architecturally distinct backbones (CNN, PANNs, CLAP vs. MERT variants), confirming that ensemble diversity requires crossing model family boundaries.

Among the top 4 models, 216 tracks (27.0\%) are misclassified by all four --- these represent a hard core of genuinely ambiguous tracks that no single-backbone probe can solve. Another 337 tracks (42.1\%) are correctly classified by all four. The remaining 247 tracks (30.9\%) are where ensemble diversity provides value: at least one model gets it right while others do not.

This quantifies why uniform ensembling works and learned fusion does not. The models' errors are moderately correlated (mean $\kappa = 0.677$) --- diverse enough for averaging to help, but not diverse enough for a learned combiner to find exploitable patterns at this data scale.

## K. Temporal Stability of Genre Predictions

Each 30-second FMA track yields 10 three-second episodes. We analyse how consistently models classify episodes from the same track.

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{episode_agreement.pdf}
\caption{(a) Distribution of inter-episode agreement (fraction of episodes per track that agree with the majority vote). 39.5\% of tracks achieve unanimous agreement across all 10 episodes. (b) Track-level accuracy conditioned on agreement: tracks with $\geq$ 90\% agreement reach 72.1\% accuracy; tracks with $<$ 50\% agreement drop to 35.9\%.}
\label{fig:agreement}
\end{figure}
```

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{temporal_accuracy.pdf}
\caption{Episode-level accuracy by temporal position for 6 representative models. Accuracy is roughly flat across positions 0--9 (first 3\,s to last 3\,s), with no model showing a consistent temporal trend. This suggests that genre-discriminative information is distributed throughout the track, not concentrated at the beginning or end.}
\label{fig:temporal}
\end{figure}
```

Mean episode agreement is 80.3\% $\pm$ 20.5\% --- higher than chance but far from unanimous. Tracks with high agreement ($\geq$ 90\%) are classified with 72.1\% accuracy, while low-agreement tracks ($<$ 50\%) achieve only 35.9\%. This 36-pp gap shows that inter-episode agreement is a strong confidence signal: a model that is consistent across temporal positions is more likely to be correct.

Per-genre agreement varies systematically: Hip-Hop (87.5\%) and International (88.6\%) achieve the highest temporal consistency, while Folk (71.6\%) and Pop (70.6\%) are the most temporally variable. The same genres that are hardest to classify are also hardest to classify *consistently* --- suggesting that the difficulty is not in any particular 3-second segment but in the genre's intrinsic acoustic diffuseness.

Accuracy is roughly flat across episode positions (range: 52.0\%--54.4\% for the best model), with no model showing a consistent improvement or degradation from the beginning to the end of a track. This flatness suggests that 3-second episodes are individually informative regardless of their position, and that the track-level aggregation we use (mean-pooling softmax probabilities) is not systematically harmed by any particular temporal position.

## L. Statistical Rigour

All accuracy numbers in this study are accompanied by bootstrap 95\% confidence intervals (10,000 resamples). The best single model (CLAP probe, 59.9\%) has CI [56.4\%, 63.2\%]. McNemar's test between all pairs of the top 8 artist-filtered models finds no significant difference ($p > 0.05$ for all 28 pairs): the top cluster of models is statistically indistinguishable.

```{=latex}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{bootstrap_forest.pdf}
\caption{Bootstrap 95\% confidence intervals for all 26 artist-filtered runs. The top cluster (7 runs above 56\%) has widely overlapping CIs, confirming that differences within the cluster are not statistically significant.}
\label{fig:forest}
\end{figure}
```

Per-genre CIs for the best model reveal the precision ceiling: Hip-Hop has the tightest CI [79.0\%, 93.0\%], while the hardest genres have wide CIs (Folk: [29.0\%, 48.0\%], Pop: [28.0\%, 46.0\%]). With only 100 test tracks per genre, detecting a 5-pp improvement in Folk with 95\% confidence would require a model to improve from 38\% to 43\% --- a bar that is meaningful but attainable.

## M. Layer-Wise Leakage Fingerprinting

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{layer_leakage.pdf}
\caption{(a) Single-layer MERT-95M probe accuracy under three protocols. Genre-discriminative accuracy (blue) peaks at layer~7 (53.0\%) and declines in later layers. Clip-random accuracy (red) peaks at layers~3--5 (87--89\%) and remains high across all depths. The shaded region shows the inflation gap. (b) Per-layer inflation. Despite the inverted-U shape in honest accuracy, inflation is remarkably uniform (33.5--41.1~pp across all 13 layers), indicating that recording identity is encoded at every transformer depth, not just in early acoustic layers.}
\label{fig:layer_leakage}
\end{figure*}
```

To localise where recording identity versus genre information resides in a pretrained transformer, we train 13 single-layer MERT-95M probes --- one per transformer layer --- under each of the three protocols (39 runs total).

Genre-discriminative accuracy follows an inverted-U pattern: it rises from layer~0 (40.9\%) through layer~7 (53.0\%), then declines to layer~12 (47.0\%). This pattern is consistent with prior findings that genre-relevant features are distributed across transformer depth, with middle layers encoding the richest semantic representations.

The key finding is that inflation is *nearly uniform* across layers: it ranges from +33.5~pp (layer~7) to +41.1~pp (layer~2), with a mean of +37.4~pp. If recording identity were concentrated in early acoustic layers (which preserve fine-grained spectral detail), we would expect inflation to decay with depth. Instead, the uniformity indicates that recording identity pervades the entire representation --- the model encodes enough recording-specific information at every layer for a simple probe to exploit it under permissive evaluation.

The slight inflation minimum at layer~7 --- the same layer with peak honest accuracy --- suggests a subtle tradeoff: the layer most useful for genre classification is also the layer that encodes proportionally more genre (vs. recording) information. But the effect is small: even at layer~7, inflation is +33.5~pp, indicating that recording identity dominates genre information at every depth.

## N. Data Efficiency and Protocol Sensitivity

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{data_efficiency.pdf}
\caption{(a) Track-level accuracy vs training data fraction for CLAP and MERT-95M flat probes under artist-filtered (solid) and clip-random (dashed) evaluation. Under honest evaluation, both models plateau early; under leakage, accuracy scales steadily with data. (b) Inflation (clip-random $-$ artist-filtered) vs training data fraction. Inflation grows monotonically with training set size: MERT flat inflates from +17.4~pp at 10\% to +35.1~pp at 75\%; CLAP inflates from +12.0~pp to +19.0~pp. More training data means more recordings to memorise, amplifying the leakage effect.}
\label{fig:data_efficiency}
\end{figure*}
```

To test whether leakage is a small-data artifact, we train CLAP and MERT-95M flat probes on subsampled training sets (10\%, 25\%, 50\%, 75\%, 100\%) under both artist-filtered and clip-random evaluation.

Under honest evaluation, accuracy is remarkably insensitive to training set size: CLAP ranges from 56.3\% (10\% data) to 59.9\% (100\%), and MERT flat from 50.1\% to 58.2\%. The frozen representations already contain the genre information; more labelled data provides diminishing returns for learning the linear mapping.

Under clip-random evaluation, accuracy scales steeply: MERT flat rises from 67.5\% (10\%) to 90.2\% (100\%). The inflation consequently grows monotonically: from +17.4~pp at 10\% to +35.1~pp at 75\% for MERT flat, and from +12.0~pp to +19.0~pp for CLAP. This scaling confirms that leakage is not a small-data artifact --- it is a memorisation effect that strengthens as the model sees more recordings. The practical implication is sobering: growing a benchmark dataset does not mitigate protocol leakage; it amplifies it.

## O. Temporal Consistency as a Protocol Diagnostic

Section 5K showed that inter-episode agreement under honest evaluation averages 80.3\%. We now compare this to agreement under clip-random evaluation across 8 paired architectures.

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{temporal_consistency.pdf}
\caption{(a) Distribution of inter-episode agreement under artist-filtered (blue) vs clip-random (red) evaluation, pooled across 8 architectures. Under leakage, agreement shifts sharply rightward: mean rises from 0.80 to 0.96. (b) Per-model agreement comparison. Every model shows higher consistency under clip-random, with MERT-330M reaching 0.98. The gap ($\Delta$) is a protocol-agnostic diagnostic: no model should achieve 96\% temporal consistency on a genuinely ambiguous 8-class task.}
\label{fig:temporal_consistency}
\end{figure*}
```

Under clip-random evaluation, mean episode agreement rises to 96.4\% $\pm$ 12.7\% --- compared to 79.8\% $\pm$ 20.5\% under honest evaluation. The shift is universal: every model shows higher consistency under leakage. MERT-330M reaches 98.2\% agreement under clip-random, meaning that on average, 9.8 of 10 episodes from the same track are classified identically. This is not genre classification --- it is recording recognition.

The consistency gap provides a model-agnostic diagnostic for protocol integrity. A model achieving $>$95\% inter-episode agreement on an 8-class genre task should raise suspicion: genuinely ambiguous genres (Folk, Pop) cannot produce such consistency from genre cues alone. Unlike calibration diagnostics (Section 5H), which require computing ECE, the consistency diagnostic requires only the raw episode predictions and no additional statistical machinery.

## P. Track Difficulty Profiling

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{track_difficulty.pdf}
\caption{(a) Per-track difficulty under artist-filtered vs clip-random evaluation across 8 models and 617 test tracks, coloured by genre. Points above the diagonal represent tracks that become easier under leakage. Hard tracks (bottom-left) migrate toward the upper-right, while already-easy tracks (upper-right) stay put. (b) Mean difficulty improvement by honest-difficulty bin. Tracks that no model classifies correctly under honest evaluation gain +61.5~pp under leakage; tracks already easy gain --1.1~pp. Leakage is a regressive subsidy: it helps where genre recognition fails.}
\label{fig:track_difficulty}
\end{figure*}
```

We define per-track difficulty as the fraction of models that classify a track correctly. Under honest evaluation, 136 of 617 test tracks are classified incorrectly by all 8 models (difficulty = 0.0); under clip-random, this drops to 35. Conversely, 229 tracks are classified correctly by all models under honest evaluation; under leakage, 374 are.

The relationship between honest difficulty and leakage benefit is monotonic and steep: tracks in the hardest bin (0.0--0.2 honest difficulty) gain +61.5 pp from leakage, while tracks in the easiest bin (0.8--1.0) gain --1.1 pp. The per-genre breakdown is equally revealing: Folk, the hardest genre under honest evaluation (33.9\% mean difficulty), gains +57.1 pp from leakage. Hip-Hop, the easiest (90.3\%), gains only +1.5 pp.

This pattern is consistent with memorisation, not generalisation. A model that learns better genre features should improve on hard and easy tracks alike; a model that memorises recordings improves only on tracks it has heard before. The monotonic difficulty--improvement relationship is thus a third independent diagnostic for protocol leakage, complementing calibration (Section 5H) and temporal consistency (Section 5O).

## Q. Representation Collapse as a Leakage Indicator

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{representation_collapse.pdf}
\caption{(a) Within-track vs between-track cosine similarity distributions for four frozen backbones. All backbones show substantially higher within-track similarity, but the gap varies: CLAP has the highest collapse ratio (2.74) while MERT-95M and BYOL-A cluster near 1.2. (b) Collapse ratio (within/between similarity) plotted against protocol inflation. The relationship is not monotonic, suggesting that raw embedding similarity is a necessary but not sufficient indicator of leakage severity --- the downstream probe architecture and genre structure also matter.}
\label{fig:representation_collapse}
\end{figure*}
```

We introduce a label-free, training-free diagnostic for protocol leakage: the *representation collapse ratio*. For each backbone, we group clips by track and compute the mean pairwise cosine similarity of episode embeddings within each track (``within-track similarity'') and between random clips from different tracks (``between-track similarity''). The ratio of these two quantities measures how much the embedding space clusters by recording identity rather than by semantic content.

Across four backbones, within-track similarity ranges from 0.831 (CLAP) to 0.929 (BYOL-A), while between-track similarity ranges from 0.303 (CLAP) to 0.772 (MERT-95M). CLAP's high collapse ratio (2.74) reflects its contrastive training: it produces well-separated embeddings in general but near-identical embeddings for clips from the same recording. MERT-95M and BYOL-A, despite lower collapse ratios (1.19 and 1.21), exhibit higher absolute within-track similarity, indicating that most of the embedding space is devoted to recording-level rather than genre-level information.

The collapse ratio does not correlate monotonically with inflation ($r = -0.81$ across four backbones; note $n = 4$ limits statistical power). This is expected: leakage requires not just embedding similarity but also that the similarity survives the probing head and maps onto class boundaries. Nevertheless, the metric provides a useful pre-training diagnostic: a backbone where within-track cosine similarity exceeds 0.85 is encoding recording identity strongly enough to create protocol sensitivity, regardless of downstream accuracy.

## R. Nearest-Neighbor Recording Identity Recovery

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{nn_identity_recovery.pdf}
\caption{(a) Nearest-neighbor identity leak rate by protocol and backbone. Under artist-filtered evaluation, the NN leak rate is exactly 0\% for all backbones (same-track clips cannot appear in both splits by construction). Under clip-random evaluation, 56.8--90.1\% of test clips have a same-track nearest neighbor in the training set. (b) Clip-random NN leak rate vs protocol inflation. MERT-95M has both the highest leak rate (90.1\%) and the highest inflation (+32.0~pp), with a strong positive correlation ($r = 0.95$) across backbones.}
\label{fig:nn_identity_recovery}
\end{figure*}
```

We directly quantify embedding-level information leakage by measuring what fraction of test clips have a same-track nearest neighbor in the training set. For each test embedding, we compute cosine similarity to all training embeddings and check whether the nearest neighbor shares the same track ID. Under artist-filtered evaluation, the NN leak rate is 0.0\% for all four backbones --- exactly as designed: no track spans the split boundary. Under clip-random evaluation, the leak rates are striking: MERT-95M at 90.1\%, CLAP at 84.4\%, BYOL-A at 84.8\%, and PANNs at 56.8\%.

The correlation between NN leak rate and protocol inflation is $r = 0.95$ across four backbones (note: with $n = 4$, this correlation is suggestive rather than statistically significant at conventional thresholds). Nonetheless, the directional relationship is clear: backbones whose embeddings most effectively recover recording identity also show the largest accuracy inflation. This makes physical sense: if a test clip's nearest training neighbor is from the same track, and the probe has learned to associate that region of embedding space with the correct genre, then the probe will classify correctly even if it has learned no transferable genre features.

This analysis provides a second label-free diagnostic: a backbone where more than 50\% of test clips recover a same-track training neighbor under permissive splitting is embedding recording identity, not genre identity. The diagnostic requires only cached embeddings and split assignments --- no training or ground-truth labels.

## S. Embedding Space Geometry Under Protocol Variation

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{embedding_tsne.pdf}
\caption{t-SNE projections of 2,000 subsampled clips for MERT-95M (top) and CLAP (bottom). Left: coloured by genre, showing cluster quality. Center: coloured by train/test membership under clip-random splitting --- black lines connect same-track clips across splits, visualising the leakage mechanism. Right: under artist-filtered splitting, no same-track connections cross the split boundary. CLAP shows tighter genre clusters (better separability) but fewer cross-split connections due to its lower NN leak rate.}
\label{fig:embedding_tsne}
\end{figure*}
```

To provide visual intuition for the leakage mechanism, we project 2,000 randomly sampled clip embeddings into 2D using t-SNE (perplexity = 30) for MERT-95M and CLAP. Three views reveal distinct aspects of the embedding geometry.

Genre clustering (left panels) shows that both backbones separate genres, though with substantial overlap --- consistent with the 55--60\% accuracy ceiling under honest evaluation. CLAP produces tighter, more separated clusters, reflecting its audio-language contrastive pretraining. MERT-95M shows more diffuse clusters with stronger inter-genre overlap, particularly between Folk/Instrumental and Electronic/Experimental.

Under clip-random splitting (center panels), same-track clips span both train and test partitions. Lines connecting these cross-split pairs reveal the leakage mechanism directly: each line represents a test clip that has a near-identical training partner. Under artist-filtered splitting (right panels), no such connections exist --- the split boundary is clean at both the track and artist level.

The visual difference between the center and right panels is the geometric explanation for the 30+ pp accuracy gap: under clip-random evaluation, the probe need only learn to recognise familiar embeddings rather than generalise across recordings.

## T. Probe Architecture Robustness

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{probe_robustness.pdf}
\caption{Protocol inflation across four probe architectures (linear, MLP-64, MLP-128-64, MLP-256-128) for CLAP and MERT-95M (flat). Inflation persists regardless of probe complexity, confirming that leakage is a property of the evaluation protocol and the backbone embeddings, not the classifier architecture. Bars show test accuracy under artist-filtered (blue) and clip-random (red) evaluation; annotations show per-probe inflation in percentage points.}
\label{fig:probe_robustness}
\end{figure*}
```

A potential concern is that our inflation measurements might be specific to the probe architecture used. To address this, we train four probe types --- linear (no hidden layers), single-layer MLP with 64 units, two-layer MLP with 128 and 64 units, and two-layer MLP with 256 and 128 units --- on two representative backbones (CLAP and MERT-95M flat) under both artist-filtered and clip-random evaluation.

Inflation is consistent across all probe types. For CLAP, inflation ranges from the linear probe to the deepest MLP with no systematic trend: the protocol gap is not an artifact of probe expressiveness. For MERT-95M, the pattern is similar: all probes show comparable inflation regardless of capacity.

This result has a clear implication: the leakage is in the evaluation protocol and the frozen embeddings, not in the downstream classifier. A linear probe --- the simplest possible classifier --- exhibits essentially the same inflation as a multi-layer network. This rules out the hypothesis that leakage arises from overfitting in the probe head, and confirms that recording identity is linearly separable in the backbone's embedding space.

# 6. Discussion

## A. The Research Gap

The finding is not that the field is cheating --- it is that the field has no reporting convention for a variable we measure at tens of points. When a paper does not state its protocol, the number becomes unplaceable, incomparable in both directions, and the honest error bar spans the full ladder range.

Across 11 verified architectures spanning four model families, the gap is universal --- it is not an artifact of any particular model choice. The inflation pattern holds for models pretrained on music (MERT-95M, MERT-330M), on general audio events (BYOL-A, PANNs), and for from-scratch CNNs. It holds for multi-layer transformer probes and single-vector MLP probes. It holds for self-supervised models and supervised ones. The only variable that consistently predicts accuracy is the evaluation protocol.

This universality has a silver lining: the fix is correspondingly simple. Because the inflation is protocol-driven rather than architecture-driven, it can be eliminated by standardising a single metadata field --- the split mechanism --- rather than requiring changes to model architectures or training procedures.

## B. Why Some Models Leak More

The leakage ladder reveals systematic differences between model families in the magnitude of inflation, even though the direction is universal. Among verified results, the MERT attention probe inflates by +41.0 pp (54.6% to 95.7%), while the from-scratch CNN inflates by only +14.1 pp (55.1% to 69.2%). Understanding these differences illuminates what the models are actually learning under permissive evaluation.

One hypothesis: models with higher temporal resolution (more time-steps per second in their latent representation) are better at fingerprinting recordings. MERT processes audio at approximately 75 frames per second, producing fine-grained temporal representations that preserve recording-specific details --- the exact reverb tail, the specific drum pattern, the precise EQ curve. A CNN operating on mel spectrograms with 43 frames per second has coarser temporal resolution and discards more recording-specific detail during pooling.

An alternative hypothesis: models with larger embedding dimensionality provide more capacity for a probe to memorise training-set identities. The MERT attention probe (95.7% clip-random) projects through a multi-head attention layer with substantial learnable capacity; the simple CNN (69.2%) has far fewer parameters between input and output. Under this explanation, the inflation difference is not about the backbone's representations but about the probe's capacity to overfit.

A third hypothesis: the pretraining objective matters. Models pretrained on diverse audio categories (PANNs on 527 AudioSet classes, CLAP on audio-text pairs spanning many domains) may develop features that generalise more broadly, making them less susceptible to recording-level memorisation. Conversely, models pretrained with reconstruction objectives (Audio-MAE, EAT) learn to preserve fine-grained audio details that may make fingerprinting easier.

Our current results partially test these hypotheses. PANNs (supervised on 527 AudioSet classes, single 2048-dim vector) inflates by only +16.0 pp, while MERT attention probe (multi-layer transformer, 95.7% clip-random) inflates by +41.0 pp --- consistent with the capacity hypothesis. Extending this to speech-pretrained models (Wav2Vec 2.0, HuBERT) and audio-language models (CLAP) in future work would provide a stronger test of the pretraining domain hypothesis.

## C. Implications for Foundation Models

The field is moving toward ever-larger audio foundation models. AudioLM [@borsos2023audiolm] generates audio continuations from acoustic and semantic tokens; EnCodec [@defossez2023encodec] compresses audio into discrete codes with high-fidelity reconstruction. These models process audio with increasing fidelity, and their successors will only improve. As representation capacity grows, the gap between honest and leaky evaluation is likely to widen --- a model that perfectly reconstructs audio will trivially achieve 100% on clip-random splits, because its representations are literal audio fingerprints.

This trajectory makes evaluation protocol disclosure increasingly urgent. A reported accuracy of 95% on FMA-small in 2027 will mean nothing without protocol metadata. The community needs standardised reporting requirements before the next generation of models arrives.

The problem extends beyond FMA-small. Any music dataset where multiple segments come from the same recording (MagnaTagATune [@law2009magnatag], GTZAN in its commonly used 3-second-clip form) is susceptible to the same inflation. Any audio dataset where multiple samples share a source identity (the same speaker in speech datasets, the same location in soundscape datasets) faces the analogous issue. The foundation model era amplifies the problem because these models are evaluated on dozens of downstream tasks, each with its own splitting conventions, and a leaky protocol on any one task inflates the overall benchmark score.

## D. A Diagnostic Suite for Protocol Integrity

This study identifies six independent diagnostics for evaluation protocol leakage, ranging from label-free embedding analyses to model-output tests:

**Calibration reversal** (Section 5H): Under honest evaluation, models are slightly overconfident ($T^* \approx 1.2$). Under clip-random evaluation, the mean optimal temperature drops to 0.76 --- the model must be *sharpened* to match its inflated accuracy. A model with $T^* < 0.9$ on a task where honest accuracy is 55--58\% should raise suspicion.

**Temporal consistency inflation** (Section 5O): Under honest evaluation, inter-episode agreement averages 80\%. Under leakage, it rises to 96\%. A model achieving $>$95\% agreement on an 8-class task with genuinely ambiguous genres is more likely recognising recordings than classifying genres.

**Difficulty--improvement monotonicity** (Section 5P): Under honest evaluation, hard tracks (difficulty $<$ 0.2) gain +61.5 pp from leakage; easy tracks (difficulty $>$ 0.8) gain --1.1 pp. Genuine improvements in genre recognition should benefit hard and easy tracks alike; a monotonic difficulty--improvement relationship signals memorisation.

**Representation collapse** (Section 5Q): Within-track embedding cosine similarity (0.83--0.93) far exceeds between-track similarity (0.30--0.77) across all backbones. A backbone where within-track similarity exceeds 0.85 is encoding recording identity strongly enough to create protocol sensitivity --- this diagnostic requires no labels or training.

**Nearest-neighbor identity recovery** (Section 5R): Under clip-random evaluation, 57--90\% of test clips have a same-track nearest neighbor in training, correlating with inflation at $r = 0.95$. Under artist-filtered evaluation, the rate is exactly 0\%. This embedding-level diagnostic directly quantifies how much recording identity is available to the probe.

**Probe architecture invariance** (Section 5T): Inflation persists identically across linear, 1-layer, and 2-layer MLP probes, confirming that leakage is in the embeddings and protocol, not the classifier. This rules out the alternative hypothesis that protocol sensitivity arises from probe overfitting.

These diagnostics are complementary: representation collapse and NN recovery test the embedding space directly; calibration, temporal consistency, and difficulty profiling test model outputs; probe invariance tests the robustness of the finding itself. A model that triggers all six warrants close examination of its evaluation protocol.

## E. Representation Diversity and Ensemble Design

The CKA and error diversity analyses (Sections 5I--J) jointly explain the ensemble dynamics we observe. MERT-95M and MERT-330M share CKA = 0.848 and kappa = 0.732--0.789: they are near-duplicates in representation space and in error patterns. Adding MERT-330M to a MERT-95M ensemble provides minimal complementary information. In contrast, PANNs (CKA = 0.498 vs MERT-95M) and CNN ($\kappa = 0.621$ vs MERT-95M flat) provide genuinely different signals.

This quantifies the intuition that ensemble diversity requires architectural diversity, not just parameter diversity. The practical implication: investing compute in multiple MERT variants (different probe heads, different seeds) yields diminishing returns compared to investing in one probe per architecturally distinct backbone.

## F. Negative Results

**Epoch-Scheduled Conditional Fusion (ESCF).** A learned fusion of CNN and MERT backbones using per-episode sigmoid gates, per-track attention, and a curriculum. Every ESCF variant lost to a fixed 50/50 uniform ensemble (McNemar p = 0.011).

**Learned attention over episodes.** Pooling with learned attention is consistently worse than averaging. The weakest MERT probe in our inventory is the attention probe.

**The lesson.** Anything learned on top of frozen features overfits at this data scale. Accuracy gains must come from better features, not new learned combiners.

## G. Recommendations for the Field

Based on our findings, we propose that future music classification publications include three metadata fields:

1. **Split mechanism**: whether the train/test boundary is artist-filtered, track-random, or clip-random. If custom, the exact splitting code or logic should be described.

2. **Evaluation unit**: whether accuracy is computed at the clip level (each 3-second segment is an independent prediction) or the track level (clip predictions are aggregated before scoring).

3. **Split provenance**: whether the dataset's shipped split was used, or a custom one was generated.

These three fields occupy a single sentence in any paper and would render all FMA-small results mutually comparable. For example: "We use FMA's official artist-filtered split and report track-level accuracy." This sentence costs 12 words and resolves an ambiguity worth 30+ accuracy points.

We further suggest that conference reviewers flag the absence of this information during review. The NeurIPS reproducibility checklist [@pineau2021neurips] could be extended with a single question: "If your evaluation dataset contains multiple samples from the same source (e.g., artist, patient, speaker, location), does your train/test split control for source identity? If so, how? If not, why not?" This question is not specific to music --- it applies to any hierarchically structured dataset.

## H. Limitations

This study covers one dataset (FMA-small), one segmentation length (3 seconds), and reimplements rather than runs original code. The 3-second window is a ceiling on what any model here can see; genre is partly a long-timescale property. FMA-small's 8 coarse genres include genuinely multi-genre tracks; the single `genre_top` label caps classifier performance.

Our study covers 14 architectures but uses a single probing protocol (frozen backbone + MLP head). Fine-tuning may change the leakage dynamics: a fine-tuned model adapts its representations to the training set, potentially increasing memorisation under permissive splits and decreasing it under honest ones. The direction of this effect is an empirical question we do not answer here.

Embedding extraction runs on a single consumer laptop. Models requiring multi-GPU inference (e.g., very large CLAP variants, full-size Whisper models) are not included. Our architecture coverage is broad but not exhaustive; notable omissions include raw waveform models (SampleCNN [@lee2018samplecnn]) and conformer architectures [@gulati2020conformer].

FMA-small's eight genre labels reflect a Western popular music taxonomy. The "International" category, as a single catch-all for non-Western music, is culturally reductive. Our findings --- including genre-confusion patterns, difficulty distributions, and accuracy ceilings --- are conditioned on this ontology and may not generalise to culturally broader or finer-grained genre systems.

The literature audit covers papers we could locate through standard academic search. We may have missed FMA-small results published in non-English venues, workshop papers, or preprints. However, the audit's conclusion --- zero of nine papers state artist-filtered evaluation --- is robust to additions: any new paper that does state its protocol would not change the finding, and any that does not would only strengthen it.

# 7. Ongoing and Future Work

## A. Fine-Tuning Experiments

All current results freeze the backbone and train only a lightweight probe head. End-to-end fine-tuning with LoRA adapters [@hu2021lora] (~2M trainable parameters on a 330M backbone), partial layer unfreezing with layer-wise learning rate decay, and multi-crop processing of full 30-second tracks would test whether the leakage pattern changes when the backbone itself adapts to the training set. Additionally, fine-tuning Audio Spectrogram Transformer [@gong2021ast] and HTS-AT [@chen2022htsat] would provide ensemble diversity through architecturally distinct pretrained backbones.

Fine-tuning Wav2Vec 2.0 and HuBERT on music data before probing would test whether the speech-to-music domain gap affects leakage magnitude. If domain-adapted speech models show different inflation patterns than their off-the-shelf counterparts, it would suggest that the pretraining domain interacts with the evaluation protocol in ways not captured by frozen probing alone.

## B. Context Window Ablation

Every embedding in this project is computed from 3 seconds of audio. A 10-second extraction grid is implemented, as is a multi-crop approach that processes multiple 5-second windows from the full 30-second track. If longer context yields measurable gains, the implication is that the entire FMA-small literature has been evaluating under a suboptimal but unreported context length.

## C. Cross-Dataset Protocol Audit

The protocol inflation phenomenon should transfer to other music benchmarks (GTZAN, MagnaTagATune [@law2009magnatag]). Measuring the same ladder on a second dataset would elevate the finding from dataset-specific to field-wide.

## D. Architecture Expansion

Raw waveform models (SampleCNN [@lee2018samplecnn], Harmonic CNN [@won2020harmonic]) and conformer architectures [@gulati2020conformer] on spectrograms would further diversify the architecture set. SampleCNN is particularly interesting because it bypasses the mel spectrogram entirely, operating directly on raw audio samples --- testing whether the leakage pattern persists when the input representation changes fundamentally.

# 8. Conclusion

We measured the cost of three evaluation protocol choices on FMA-small music genre classification across 14 architectures, 62 training runs, and 4 model families: from-scratch CNNs, music-specific self-supervised models (MERT-95M, MERT-330M), general audio and audio-language models (BYOL-A, PANNs, CLAP), and non-neural baselines (XGBoost). The effect is large (+30.8 pp mean inflation across 11 verified architectures), systematic (architecture-independent), and undisclosed (0 of 9 published papers state artist-filtered evaluation).

Our best model reaches 59.9\% under the artist-filtered, track-level protocol --- 5.1 pp above the only prior sourceable artist-filtered number. The contribution of this work is not the accuracy number but the demonstration that most published numbers are not what they appear to be.

Beyond accuracy, the protocol choice distorts multiple measurable properties. Calibration reverses: under honest evaluation, models are approximately calibrated; under clip-random evaluation, all models are underconfident ($T^* < 1$). Temporal consistency inflates from 80\% to 96\%, far beyond what genre classification alone can sustain. Track difficulty profiling reveals that tracks no model classifies correctly under honest evaluation gain +61.5 pp under leakage, while already-easy tracks gain nothing (--1.1 pp).

At the representation level, leakage leaves clear signatures. Confusion matrix analysis shows that leakage eliminates genuine genre ambiguity (e.g., Folk--Instrumental confusion drops by 26 pp) by substituting recording recognition for genre recognition. Layer-wise probing of all 13 MERT transformer layers shows that recording identity is encoded uniformly across depth (inflation range: 33.5--41.1 pp). Data efficiency analysis confirms that inflation scales with training set size, ruling out a small-data artifact.

Embedding-level analyses provide the most direct evidence. Within-track cosine similarity (0.83--0.93) vastly exceeds between-track similarity (0.30--0.77), and 57--90\% of test clips under clip-random evaluation have a same-track nearest neighbor in the training set. t-SNE visualisations make the leakage mechanism visible: same-track clips span both partitions under permissive splitting but are cleanly separated under honest splitting. The inflation is probe-invariant: linear, 1-layer, and 2-layer MLP probes all show identical inflation, ruling out probe overfitting as an explanation.

Together, these six diagnostics --- calibration, temporal consistency, difficulty profiling, representation collapse, nearest-neighbor recovery, and probe invariance --- provide independent, model-agnostic tests for protocol integrity. CKA and error diversity analyses further reveal that architectural diversity, not parameter scaling, drives ensemble performance.

Three metadata fields --- split mechanism, evaluation unit, split provenance --- would render all future FMA-small results mutually comparable. These fields cost a single sentence per paper. Until that sentence becomes standard, any published FMA-small accuracy carries an implicit error bar of +31 percentage points: the distance between an honest protocol and a permissive one.

The gap is not in the models --- it is in the reporting convention.

# Appendix: Reproduction Guide

This appendix provides the exact steps, commands, and configuration values needed to reproduce every result in this paper from raw audio files.

## A. Environment and Dependencies

The full pipeline runs on a single consumer laptop (Apple M-series, 16 GB RAM, MPS backend). No GPU cluster is required. Setup:

```
git clone <repository-url> && cd music
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

A `.env` file with Kaggle API credentials (`KAGGLE_USERNAME`, `KAGGLE_KEY`) is required for automated FMA-small download. The complete dependency list is pinned in `pyproject.toml`.

## B. Data Pipeline

**Step 1: Download FMA-small** (7.2 GB compressed):

```
./scripts/01_fetch_data.sh
```

This downloads and extracts 8,000 MP3 tracks (7,996 readable after excluding 4 corrupt files) balanced across eight genres.

**Step 2: Preprocess audio** --- decode MP3s to waveforms, segment each 30-second track into ten non-overlapping 3-second episodes:

```
./scripts/02_preprocess.sh
```

Produces 75,296 total episodes (60,188 train / 7,544 validation / 7,564 test under the official artist-filtered split).

## C. Embedding Extraction

Frozen backbone embeddings are cached once and reused across all downstream experiments. Each backbone requires a single extraction pass:

```
python -m src.data.extract_embeddings --workers 4
python -m src.data.extract_mert_large --workers 4
python -m src.data.extract_byola
python -m src.data.extract_panns
python -m src.data.extract_clap
```

Approximate wall-clock times: MERT-95M 45 min, MERT-330M 90 min, BYOL-A/PANNs/CLAP 30 min each.

Extracted embeddings are stored in `data/embeddings/<backbone>/` as memory-mapped arrays. Total extraction time is approximately 4 hours sequential on a single machine.

## D. Training Configuration

All 72 YAML configuration files are in `src/training/configs/`. Each file fully specifies a single training run. Representative configuration (CLAP probe, artist-filtered):

```{=latex}
\begin{table}[h]
\centering
\footnotesize
\caption{Representative Training Configuration (CLAP Probe)}
\begin{tabular}{ll}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Optimizer & AdamW \\
Learning rate & 0.0003 \\
Weight decay & 0.001 \\
Scheduler & Cosine annealing \\
Batch size & 128 \\
Max epochs & 50 \\
Early stopping patience & 10 \\
Label smoothing & 0.1 \\
Mixup alpha & 0.3 \\
Dropout & 0.4 \\
Hidden layers & [256, 128] \\
Random seed & 42 \\
\bottomrule
\end{tabular}
\end{table}
```

To train a single configuration:

```
python -m src.training.train \
  --config src/training/configs/clap_probe_v1.yaml
```

To train all configurations sequentially:

```
./scripts/03_train_all.sh
```

Each probe training run completes in 1--3 minutes; from-scratch CNN runs take 30--60 minutes. Total training time for all 62 evaluated runs is under 40 GPU-hours equivalent on MPS.

## E. Protocol Ladder Reproduction

The three-rung protocol ladder (artist-filtered, random-track, clip-random) is reproduced by training each architecture under all three split modes. For each architecture `X`, three config files exist:

- `X_v1.yaml` --- artist-filtered (official FMA split)
- `X_v1_randomtrack.yaml` --- random 80/10/10 over whole tracks
- `X_v1_cliprandom.yaml` --- random 80/10/10 over individual 3-second episodes

The ladder queue runs all pairs:

```
./scripts/run_ladder_queue.sh
```

## F. Evaluation and Analysis

**Track-level evaluation** (per-episode softmax probabilities averaged within each track, then argmax):

```
./scripts/04_evaluate_all.sh
```

**Analysis scripts** produce the JSON artifacts from which all figures and tables are generated:

```
python scripts/calibration_analysis.py
python scripts/cka_analysis.py
python scripts/error_diversity.py
python scripts/statistical_tests.py
python scripts/temporal_analysis.py
python scripts/multigenre_analysis.py
```

Each script writes its output to `outputs/<name>.json`.

**Figure generation** from analysis JSONs:

```
python scripts/generate_paper_figures.py
```

**Results table generation** (single source of truth --- no number in the paper is hand-edited):

```
python scripts/update_results.py
python scripts/protocol_leakage_analysis.py
```

## G. Seed Stability Verification

To verify that results are stable across random seeds, six seed-varied runs are provided for the MERT-95M flat probe (`mert_flat_diverse_s1.yaml` through `mert_flat_diverse_s6.yaml`, seeds 42, 123, 256, 789, 1024, 2048). The standard deviation across these runs is reported in Section 5B.

## H. Key Numerical Summary

```{=latex}
\begin{table}[h]
\centering
\footnotesize
\caption{Study-Wide Numerical Summary}
\begin{tabular}{ll}
\toprule
\textbf{Quantity} & \textbf{Value} \\
\midrule
Total training runs evaluated & 62 \\
Unique architectures & 14 \\
Model families & 4 \\
Backbone models & 12 \\
Configuration files & 72 \\
Tracks in FMA-small & 8,000 (7,996 readable) \\
Episodes per track & 10 (3 seconds each) \\
Total episodes & 75,296 \\
Train / Val / Test episodes & 60,188 / 7,544 / 7,564 \\
Genres & 8 \\
Best artist-filtered accuracy & 59.9\% (CLAP probe) \\
Mean protocol inflation & +30.8 pp \\
Maximum protocol inflation & +41.2 pp (MERT-330M deep probe) \\
Total compute & $<$40 GPU-hours (MPS) \\
\bottomrule
\end{tabular}
\end{table}
```

