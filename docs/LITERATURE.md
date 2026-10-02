# FMA-small: what the literature actually states

A registry of every FMA-small genre-classification result we could trace to a primary
source, recording **what each paper says about its evaluation protocol** — not what its
accuracy implies.

That distinction is the whole discipline of this page, so it is worth stating plainly
before the table:

> **We never infer a protocol from an accuracy.** The thesis of this project is that
> protocol choice moves FMA-small accuracy by tens of points. If we then used a high
> accuracy as evidence that a paper leaked, we would be assuming the conclusion. A paper
> that does not disclose its split is *unplaceable*, not *guilty*. Several of the papers
> below report numbers we cannot reach under the artist-filtered split; that is a fact
> about the comparison, not an accusation about the authors, and nothing here should be
> read as one.

Every row was read from the paper's own text, not from another paper's comparison table.
That matters: the third section of this page documents three widely-cited FMA-small
numbers that do not survive being looked up.

---

## 1. The registry

| # | Method | Year | Accuracy | What the paper states about its split |
|---|--------|------|----------|----------------------------------------|
| 1 | TS-Resformer (ResNet+Transformer), Zhang | 2025 | 90.23% | **Random 8:1:1 over songs.** "the dataset is sliced into training set, validation set, and test set according to 8:1:1, and the number of **songs** of different genres in the three sets is guaranteed to be balanced." Their 12s/50%-overlap segmentation therefore happens *inside* each partition, so clips cannot cross. Artist overlap is not addressed. |
| 2 | CT-GateNet (CNN+Transformer), Ma et al. | 2026 | 89.42% | **Ratio only, unit unnamed.** "all datasets were evenly split into training, validation and test sets at an 8:1:1 ratio." The paper never says 8:1:1 *of what*. |
| 3 | DenseNet + SVM stacking, Bian et al. | 2019 | 68.9% | **Nothing.** No split sentence of any kind. Scoring is track-level: segment predictions are averaged into "a single song-level prediction". |
| 4 | Dual residual CNN, Ashraf et al. | 2025 | 68.5% | **Clip-random, stated outright.** "we segmented all music files into 3-second clips and split them into training, validation, and test sets using an 8:1:1 ratio" — segmentation precedes splitting. The only paper in the registry that says so. |
| 5 | BYOL-A + DNN, Rai & Bhattacharjee | 2025 | 64.3% | **Nothing.** The strings "split", "artist", "fold", "partition", "random" and "stratif" do not occur anywhere in the paper. |
| 6 | PANNs CNN14, Liu, DeMori & Abayomi | 2022 | 60.9%\* | **Partitions described, artist-conditioning not.** "each dataset is split into a training set, an evaluation set, and a test set"; the Experiment 2 sizes of 6,400/800 match FMA's official partition exactly, and the text references "the valid split". The word "artist" appears only in a discussion of label quality. |
| 7 | Transfer-learning CNN from artist labels, Park et al. | 2018 | 56.9% | **Nothing for FMA.** Notably, the same paper *does* specify a "fault-filtered" GTZAN split "where the dataset was divided to prevent artist repetition". The awareness is present; the FMA disclosure is not. |
| 8 | CNN/RNN/CRNN ensembles, Kostrzewa et al. | 2021 | 56.4% (54.9 F1) | **Ratio only.** "divided into three independent subsets in an 80:10:10 ratio", no mechanism. They use one 128×128 spectrogram per song, so same-track leakage is structurally impossible; artist overlap is the open question. The same paper criticises a competing result for "data leakage … which leads to over-optimistic outcomes". |
| 9 | Features + SVM, **re-run by Park et al.** of FMA's own baseline code | 2018 | 54.8% | **Artist-filtered by construction, not by claim.** Park Table 4: `SVM [7] — 0.5482 †`, footnoted "this result was obtained using the provided code and dataset in [7]". That code reads FMA's `split` column, so the property is inherited from the dataset rather than asserted by an author. |

\* 60.9% is the closed-set sub-metric `Acc_c` of an **open-set** experiment (their Table 3),
whose test set also contains unknown classes drawn from fma_medium. It is not a standalone
8-way fma_small result, and we flag it as such wherever it appears.

### The count

Of the 9 external results: **0 state artist-filtered evaluation in prose.** One states
clip-random, one states a track-level random split, one describes its partitions without
stating artist-conditioning, two give a ratio with no unit, and four say nothing at all.
One is artist-filtered by inheritance from the dataset's own code.

This is the finding. Not that the field is cheating — that the field has no reporting
convention for a variable we measure at tens of points
(`outputs/protocol_leakage_table.md`), and so a reader cannot place most of these numbers
on a common axis even in principle.

---

## 2. Why "unspecified" is a problem rather than an accusation

When a paper does not state its protocol:

1. **The number becomes unplaceable.** Not "probably leaked" — unplaceable. We have no
   admissible evidence about which rung it belongs on.
2. **It is incomparable in both directions.** An undisclosed 64.3% is neither above nor
   below an artist-filtered 60.3% in any meaningful sense. Ranking them is not a
   conservative move; it is a category error.
3. **The honest error bar is the full ladder span.** For an architecture of that family,
   that span is measured in `outputs/protocol_leakage_table.md` and is wider than the
   entire range of accuracies this literature competes over.

The practical consequence for this project: we do not report a "state of the art". We
report our accuracy under a named protocol at a named scoring unit, and we publish the
measured cost of the alternatives so a future reader can do the placement we cannot.

---

## 3. Four circulating "FMA-small records" that do not survive checking

Kept here with reasons, rather than silently omitted — a reader who has seen these numbers
elsewhere deserves to know why they are absent from the registry above.

### "Defferrard et al., 58.0%" — the baseline the field treats as its honest reference

**It does not exist.** The FMA paper proposes "single top genre on the balanced small
subset" as problem 1 and never reports a result for it. Its only accuracy table, Table 6,
is captioned "Test set accuracies … on the **medium** subset" — 25,000 tracks, 16 genres —
and 58 is the MLP/all-features cell *there*. The repository's executed `baselines.ipynb`
(on the `outputs` branch) subsets to `<= 'medium'` and prints partition sizes of
19,922/2,505/2,573 over 16 genres; its best cell is `SVCrbf` at 63.39%, also on medium.
Issue [mdeff/fma#69](https://github.com/mdeff/fma/issues/69) (December 2023) asks the
authors directly for small-subset baselines and has gone unanswered.

The 58.0 attribution traces to Rai & Bhattacharjee's Tables 1 and 3, cited to no page.
**Do not cite it.**

A related correction: the 54.8% figure often attributed to Defferrard et al. is real but
is Park et al.'s re-run of FMA's released code (row 9 above). Bian et al. reprint it as
"SVM (Defferrard, 2017) 54.8", which is how the attribution drifted.

### "CapsNet + UIGMM, 68.87%" (Scientific Reports 2024)

That paper runs only on ISMIR2004, GTZAN and Extended Ballroom, and never touches FMA. The
68.87% appears in its *related-work* discussion of Ashraf et al. It is a quotation, not a
result.

### "DPADMM-G, 97.27%"

A **binary, two-genre** task on FMA's precomputed feature vectors: "we randomly selected
two categories from the eight categories in fma_small… from each category, we randomly
chose 800 samples as the training set and the remaining 200 samples as the test set". Not
8-way audio classification.

That a leaderboard absorbed a binary-subtask number into an 8-way benchmark is, in
miniature, the same failure this project documents: a number detached from the protocol
that produced it travels further than it should.

### "Lee, Lee, Park & Lee, 51.2%"

Carried in our own registry for a long time on the strength of a row in Bian et al.'s
comparison table. The underlying citation could not be resolved to any paper reporting an
fma_small number. Dropped rather than guessed at — and noted here because it is the one
unverifiable row that was *ours*, inherited by exactly the mechanism described above.

---

## 4. Precedent in other fields

Source-level leakage is well understood elsewhere, which is part of why its absence from
the music-genre reporting convention is worth flagging. Wherever data points share a
common origin, the correct split is at the **source** level, not the instance level:

| Field | Shared source | Standard fix |
|---|---|---|
| Medical imaging | patient (multiple scans) | patient-level split |
| Speaker verification | speaker (multiple utterances) | speaker-level split |
| Document classification | author (writing style, not topic) | author-level split |
| Satellite imagery | location (adjacent tiles) | spatial hold-out |

Music genre classification has all three forms at once: the same recording (clip-level),
the same performance (track-level), and the same production signature (artist-level). FMA
ships the artist-level fix in the dataset itself. See `docs/METHODOLOGY.md` for why we use
it and what it costs.

---

## References

1. M. Defferrard, K. Benzi, P. Vandergheynst, X. Bresson. "FMA: A Dataset for Music
   Analysis." *ISMIR*, 2017. https://arxiv.org/abs/1612.01840
2. J. Park, J. Lee, J. Park, J. Ha, J. Nam. "Representation Learning of Music Using Artist
   Labels." *ISMIR*, 2018. https://arxiv.org/abs/1710.06648
3. W. Bian, J. Wang, B. Zhuang, J. Yang, S. Wang, J. Xiao. "Audio-Based Music
   Classification with DenseNet and Data Augmentation." *PAKDD*, 2019.
   https://arxiv.org/pdf/1906.11620
4. A. Kostrzewa, P. Kaminski, R. Brzeski. "Music Genre Classification: Looking for the
   Perfect Network." *ICCS*, 2021. https://doi.org/10.1007/978-3-030-77961-0_6
5. Q. Liu, R. DeMori, T. Abayomi. "Open Set Recognition for Music Genre Classification."
   2022. https://arxiv.org/html/2209.07548v2
6. M. Ashraf, F. Abid, M. A. Raza et al. "Music genre classification with modified residual
   learning and dual neural network." *PLOS ONE*, 2025.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12520411/
7. Rai, Bhattacharjee. "Evaluating Pretrained General-Purpose Audio Representations for
   Music Genre Classification." *PReMI*, 2025.
8. Zhang. "TS-Resformer for music genre classification." 2025.
9. Ma et al. "CT-GateNet." 2026.
