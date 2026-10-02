# Results

<!-- GENERATED FILE — do not edit. `python scripts/update_results.py`. Last run 2026-10-02. -->

> **Generated file — do not edit by hand.** Every number in this project lives here
> or in `outputs/protocol_leakage_table.md`. Prose elsewhere references these and
> never restates them. Regenerate with `python scripts/update_results.py`.

All accuracies are **track-level**: per-episode softmax averaged within a track, then
argmax. See `docs/METHODOLOGY.md` §2.2 for why this is the only unit reported.

---

## 1. Our models, official artist-filtered split

Best run per architecture, ranked by accuracy. 36 artist-filtered runs across 13 architectures.

| Architecture | Best run | Accuracy | Macro-F1 |
|---|---|---|---|
| Ensemble of per-architecture bests | `ensemble_uniform` | 58.5% | 58.2% |
| MERT-95M flat probe (13-layer concat) | `mert_flat_probe_v1` | 58.2% | 57.8% |
| BYOL-A + DNN probe | `byola_probe_v1` | 57.4% | 57.0% |
| PANNs CNN14 probe | `panns_probe_v1` | 57.4% | 56.6% |
| MERT-330M probe (learned layer weights) | `mert_large_probe_v1` | 56.6% | 56.5% |
| MERT-330M flat probe (25-layer concat) | `mert_large_flat_v1` | 56.5% | 55.9% |
| MERT-95M probe (learned layer weights) | `mert_probe_tuned` | 56.4% | 55.7% |
| MERT-95M deep probe (multi-scale) | `mert_deep_probe_v2` | 56.0% | 55.9% |
| MERT-95M attention probe | `mert_attn_probe_v2` | 55.2% | 54.4% |
| GenreCNN (log-mel, from scratch) | `cnn_improved` | 55.1% | 54.2% |
| MERT-330M deep probe (multi-scale) | `mert_large_deep_probe_v1` | 54.9% | 55.5% |
| DenseNet121 | `densenet_v1` | 53.9% | 52.2% |
| Tree ensemble on MERT embeddings | `xgboost_mert` | 52.5% | 51.4% |

## 2. Against the published literature

Ordered by accuracy, with what each paper discloses about its split. The protocol
column records what the source *states*, never what its accuracy implies — see
`docs/LITERATURE.md` for the full audit and the three excluded claims.

| Result | Accuracy | Macro-F1 | Stated protocol | Source |
|---|---|---|---|---|
| Zhang, TS-Resformer (2025) | 90.2% | — | random 8:1:1 over *songs*, so the 12s/50%-overlap segmentation sits inside each partition rather than across them. Track-level random; artists unaddressed | https://doi.org/10.3389/fnbot.2025.1568811 |
| Ma et al., CT-GateNet (2026) | 89.4% | — | "all datasets were evenly split into training, validation and test sets at an 8:1:1 ratio" -- the paper never says 8:1:1 *of what* | https://doi.org/10.1371/journal.pone.0344606 |
| Bian, Wang, Zhuang et al., DenseNet + augmentation (2019) | 68.9% | — | none -- no split sentence anywhere. Scoring is track-level: segment predictions averaged into "a single song-level prediction" | https://arxiv.org/pdf/1906.11620 |
| Ashraf, Abid, Raza et al., dual residual CNN (2025) | 68.5% | 70.5% | clip-random, stated outright: "we segmented all music files into 3-second clips and split them into training, validation, and test sets using an 8:1:1 ratio" -- segmentation precedes splitting | https://doi.org/10.1371/journal.pone.0333808 |
| K. Rai & M. Bhattacharjee, BYOL-A + DNN (PReMI) (2025) | 64.3% | — | none -- the strings "split", "artist", "fold", "partition", "random" and "stratif" do not occur in the paper | https://arxiv.org/pdf/2603.13871 |
| Liu, DeMori & Abayomi, CNN14 (PANNs), open-set study (2022) | 60.9% | — | partitions *are* described and the 6,400/800 counts match FMA's official sizes; whether those splits are artist-conditional is never stated | https://arxiv.org/html/2209.07548v2 |
| **Ours — Ensemble of per-architecture bests** (`ensemble_uniform`) | **58.5%** | **58.2%** | **official artist-filtered, track-level** | this work |
| Park, Lee, Park, Ha & Nam, transfer learning from artist labels (2018) | 56.9% | — | none for FMA; the same paper *does* specify an artist-separated split for GTZAN, so the awareness is present and the disclosure is not | https://arxiv.org/abs/1710.06648 |
| Kostrzewa, Kaminski & Brzeski, CNN/RNN/CRNN ensembles (2021) | 56.4% | 54.9% | "divided into three independent subsets in an 80:10:10 ratio", mechanism unstated. One spectrogram per song, so clip leakage is structurally impossible; artist overlap is the open question | https://doi.org/10.1007/978-3-030-77961-0_6 |
| Features+SVM, re-run by Park et al. of FMA's released baseline code (2018) | 54.8% | — | artist-filtered *by construction* -- the code reads FMA's `split` column -- but not asserted in prose | https://arxiv.org/abs/1710.06648 |

### How to read that table

Our best artist-filtered model reaches **58.5%**. The only published number carrying an artist-filtered guarantee is 54.8%, a margin of **+3.7pp** — and that one is artist-filtered by construction (it inherits FMA's own `split` column) rather than by authorial claim.

The remaining 6 of 9 external results do not disclose their split. We deliberately do **not** report a 'beats N of M' count: ranking against numbers whose protocol is unknown would treat incomparable quantities as comparable, which is the failure this project documents. The measured cost of each protocol choice is in `outputs/protocol_leakage_table.md`.

## 3. Protocol inflation

The same architectures run down the three-rung ladder (`official` → `random_track` →
`random_clip`) are tabulated in **`outputs/protocol_leakage_table.md`**, regenerated by
`python scripts/protocol_leakage_analysis.py`. That file is the single source for every
inflation figure quoted anywhere in this repository.

## 4. Everything else

- Per-run hyperparameters and the full run inventory: `docs/EXPERIMENT_LOG.md`
- What failed and why: `docs/NEGATIVE_RESULTS.md`
- Definitions of every term above: `docs/METHODOLOGY.md`
