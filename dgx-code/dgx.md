# DGX Usage Guide

Complete guide for running FMA-small genre classification experiments on a DGX system.

## Directory Structure

After setup, everything lives under `/dgx-code/`:

```
dgx-code/
├── dgx.md                    ← you are here
├── README.md                 ← quick start
├── setup.sh                  ← one-time install
├── train.py                  ← main training entrypoint (DDP-aware)
├── evaluate.py               ← standalone evaluation with track-level scoring
├── ensemble.py               ← multi-model ensemble
├── run_all.sh                ← queue all experiments in priority order
│
├── configs/                  ← one YAML per experiment
│   ├── mert_lora_330m.yaml       ← [PRIMARY] MERT-330M + LoRA
│   ├── mert_lora_95m.yaml        ← MERT-95M + LoRA
│   ├── mert_unfreeze_330m.yaml   ← MERT-330M partial unfreeze
│   ├── mert_multicrop_330m.yaml  ← MERT-330M multi-crop (full 30s)
│   ├── ast_finetune.yaml         ← Audio Spectrogram Transformer
│   ├── htsat_genre.yaml          ← HTS-AT (Swin Transformer for audio)
│   └── efficientat_genre.yaml    ← EfficientAT (lightweight CNN)
│
├── models/
│   ├── mert_finetune.py      ← MERT LoRA / partial unfreeze / multi-crop
│   ├── ast_finetune.py       ← AST + EfficientAT models
│   └── htsat.py              ← HTS-AT (Swin-based audio transformer)
│
├── data/
│   ├── download.py           ← download + verify FMA-small
│   └── dataset.py            ← FMAFullTrackDataset, FMAEpisodeDataset
│
├── fma_data/                 ← created by setup.sh (NOT in git)
│   └── raw/
│       ├── fma_small/        ← 8000 MP3 tracks
│       └── fma_metadata/     ← tracks.csv with genre labels + splits
│
└── outputs/                  ← created per run (NOT in git)
    ├── mert_lora_330m/
    │   ├── config.json       ← frozen copy of the YAML config
    │   ├── metrics.csv       ← per-epoch: loss, acc, f1, lr, time
    │   ├── best.pt           ← best checkpoint (by val macro-F1)
    │   ├── last.pt           ← last checkpoint (for resume)
    │   ├── summary.json      ← final test metrics
    │   └── eval_metrics.json ← full eval: clip-level + track-level + per-genre
    ├── ast_finetune/
    │   └── ...
    └── ensemble_mert_lora_330m+mert_lora_95m+ast_finetune/
        └── eval_metrics.json
```

## Step-by-Step Instructions

### 1. Initial Setup

```bash
cd dgx-code

# Install all dependencies
bash setup.sh
```

This downloads FMA-small (~7.2 GB) and installs PyTorch, transformers, peft, etc.

**If FMA is already downloaded** (e.g., from the laptop pipeline), set the env var:
```bash
export FMA_DATA_DIR=/path/to/existing/fma_data
```

### 2. Run a Single Experiment

**Single GPU:**
```bash
python train.py --config configs/mert_lora_330m.yaml
```

**Multi-GPU (DDP):**
```bash
# 4 GPUs (adjust to your DGX config)
torchrun --nproc_per_node=4 train.py --config configs/mert_lora_330m.yaml
```

**Resume from checkpoint** (if interrupted):
```bash
torchrun --nproc_per_node=4 train.py \
    --config configs/mert_lora_330m.yaml \
    --resume outputs/mert_lora_330m/last.pt
```

### 3. Run All Experiments

```bash
# Set GPU count
export NGPU=4

# Run everything in priority order + ensemble at the end
bash run_all.sh
```

This takes approximately:
- MERT LoRA runs: ~30-60 min each
- AST fine-tune: ~30 min
- HTS-AT from scratch: ~1-2 hours
- Multi-crop: ~1-2 hours
- Total: ~4-6 hours on 4x A100

### 4. Evaluate a Single Run

```bash
python evaluate.py --run outputs/mert_lora_330m
```

This produces:
- `eval_metrics.json` — clip-level AND track-level accuracy/F1/per-genre breakdown
- `mert_lora_330m_confusion_matrix.png` — visual confusion matrix

### 5. Build an Ensemble

```bash
# Uniform weights (default)
python ensemble.py --runs outputs/mert_lora_330m outputs/ast_finetune

# Custom weights
python ensemble.py \
    --runs outputs/mert_lora_330m outputs/mert_lora_95m outputs/ast_finetune \
    --weights 0.4 0.3 0.3
```

### 6. Check Results

Quick summary of all runs:
```bash
for run in outputs/*/eval_metrics.json; do
    name=$(basename $(dirname $run))
    python -c "
import json
d = json.load(open('$run'))
t = d.get('track_level', {})
print(f'  {\"$name\":<35s} acc={t.get(\"accuracy\",0):.4f}  f1={t.get(\"macro_f1\",0):.4f}')
"
done
```

## Understanding the Results

### What matters: track-level accuracy under artist-filtered split

Every number that goes into the paper comes from `eval_metrics.json → track_level → accuracy`. The episode-level numbers are logged for diagnostics only.

The artist-filtered split is hard-coded — `data/dataset.py` always uses FMA's official split from `tracks.csv`. There is no clip-random or random-track mode here; those diagnostic protocols are only in the laptop pipeline.

### Why these models should beat 58.5%

The laptop pipeline's 58.5% ceiling comes from **frozen probes** — no backbone weights change, so the model can only linearly combine what MERT already represents. The DGX models break this ceiling by:

| Model | Strategy | Why it should help |
|-------|----------|-------------------|
| MERT LoRA 330M | Adapt attention layers with ~2M LoRA params | Backbone learns genre-relevant representations beyond frozen features |
| MERT Unfreeze 330M | Unfreeze top 4 layers | Higher capacity adaptation, with layer-wise LR decay to prevent catastrophic forgetting |
| MERT Multi-crop | 5s crops from full 30s | Sees more context than 3s — genre is partly a long-timescale property |
| AST | AudioSet-supervised ViT | Different pretraining (supervised classification vs self-supervised music) |
| HTS-AT | Hierarchical Swin for audio | Captures multi-scale temporal structure |
| EfficientAT | Lightweight depthwise CNN | Fast, diverse architecture for ensemble |

### What could go wrong

- **Overfitting**: Only 6,400 training tracks. Watch for val F1 diverging from train F1 early. The configs include strong regularisation (dropout, label smoothing, mixup, weight decay).
- **MERT 330M OOM**: Gradient checkpointing is enabled by default. If still OOM, reduce `batch_size` and increase `grad_accum_steps` proportionally.
- **FMA corrupt tracks**: 4 of 8000 tracks are corrupt. The dataset silently skips them via try/except.

## Config Reference

Key YAML fields (all optional except `run_name`, `model`):

| Field | Default | Description |
|-------|---------|-------------|
| `run_name` | required | Output subdirectory name |
| `model` | required | Model type: `mert_finetune`, `mert_multicrop`, `ast_finetune`, `htsat`, `efficientat` |
| `checkpoint` | varies | HuggingFace model name for MERT/AST |
| `use_lora` | false | Enable LoRA adapters |
| `lora_rank` | 16 | LoRA rank (higher = more params) |
| `unfreeze_layers` | 0 | Number of top transformer layers to unfreeze |
| `pool_mode` | "weighted" | How to combine MERT layers: "last", "mean_all", "weighted", "concat_last_4" |
| `dataset_type` | "episode" | "episode" (3s clips) or "full_track" (30s) |
| `target_sr` | 24000 | Audio sample rate (24k for MERT, 16k for AST, 32k for HTS-AT) |
| `batch_size` | 64 | Per-GPU batch size |
| `grad_accum_steps` | 1 | Gradient accumulation (effective batch = batch_size × accum × n_gpus) |
| `lr` | 0.0002 | Learning rate |
| `layer_lr_decay` | 1.0 | Layer-wise LR decay (0.75 = bottom layers get 75% of top layer's LR) |
| `scheduler` | "cosine_warmup" | LR schedule: "cosine_warmup", "cosine", "none" |
| `warmup_steps` | 500 | Linear warmup steps |
| `epochs` | 30 | Maximum epochs |
| `early_stopping_patience` | 15 | Stop if val F1 doesn't improve for N epochs |
| `use_amp` | true | Mixed precision (FP16) training |
| `seed` | 42 | Random seed |

## Customising Runs

To create a new experiment, copy an existing config and change what you want:

```bash
cp configs/mert_lora_330m.yaml configs/mert_lora_330m_r32.yaml
# Edit: change lora_rank to 32, run_name to mert_lora_330m_r32
torchrun --nproc_per_node=4 train.py --config configs/mert_lora_330m_r32.yaml
```

## Monitoring Training

Watch the metrics CSV in real-time:
```bash
# Tail the latest metrics
tail -f outputs/mert_lora_330m/metrics.csv

# Or use a quick plot
python -c "
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv('outputs/mert_lora_330m/metrics.csv')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(df['epoch'], df['train_loss'], label='train')
ax1.plot(df['epoch'], df['val_loss'], label='val')
ax1.set_title('Loss'); ax1.legend()
ax2.plot(df['epoch'], df['train_f1'], label='train')
ax2.plot(df['epoch'], df['val_f1'], label='val')
ax2.set_title('Macro F1'); ax2.legend()
plt.savefig('training_progress.png'); print('Saved training_progress.png')
"
```

## Copying Results Back

After all runs complete, the key files to copy back to the laptop repo:

```bash
# From DGX
scp -r outputs/*/eval_metrics.json outputs/*/summary.json outputs/*/config.json user@laptop:~/Code/clgprjcts/music/outputs/
```

The `eval_metrics.json` files are compatible with the laptop pipeline's `scripts/update_results.py`.
