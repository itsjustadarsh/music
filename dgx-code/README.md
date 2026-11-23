# DGX Training Code for FMA-Small Genre Classification

GPU-scale training to beat the current best of 58.5% (artist-filtered, track-level).

## Strategy

The laptop pipeline uses **frozen probes** over cached embeddings. The ceiling there
is the quality of the pretrained representation. On a DGX we can:

1. **Fine-tune MERT-330M with LoRA** — adapt the backbone to genre, not just probe it
2. **Fine-tune Audio Spectrogram Transformer (AST)** — strong AudioSet-pretrained model
3. **Use full 30-second tracks** — no 3s windowing ceiling
4. **Multi-GPU DDP** — larger batches, faster convergence
5. **Ensemble** fine-tuned models with diverse inductive biases

## Setup

```bash
bash setup.sh
```

## Training

Single-GPU:
```bash
python train.py --config configs/mert_lora_330m.yaml
```

Multi-GPU DDP:
```bash
torchrun --nproc_per_node=4 train.py --config configs/mert_lora_330m.yaml
```

## Evaluation

```bash
python evaluate.py --run outputs/mert_lora_330m
```

## Run All Experiments

```bash
bash run_all.sh
```

## Single-GPU, in-memory run (shared server / low disk)

`run_pipeline.py` streams FMA-small (zip + metadata) straight into RAM, verifies
SHA1, decodes all MP3s in memory, and trains every config on one GPU in order,
then ensembles at track level. No dataset or per-epoch checkpoint touches disk.

```bash
export CUDA_VISIBLE_DEVICES=<free gpu>
python run_pipeline.py              # all 7 configs + ensembles
python run_pipeline.py --smoke      # synthetic-audio code check (1 tiny epoch each)
```

Per run, `outputs/<run_name>/` gets `metrics.csv`, `eval_metrics.json`,
`summary.json`, `test_probs.npz`, and `trainable_weights.pt` only if < 50 MB
(not committed). Overall results: `outputs/results_summary.json`.

### Results (H200, single GPU, artist-filtered test split, 800 tracks)

| Run | Track acc | Track macro-F1 |
|-----|-----------|----------------|
| ast_finetune | 0.581 | 0.583 |
| efficientat_genre | 0.554 | 0.546 |
| mert_lora_330m | 0.544 | 0.551 |
| mert_multicrop_330m | 0.539 | 0.541 |
| mert_unfreeze_330m | 0.534 | 0.530 |
| mert_lora_95m | 0.529 | 0.519 |
| htsat_genre | 0.474 | 0.470 |
| **Ensemble** (330m-LoRA + 95m-LoRA + AST + HTS-AT) | **0.589** | **0.588** |
| Ensemble (all 7) | 0.585 | 0.582 |
