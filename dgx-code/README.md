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
