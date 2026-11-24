"""DGX training entrypoint with DDP, mixed precision, and gradient accumulation.

Usage:
    # Single GPU
    python train.py --config configs/mert_lora_330m.yaml

    # Multi-GPU DDP
    torchrun --nproc_per_node=4 train.py --config configs/mert_lora_330m.yaml

    # Resume from checkpoint
    python train.py --config configs/mert_lora_330m.yaml --resume outputs/mert_lora_330m/last.pt

Logs to outputs/<run_name>/ with metrics CSV, checkpoints, and eval results.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# bf16 autocast (Hopper-native, no loss scaling needed, avoids fp16 overflow in MERT)
AMP_DTYPE = torch.bfloat16

# Only save trainable weights to disk if they are smaller than this (disk is tight)
MAX_SAVE_MB = 50


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_distributed() or dist.get_rank() == 0


def setup_distributed():
    """Initialize DDP if launched with torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def build_model(cfg: dict) -> nn.Module:
    model_type = cfg["model"]

    if model_type == "mert_finetune":
        from models.mert_finetune import MertFinetune
        return MertFinetune(
            checkpoint=cfg.get("checkpoint", "m-a-p/MERT-v1-330M"),
            num_classes=cfg.get("num_classes", 8),
            pool_mode=cfg.get("pool_mode", "weighted"),
            unfreeze_layers=cfg.get("unfreeze_layers", 0),
            use_lora=cfg.get("use_lora", False),
            lora_rank=cfg.get("lora_rank", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            classifier_hidden=cfg.get("classifier_hidden", 512),
            dropout=cfg.get("dropout", 0.3),
            gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        )

    elif model_type == "mert_multicrop":
        from models.mert_finetune import MertMultiCrop
        return MertMultiCrop(
            checkpoint=cfg.get("checkpoint", "m-a-p/MERT-v1-330M"),
            num_classes=cfg.get("num_classes", 8),
            crop_seconds=cfg.get("crop_seconds", 5.0),
            n_crops_train=cfg.get("n_crops_train", 6),
            n_crops_eval=cfg.get("n_crops_eval", 10),
            use_lora=cfg.get("use_lora", True),
            lora_rank=cfg.get("lora_rank", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            dropout=cfg.get("dropout", 0.3),
        )

    elif model_type == "ast_finetune":
        from models.ast_finetune import ASTFinetune
        return ASTFinetune(
            num_classes=cfg.get("num_classes", 8),
            target_sr=cfg.get("target_sr", 16000),
            max_duration_sec=cfg.get("max_duration_sec", 10.0),
            freeze_layers=cfg.get("freeze_layers", 0),
            dropout=cfg.get("dropout", 0.3),
            spec_augment=cfg.get("spec_augment", False),
        )

    elif model_type == "efficientat":
        from models.ast_finetune import EfficientATFinetune
        return EfficientATFinetune(
            num_classes=cfg.get("num_classes", 8),
            target_sr=cfg.get("target_sr", 32000),
            max_duration_sec=cfg.get("max_duration_sec", 10.0),
            dropout=cfg.get("dropout", 0.3),
        )

    elif model_type == "htsat":
        from models.htsat import HTSATGenre
        return HTSATGenre(
            num_classes=cfg.get("num_classes", 8),
            target_sr=cfg.get("target_sr", 32000),
            max_duration_sec=cfg.get("max_duration_sec", 10.0),
            embed_dim=cfg.get("embed_dim", 96),
            depths=tuple(cfg.get("depths", [2, 2, 6, 2])),
            num_heads=tuple(cfg.get("num_heads", [3, 6, 12, 24])),
            dropout=cfg.get("dropout", 0.1),
        )

    else:
        raise ValueError(f"Unknown model: {model_type}")


def build_dataset(cfg: dict, split: str):
    dataset_type = cfg.get("dataset_type", "episode")
    target_sr = cfg.get("target_sr", 24000)
    augment = split == "train"

    if dataset_type == "random_crop":
        from data.dataset import FMAEpisodeDataset, FMARandomCropDataset
        if split == "train":
            return FMARandomCropDataset(
                split=split, target_sr=target_sr, crop_sec=cfg.get("episode_sec", 10.0),
                repeats=cfg.get("crop_repeats", 3),
            )
        # val/test: fixed non-overlapping windows, averaged per track
        return FMAEpisodeDataset(split=split, target_sr=target_sr,
                                 episode_sec=cfg.get("episode_sec", 10.0), augment=False)
    if dataset_type == "full_track":
        from data.dataset import FMAFullTrackDataset
        return FMAFullTrackDataset(
            split=split,
            target_sr=target_sr,
            max_duration_sec=cfg.get("max_duration_sec", 30.0),
            augment=augment,
        )
    else:
        from data.dataset import FMAEpisodeDataset
        return FMAEpisodeDataset(
            split=split,
            target_sr=target_sr,
            episode_sec=cfg.get("episode_sec", 3.0),
            augment=augment,
        )


def build_loaders(cfg: dict):
    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    test_ds = build_dataset(cfg, "test")

    bs = cfg["batch_size"]
    num_workers = cfg.get("num_workers", 4)
    from data import memory_cache
    if memory_cache.is_loaded():
        num_workers = 0  # audio already decoded in RAM; slicing is cheap

    if is_distributed():
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
        test_sampler = DistributedSampler(test_ds, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=num_workers, pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        sampler=val_sampler, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        sampler=test_sampler, num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, train_sampler


def build_optimizer(cfg: dict, model: nn.Module):
    """Build optimizer with optional layer-wise learning rate decay."""
    base_lr = cfg["lr"]
    weight_decay = cfg.get("weight_decay", 0.01)
    lr_decay = cfg.get("layer_lr_decay", 1.0)

    if lr_decay < 1.0 and hasattr(model, "backbone"):
        # Layer-wise LR decay: lower layers get smaller LR
        param_groups = []

        # Backbone layers (decaying LR)
        if hasattr(model, "backbone"):
            backbone = model.backbone
            if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layers"):
                layers = list(backbone.encoder.layers)
                n_layers = len(layers)
                for i, layer in enumerate(layers):
                    layer_lr = base_lr * (lr_decay ** (n_layers - i))
                    trainable = [p for p in layer.parameters() if p.requires_grad]
                    if trainable:
                        param_groups.append({
                            "params": trainable,
                            "lr": layer_lr,
                            "weight_decay": weight_decay,
                        })

        # Classifier and other params at full LR
        classifier_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and "backbone" not in name:
                classifier_params.append(param)
            elif param.requires_grad and "layer_weights" in name:
                classifier_params.append(param)

        if classifier_params:
            param_groups.append({
                "params": classifier_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
            })

        optimizer = torch.optim.AdamW(param_groups)
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=base_lr, weight_decay=weight_decay)

    return optimizer


def build_scheduler(cfg: dict, optimizer, num_training_steps: int):
    name = cfg.get("scheduler", "cosine_warmup")

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    elif name == "cosine_warmup":
        warmup_steps = cfg.get("warmup_steps", 500)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
            return max(0.01, 0.5 * (1 + np.cos(np.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif name == "none":
        return None

    else:
        raise ValueError(f"Unknown scheduler: {name}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    scaler,
    train: bool,
    scheduler=None,
    grad_accum_steps: int = 1,
    mixup_alpha: float = 0.0,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
):
    model.train(mode=train)
    total_loss = 0.0
    all_preds, all_labels = [], []
    n_steps = 0

    for step, batch in enumerate(loader):
        waveforms, labels, track_ids = batch
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        # Mixup augmentation
        if train and mixup_alpha > 0:
            lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample())
            index = torch.randperm(waveforms.size(0), device=device)
            waveforms = lam * waveforms + (1 - lam) * waveforms[index]

        with torch.set_grad_enabled(train):
            with autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=use_amp):
                logits = model(waveforms)

                if train and mixup_alpha > 0:
                    loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[index])
                else:
                    loss = criterion(logits, labels)

                loss = loss / grad_accum_steps

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if (step + 1) % grad_accum_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            max_grad_norm,
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()
                else:
                    loss.backward()
                    if (step + 1) % grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            max_grad_norm,
                        )
                        optimizer.step()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()

        total_loss += loss.item() * grad_accum_steps * waveforms.size(0)
        all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        n_steps += 1

    n = len(all_labels)
    avg_loss = total_loss / max(n, 1)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(n, 1)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1


def aggregate_track_preds(
    all_probs: list[np.ndarray],
    all_labels: list[int],
    all_track_ids: list[int],
) -> tuple[list[int], list[int]]:
    """Mean-pool episode probabilities within a track."""
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    label_of: dict[int, int] = {}

    for prob, label, tid in zip(all_probs, all_labels, all_track_ids):
        if tid not in sums:
            sums[tid] = np.zeros_like(prob)
            counts[tid] = 0
            label_of[tid] = label
        sums[tid] += prob
        counts[tid] += 1

    ordered = sorted(sums)
    track_labels = [label_of[t] for t in ordered]
    track_preds = [int((sums[t] / counts[t]).argmax()) for t in ordered]
    return track_labels, track_preds


def evaluate_track_level(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    """Full evaluation with track-level aggregation."""
    model.eval()
    all_probs, all_labels, all_track_ids = [], [], []

    with torch.no_grad():
        for waveforms, labels, track_ids in loader:
            waveforms = waveforms.to(device)
            with autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=use_amp):
                logits = model(waveforms)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())
            all_track_ids.extend(track_ids.tolist())

    # Episode-level metrics
    episode_preds = [int(p.argmax()) for p in all_probs]
    episode_acc = sum(p == l for p, l in zip(episode_preds, all_labels)) / len(all_labels)
    episode_f1 = f1_score(all_labels, episode_preds, average="macro", zero_division=0)

    # Track-level metrics
    track_labels, track_preds = aggregate_track_preds(all_probs, all_labels, all_track_ids)
    track_acc = sum(p == l for p, l in zip(track_preds, track_labels)) / len(track_labels)
    track_f1 = f1_score(track_labels, track_preds, average="macro", zero_division=0)

    return {
        "probs": np.stack(all_probs),
        "labels": np.array(all_labels),
        "track_ids": np.array(all_track_ids),
        "episode_accuracy": episode_acc,
        "episode_macro_f1": episode_f1,
        "track_accuracy": track_acc,
        "track_macro_f1": track_f1,
        "n_episodes": len(all_labels),
        "n_tracks": len(track_labels),
    }


def train(cfg: dict, resume_path: str | None = None):
    distributed = setup_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        log.info("Config: %s", json.dumps(cfg, indent=2))
        log.info("Device: %s, Distributed: %s", device, distributed)

    torch.manual_seed(cfg.get("seed", 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.get("seed", 42))

    # Output directory
    run_dir = Path(cfg.get("_out_root", "outputs")) / cfg["run_name"]
    if is_main_process():
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    # Build model
    model = build_model(cfg)
    model.to(device)

    if is_main_process():
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log.info("Parameters: %d trainable / %d total (%.1f%%)", trainable, total, 100 * trainable / total)

    if distributed:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=True)

    # Build data
    train_loader, val_loader, test_loader, train_sampler = build_loaders(cfg)

    # Build optimizer and scheduler
    base_model = model.module if distributed else model
    optimizer = build_optimizer(cfg, base_model)

    num_training_steps = len(train_loader) * cfg["epochs"] // cfg.get("grad_accum_steps", 1)
    scheduler = build_scheduler(cfg, optimizer, num_training_steps)

    # Loss
    label_smoothing = cfg.get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Mixed precision
    use_amp = cfg.get("use_amp", True) and torch.cuda.is_available()
    scaler = None  # bf16 autocast does not need a GradScaler

    # Resume from checkpoint
    start_epoch = 1
    best_val_f1 = -1.0
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device)
        base_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", -1.0)
        if scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if is_main_process():
            log.info("Resumed from epoch %d (best val F1: %.4f)", start_epoch - 1, best_val_f1)

    # Training loop
    if is_main_process():
        metrics_path = run_dir / "metrics.csv"
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "train_acc", "train_f1",
                "val_loss", "val_acc", "val_f1", "lr", "seconds",
            ])

    patience = cfg.get("early_stopping_patience", 15)
    epochs_since_improvement = 0
    grad_accum = cfg.get("grad_accum_steps", 1)
    mixup_alpha = cfg.get("mixup_alpha", 0.0)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    best_epoch = start_epoch - 1
    best_state = None

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()

        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            train=True, scheduler=scheduler, grad_accum_steps=grad_accum,
            mixup_alpha=mixup_alpha, max_grad_norm=max_grad_norm, use_amp=use_amp,
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, optimizer, device, scaler,
            train=False, use_amp=use_amp,
        )
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]

        if is_main_process():
            log.info(
                "epoch %d/%d  train_loss=%.4f train_f1=%.4f  val_loss=%.4f val_acc=%.4f val_f1=%.4f  lr=%.2e  (%.1fs)",
                epoch, cfg["epochs"], train_loss, train_f1, val_loss, val_acc, val_f1, current_lr, elapsed,
            )
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch, train_loss, train_acc, train_f1,
                    val_loss, val_acc, val_f1, current_lr, elapsed,
                ])

            # No per-epoch checkpoints on disk: keep the best weights in CPU RAM.
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                epochs_since_improvement = 0
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in base_model.state_dict().items()}
                log.info("  New best val F1: %.4f (epoch %d)", best_val_f1, epoch)
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= patience:
                log.info("Early stopping at epoch %d", epoch)
                break

    # Final evaluation on test set
    if is_main_process():
        log.info("Loading best checkpoint (epoch %d) for test evaluation...", best_epoch)
        base_model.load_state_dict(best_state)

        test_metrics = evaluate_track_level(base_model, test_loader, device, use_amp)
        save_run_outputs(run_dir, base_model, test_metrics)
        # Val probabilities too, so ensemble weights can be chosen on val (not test)
        val_metrics = evaluate_track_level(base_model, val_loader, device, use_amp)
        np.savez_compressed(run_dir / "val_probs.npz", probs=val_metrics["probs"].astype(np.float32),
                            labels=val_metrics["labels"], track_ids=val_metrics["track_ids"])

        log.info(
            "TEST  episode_acc=%.4f episode_f1=%.4f  track_acc=%.4f track_f1=%.4f  (%d episodes, %d tracks)",
            test_metrics["episode_accuracy"], test_metrics["episode_macro_f1"],
            test_metrics["track_accuracy"], test_metrics["track_macro_f1"],
            test_metrics["n_episodes"], test_metrics["n_tracks"],
        )

        summary = {
            "run_name": cfg["run_name"],
            "model": cfg["model"],
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "test": {k: v for k, v in test_metrics.items() if not isinstance(v, np.ndarray)},
            "config": cfg,
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        from evaluate import aggregate_to_tracks, compute_metrics
        probs, labels, tids = test_metrics["probs"], test_metrics["labels"].tolist(), test_metrics["track_ids"].tolist()
        track_labels, track_preds = aggregate_to_tracks(probs, labels, tids)
        eval_metrics = {
            "run_name": cfg["run_name"],
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "clip_level": compute_metrics(labels, probs.argmax(1).tolist()),
            "track_level": compute_metrics(track_labels, track_preds),
            "n_clips": test_metrics["n_episodes"],
            "n_tracks": test_metrics["n_tracks"],
        }
        with open(run_dir / "eval_metrics.json", "w") as f:
            json.dump(eval_metrics, f, indent=2)

    cleanup_distributed()
    if is_main_process():
        del model, base_model, optimizer, best_state
        torch.cuda.empty_cache()
        return eval_metrics


def save_run_outputs(run_dir: Path, model: nn.Module, test_metrics: dict) -> None:
    """Save test probabilities (small) and trainable weights only if they are small."""
    np.savez_compressed(
        run_dir / "test_probs.npz",
        probs=test_metrics["probs"].astype(np.float32),
        labels=test_metrics["labels"],
        track_ids=test_metrics["track_ids"],
    )
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    state = {k: v.cpu() for k, v in model.state_dict().items()
             if k in trainable or "running_" in k or "num_batches_tracked" in k}
    size_mb = sum(v.numel() * v.element_size() for v in state.values()) / 1e6
    if size_mb <= MAX_SAVE_MB:
        torch.save(state, run_dir / "trainable_weights.pt")
        log.info("Saved trainable weights (%.1f MB) to %s", size_mb, run_dir / "trainable_weights.pt")
    else:
        log.info("Trainable weights are %.1f MB (> %d MB) — NOT saved to disk", size_mb, MAX_SAVE_MB)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
