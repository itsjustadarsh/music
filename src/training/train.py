"""Train any registered genre model from a YAML config.

Usage:
    python -m src.training.train --config src/training/configs/cnn_baseline.yaml
    python -m src.training.train --config src/training/configs/ast_baseline.yaml

Logs per-epoch metrics to outputs/<run_name>/metrics.csv and a final summary to
outputs/<run_name>/summary.json; saves the best checkpoint (by val macro-F1) to
outputs/<run_name>/best.pt.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.config import DATA_DIR, OUTPUT_DIR, PROCESSED_DIR
from src.models.cnn import GenreCNN
from src.models.common import NUM_CLASSES, get_device
from src.training.dataset import LogMelDataset, MertEmbeddingDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"
MERT_EMBEDDINGS_DIR = DATA_DIR / "embeddings" / "mert"
MERT_MANIFEST_PATH = MERT_EMBEDDINGS_DIR / "manifest.csv"


def build_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(name: str, optimizer, epochs: int):
    name = (name or "none").lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"Unknown scheduler: {name}")


def build_cnn_loaders(cfg: dict):
    split_mode = cfg.get("split_mode", "official")
    train_ds = LogMelDataset(MANIFEST_PATH, PROCESSED_DIR, "train", augment=cfg.get("augment_spec", False), split_mode=split_mode)
    val_ds = LogMelDataset(MANIFEST_PATH, PROCESSED_DIR, "val", augment=False, split_mode=split_mode)
    test_ds = LogMelDataset(MANIFEST_PATH, PROCESSED_DIR, "test", augment=False, split_mode=split_mode)
    bs = cfg["batch_size"]
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2),
        DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=2),
    )


def build_mert_loaders(cfg: dict):
    """Loaders over cached MERT embeddings -- no backbone forward pass, so these are
    fast enough to run the whole multi-seed ablation grid on a laptop.

    `embeddings_dir` selects the cache. It exists so a probe can be trained on a
    longer episode grid (`extract_embeddings.py --episode-seconds`) with everything
    else held fixed, which is the only way the window length can be read as the cause
    of any difference.
    """
    split_mode = cfg.get("split_mode", "official")
    embed_dir = Path(cfg["embeddings_dir"]) if cfg.get("embeddings_dir") else MERT_EMBEDDINGS_DIR
    manifest = embed_dir / "manifest.csv"
    kwargs = dict(split_mode=split_mode)
    train_ds = MertEmbeddingDataset(manifest, embed_dir, "train", **kwargs)
    val_ds = MertEmbeddingDataset(manifest, embed_dir, "val", **kwargs)
    test_ds = MertEmbeddingDataset(manifest, embed_dir, "test", **kwargs)
    bs = cfg["batch_size"]
    # num_workers=0: each dataset caches one track's array, which worker processes
    # would duplicate rather than share.
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0),
        DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0),
    )


def run_epoch(model, loader, criterion, optimizer, device, train: bool, mixup_alpha: float = 0.0):
    model.train(mode=train)
    total_loss, all_preds, all_labels = 0.0, [], []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.set_grad_enabled(train):
            if train and mixup_alpha > 0:
                lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample())
                index = torch.randperm(inputs.size(0), device=device)
                inputs = lam * inputs + (1 - lam) * inputs[index]
                logits = model(inputs)
                loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[index])
            else:
                logits = model(inputs)
                loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    n = len(all_labels)
    avg_loss = total_loss / max(n, 1)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if n else 0.0
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(n, 1)
    return avg_loss, acc, macro_f1


def train(cfg: dict) -> Path:
    torch.manual_seed(cfg.get("seed", 42))
    device = get_device()
    log.info("Using device: %s", device)

    run_dir = OUTPUT_DIR / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    if cfg["model"] == "cnn":
        model = GenreCNN(num_classes=NUM_CLASSES)
        train_loader, val_loader, test_loader = build_cnn_loaders(cfg)
    elif cfg["model"] == "mert_probe":
        from src.models.mert_probe import MertProbe

        model = MertProbe(
            num_classes=NUM_CLASSES,
            proj_size=cfg.get("proj_size", 512),
            dropout=cfg.get("dropout", 0.3),
            learn_layer_weights=cfg.get("learn_layer_weights", True),
            layer_index=cfg.get("layer_index"),
        )
        train_loader, val_loader, test_loader = build_mert_loaders(cfg)
    elif cfg["model"] == "mert_attn_probe":
        from src.models.mert_attn_probe import MertAttentionProbe

        model = MertAttentionProbe(
            num_classes=NUM_CLASSES,
            n_heads=cfg.get("n_heads", 8),
            attn_layers=cfg.get("attn_layers", 2),
            mlp_size=cfg.get("mlp_size", 512),
            dropout=cfg.get("dropout", 0.3),
        )
        train_loader, val_loader, test_loader = build_mert_loaders(cfg)
    elif cfg["model"] == "mert_flat_probe":
        from src.models.mert_flat_probe import MertFlatProbe

        model = MertFlatProbe(
            num_classes=NUM_CLASSES,
            proj_size=cfg.get("proj_size", 512),
            dropout=cfg.get("dropout", 0.5),
        )
        train_loader, val_loader, test_loader = build_mert_loaders(cfg)
    elif cfg["model"] == "resnet":
        from src.models.resnet_cnn import GenreResNet

        model = GenreResNet(
            num_classes=NUM_CLASSES,
            dropout=cfg.get("dropout", 0.4),
            embed_dim=cfg.get("embed_dim", 256),
        )
        train_loader, val_loader, test_loader = build_cnn_loaders(cfg)
    elif cfg["model"] == "mert_deep_probe":
        from src.models.mert_deep_probe import MertDeepProbe

        model = MertDeepProbe(
            num_classes=NUM_CLASSES,
            proj_size=cfg.get("proj_size", 512),
            n_groups=cfg.get("n_groups", 3),
            n_residual_blocks=cfg.get("n_residual_blocks", 2),
            dropout=cfg.get("dropout", 0.4),
        )
        train_loader, val_loader, test_loader = build_mert_loaders(cfg)
    elif cfg["model"] == "mert_large_flat_probe":
        from src.models.mert_flat_probe import MertFlatProbe
        from src.models.mert_embed_large import MERT_LARGE_HIDDEN_SIZE, MERT_LARGE_NUM_LAYERS

        model = MertFlatProbe(
            num_classes=NUM_CLASSES,
            num_layers=MERT_LARGE_NUM_LAYERS,
            hidden_size=MERT_LARGE_HIDDEN_SIZE,
            proj_size=cfg.get("proj_size", 512),
            dropout=cfg.get("dropout", 0.5),
        )
        mert_large_dir = DATA_DIR / "embeddings" / "mert_large"
        mert_large_manifest = mert_large_dir / "manifest.csv"
        split_mode = cfg.get("split_mode", "official")
        train_ds = MertEmbeddingDataset(mert_large_manifest, mert_large_dir, "train", split_mode=split_mode)
        val_ds = MertEmbeddingDataset(mert_large_manifest, mert_large_dir, "val", split_mode=split_mode)
        test_ds = MertEmbeddingDataset(mert_large_manifest, mert_large_dir, "test", split_mode=split_mode)
        bs = cfg["batch_size"]
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)
    elif cfg["model"] == "mert_large_deep_probe":
        from src.models.mert_deep_probe import MertDeepProbe
        from src.models.mert_embed_large import MERT_LARGE_HIDDEN_SIZE, MERT_LARGE_NUM_LAYERS

        model = MertDeepProbe(
            num_classes=NUM_CLASSES,
            num_layers=MERT_LARGE_NUM_LAYERS,
            hidden_size=MERT_LARGE_HIDDEN_SIZE,
            proj_size=cfg.get("proj_size", 512),
            n_groups=cfg.get("n_groups", 5),
            n_residual_blocks=cfg.get("n_residual_blocks", 2),
            dropout=cfg.get("dropout", 0.4),
        )
        mert_large_dir = DATA_DIR / "embeddings" / "mert_large"
        mert_large_manifest = mert_large_dir / "manifest.csv"
        split_mode = cfg.get("split_mode", "official")
        train_ds = MertEmbeddingDataset(mert_large_manifest, mert_large_dir, "train", split_mode=split_mode)
        val_ds = MertEmbeddingDataset(mert_large_manifest, mert_large_dir, "val", split_mode=split_mode)
        test_ds = MertEmbeddingDataset(mert_large_manifest, mert_large_dir, "test", split_mode=split_mode)
        bs = cfg["batch_size"]
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)
    elif cfg["model"] == "mert_large_probe":
        from src.models.mert_probe import MertProbe
        from src.models.mert_embed_large import MERT_LARGE_HIDDEN_SIZE, MERT_LARGE_NUM_LAYERS

        model = MertProbe(
            num_classes=NUM_CLASSES,
            num_layers=MERT_LARGE_NUM_LAYERS,
            hidden_size=MERT_LARGE_HIDDEN_SIZE,
            proj_size=cfg.get("proj_size", 512),
            dropout=cfg.get("dropout", 0.3),
            learn_layer_weights=cfg.get("learn_layer_weights", True),
            layer_index=cfg.get("layer_index"),
        )
        mert_large_dir = DATA_DIR / "embeddings" / "mert_large"
        mert_large_manifest = mert_large_dir / "manifest.csv"
        split_mode = cfg.get("split_mode", "official")
        train_ds = MertEmbeddingDataset(mert_large_manifest, mert_large_dir, "train", split_mode=split_mode)
        val_ds = MertEmbeddingDataset(mert_large_manifest, mert_large_dir, "val", split_mode=split_mode)
        test_ds = MertEmbeddingDataset(mert_large_manifest, mert_large_dir, "test", split_mode=split_mode)
        bs = cfg["batch_size"]
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)
    elif cfg["model"] == "densenet":
        from src.models.densenet import GenreDenseNet

        model = GenreDenseNet(
            num_classes=NUM_CLASSES,
            dropout=cfg.get("dropout", 0.3),
            pretrained=cfg.get("pretrained", True),
        )
        train_loader, val_loader, test_loader = build_cnn_loaders(cfg)
    elif cfg["model"] == "dual_cnn":
        from src.models.dual_cnn import DualCNN

        model = DualCNN(
            num_classes=NUM_CLASSES,
            dropout=cfg.get("dropout", 0.3),
        )
        train_loader, val_loader, test_loader = build_cnn_loaders(cfg)
    elif cfg["model"] == "cnn_bigru":
        from src.models.cnn_bigru import CNNBiGRU

        model = CNNBiGRU(
            num_classes=NUM_CLASSES,
            dropout=cfg.get("dropout", 0.3),
            gru_hidden=cfg.get("gru_hidden", 128),
            gru_layers=cfg.get("gru_layers", 2),
        )
        train_loader, val_loader, test_loader = build_cnn_loaders(cfg)
    elif cfg["model"] == "capsnet":
        from src.models.capsnet import CapsNet

        model = CapsNet(
            num_classes=NUM_CLASSES,
            num_routing=cfg.get("num_routing", 3),
        )
        train_loader, val_loader, test_loader = build_cnn_loaders(cfg)
    elif cfg["model"] == "cnn_transformer":
        from src.models.cnn_transformer import CNNTransformer

        model = CNNTransformer(
            num_classes=NUM_CLASSES,
            dropout=cfg.get("dropout", 0.3),
            d_model=cfg.get("d_model", 256),
            nhead=cfg.get("nhead", 4),
            num_transformer_layers=cfg.get("num_transformer_layers", 2),
        )
        train_loader, val_loader, test_loader = build_cnn_loaders(cfg)
    elif cfg["model"] == "embedding_probe":
        from src.models.embedding_probe import EmbeddingProbe

        model = EmbeddingProbe(
            num_classes=NUM_CLASSES,
            embed_dim=cfg["embed_dim"],
            hidden_sizes=tuple(cfg.get("hidden_sizes", (512, 256))),
            dropout=cfg.get("dropout", 0.4),
        )
        # BYOL-A and PANNs store one flat vector per episode, so MertEmbeddingDataset's
        # "load the track's array, index the episode" logic applies unchanged.
        embed_dir = DATA_DIR / "embeddings" / cfg["embedding_name"]
        embed_manifest = embed_dir / "manifest.csv"
        split_mode = cfg.get("split_mode", "official")
        bs = cfg["batch_size"]
        loaders = []
        for split, shuffle in (("train", True), ("val", False), ("test", False)):
            ds = MertEmbeddingDataset(embed_manifest, embed_dir, split, split_mode=split_mode)
            loaders.append(DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0))
        train_loader, val_loader, test_loader = loaders
    else:
        raise ValueError(f"Unknown model type: {cfg['model']}")

    model.to(device)
    from src.training.losses import build_criterion
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"], cfg.get("weight_decay", 0.0))
    scheduler = build_scheduler(cfg.get("scheduler"), optimizer, cfg["epochs"])

    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1", "seconds"]
        )

    best_val_f1 = -1.0
    best_epoch = -1
    patience = cfg.get("early_stopping_patience", 10)
    epochs_since_improvement = 0

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        mixup_alpha = cfg.get("mixup_alpha", 0.0)
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer, device, train=True, mixup_alpha=mixup_alpha)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        if scheduler is not None:
            scheduler.step()
        elapsed = time.time() - t0

        log.info(
            "epoch %d/%d train_loss=%.4f train_f1=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f (%.1fs)",
            epoch, cfg["epochs"], train_loss, train_f1, val_loss, val_acc, val_f1, elapsed,
        )
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, elapsed])

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                log.info("Early stopping at epoch %d (no val improvement for %d epochs)", epoch, patience)
                break

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    test_loss, test_acc, test_f1 = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    log.info("Test (best checkpoint, epoch %d): acc=%.4f macro_f1=%.4f", best_epoch, test_acc, test_f1)

    summary = {
        "run_name": cfg["run_name"],
        "model": cfg["model"],
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "device": str(device),
        "config": cfg,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote summary to %s", run_dir / "summary.json")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
