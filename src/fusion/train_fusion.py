"""Train and evaluate one ESCF ablation row, over cached per-episode outputs.

Supervision is at track level, which is also the evaluation unit, so the loss the
model minimises is the quantity the paper reports. Because both backbones are frozen
and cached, a full run here takes seconds -- which is what makes 5 seeds per row
affordable, and multi-seed variance is not optional for a peer-reviewed claim.

Usage:
    python -m src.fusion.train_fusion --variant conditional --schedule cosine --seed 0
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.config import OUTPUT_DIR
from src.evaluation.metrics import compute_metrics
from src.fusion.data import build_track_batch
from src.fusion.escf import ESCF
from src.models.common import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FUSION_DIR = OUTPUT_DIR / "fusion"


def iterate_minibatches(batch, batch_size: int, shuffle: bool, generator=None):
    n = len(batch)
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield type(batch)(
            batch.emb_a[idx], batch.emb_b[idx], batch.logits_a[idx], batch.logits_b[idx],
            batch.mask[idx], batch.labels[idx], batch.track_ids[idx.numpy()],
        )


@torch.no_grad()
def evaluate(model, batch, epoch, total_epochs, batch_size=256) -> tuple[dict, np.ndarray]:
    model.eval()
    preds = []
    for chunk in iterate_minibatches(batch, batch_size, shuffle=False):
        logits = model(chunk, epoch=epoch, total_epochs=total_epochs)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(preds)
    return compute_metrics(batch.labels.cpu().tolist(), preds.tolist()), preds


def train_fusion(cfg: dict) -> dict:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = get_device()

    run_a, run_b = cfg["run_a"], cfg["run_b"]
    splits = {s: build_track_batch(run_a, run_b, s).to(device) for s in ("train", "val", "test")}

    model = ESCF(
        dim_a=splits["train"].emb_a.shape[-1],
        dim_b=splits["train"].emb_b.shape[-1],
        variant=cfg["variant"],
        use_attention=cfg.get("use_attention", True),
        hidden=cfg.get("hidden", 128),
        dropout=cfg.get("dropout", 0.2),
        alpha_prior=cfg.get("alpha_prior", 0.5),
        fixed_alpha=cfg.get("fixed_alpha", 0.5),
        schedule=cfg.get("schedule", "cosine"),
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    epochs = cfg.get("epochs", 60)
    criterion = nn.CrossEntropyLoss()

    # Variants with no parameters (cnn_only/mert_only/fixed without attention) still
    # need an optimizer object for the loop; give it a dummy so the code path is one.
    optimizer = (
        torch.optim.AdamW(trainable, lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4))
        if trainable
        else None
    )

    generator = torch.Generator().manual_seed(cfg["seed"])
    best_val_f1, best_state, best_epoch = -1.0, None, 0
    patience = cfg.get("early_stopping_patience", 12)
    since_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        if optimizer is not None:
            for chunk in iterate_minibatches(splits["train"], cfg.get("batch_size", 64), True, generator):
                logits = model(chunk, epoch=epoch, total_epochs=epochs)
                loss = criterion(logits, chunk.labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        val_metrics, _ = evaluate(model, splits["val"], epoch, epochs)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= patience:
                break
        if optimizer is None:  # nothing to learn; one pass is the whole run
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate the selected model at the schedule's endpoint: at inference there is no
    # "current epoch", and the deployed gate is the fully-trusted one (lambda = 1).
    val_metrics, _ = evaluate(model, splits["val"], epochs, epochs)
    test_metrics, test_preds = evaluate(model, splits["test"], epochs, epochs)

    summary = {
        "row": cfg.get("row", cfg["variant"]),
        "variant": cfg["variant"],
        "schedule": cfg.get("schedule", "cosine"),
        "use_attention": cfg.get("use_attention", True),
        "seed": cfg["seed"],
        "best_epoch": best_epoch,
        "val_macro_f1": val_metrics["macro_f1"],
        "val_accuracy": val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "per_genre": test_metrics["per_genre"],
        "n_test_tracks": len(splits["test"]),
        "config": cfg,
    }

    out_dir = FUSION_DIR / f"{summary['row']}_seed{cfg['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.save(out_dir / "test_preds.npy", test_preds)
    np.save(out_dir / "test_labels.npy", splits["test"].labels.cpu().numpy())
    torch.save(model.state_dict(), out_dir / "fusion.pt")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", default="cnn_tuned_lr")
    parser.add_argument("--run-b", default="mert_probe_baseline")
    parser.add_argument("--variant", default="conditional", choices=ESCF.VARIANTS)
    parser.add_argument("--schedule", default="cosine")
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--fixed-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--row", default=None)
    args = parser.parse_args()

    summary = train_fusion(
        {
            "run_a": args.run_a,
            "run_b": args.run_b,
            "variant": args.variant,
            "schedule": args.schedule,
            "use_attention": not args.no_attention,
            "fixed_alpha": args.fixed_alpha,
            "seed": args.seed,
            "epochs": args.epochs,
            "row": args.row or args.variant,
        }
    )
    log.info(
        "%s seed=%d: test_acc=%.4f test_macro_f1=%.4f",
        summary["row"], summary["seed"], summary["test_accuracy"], summary["test_macro_f1"],
    )


if __name__ == "__main__":
    main()
