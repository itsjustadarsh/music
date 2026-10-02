"""Dump per-episode logits and embeddings for a trained backbone, for every split.

This is what makes the fusion study affordable. Each ablation row in the paper
otherwise needs a forward pass over ~75k episodes through both backbones; cached
once, every fusion variant trains on arrays in memory, so a 10-row grid at 5 seeds
costs minutes instead of days.

Outputs `outputs/<run>/episode_outputs.npz` with parallel arrays keyed by
(track_id, subclip_idx) so two backbones' caches can be joined exactly.

Usage:
    python -m src.evaluation.cache_outputs --run cnn_tuned_lr
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import torch
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
SPLITS = ("train", "val", "test")


def _build(run_name: str, cfg: dict, split: str):
    """Return (model, dataset, collate_fn) for one split of a trained run.

    The run's own `split_mode` is honoured. Defaulting it to `official` here would
    silently cache a leaky run against the artist-filtered partition, producing a
    file that looks comparable to every other cache and is not.
    """
    collate_fn = None
    split_mode = cfg.get("split_mode", "official")
    if cfg["model"] == "cnn":
        model = GenreCNN(num_classes=NUM_CLASSES)
        ds = LogMelDataset(MANIFEST_PATH, PROCESSED_DIR, split, augment=False,
                           return_track_ids=True, split_mode=split_mode)
    elif cfg["model"] == "mert_probe":
        from src.models.mert_probe import MertProbe

        model = MertProbe(
            num_classes=NUM_CLASSES,
            proj_size=cfg.get("proj_size", 512),
            dropout=cfg.get("dropout", 0.3),
            learn_layer_weights=cfg.get("learn_layer_weights", True),
            layer_index=cfg.get("layer_index"),
        )
        ds = MertEmbeddingDataset(
            MERT_MANIFEST_PATH, MERT_EMBEDDINGS_DIR, split,
            return_track_ids=True, split_mode=split_mode,
        )
    elif cfg["model"] == "mert_attn_probe":
        from src.models.mert_attn_probe import MertAttentionProbe

        model = MertAttentionProbe(
            num_classes=NUM_CLASSES,
            n_heads=cfg.get("n_heads", 8),
            attn_layers=cfg.get("attn_layers", 2),
            mlp_size=cfg.get("mlp_size", 512),
            dropout=cfg.get("dropout", 0.3),
        )
        ds = MertEmbeddingDataset(
            MERT_MANIFEST_PATH, MERT_EMBEDDINGS_DIR, split,
            return_track_ids=True, split_mode=split_mode,
        )
    elif cfg["model"] == "resnet":
        from src.models.resnet_cnn import GenreResNet

        model = GenreResNet(
            num_classes=NUM_CLASSES,
            dropout=cfg.get("dropout", 0.4),
            embed_dim=cfg.get("embed_dim", 256),
        )
        ds = LogMelDataset(MANIFEST_PATH, PROCESSED_DIR, split, augment=False,
                           return_track_ids=True, split_mode=split_mode)
    elif cfg["model"] == "mert_deep_probe":
        from src.models.mert_deep_probe import MertDeepProbe

        model = MertDeepProbe(
            num_classes=NUM_CLASSES,
            proj_size=cfg.get("proj_size", 512),
            n_groups=cfg.get("n_groups", 3),
            n_residual_blocks=cfg.get("n_residual_blocks", 2),
            dropout=cfg.get("dropout", 0.4),
        )
        ds = MertEmbeddingDataset(
            MERT_MANIFEST_PATH, MERT_EMBEDDINGS_DIR, split,
            return_track_ids=True, split_mode=split_mode,
        )
    elif cfg["model"] in ("mert_flat_probe",):
        from src.models.mert_flat_probe import MertFlatProbe

        model = MertFlatProbe(
            num_classes=NUM_CLASSES,
            proj_size=cfg.get("proj_size", 512),
            dropout=cfg.get("dropout", 0.5),
        )
        ds = MertEmbeddingDataset(
            MERT_MANIFEST_PATH, MERT_EMBEDDINGS_DIR, split,
            return_track_ids=True, split_mode=split_mode,
        )
    elif cfg["model"] == "mert_large_probe":
        from src.models.mert_probe import MertProbe
        from src.models.mert_embed_large import MERT_LARGE_HIDDEN_SIZE, MERT_LARGE_NUM_LAYERS

        mert_large_dir = DATA_DIR / "embeddings" / "mert_large"
        mert_large_manifest = mert_large_dir / "manifest.csv"
        model = MertProbe(
            num_classes=NUM_CLASSES,
            num_layers=MERT_LARGE_NUM_LAYERS,
            hidden_size=MERT_LARGE_HIDDEN_SIZE,
            proj_size=cfg.get("proj_size", 512),
            dropout=cfg.get("dropout", 0.3),
            learn_layer_weights=cfg.get("learn_layer_weights", True),
            layer_index=cfg.get("layer_index"),
        )
        ds = MertEmbeddingDataset(
            mert_large_manifest, mert_large_dir, split,
            return_track_ids=True, split_mode=split_mode,
        )
    elif cfg["model"] == "mert_large_flat_probe":
        from src.models.mert_flat_probe import MertFlatProbe
        from src.models.mert_embed_large import MERT_LARGE_HIDDEN_SIZE, MERT_LARGE_NUM_LAYERS

        mert_large_dir = DATA_DIR / "embeddings" / "mert_large"
        mert_large_manifest = mert_large_dir / "manifest.csv"
        model = MertFlatProbe(
            num_classes=NUM_CLASSES,
            num_layers=MERT_LARGE_NUM_LAYERS,
            hidden_size=MERT_LARGE_HIDDEN_SIZE,
            proj_size=cfg.get("proj_size", 512),
            dropout=cfg.get("dropout", 0.5),
        )
        ds = MertEmbeddingDataset(
            mert_large_manifest, mert_large_dir, split,
            return_track_ids=True, split_mode=split_mode,
        )
    elif cfg["model"] == "mert_large_deep_probe":
        from src.models.mert_deep_probe import MertDeepProbe
        from src.models.mert_embed_large import MERT_LARGE_HIDDEN_SIZE, MERT_LARGE_NUM_LAYERS

        mert_large_dir = DATA_DIR / "embeddings" / "mert_large"
        mert_large_manifest = mert_large_dir / "manifest.csv"
        model = MertDeepProbe(
            num_classes=NUM_CLASSES,
            num_layers=MERT_LARGE_NUM_LAYERS,
            hidden_size=MERT_LARGE_HIDDEN_SIZE,
            proj_size=cfg.get("proj_size", 512),
            n_groups=cfg.get("n_groups", 5),
            n_residual_blocks=cfg.get("n_residual_blocks", 2),
            dropout=cfg.get("dropout", 0.4),
        )
        ds = MertEmbeddingDataset(
            mert_large_manifest, mert_large_dir, split,
            return_track_ids=True, split_mode=split_mode,
        )
    elif cfg["model"] == "embedding_probe":
        from src.models.embedding_probe import EmbeddingProbe

        # BYOL-A and PANNs store one flat vector per episode, so MertEmbeddingDataset's
        # "load the track's array, index the episode" logic applies unchanged.
        embed_dir = DATA_DIR / "embeddings" / cfg["embedding_name"]
        model = EmbeddingProbe(
            num_classes=NUM_CLASSES,
            embed_dim=cfg["embed_dim"],
            hidden_sizes=tuple(cfg.get("hidden_sizes", (512, 256))),
            dropout=cfg.get("dropout", 0.4),
        )
        ds = MertEmbeddingDataset(
            embed_dir / "manifest.csv", embed_dir, split,
            return_track_ids=True, split_mode=split_mode,
        )
    else:
        raise ValueError(f"Caching not supported for model type {cfg['model']!r}")
    return model, ds, collate_fn


def cache_run(run_name: str) -> dict:
    run_dir = OUTPUT_DIR / run_name
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)

    device = get_device()
    arrays: dict[str, np.ndarray] = {}

    for split in SPLITS:
        model, ds, collate_fn = _build(run_name, cfg, split)
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
        model.to(device).eval()

        # shuffle=False, so dataset row order is the emission order; we still verify
        # the track ids coming back match the manifest rather than assuming it.
        loader_kwargs = dict(batch_size=cfg["batch_size"], shuffle=False)
        if collate_fn is not None:
            loader_kwargs["collate_fn"] = collate_fn
        loader = DataLoader(ds, **loader_kwargs)

        logits_out, embed_out, labels_out, track_out = [], [], [], []
        with torch.no_grad():
            for inputs, labels, track_ids in loader:
                logits, embedding = model(inputs.to(device), return_embedding=True)
                logits_out.append(logits.cpu().numpy().astype(np.float32))
                embed_out.append(embedding.cpu().numpy().astype(np.float32))
                labels_out.append(labels.numpy())
                track_out.append(track_ids.numpy())

        track_ids = np.concatenate(track_out)
        expected = ds.df["track_id"].to_numpy()
        if not np.array_equal(track_ids, expected):
            raise RuntimeError(
                f"{run_name}/{split}: loader order does not match manifest order; "
                "per-episode caches would be misaligned across backbones"
            )

        arrays[f"{split}_logits"] = np.concatenate(logits_out)
        arrays[f"{split}_embeddings"] = np.concatenate(embed_out)
        arrays[f"{split}_labels"] = np.concatenate(labels_out)
        arrays[f"{split}_track_ids"] = track_ids
        arrays[f"{split}_subclip_idx"] = ds.df["subclip_idx"].to_numpy()
        log.info("%s/%s: %d episodes", run_name, split, len(track_ids))

    out_path = run_dir / "episode_outputs.npz"
    np.savez_compressed(out_path, **arrays)
    log.info("Wrote %s", out_path)
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    cache_run(args.run)


if __name__ == "__main__":
    main()
