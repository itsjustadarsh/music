"""Single-process, single-GPU driver: load FMA-small into RAM once, train every
config in order, evaluate on the test split, and ensemble — no dataset on disk.

Usage:
    export CUDA_VISIBLE_DEVICES=<gpu id>
    python run_pipeline.py                         # all configs, full data
    python run_pipeline.py --only mert_lora_95m    # subset (comma-separated run names)
    python run_pipeline.py --smoke                 # synthetic audio, 1 tiny epoch per config

Outputs per run in outputs/<run_name>/: config.json, metrics.csv, eval_metrics.json,
summary.json, test_probs.npz, and trainable_weights.pt (only if < 50 MB).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pipeline")

RUN_ORDER = [
    "mert_lora_330m",
    "mert_lora_95m",
    "mert_unfreeze_330m",
    "ast_finetune",
    "htsat_genre",
    "efficientat_genre",
    "mert_multicrop_330m",
]
# Ensemble used in run_all.sh
RUN_ALL_ENSEMBLE = ["mert_lora_330m", "mert_lora_95m", "ast_finetune", "htsat_genre"]


def load_cfg(name: str) -> dict:
    with open(Path("configs") / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def fake_cache(sample_rates: list[int], per_split: int = 16) -> None:
    """Synthetic in-memory dataset for a fast end-to-end code check."""
    from data import memory_cache
    from data.dataset import FMA_GENRES
    rng = np.random.default_rng(0)
    rows = []
    tid = 1
    for split in ("train", "val", "test"):
        for i in range(per_split):
            rows.append({"track_id": tid, "genre": FMA_GENRES[i % 8], "split": split, "subset": "small"})
            tid += 1
    meta = pd.DataFrame(rows)
    memory_cache.METADATA = meta
    memory_cache.ROW_OF = {int(t): i for i, t in enumerate(meta["track_id"])}
    memory_cache.AUDIO = {
        sr: (0.1 * rng.standard_normal((len(meta), int(30 * sr)))).astype(np.float32)
        for sr in sample_rates
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="comma-separated run names")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    runs = [r for r in RUN_ORDER if not args.only or r in args.only.split(",")]
    cfgs = {r: load_cfg(r) for r in runs}
    srs = sorted({c.get("target_sr", 24000) for c in cfgs.values()})

    from data import memory_cache
    if args.smoke:
        fake_cache(srs)
    else:
        memory_cache.load(srs)

    import torch
    from train import train
    from ensemble import ensemble_from_track_probs, load_run

    out_root = Path("outputs_smoke" if args.smoke else "outputs")
    results, failed = {}, {}
    for i, name in enumerate(runs, 1):
        cfg = cfgs[name]
        if args.smoke:
            cfg.update(epochs=1, batch_size=min(cfg["batch_size"], 8), warmup_steps=1, grad_accum_steps=1)
        cfg["_out_root"] = str(out_root)
        log.info("=== [%d/%d] %s ===", i, len(runs), name)
        t0 = time.time()
        try:
            m = train(cfg)
            results[name] = {
                "track_acc": m["track_level"]["accuracy"],
                "track_f1": m["track_level"]["macro_f1"],
                "clip_acc": m["clip_level"]["accuracy"],
                "best_epoch": m["best_epoch"],
                "minutes": round((time.time() - t0) / 60, 1),
            }
            log.info("=== %s done: %s", name, results[name])
        except Exception as e:  # keep going with the next model
            failed[name] = f"{type(e).__name__}: {e}"
            log.error("=== %s FAILED after %.1f min:\n%s", name, (time.time() - t0) / 60, traceback.format_exc())
        torch.cuda.empty_cache()

    ensembles = {}
    done = [r for r in runs if r in results]
    for members in ({*RUN_ALL_ENSEMBLE} & set(done), set(done)):
        members = [r for r in runs if r in members]
        if len(members) >= 2 and "+".join(members) not in ensembles:
            per_run = {r: load_run(out_root / r) for r in members}
            e = ensemble_from_track_probs(per_run, out_root=out_root)
            ensembles["+".join(members)] = {
                "track_acc": e["track_level"]["accuracy"],
                "track_f1": e["track_level"]["macro_f1"],
            }

    summary = {"runs": results, "failed": failed, "ensembles": ensembles}
    with open(out_root / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("RESULTS SUMMARY:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
