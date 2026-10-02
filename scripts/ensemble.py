"""Ensemble cached per-episode outputs into a single track-level prediction.

Replaces nine near-duplicate ensemble scripts. They differed mainly in which
combination rule they tried, and each one printed a table of test accuracies for
every rule it tried — so picking "the best ensemble" across those scripts was
model selection on the test set. This script makes that structurally impossible:

  * **Everything is decided on validation.** Member selection, the number of
    members, and (for `--strategy weighted`) the combination weights are all
    fitted or ranked on `val`. Test is touched once, at the end, for the single
    configuration that validation chose.
  * **Uniform weights are the default**, because `docs/NEGATIVE_RESULTS.md` §3
    measured learned combination weights losing to fixed ones at this data scale
    — 6,400 train / 800 val tracks is not enough to fit a combiner on top of
    frozen features. `--strategy weighted` exists so that finding stays
    reproducible, not because it is recommended.
  * **One member per architecture** by default. Six seeds of the same probe are a
    seed ensemble, not a diverse one, and averaging them mostly measures seed
    noise. `--per-architecture all` opts out.
  * **Artist-filtered members only.** A run with a non-`official` split_mode is a
    diagnostic instrument (`docs/METHODOLOGY.md` §2.1) and may not enter a
    reported result; including one would be the exact substitution this project
    documents.

Writes `outputs/<name>/{config.json,eval_metrics.json}` in the same shape a
trained run emits, so `scripts/update_results.py` picks the ensemble up with no
special-casing and it lands in the headline table on equal terms.

Usage:
    python scripts/ensemble.py
    python scripts/ensemble.py --strategy weighted --name ensemble_weighted
    python scripts/ensemble.py --members cnn_improved mert_flat_probe_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class Member:
    """One candidate ensemble member, reduced to track-level probabilities."""

    def __init__(self, run_name: str):
        self.name = run_name
        d = OUTPUT_DIR / run_name
        self.cfg = json.loads((d / "config.json").read_text())
        npz = np.load(d / "episode_outputs.npz")
        self.probs: dict[str, dict[int, np.ndarray]] = {}
        self.labels: dict[str, dict[int, int]] = {}
        for split in ("val", "test"):
            per_track: dict[int, list[np.ndarray]] = defaultdict(list)
            lab: dict[int, int] = {}
            for p, l, t in zip(softmax(npz[f"{split}_logits"]),
                               npz[f"{split}_labels"], npz[f"{split}_track_ids"]):
                per_track[int(t)].append(p)
                lab[int(t)] = int(l)
            self.probs[split] = {t: np.mean(ps, axis=0) for t, ps in per_track.items()}
            self.labels[split] = lab

    @property
    def split_mode(self) -> str:
        return self.cfg.get("split_mode", "official")

    @property
    def architecture(self) -> str:
        m = self.cfg.get("model", "unknown")
        if m == "embedding_probe":
            return f"{m}:{self.cfg.get('embedding_name', '?')}"
        return m

    def score(self, split: str, tracks: list[int]) -> tuple[float, float]:
        preds = [int(self.probs[split][t].argmax()) for t in tracks]
        truth = [self.labels[split][t] for t in tracks]
        m = compute_metrics(truth, preds)
        return m["accuracy"], m["macro_f1"]


def discover() -> list[Member]:
    members = []
    for d in sorted(OUTPUT_DIR.iterdir()):
        if d.is_dir() and (d / "episode_outputs.npz").exists() and (d / "config.json").exists():
            members.append(Member(d.name))
    return members


def choose_members(candidates: list[Member], per_architecture: str,
                   val_tracks: list[int]) -> list[Member]:
    """Rank on validation macro-F1 only. Test is not consulted here or anywhere upstream."""
    if per_architecture == "all":
        return candidates
    best: dict[str, tuple[float, Member]] = {}
    for m in candidates:
        f1 = m.score("val", val_tracks)[1]
        if m.architecture not in best or f1 > best[m.architecture][0]:
            best[m.architecture] = (f1, m)
    return [m for _, m in sorted(best.values(), key=lambda x: -x[0])]


def fit_weights(members: list[Member], val_tracks: list[int], strategy: str) -> np.ndarray:
    n = len(members)
    if strategy == "uniform":
        return np.ones(n) / n

    truth = np.array([members[0].labels["val"][t] for t in val_tracks])
    stacked = np.stack([[m.probs["val"][t] for t in val_tracks] for m in members])  # (n, T, C)

    def neg_acc(raw: np.ndarray) -> float:
        w = np.abs(raw)
        w = w / w.sum()
        preds = np.tensordot(w, stacked, axes=(0, 0)).argmax(axis=1)
        return -float((preds == truth).mean())

    res = minimize(neg_acc, np.ones(n) / n, method="Nelder-Mead",
                   options={"maxiter": 5000, "adaptive": True})
    w = np.abs(res.x)
    return w / w.sum()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default="uniform", choices=["uniform", "weighted"],
                    help="uniform (default, see NEGATIVE_RESULTS.md §3) or Nelder-Mead on val")
    ap.add_argument("--members", nargs="*", default=None,
                    help="explicit run names; default is automatic selection on val")
    ap.add_argument("--per-architecture", default="best", choices=["best", "all"],
                    help="'best' keeps one run per architecture (default); 'all' keeps every run")
    ap.add_argument("--top-k", type=int, default=None,
                    help="keep only the K members with the highest val macro-F1")
    ap.add_argument("--name", default=None, help="output run name")
    args = ap.parse_args()

    candidates = discover()
    if args.members:
        missing = [r for r in args.members if r not in {m.name for m in candidates}]
        if missing:
            raise SystemExit(f"No cached episode_outputs.npz for: {', '.join(missing)}")
        candidates = [m for m in candidates if m.name in args.members]

    leaky = [m for m in candidates if m.split_mode != "official"]
    if leaky:
        log.info("Excluding %d non-official run(s) — diagnostic only: %s",
                 len(leaky), ", ".join(m.name for m in leaky))
        candidates = [m for m in candidates if m.split_mode == "official"]
    if not candidates:
        raise SystemExit("No artist-filtered runs with cached outputs. Run src.evaluation.cache_outputs.")

    val_tracks = sorted(set.intersection(*[set(m.probs["val"]) for m in candidates]))
    test_tracks = sorted(set.intersection(*[set(m.probs["test"]) for m in candidates]))
    log.info("%d candidate runs; %d val / %d test tracks in common",
             len(candidates), len(val_tracks), len(test_tracks))

    members = choose_members(candidates, args.per_architecture, val_tracks)
    if args.top_k:
        members = sorted(members, key=lambda m: -m.score("val", val_tracks)[1])[:args.top_k]
    log.info("Members (%d): %s", len(members), ", ".join(m.name for m in members))

    print("\nMembers, ranked by validation macro-F1 (test not consulted):")
    for m in members:
        v_acc, v_f1 = m.score("val", val_tracks)
        print(f"  {m.name:32s} {m.architecture:24s} val acc={v_acc:.4f} f1={v_f1:.4f}")

    weights = fit_weights(members, val_tracks, args.strategy)
    if args.strategy != "uniform":
        print("\nWeights fitted on validation:")
        for m, w in sorted(zip(members, weights), key=lambda x: -x[1]):
            print(f"  {m.name:32s} {w:.4f}")

    # Test is read exactly once, for the configuration validation already chose.
    def combine(split: str, tracks: list[int]) -> tuple[list[int], list[int]]:
        preds = [int(sum(w * m.probs[split][t] for w, m in zip(weights, members)).argmax())
                 for t in tracks]
        truth = [members[0].labels[split][t] for t in tracks]
        return truth, preds

    val_metrics = compute_metrics(*combine("val", val_tracks))
    test_metrics = compute_metrics(*combine("test", test_tracks))

    best_single = max(members, key=lambda m: m.score("val", val_tracks)[1])
    bs_val, _ = best_single.score("val", val_tracks)
    bs_test, bs_test_f1 = best_single.score("test", test_tracks)

    print(f"\nBest single member ({best_single.name}): "
          f"val acc={bs_val:.4f} | test acc={bs_test:.4f} f1={bs_test_f1:.4f}")
    print(f"Ensemble ({args.strategy}, {len(members)} members): "
          f"val acc={val_metrics['accuracy']:.4f} | "
          f"test acc={test_metrics['accuracy']:.4f} f1={test_metrics['macro_f1']:.4f}")
    print(f"Delta over best single member, test: {100 * (test_metrics['accuracy'] - bs_test):+.1f}pp")

    name = args.name or f"ensemble_{args.strategy}"
    out = OUTPUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "run_name": name,
        "model": "ensemble",
        "split_mode": "official",
        "strategy": args.strategy,
        "members": [m.name for m in members],
        "weights": [float(w) for w in weights],
        "per_architecture": args.per_architecture,
        "top_k": args.top_k,
    }, indent=2))
    (out / "eval_metrics.json").write_text(json.dumps({
        "run_name": name,
        "track_level": test_metrics,
        "val_track_level": val_metrics,
        "n_tracks": len(test_tracks),
    }, indent=2))
    log.info("Wrote %s — rerun scripts/update_results.py to pick it up", out)


if __name__ == "__main__":
    main()
