"""Round 2: aim for > 60% track accuracy on the artist-filtered FMA-small test split.

1. Frozen-encoder probes: extract layer-wise embeddings (kept in RAM) from
   Whisper-large-v3 encoder, MERT-330M, CLAP-music and AST, then train a small
   layer-weighted MLP probe per encoder (3 seeds averaged), plus a fusion probe
   over all encoders.
2. AST v2 fine-tune (configs/ast_v2.yaml): random crops + SpecAugment.
3. Ensemble chosen on the VALIDATION split (greedy forward selection with
   replacement); test is only scored at the end.

Usage:
    export CUDA_VISIBLE_DEVICES=<gpu>
    python run_round2.py                 # everything
    python run_round2.py --skip-finetune # probes + ensemble only
    python run_round2.py --smoke         # synthetic audio, tiny, for code checks
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("round2")

DEV = torch.device("cuda")
SPLITS = ("train", "val", "test")
TRACK_IDS: dict = {}


# --------------------------------------------------------------------------- data
def clips(arr: np.ndarray, rows: np.ndarray, sr: int, sec: float) -> torch.Tensor:
    """(n_rows, n_clips, clip_samples) non-overlapping windows from 30s tracks."""
    n = int(sec * sr)
    k = arr.shape[1] // n
    return torch.from_numpy(arr[rows, : k * n].reshape(len(rows), k, n))


# --------------------------------------------------------------------- extractors
@torch.no_grad()
def extract_whisper(audio16: np.ndarray, bs: int = 32) -> np.ndarray:
    """Whisper-large-v3 encoder on full 30s: (N, 1, L=33, 1280), time-mean per layer."""
    from transformers import WhisperFeatureExtractor, WhisperModel
    name = "openai/whisper-large-v3"
    fe = WhisperFeatureExtractor.from_pretrained(name)
    enc = WhisperModel.from_pretrained(name, torch_dtype=torch.bfloat16).encoder.to(DEV).eval()
    out = []
    for i in range(0, len(audio16), bs):
        feats = fe(list(audio16[i : i + bs]), sampling_rate=16000, return_tensors="pt")
        x = feats["input_features"].to(DEV, torch.bfloat16)
        hs = enc(x, output_hidden_states=True).hidden_states
        out.append(torch.stack([h.float().mean(1) for h in hs], 1).half().cpu())
    del enc
    return torch.cat(out).unsqueeze(1).numpy()


@torch.no_grad()
def extract_mert(audio24: np.ndarray, bs: int = 16) -> np.ndarray:
    """MERT-330M on 10 x 3s clips: (N, 10, L=25, 1024)."""
    from models.mert_finetune import load_mert_backbone
    m, _ = load_mert_backbone("m-a-p/MERT-v1-330M", gradient_checkpointing=False)
    m = m.to(DEV).eval()
    out = []
    rows = np.arange(len(audio24))
    for i in range(0, len(rows), bs):
        c = clips(audio24, rows[i : i + bs], 24000, 3.0).to(DEV)
        b, k, n = c.shape
        x = c.reshape(b * k, n)
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-7)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = m(input_values=x, output_hidden_states=True).hidden_states
        e = torch.stack([h.float().mean(1) for h in hs], 1)
        out.append(e.reshape(b, k, *e.shape[1:]).half().cpu())
    del m
    return torch.cat(out).numpy()


@torch.no_grad()
def extract_clap(audio48: np.ndarray, bs: int = 16) -> np.ndarray:
    """CLAP (laion/larger_clap_music) on 3 x 10s clips: (N, 3, L=1, 512)."""
    from transformers import ClapFeatureExtractor, ClapModel
    name = "laion/larger_clap_music"
    fe = ClapFeatureExtractor.from_pretrained(name)
    m = ClapModel.from_pretrained(name).to(DEV).eval()
    out = []
    rows = np.arange(len(audio48))
    for i in range(0, len(rows), bs):
        c = clips(audio48, rows[i : i + bs], 48000, 10.0)
        b, k, n = c.shape
        feats = fe(list(c.reshape(b * k, n).numpy()), sampling_rate=48000, return_tensors="pt")
        feats = {kk: v.to(DEV) for kk, v in feats.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = m.get_audio_features(**feats).float()
        out.append(e.reshape(b, k, 1, -1).half().cpu())
    del m
    return torch.cat(out).numpy()


@torch.no_grad()
def extract_ast(audio16: np.ndarray, bs: int = 16) -> np.ndarray:
    """AudioSet AST on 3 x 10s clips: (N, 3, L=13, 768), token-mean per layer."""
    from transformers import ASTFeatureExtractor, ASTModel
    name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    fe = ASTFeatureExtractor.from_pretrained(name)
    m = ASTModel.from_pretrained(name).to(DEV).eval()
    out = []
    rows = np.arange(len(audio16))
    for i in range(0, len(rows), bs):
        c = clips(audio16, rows[i : i + bs], 16000, 10.0)
        b, k, n = c.shape
        feats = fe(list(c.reshape(b * k, n).numpy()), sampling_rate=16000, return_tensors="pt")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = m(input_values=feats["input_values"].to(DEV), output_hidden_states=True).hidden_states
        e = torch.stack([h.float().mean(1) for h in hs], 1)
        out.append(e.reshape(b, k, *e.shape[1:]).half().cpu())
    del m
    return torch.cat(out).numpy()


# -------------------------------------------------------------------------- probes
class LayerProbe(nn.Module):
    """Softmax-weighted sum over layers -> LayerNorm -> MLP."""

    def __init__(self, n_layers: int, dim: int, n_classes: int = 8, hidden: int = 512, p: float = 0.4):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Dropout(0.2), nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(p), nn.Linear(hidden, n_classes),
        )

    def forward(self, x):  # x: (B, L, D)
        return self.net((x * torch.softmax(self.w, 0).view(1, -1, 1)).sum(1))


def track_probs_from_clips(model, X: torch.Tensor, bs: int = 512) -> np.ndarray:
    """X: (N, K, L, D) -> mean clip softmax per track (N, C)."""
    model.eval()
    n, k = X.shape[:2]
    flat = X.reshape(n * k, *X.shape[2:])
    with torch.no_grad():
        p = torch.cat([torch.softmax(model(flat[i : i + bs].float()), -1) for i in range(0, len(flat), bs)])
    return p.reshape(n, k, -1).mean(1).cpu().numpy()


def train_probe(X: dict, y: dict, seed: int, epochs: int = 80, patience: int = 12,
                lr: float = 1e-3, wd: float = 0.05, bs: int = 256, mixup: float = 0.2):
    """Train on clips (each clip gets its track label); select on val track accuracy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr = X["train"], y["train"]
    n, k = Xtr.shape[:2]
    flat = Xtr.reshape(n * k, *Xtr.shape[2:])
    yflat = ytr.repeat_interleave(k)
    model = LayerProbe(Xtr.shape[2], Xtr.shape[3]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * ((len(flat) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    best, best_state, bad = -1.0, None, 0
    yval = y["val"].cpu().numpy()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(flat), device=DEV)
        for i in range(0, len(flat), bs):
            idx = perm[i : i + bs]
            xb, yb = flat[idx].float(), yflat[idx]
            xb = xb + 0.05 * torch.randn_like(xb) * xb.std()
            lam = float(np.random.beta(mixup, mixup)) if mixup > 0 else 1.0
            j = torch.randperm(len(xb), device=DEV)
            logits = model(lam * xb + (1 - lam) * xb[j])
            loss = lam * F.cross_entropy(logits, yb, label_smoothing=0.1) + \
                (1 - lam) * F.cross_entropy(logits, yb[j], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            getattr(loss, "backward")()
            opt.step()
            sched.step()
        acc = (track_probs_from_clips(model, X["val"]).argmax(1) == yval).mean()
        if acc > best:
            best, bad = acc, 0
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best


def save_probs(run_dir: Path, res: dict, y: dict):
    """Track-level probs (one row per track) in the same npz format as train.py."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for s in ("val", "test"):
        np.savez_compressed(run_dir / f"{s}_probs.npz", probs=res[s].astype(np.float32),
                            labels=y[s].cpu().numpy(), track_ids=TRACK_IDS[s])


def run_probe(name: str, X: dict, y: dict, out_root: Path, seeds=(0, 1, 2)) -> dict:
    t0 = time.time()
    probs = {s: [] for s in ("val", "test")}
    vals = []
    for seed in seeds:
        model, best = train_probe(X, y, seed)
        vals.append(float(best))
        for s in probs:
            probs[s].append(track_probs_from_clips(model, X[s]))
        del model
    res = {s: np.mean(probs[s], 0) for s in probs}
    save_probs(out_root / name, res, y)
    m = {f"{s}_acc": float((res[s].argmax(1) == y[s].cpu().numpy()).mean()) for s in res}
    log.info("PROBE %-14s val_acc=%.4f test_acc=%.4f (seed vals %s, %.1f min)",
             name, m["val_acc"], m["test_acc"], [round(v, 3) for v in vals], (time.time() - t0) / 60)
    return m


# -------------------------------------------------------------------- ensembling
def load_track_probs(run_dir: Path, split: str):
    from ensemble import track_probs
    d = np.load(run_dir / f"{split}_probs.npz")
    tp = track_probs(d["probs"], d["labels"], d["track_ids"])
    tids = sorted(tp)
    return tids, np.stack([tp[t][0] for t in tids]), np.array([tp[t][1] for t in tids])


def greedy_ensemble(cands: dict, iters: int = 25) -> dict:
    """Caruana forward selection with replacement on val track accuracy
    (val log-loss as tie-break). Returns {name: weight}."""
    names = list(cands)
    yv = cands[names[0]]["val_y"]
    chosen, cur = [], None
    for _ in range(iters):
        best = None
        for nme in names:
            p = cands[nme]["val"] if cur is None else (cur * len(chosen) + cands[nme]["val"]) / (len(chosen) + 1)
            acc = (p.argmax(1) == yv).mean()
            nll = -np.log(p[np.arange(len(yv)), yv] + 1e-9).mean()
            if best is None or (acc, -nll) > best[0]:
                best = ((acc, -nll), nme, p)
        chosen.append(best[1])
        cur = best[2]
    w = {n: chosen.count(n) / len(chosen) for n in set(chosen)}
    return dict(sorted(w.items(), key=lambda kv: -kv[1]))


def ensemble_report(run_dirs: dict, out_root: Path) -> dict:
    from evaluate import compute_metrics
    cands = {}
    for name, d in sorted(run_dirs.items()):
        vt, vp, vy = load_track_probs(d, "val")
        tt, tp, ty = load_track_probs(d, "test")
        cands[name] = {"val": vp, "val_y": vy, "val_t": vt, "test": tp, "test_y": ty, "test_t": tt}
    ref = next(iter(cands.values()))
    for c in cands.values():  # all models must cover the same tracks in the same order
        assert c["val_t"] == ref["val_t"] and c["test_t"] == ref["test_t"]
    weights = greedy_ensemble(cands)
    combine = lambda s: sum(w * cands[n][s] for n, w in weights.items())
    val_acc = float((combine("val").argmax(1) == ref["val_y"]).mean())
    test = compute_metrics(ref["test_y"].tolist(), combine("test").argmax(1).tolist())
    uniform = sum(c["test"] for c in cands.values()) / len(cands)
    res = {
        "method": "greedy forward selection (with replacement) on val track accuracy",
        "weights": weights,
        "val_accuracy": val_acc,
        "test_track_level": test,
        "uniform_all_test_accuracy": float((uniform.argmax(1) == ref["test_y"]).mean()),
        "candidates_val_acc": {n: float((c["val"].argmax(1) == c["val_y"]).mean()) for n, c in cands.items()},
        "candidates_test_acc": {n: float((c["test"].argmax(1) == c["test_y"]).mean()) for n, c in cands.items()},
    }
    out = out_root / "ensemble_round2"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "eval_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    log.info("ENSEMBLE (val-selected) weights=%s val_acc=%.4f TEST acc=%.4f f1=%.4f | uniform-all test=%.4f",
             weights, val_acc, test["accuracy"], test["macro_f1"], res["uniform_all_test_accuracy"])
    return res


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-finetune", action="store_true")
    ap.add_argument("--encoders", default="whisper,mert,clap,ast")
    args = ap.parse_args()

    from data import memory_cache
    if args.smoke:
        from run_pipeline import fake_cache
        fake_cache([16000, 24000, 48000], per_split=8)
    else:
        memory_cache.load([16000, 24000, 48000])

    meta = memory_cache.METADATA
    from data.dataset import GENRE_TO_IDX
    rows = {s: np.flatnonzero(meta["split"].to_numpy() == s) for s in SPLITS}
    for s in SPLITS:
        TRACK_IDS[s] = meta["track_id"].to_numpy()[rows[s]].astype(int)
    y = {s: torch.tensor(meta["genre"].map(GENRE_TO_IDX).to_numpy()[rows[s]], device=DEV) for s in SPLITS}

    out_root = Path("outputs_smoke" if args.smoke else "outputs")
    A = memory_cache.AUDIO
    extractors = {
        "whisper": lambda: extract_whisper(A[16000]),
        "mert": lambda: extract_mert(A[24000]),
        "clap": lambda: extract_clap(A[48000]),
        "ast": lambda: extract_ast(A[16000]),
    }
    feats, results = {}, {}
    for enc in args.encoders.split(","):
        t0 = time.time()
        log.info("=== extracting %s ===", enc)
        try:
            Z = extractors[enc]()
        except Exception:
            log.exception("=== %s extraction FAILED", enc)
            continue
        torch.cuda.empty_cache()
        log.info("  %s features %s in %.1f min", enc, Z.shape, (time.time() - t0) / 60)
        X = {s: torch.from_numpy(Z[rows[s]]).to(DEV) for s in SPLITS}
        try:
            results[f"probe_{enc}"] = run_probe(f"probe_{enc}", X, y, out_root)
        except Exception:
            log.exception("=== probe_%s FAILED", enc)
        # track-level, layer-averaged, normalised vector per encoder for the fusion probe
        feats[enc] = {s: F.layer_norm(X[s].float().mean((1, 2)), X[s].shape[-1:]) for s in SPLITS}
        del X, Z
        torch.cuda.empty_cache()

    if len(feats) >= 2:
        Xf = {s: torch.cat([feats[e][s] for e in feats], -1).view(len(rows[s]), 1, 1, -1) for s in SPLITS}
        try:
            results["probe_fusion"] = run_probe("probe_fusion", Xf, y, out_root)
        except Exception:
            log.exception("=== probe_fusion FAILED")
        del Xf
    del feats
    torch.cuda.empty_cache()

    if not args.skip_finetune:
        import yaml
        from train import train
        cfg = yaml.safe_load(open("configs/ast_v2.yaml"))
        if args.smoke:
            cfg.update(epochs=1, batch_size=4, warmup_steps=1, crop_repeats=1)
        cfg["_out_root"] = str(out_root)
        log.info("=== fine-tune ast_v2 ===")
        try:
            m = train(cfg)
            results["ast_v2"] = {"test_acc": m["track_level"]["accuracy"], "best_epoch": m["best_epoch"]}
            log.info("=== ast_v2 done: %s", results["ast_v2"])
        except Exception:
            log.exception("=== ast_v2 FAILED")
        torch.cuda.empty_cache()

    run_dirs = {d.name: d for d in out_root.iterdir() if d.is_dir() and (d / "val_probs.npz").exists()}
    ens = ensemble_report(run_dirs, out_root) if len(run_dirs) >= 2 else None
    summary = {"models": results, "ensemble": ens}
    with open(out_root / "results_round2.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log.info("ROUND2 SUMMARY:\n%s", json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
