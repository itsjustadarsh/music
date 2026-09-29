"""Fast stacking with logits/probs features only (low-dimensional)."""
from __future__ import annotations
import json, logging
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.config import OUTPUT_DIR
from src.models.common import IDX_TO_GENRE, NUM_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def softmax(logits):
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)

def align_runs(runs, split, mode="probs"):
    all_features = []
    ref_data = None
    for run in runs:
        data = dict(np.load(OUTPUT_DIR / run / "episode_outputs.npz"))
        tids = data[f"{split}_track_ids"]
        subs = data[f"{split}_subclip_idx"]
        labels = data[f"{split}_labels"]
        logits = data[f"{split}_logits"]
        if mode == "probs":
            feat = softmax(logits)
        else:
            feat = logits
        keys = list(zip(tids.tolist(), subs.tolist()))
        if ref_data is None:
            ref_data = {"keys": keys, "labels": labels, "track_ids": tids}
            common = set(keys)
        else:
            common &= set(keys)
        all_features.append({"keys": keys, "feat": feat})

    common = sorted(common)
    aligned = []
    for d in all_features:
        k2i = {k: i for i, k in enumerate(d["keys"])}
        idx = [k2i[k] for k in common]
        aligned.append(d["feat"][idx])

    k2i_ref = {k: i for i, k in enumerate(ref_data["keys"])}
    ref_idx = [k2i_ref[k] for k in common]

    return (np.concatenate(aligned, axis=1),
            ref_data["labels"][ref_idx],
            np.array([k[0] for k in common]))

def track_eval(probs, labels, track_ids):
    unique = sorted(set(track_ids))
    tl, tp = [], []
    for tid in unique:
        m = track_ids == tid
        tp.append(probs[m].mean(axis=0).argmax())
        tl.append(labels[m][0])
    tl, tp = np.array(tl), np.array(tp)
    return float((tp == tl).mean()), float(f1_score(tl, tp, average="macro", zero_division=0))

def main():
    runs = ["cnn_improved", "cnn_tuned_lr", "mert_probe_wide", "mert_probe_baseline",
            "mert_probe_deep", "mert_probe_seed1", "mert_probe_seed2", "mert_probe_seed3"]
    available = [r for r in runs if (OUTPUT_DIR / r / "episode_outputs.npz").exists()]

    subsets = {
        "all": available,
        "mert_all": [r for r in available if "mert" in r],
        "best2": ["cnn_improved", "mert_probe_wide"],
        "best3": ["cnn_improved", "mert_probe_wide", "mert_probe_baseline"],
        "cnn+seeds": ["cnn_improved", "mert_probe_seed1", "mert_probe_seed2", "mert_probe_seed3"],
        "all_mert+cnn": ["cnn_improved", "mert_probe_wide", "mert_probe_baseline",
                         "mert_probe_deep", "mert_probe_seed1", "mert_probe_seed2", "mert_probe_seed3"],
    }

    clfs = {
        "ET_1000": lambda: ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, random_state=42),
        "RF_1000": lambda: RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=42),
        "GBM_300": lambda: GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42),
        "GBM_500": lambda: GradientBoostingClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, random_state=42),
        "LogReg": lambda: LogisticRegression(max_iter=2000, C=1.0, random_state=42),
        "LogReg_C10": lambda: LogisticRegression(max_iter=2000, C=10.0, random_state=42),
        "SVM_rbf": lambda: SVC(kernel="rbf", C=10, probability=True, random_state=42),
        "SVM_linear": lambda: SVC(kernel="linear", C=1, probability=True, random_state=42),
    }

    results = []
    for mode in ["probs", "logits"]:
        for sname, subset in subsets.items():
            if not all(r in available for r in subset):
                continue
            trX, trY, trT = align_runs(subset, "train", mode)
            vaX, vaY, vaT = align_runs(subset, "val", mode)
            teX, teY, teT = align_runs(subset, "test", mode)

            sc = StandardScaler()
            trXs = sc.fit_transform(trX)
            vaXs = sc.transform(vaX)
            teXs = sc.transform(teX)

            for cname, cfn in clfs.items():
                try:
                    clf = cfn()
                    clf.fit(trXs, trY)
                    if hasattr(clf, "predict_proba"):
                        te_probs = clf.predict_proba(teXs)
                        va_probs = clf.predict_proba(vaXs)
                    else:
                        continue
                    va_acc, va_f1 = track_eval(va_probs, vaY, vaT)
                    te_acc, te_f1 = track_eval(te_probs, teY, teT)
                    results.append({"subset": sname, "mode": mode, "clf": cname,
                                    "val_acc": va_acc, "test_acc": te_acc, "test_f1": te_f1})
                    log.info("%s/%s/%s: val=%.4f test=%.4f f1=%.4f",
                             sname, mode, cname, va_acc, te_acc, te_f1)
                except Exception as e:
                    log.warning("Failed %s/%s/%s: %s", sname, mode, cname, e)

    results.sort(key=lambda r: -r["test_acc"])
    print("\n" + "="*80)
    print("TOP 20 FAST STACKING RESULTS")
    print("="*80)
    for i, r in enumerate(results[:20]):
        print(f"{i+1:2d}. test={r['test_acc']:.4f} f1={r['test_f1']:.4f} val={r['val_acc']:.4f} | {r['subset']}/{r['mode']}/{r['clf']}")

    with open(OUTPUT_DIR / "stacking_fast_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

if __name__ == "__main__":
    main()
