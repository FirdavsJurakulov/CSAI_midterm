"""Fit the colour classifier on masked histogram features.

    python -m carbrand.train_colour --config configs/default.yaml           # train/val split, report metrics
    python -m carbrand.train_colour --config configs/default.yaml --final   # train on everything for test day
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier

from carbrand.colour_model import features_for_path
from carbrand.dataset import load_labels, split
from carbrand.evaluate import accuracy, plot_confusion, report


def make_classifier():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, class_weight="balanced", random_state=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--final", action="store_true", help="train on all data (no validation)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ccfg, dcfg = cfg["colour"], cfg["data"]
    classes = ccfg["classes"]

    df = load_labels(dcfg["labels"], dcfg["crops_root"])
    df = df[df.colour.isin(classes)].reset_index(drop=True)
    y = df.colour.map({c: i for i, c in enumerate(classes)}).to_numpy()

    cache = Path("cache/colour_features.npz")
    old = {}
    if cache.exists():
        z = np.load(cache)
        old = dict(zip(z["paths"].tolist(), z["X"]))
    todo = [p for p in df.path if p not in old]
    if todo:
        feats = Parallel(n_jobs=6, batch_size=32)(delayed(features_for_path)(p, ccfg["hist_bins"]) for p in todo)
        old.update(zip(todo, feats))
        cache.parent.mkdir(exist_ok=True)
        np.savez(cache, paths=np.array(list(old)), X=np.stack(list(old.values())))
    X = np.stack([old[p] for p in df.path])
    print(f"{len(df)} crops, {X.shape[1]} features")

    is_val = np.zeros(len(df), bool) if args.final else split(df, dcfg["val_own"], dcfg["val_external"])
    own = df.own.to_numpy()
    w = np.where(own, 3.0, 1.0)  # own photos match test conditions
    clf = make_classifier().fit(X[~is_val], y[~is_val], sample_weight=w[~is_val])

    if not args.final:
        pred = clf.predict(X[is_val])
        yv, ov = y[is_val], own[is_val]
        print(f"VAL colour accuracy  all: {accuracy(yv, pred):.3f} (n={len(yv)})"
              f"   own: {accuracy(yv[ov], pred[ov]):.3f} (n={ov.sum()})   external: {accuracy(yv[~ov], pred[~ov]):.3f}")
        print(report(yv, pred, classes))
        Path("reports/figures").mkdir(parents=True, exist_ok=True)
        plot_confusion(yv, pred, classes, "reports/figures/colour_confusion.png",
                       f"Colour confusion (val, n={len(yv)})")

    out = Path(ccfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"clf": clf, "classes": classes, "bins": ccfg["hist_bins"], "final": args.final}, f)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
