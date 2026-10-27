"""Compare classifier heads on the same cached, frozen DINOv2 features and the same train/val split.

    python scripts/compare_heads.py

Writes reports/head_comparison.md (summary + per-class report of every head) and one confusion matrix per head.
"""
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from carbrand.dataset import load_labels, split
from carbrand.evaluate import accuracy, plot_confusion, report
from carbrand.train_brand import sample_weights, train_head


def main():
    cfg = yaml.safe_load(open("configs/default.yaml"))
    bcfg, dcfg = cfg["brand"], cfg["data"]
    df = load_labels(dcfg["labels"], dcfg["crops_root"])
    classes = sorted(df.brand.unique())
    y = df.brand.map({c: i for i, c in enumerate(classes)}).to_numpy()
    is_val = split(df, dcfg["val_own"], dcfg["val_external"])
    own = df.own.to_numpy()
    cache = torch.load(f"cache/features_{bcfg['backbone']}_{dcfg['image_size']}.pt")

    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    feats = lambda idx, v: torch.stack([cache[p][v] for p in df.path.iloc[idx]])
    X_tr = torch.cat([feats(tr, 0), feats(tr, 1)])          # original + mirrored view
    y_tr = np.concatenate([y[tr], y[tr]])
    own_tr = np.concatenate([own[tr], own[tr]])
    w_tr = sample_weights(y_tr, own_tr, len(classes), bcfg["own_weight"])
    X_va = feats(va, 0)
    l2 = lambda Z: (Z / Z.norm(dim=1, keepdim=True)).numpy()

    def mlp():
        torch.manual_seed(0)
        head = train_head(X_tr, torch.tensor(y_tr), w_tr, len(classes), bcfg, "cuda")
        with torch.no_grad():
            return head(X_va.cuda()).argmax(1).cpu().numpy()

    heads = {
        "MLP head (ours)": mlp,
        "Logistic regression": lambda: LogisticRegression(C=2, max_iter=3000, class_weight="balanced")
            .fit(l2(X_tr), y_tr, sample_weight=w_tr.numpy()).predict(l2(X_va)),
        "kNN (k=10, cosine)": lambda: KNeighborsClassifier(10, metric="cosine", weights="distance")
            .fit(X_tr.numpy(), y_tr).predict(X_va.numpy()),
        "XGBoost": lambda: XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.1, subsample=0.8,
                                         colsample_bytree=0.5, tree_method="hist", device="cuda")
            .fit(X_tr.numpy(), y_tr, sample_weight=w_tr.numpy()).predict(X_va.numpy()),
        "CatBoost": lambda: CatBoostClassifier(iterations=1500, depth=6, learning_rate=0.1, task_type="GPU",
                                               loss_function="MultiClass", verbose=0, random_seed=0)
            .fit(X_tr.numpy(), y_tr, sample_weight=w_tr.numpy()).predict(X_va.numpy()).ravel().astype(int),
        "LightGBM": lambda: LGBMClassifier(n_estimators=600, num_leaves=31, learning_rate=0.05,
                                           colsample_bytree=0.5, subsample=0.8, subsample_freq=1, verbose=-1)
            .fit(X_tr.numpy(), y_tr, sample_weight=w_tr.numpy()).predict(X_va.numpy()),
    }

    rows, details = [], []
    out_dir = Path("reports/figures/heads")
    out_dir.mkdir(parents=True, exist_ok=True)
    yv, ov = y[va], own[va]
    for name, fit_predict in heads.items():
        t0 = time.time()
        pred = np.asarray(fit_predict())
        secs = time.time() - t0
        row = (name, accuracy(yv, pred), accuracy(yv[ov], pred[ov]), accuracy(yv[~ov], pred[~ov]),
               f1_score(yv, pred, average="macro"), secs)
        rows.append(row)
        print(f"{name:22s} acc {row[1]:.3f}  own {row[2]:.3f}  external {row[3]:.3f}  macro-F1 {row[4]:.3f}  {secs:6.1f}s",
              flush=True)
        slug = name.split(" ")[0].lower()
        plot_confusion(yv, pred, classes, out_dir / f"confusion_{slug}.png", f"{name}: brand confusion (val, n={len(va)})")
        details.append(f"### {name}\n\n```\n{report(yv, pred, classes)}```\n")

    md = ["# Brand head comparison (frozen DINOv2 ViT-S/14 features, 224 px)\n",
          f"Train {len(tr)} crops (x2 with mirror), validation {len(va)} crops "
          f"({ov.sum()} own Tashkent photos, {(~ov).sum()} external). Same split for every head.\n",
          "| Head | Val accuracy | Own photos | External | Macro-F1 | Train time |",
          "|---|---|---|---|---|---|"]
    md += [f"| {n} | {a:.3f} | {o:.3f} | {e:.3f} | {f:.3f} | {s:.0f}s |" for n, a, o, e, f, s in rows]
    Path("reports/head_comparison.md").write_text("\n".join(md) + "\n\n" + "\n".join(details), encoding="utf-8")
    print("wrote reports/head_comparison.md")


if __name__ == "__main__":
    main()
