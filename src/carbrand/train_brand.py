"""Brand classifier training.

Stage A (this file): the DINOv2 backbone is frozen. Every crop is embedded once (original + horizontal flip)
and cached to disk; then only the small MLP head is trained on the cached features, which takes seconds.

    python -m carbrand.train_brand --config configs/default.yaml           # train/val split, report metrics
    python -m carbrand.train_brand --config configs/default.yaml --final   # train on everything for test day
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from tqdm import tqdm

from carbrand.brand_model import BrandHead, embed, make_backbone
from carbrand.dataset import eval_transform, load_labels, split
from carbrand.evaluate import accuracy, plot_confusion, report


@torch.no_grad()
def cache_features(paths, backbone_name, img_size, cache_file, device, batch_size=64):
    """Returns {path: tensor[2, D]} (original, flipped); only embeds paths missing from the cache."""
    cache = torch.load(cache_file) if Path(cache_file).exists() else {}
    todo = [p for p in paths if p not in cache]
    if todo:
        backbone = make_backbone(backbone_name, img_size).to(device).eval()
        tf = eval_transform(img_size)
        for i in tqdm(range(0, len(todo), batch_size), desc="embedding crops"):
            chunk = todo[i:i + batch_size]
            x = torch.stack([tf(Image.open(p)) for p in chunk]).to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                f = torch.stack([embed(backbone, x), embed(backbone, x.flip(-1))], dim=1)
            for p, v in zip(chunk, f.float().cpu()):
                cache[p] = v
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_file)
    return cache


def train_head(X, y, w, n_classes, cfg, device):
    head = BrandHead(X.shape[1], n_classes, cfg["hidden"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["lr_head"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    loss_fn = nn.CrossEntropyLoss(reduction="none", label_smoothing=cfg["label_smoothing"])
    X, y, w = X.to(device), y.to(device), w.to(device)
    for _ in range(cfg["epochs"]):
        head.train()
        for idx in torch.randperm(len(X), device=device).split(cfg["batch_size"]):
            loss = (loss_fn(head(X[idx]), y[idx]) * w[idx]).sum() / w[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return head.eval()


def sample_weights(y, own, n_classes, own_weight):
    """Inverse-sqrt class frequency (rare brands are not drowned out) x extra weight for own Tashkent photos."""
    counts = np.bincount(y, minlength=n_classes)
    w = 1.0 / np.sqrt(np.maximum(counts[y], 1))
    w = w * np.where(own, own_weight, 1.0)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--final", action="store_true", help="train on all data (no validation)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    bcfg, dcfg = cfg["brand"], cfg["data"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    df = load_labels(dcfg["labels"], dcfg["crops_root"])
    classes = sorted(df.brand.unique())
    y_all = df.brand.map({c: i for i, c in enumerate(classes)}).to_numpy()
    df["is_val"] = False if args.final else split(df, dcfg["val_own"], dcfg["val_external"])

    t0 = time.time()
    cache = cache_features(df.path.tolist(), bcfg["backbone"], dcfg["image_size"],
                           f"cache/features_{bcfg['backbone']}_{dcfg['image_size']}.pt", device)
    print(f"features ready in {time.time() - t0:.0f}s")

    tr = np.flatnonzero(~df.is_val)
    # both views (original + flip) of every training crop
    X_tr = torch.cat([torch.stack([cache[p][v] for p in df.path.iloc[tr]]) for v in (0, 1)])
    y_tr = np.concatenate([y_all[tr], y_all[tr]])
    own_tr = np.concatenate([df.own.to_numpy()[tr]] * 2)
    w_tr = sample_weights(y_tr, own_tr, len(classes), bcfg["own_weight"])

    t0 = time.time()
    head = train_head(X_tr, torch.tensor(y_tr), w_tr, len(classes), bcfg, device)
    print(f"head trained in {time.time() - t0:.1f}s on {len(tr)} crops")

    if not args.final:
        va = np.flatnonzero(df.is_val)
        X_va = torch.stack([cache[p][0] for p in df.path.iloc[va]]).to(device)
        with torch.no_grad():
            pred = head(X_va).argmax(1).cpu().numpy()
        y_va, own_va = y_all[va], df.own.to_numpy()[va]
        print(f"\nVAL brand accuracy  all: {accuracy(y_va, pred):.3f} (n={len(va)})"
              f"   own Tashkent photos: {accuracy(y_va[own_va], pred[own_va]):.3f} (n={own_va.sum()})"
              f"   external: {accuracy(y_va[~own_va], pred[~own_va]):.3f} (n={(~own_va).sum()})")
        print(report(y_va, pred, classes))
        Path("reports/figures").mkdir(parents=True, exist_ok=True)
        plot_confusion(y_va, pred, classes, "reports/figures/brand_confusion_stageA.png",
                       f"Stage A brand confusion (val, n={len(va)})")
        for name, m in [("own", own_va), ("external", ~own_va)]:
            wrong = [(classes[t], classes[p]) for t, p in zip(y_va[m], pred[m]) if t != p]
            pairs = {}
            for k in wrong:
                pairs[k] = pairs.get(k, 0) + 1
            top = sorted(pairs.items(), key=lambda kv: -kv[1])[:8]
            print(f"top confusions ({name}):", ", ".join(f"{t}->{p} x{n}" for (t, p), n in top))

    out = Path(bcfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "classes": classes, "in_dim": X_tr.shape[1],
                "hidden": bcfg["hidden"], "dropout": bcfg["dropout"], "backbone": bcfg["backbone"],
                "img_size": dcfg["image_size"], "final": args.final}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
