"""Brand classifier, stage B: fine-tune the last DINOv2 blocks together with the head on augmented crops.

Run stage A first (its head is the warm start), then:

    python -m carbrand.finetune_brand --config configs/default.yaml           # train/val split, report metrics
    python -m carbrand.finetune_brand --config configs/default.yaml --final   # train on everything for test day
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from carbrand.brand_model import BrandHead, BrandNet, make_backbone
from carbrand.dataset import eval_transform, load_labels, split, train_transform
from carbrand.evaluate import accuracy, plot_confusion, report
from carbrand.train_brand import sample_weights


class Crops(Dataset):
    def __init__(self, paths, y, w, tf):
        self.paths, self.y, self.w, self.tf = paths, y, w, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i])), self.y[i], self.w[i]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for x, _, _ in loader:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        out.append(logits.float().argmax(1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--final", action="store_true", help="train on all data (no validation)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    bcfg, dcfg, fcfg = cfg["brand"], cfg["data"], cfg["finetune"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)

    df = load_labels(dcfg["labels"], dcfg["crops_root"])
    classes = sorted(df.brand.unique())
    y_all = df.brand.map({c: i for i, c in enumerate(classes)}).to_numpy()
    is_val = np.zeros(len(df), bool) if args.final else split(df, dcfg["val_own"], dcfg["val_external"])
    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    own = df.own.to_numpy()

    w_tr = sample_weights(y_all[tr], own[tr], len(classes), bcfg["own_weight"]).numpy()
    size, bs, nw = fcfg["image_size"], fcfg["batch_size"], fcfg["workers"]
    train_dl = DataLoader(Crops(df.path.iloc[tr].tolist(), y_all[tr], w_tr, train_transform(size)), bs,
                          shuffle=True, num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=True)
    val_dl = DataLoader(Crops(df.path.iloc[va].tolist(), y_all[va], np.ones(len(va), np.float32), eval_transform(size)),
                        64, num_workers=nw, pin_memory=True, persistent_workers=True) if len(va) else None

    head = BrandHead(2 * 384, len(classes), bcfg["hidden"], bcfg["dropout"])
    init = Path(fcfg["init_head"])
    if init.exists():
        ck = torch.load(init)
        if ck.get("final") and not args.final:
            print(f"{init} was trained on all data (incl. val): head starts from scratch to keep val honest")
        elif ck["classes"] == classes:
            head.load_state_dict(ck["head"])
            print(f"warm start from {init}")
        else:
            print(f"{init} has different classes, head starts from scratch")
    model = BrandNet(make_backbone(bcfg["backbone"], size), head, fcfg["unfreeze_blocks"]).to(device)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": backbone_params, "lr": fcfg["lr_backbone"]},
                             {"params": head.parameters(), "lr": fcfg["lr_head"]}],
                            weight_decay=fcfg["weight_decay"])
    steps, warm = fcfg["epochs"] * len(train_dl), fcfg["warmup_epochs"] * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    loss_fn = nn.CrossEntropyLoss(reduction="none", label_smoothing=bcfg["label_smoothing"])
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{len(tr)} train / {len(va)} val crops, {n_train / 1e6:.1f}M trainable params, {len(train_dl)} steps/epoch")

    out = Path(fcfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    best, y_va, own_va = -1.0, y_all[va], own[va]
    for epoch in range(fcfg["epochs"]):
        model.train()
        model.backbone.blocks[:model.n_frozen].eval()  # frozen part: no dropout/drop-path
        t0, total = time.time(), 0.0
        for x, y, w in train_dl:
            x, y, w = x.to(device, non_blocking=True), y.to(device), w.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                loss = (loss_fn(model(x).float(), y) * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        msg = f"epoch {epoch + 1:2d}/{fcfg['epochs']}  loss {total / len(train_dl):.3f}  {time.time() - t0:.0f}s"
        score = epoch
        if val_dl is not None:
            pred = predict(model, val_dl, device)
            acc, acc_own = accuracy(y_va, pred), accuracy(y_va[own_va], pred[own_va])
            msg += f"  val {acc:.3f}  own {acc_own:.3f}  external {accuracy(y_va[~own_va], pred[~own_va]):.3f}"
            score = acc
        print(msg, flush=True)
        if score > best:  # with --final there is no val: the last epoch is kept
            best = score
            torch.save({"model": model.state_dict(), "classes": classes, "hidden": bcfg["hidden"],
                        "dropout": bcfg["dropout"], "backbone": bcfg["backbone"], "img_size": size,
                        "unfreeze_blocks": fcfg["unfreeze_blocks"], "final": args.final, "epoch": epoch + 1}, out)

    if val_dl is not None:
        model.load_state_dict(torch.load(out)["model"])
        pred = predict(model, val_dl, device)
        print(f"\nBEST stage B  all: {accuracy(y_va, pred):.3f} (n={len(va)})"
              f"   own Tashkent photos: {accuracy(y_va[own_va], pred[own_va]):.3f} (n={own_va.sum()})"
              f"   external: {accuracy(y_va[~own_va], pred[~own_va]):.3f} (n={(~own_va).sum()})")
        print(report(y_va, pred, classes))
        Path("reports/figures").mkdir(parents=True, exist_ok=True)
        plot_confusion(y_va, pred, classes, "reports/figures/brand_confusion_stageB.png",
                       f"Stage B brand confusion (val, n={len(va)})")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
