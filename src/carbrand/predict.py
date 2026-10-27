"""Test-day entry point: photo(s) -> one (brand, colour) per detected car. Runs fully offline.

    python -m carbrand.predict photo.jpg [more.jpg | folder ...]
    python -m carbrand.predict data/test --out runs/predict

Prints "photo.jpg: BYD white; Chevrolet black" per photo, writes predictions.csv and an annotated copy of
every photo to --out. Cars are listed left to right.
"""
import argparse
import csv
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never touch the network on test day

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

from carbrand.brand_model import BrandHead, BrandNet, embed, make_backbone
from carbrand.colour_model import colour_features, imread
from carbrand.dataset import eval_transform

VEHICLE = [2, 5, 7]  # COCO car, bus, truck
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class BrandClassifier:
    """Stage-B checkpoint if present (whole fine-tuned backbone + head), otherwise frozen backbone + stage-A head."""

    def __init__(self, stage_b, stage_a, backbone_file, device):
        self.device = device
        if Path(stage_b).exists():
            ck = torch.load(stage_b, map_location="cpu")
            head = BrandHead(768, len(ck["classes"]), ck["hidden"], ck["dropout"])
            self.model = BrandNet(make_backbone(ck["backbone"], ck["img_size"], pretrained=False), head,
                                  ck["unfreeze_blocks"])
            self.model.load_state_dict(ck["model"])
            self.forward = self.model
            self.source = f"stage B ({stage_b})"
        else:
            ck = torch.load(stage_a, map_location="cpu")
            local = Path(backbone_file).exists()
            backbone = make_backbone(ck["backbone"], ck["img_size"], pretrained=not local)
            if local:
                backbone.load_state_dict(torch.load(backbone_file, map_location="cpu"))
            head = BrandHead(ck["in_dim"], len(ck["classes"]), ck["hidden"], ck["dropout"])
            head.load_state_dict(ck["head"])
            self.model = torch.nn.ModuleDict({"backbone": backbone, "head": head})
            self.forward = lambda x: head(embed(backbone, x))
            self.source = f"stage A ({stage_a})"
        self.model.to(device).eval()
        self.classes = ck["classes"]
        self.tf = eval_transform(ck["img_size"])

    @torch.no_grad()
    def __call__(self, crops_rgb):
        x = torch.stack([self.tf(Image.fromarray(c)) for c in crops_rgb]).to(self.device)
        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.device == "cuda"):
            logits = self.forward(x) + self.forward(x.flip(-1))  # mirror test-time augmentation
        prob = logits.float().softmax(1).cpu().numpy()
        return [(self.classes[i], float(p[i])) for p, i in zip(prob, prob.argmax(1))]


def detect(yolo, img, conf, min_area, pad):
    H, W = img.shape[:2]
    res = yolo.predict(img, conf=conf, classes=VEHICLE, retina_masks=True, verbose=False)[0]
    cars = []
    if res.masks is None:
        return cars
    for box, c, mask in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.masks.data.cpu().numpy()):
        x1, y1, x2, y2 = box
        if (x2 - x1) * (y2 - y1) / (W * H) < min_area:
            continue
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        a, b, cc, d = int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(W, x2 + px)), int(min(H, y2 + py))
        m = (mask[b:d, a:cc] > 0.5).astype(np.uint8) * 255
        cars.append({"box": (a, b, cc, d), "det_conf": float(c), "crop": img[b:d, a:cc], "mask": m})
    return sorted(cars, key=lambda car: car["box"][0])


def collect(inputs):
    paths = []
    for s in inputs:
        p = Path(s)
        paths += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_EXT) if p.is_dir() else [p]
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="photos and/or folders of photos")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="runs/predict")
    ap.add_argument("--conf", type=float, help="detector confidence (default from config)")
    ap.add_argument("--min-area", type=float, help="min car box area as fraction of photo (default from config)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    pcfg = cfg["predict"]
    conf = args.conf if args.conf is not None else pcfg["det_conf"]
    min_area = args.min_area if args.min_area is not None else pcfg["min_area_frac"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    yolo = YOLO(pcfg["detector"])
    brand = BrandClassifier(cfg["finetune"]["out"], cfg["brand"]["out"], pcfg["backbone_weights"], device)
    with open(cfg["colour"]["out"], "rb") as f:
        colour = pickle.load(f)
    print(f"models loaded in {time.time() - t0:.1f}s on {device}; brand model: {brand.source}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in collect(args.inputs):
        t1 = time.time()
        img = imread(path)
        if img is None:
            print(f"{path.name}: cannot read")
            continue
        cars = detect(yolo, img, conf, min_area, pcfg["pad"])
        if cars:
            brands = brand([cv2.cvtColor(c["crop"], cv2.COLOR_BGR2RGB) for c in cars])
            X = np.stack([colour_features(c["crop"], c["mask"], colour["bins"]) for c in cars])
            colours = [colour["classes"][i] for i in colour["clf"].predict(X)]
        else:
            brands, colours = [], []
        vis = img.copy()
        for i, (car, (b, p), col) in enumerate(zip(cars, brands, colours)):
            rows.append([path.name, i, b, col, round(p, 3), round(car["det_conf"], 3), *car["box"]])
            a, y, c, d = car["box"]
            cv2.rectangle(vis, (a, y), (c, d), (0, 255, 0), 3)
            cv2.putText(vis, f"{b} {col}", (a + 4, max(30, y + 30)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imencode(".jpg", vis)[1].tofile(str(out / f"{path.stem}_pred.jpg"))
        listing = "; ".join(f"{b} {col}" for (b, _), col in zip(brands, colours)) or "(no cars)"
        print(f"{path.name}: {listing}   [{time.time() - t1:.1f}s]", flush=True)

    with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["photo", "car", "brand", "colour", "brand_prob", "det_conf", "x1", "y1", "x2", "y2"])
        w.writerows(rows)
    print(f"wrote {out / 'predictions.csv'}")


if __name__ == "__main__":
    main()
