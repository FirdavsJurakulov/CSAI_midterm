"""Run the detector over photos and save one crop (+ binary mask) per car.

    python scripts/extract_crops.py data/raw data/crops/own --session own_2026-10-26
    python scripts/extract_crops.py data/external/commons data/crops/commons --largest-only

Writes <out>/manifest.csv: crop, mask, source_image, box, area fraction, detector confidence.
"""
import argparse, csv, os, sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

VEHICLE = {2: "car", 5: "bus", 7: "truck"}  # COCO ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--weights", default="yolo11s-seg.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min-side", type=int, default=90, help="min crop width/height in px")
    ap.add_argument("--min-area", type=float, default=0.004, help="min box area as fraction of image")
    ap.add_argument("--pad", type=float, default=0.08)
    ap.add_argument("--largest-only", action="store_true", help="keep only the biggest car per image")
    ap.add_argument("--session", default="")
    args = ap.parse_args()

    model = YOLO(args.weights)
    src = Path(args.src)
    images = sorted(p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "mask").mkdir(parents=True, exist_ok=True)

    rows = []
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            continue
        H, W = img.shape[:2]
        res = model.predict(img, conf=args.conf, classes=list(VEHICLE), retina_masks=True, verbose=False)[0]
        if res.masks is None:
            continue
        dets = []
        for box, cls, conf, mask in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(),
                                        res.boxes.conf.cpu().numpy(), res.masks.data.cpu().numpy()):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if min(w, h) < args.min_side or w * h / (W * H) < args.min_area:
                continue
            dets.append((w * h, box, int(cls), float(conf), mask))
        dets.sort(key=lambda d: -d[0])
        if args.largest_only:
            dets = dets[:1]

        stem = path.stem if src == path.parent else "_".join(path.relative_to(src).with_suffix("").parts)
        for i, (area, (x1, y1, x2, y2), cls, conf, mask) in enumerate(dets):
            px, py = (x2 - x1) * args.pad, (y2 - y1) * args.pad
            a, b = int(max(0, x1 - px)), int(max(0, y1 - py))
            c, d = int(min(W, x2 + px)), int(min(H, y2 + py))
            name = f"{stem}_c{i}"
            cv2.imwrite(str(out / "img" / f"{name}.jpg"), img[b:d, a:c], [cv2.IMWRITE_JPEG_QUALITY, 92])
            m = (mask[b:d, a:c] > 0.5).astype(np.uint8) * 255
            cv2.imwrite(str(out / "mask" / f"{name}.png"), m)
            border = int(a == 0 or b == 0 or c == W or d == H)
            rows.append([f"img/{name}.jpg", f"mask/{name}.png", str(path.relative_to(src)),
                         a, b, c, d, round(area / (W * H), 4), round(conf, 3), VEHICLE[cls], border, args.session])
        print(f"{path.name}: {len(dets)} cars", file=sys.stderr)

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["crop", "mask", "source_image", "x1", "y1", "x2", "y2", "area_frac", "conf", "det_class",
                    "touches_border", "session"])
        w.writerows(rows)
    print(f"{len(rows)} crops from {len(images)} images -> {out}")


if __name__ == "__main__":
    main()
