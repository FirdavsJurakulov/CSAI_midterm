"""Append labels for externally-sourced crops (e.g. Wikimedia Commons) to data/labels.csv.

Joins <crops>/manifest.csv (from extract_crops.py --largest-only) with the downloader's metadata.csv,
so each crop inherits the brand/colour of its source image.

    python scripts/extract_crops.py data/external/commons data/crops/commons --largest-only --min-area 0.05
    python scripts/merge_external_labels.py data/external/commons data/crops/commons --source commons
"""
import argparse, csv, os


def norm(p):
    return p.replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("external_dir", help="folder with metadata.csv")
    ap.add_argument("crops_dir", help="folder with manifest.csv")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--source", default="commons")
    ap.add_argument("--min-conf", type=float, default=0.6, help="drop weak detections (interiors, details)")
    args = ap.parse_args()

    meta = {norm(r["file"]): r for r in csv.DictReader(open(os.path.join(args.external_dir, "metadata.csv"), encoding="utf-8"))}
    crops_rel = norm(os.path.relpath(args.crops_dir, os.path.dirname(args.labels)))

    existing = set()
    if os.path.exists(args.labels):
        existing = {r["crop_path"] for r in csv.DictReader(open(args.labels, encoding="utf-8"))}

    added = 0
    with open(args.labels, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in csv.DictReader(open(os.path.join(args.crops_dir, "manifest.csv"), encoding="utf-8")):
            m = meta.get(norm(r["source_image"]))
            crop_path = f"{crops_rel}/{r['crop']}"
            # buses and low-confidence boxes are mostly wrong-category uploads (boxy SUVs/pickups come out as "truck")
            if m is None or crop_path in existing or r["det_class"] not in ("car", "truck") or float(r["conf"]) < args.min_conf:
                continue
            # one session per source photo: different uploads are (mostly) different cars
            session = f"{args.source}_{os.path.splitext(os.path.basename(norm(r['source_image'])))[0]}"
            w.writerow([crop_path, m["brand"], m["colour"], "", args.source, session, norm(r["source_image"])])
            added += 1
    print(f"added {added} rows to {args.labels}")


if __name__ == "__main__":
    main()
