# Car Brand & Colour Classification — Tashkent (CSAI-3210 Midterm)

Takes a phone photo of a Tashkent street and prints the brand and colour of every car in it, e.g.
`BYD white; Chevrolet grey`. Runs fully offline.

Pipeline: YOLO11-seg (car detection + masks) → DINOv2 ViT-S/14 brand classifier → masked-histogram colour classifier.
See [docs/architecture.md](docs/architecture.md).

## Install (clean laptop)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

<!-- TODO: weights download link / instructions -->

## Test-day usage

```bash
python -m carbrand.predict path/to/photo.jpg
python -m carbrand.predict --watch incoming/    # predict every new photo dropped in a folder
```

## Training

```bash
# own photos -> crops (labelled by hand into data/labels.csv)
python scripts/extract_crops.py data/raw data/crops/own --weights weights/yolo11s-seg.pt --session own_2026-10-26
# supplementary Wikimedia Commons photos (CC-licensed; attribution in data/external/*/metadata.csv)
python scripts/fetch_commons.py --out data/external/commons --per-brand 200
python scripts/extract_crops.py data/external/commons data/crops/commons --weights weights/yolo11s-seg.pt --largest-only --min-area 0.05
python scripts/merge_external_labels.py data/external/commons data/crops/commons --source commons
python -m carbrand.train_brand --config configs/default.yaml
python -m carbrand.train_colour --config configs/default.yaml
python -m carbrand.evaluate --config configs/default.yaml
```

## Repository layout

```
configs/     hyperparameters
data/        raw photos, crops, labels.csv (dataset link: TODO)
docs/        architecture & data plan
scripts/     data preparation tools
src/carbrand detection, models, training, evaluation, inference
weights/     trained weights
reports/     figures and the final PDF report
```
