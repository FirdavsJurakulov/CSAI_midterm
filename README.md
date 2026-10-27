# Car Brand & Colour Classification — Tashkent (CSAI-3210 Midterm)

Takes a phone photo of a Tashkent street and prints the brand and colour of every car in it, e.g.
`BYD white; Chevrolet grey`. Runs fully offline on the laptop (about 1 s per photo on a GTX 1650, a few seconds on CPU).

Pipeline: YOLO11s-seg (finds every car + its pixel mask) → DINOv2 ViT-S/14 brand classifier (trained by us)
→ masked HSV/Lab colour features + gradient-boosted colour classifier (trained by us).
Details: [docs/architecture.md](docs/architecture.md).

## 1. Install (clean laptop, Python 3.10–3.12)

```bash
git clone https://github.com/FirdavsJurakulov/CSAI_midterm.git
cd CSAI_midterm
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs the CPU/CUDA build that pip picks for your machine. For an NVIDIA GPU on Windows, install
the CUDA build of PyTorch first: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`.

All weights are in the repository (`weights/`), nothing is downloaded at run time:

| File | What it is |
|---|---|
| `weights/yolo11s-seg.pt` | COCO-pretrained YOLO11s-seg detector (Ultralytics), used as-is |
| `weights/dinov2_vits14.pt` | pretrained DINOv2 ViT-S/14 backbone (for the stage-A model) |
| `weights/brand_head.pt` | our stage-A brand head (frozen backbone) |
| `weights/brand_stageB.pt` | our stage-B brand model (fine-tuned backbone + head), used automatically if present |
| `weights/colour.pkl` | our colour classifier |

## 2. Test day

1. Take the photo with the phone and send it to the laptop (Telegram "Saved messages" / cable), save it into `incoming/`.
2. Run:

```bash
python -m carbrand.predict incoming/photo.jpg
```

Output, one line per photo, cars from left to right:

```
photo.jpg: BYD white; Chevrolet grey   [1.0s]
```

A folder works too (`python -m carbrand.predict incoming/`). Every run also writes `runs/predict/predictions.csv`
and an annotated copy of each photo (`runs/predict/<name>_pred.jpg`) to check which cars were counted.
Detection thresholds (every extra prediction costs a point) are in `configs/default.yaml → predict`, or on the
command line: `--conf 0.5 --min-area 0.01`.

## 3. Reproduce training

```bash
# own photos -> one crop + mask per car (then labelled by hand into data/labels.csv)
python scripts/extract_crops.py data/raw data/crops/own --weights weights/yolo11s-seg.pt --session own_2026-10-26
# supplementary Wikimedia Commons photos (CC-licensed; author/licence per file in data/external/*/metadata.csv)
python scripts/fetch_commons.py --out data/external/commons --per-brand 200
python scripts/extract_crops.py data/external/commons data/crops/commons --weights weights/yolo11s-seg.pt --largest-only --min-area 0.05
python scripts/merge_external_labels.py data/external/commons data/crops/commons --source commons

# brand, stage A: frozen DINOv2 features (cached) + MLP head; prints val accuracy + confusion matrix
python -m carbrand.train_brand --config configs/default.yaml
# brand, stage B: fine-tune the last 4 DINOv2 blocks + head (warm-started from stage A)
python -m carbrand.finetune_brand --config configs/default.yaml
# colour
python -m carbrand.train_colour --config configs/default.yaml
# comparison of heads on the same features (MLP vs logistic regression, kNN, XGBoost, CatBoost, LightGBM)
python scripts/compare_heads.py

# final models for test day: same commands with --final (train on train + validation)
python -m carbrand.train_brand --final
python -m carbrand.train_colour --final
```

## Data

`data/labels.csv` — one line per car crop: `crop_path, brand, colour, view, source, session, source_image`.
`source` is `own` (our phone photos of Tashkent streets) or `commons` (Wikimedia Commons, see
`data/external/*/metadata.csv` for author, licence and URL of every file). Crops and masks are in `data/crops/`.

## Repository layout

```
configs/      hyperparameters (one YAML)
data/         labels.csv, crops + masks, Commons metadata
docs/         architecture & data plan
scripts/      data preparation (crop extraction, Commons crawler, label merge, head comparison)
src/carbrand/ dataset, models, training (stage A / stage B / colour), evaluation, inference (predict.py)
weights/      trained weights
reports/      figures, metrics and the PDF report
```
