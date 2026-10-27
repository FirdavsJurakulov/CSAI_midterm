# Architecture: Car Brand & Colour Classification (Tashkent)

## Pipeline

```
phone photo
   │
   ▼
[1] Car detector + segmenter ── YOLO11s-seg, COCO-pretrained, NOT fine-tuned
   │   keep classes: car, truck, bus
   │   drop boxes that are too small / cut off / low confidence  (extra predictions cost −1)
   ▼
per-car crop (box + 10% padding)  +  per-car pixel mask
   │                                   │
   ▼                                   ▼
[2] Brand classifier                 [3] Colour classifier
    DINOv2 ViT-S/14 backbone             masked HSV + Lab histogram of body pixels
    stage A: frozen, cached features     (windows/tyres/shadows removed by
    stage B: last 2-4 blocks fine-tuned   masking dark/low-saturation outliers)
    head: LayerNorm → Linear(384→256)    → small MLP / logistic regression
          → GELU → Dropout → Linear(N)     trained on our colour labels (8 classes)
   │                                   │
   └───────────────┬───────────────────┘
                   ▼
        list of (brand, colour) pairs  →  "BYD white; Chevrolet grey"
```

## Why this design

| Decision | Reason |
|---|---|
| Detector → per-car classifier, not multi-label on the whole photo | Scoring is per car, and every extra prediction costs a point. You need one prediction per car and control over how many cars you report. |
| YOLO11-seg (COCO weights, used as-is) | Allowed as a general-purpose pretrained detector. The mask gives the colour model clean body pixels for free. |
| DINOv2 ViT-S/14 backbone | Self-supervised features hold up well on fine-grained, few-shot tasks, which suits a few hundred crops per brand. 22M params runs on a GTX 1650. It is also an unusual choice, which helps with the originality rule (most teams will pick ResNet/EfficientNet/MobileNet + YOLO). |
| Two-stage training (frozen → partial fine-tune) | Stage A trains in seconds on cached features and gives the baseline to keep. Stage B is used only if it beats A on validation. |
| Separate, non-CNN colour model | DINOv2 was trained with colour-jitter augmentation, so its features are partly colour-invariant. Histograms of masked body pixels are explicit, cheap, easy to explain, and are our own model. |
| Never output `unknown` | A wrong brand and `unknown` both score 0 for brand, and both still earn the colour share. A top-1 guess can only help. Confidence thresholds belong on the *detector* (whether to report a car at all), not on the brand. |

## Multi-car filtering (matters for scoring)

The instructor leaves out cars "too small, too blurry or too hidden" to identify. Each extra car you report costs 1 point. Tune these on your own multi-car photos:
- min box area (start at ~2% of the image area)
- detector confidence (start at 0.5)
- drop boxes cut off by the image border by more than ~40%
- drop boxes covered by another box (IoU/containment) by more than ~50%

## Approx. parameter count
- YOLO11s-seg: ~10M (frozen)
- DINOv2 ViT-S/14: ~22M (frozen or partially tuned)
- Brand head: ~0.1M
- Colour MLP: <0.01M
