"""Car body colour from the segmentation mask: hand-made HSV/Lab features + a gradient-boosted classifier.

Only pixels inside the (eroded) YOLO mask are used, so road, sky and neighbouring cars do not leak in.
Windows, tyres and shadows are dark in every car; features are therefore computed twice, on the whole mask
and on its lower-middle band (doors/bumper, mostly paint), and the classifier learns which to trust.
"""
from pathlib import Path

import cv2
import numpy as np

COLOURS = ["white", "black", "grey", "red", "blue", "green", "brown", "yellow"]
MAX_SIDE = 192


def mask_path(crop_path):
    """data/crops/X/img/name.jpg -> data/crops/X/mask/name.png"""
    p = Path(crop_path)
    return p.parent.parent / "mask" / (p.stem + ".png")


def imread(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread cannot open non-ASCII (e.g. Chinese) file names on Windows; decode from bytes instead."""
    return cv2.imdecode(np.fromfile(str(path), np.uint8), flags)


def load_crop(crop_path):
    img = imread(crop_path)
    mp = mask_path(crop_path)
    mask = imread(mp, cv2.IMREAD_GRAYSCALE) if mp.exists() else None
    return img, mask


def _region_features(hsv, lab, sel, bins):
    if sel.sum() < 30:
        return np.zeros(bins["h"] + bins["s"] + bins["v"] + 1 + 15, np.float32)
    h, s, v = (hsv[..., i][sel].astype(np.float32) for i in range(3))
    chroma = (s > 50) & (v > 50)
    # hue histogram only over clearly coloured pixels, weighted by saturation
    hh = np.histogram(h[chroma], bins["h"], (0, 180), weights=s[chroma])[0] if chroma.any() else np.zeros(bins["h"])
    hh = hh / max(s[chroma].sum(), 1.0) if chroma.any() else hh
    sh = np.histogram(s, bins["s"], (0, 256))[0] / len(s)
    vh = np.histogram(v, bins["v"], (0, 256))[0] / len(v)
    L, a, b = (lab[..., i][sel].astype(np.float32) for i in range(3))
    stats = [f(x) for x in (L, a, b) for f in (np.mean, np.std, lambda z: np.percentile(z, 10),
                                               lambda z: np.percentile(z, 50), lambda z: np.percentile(z, 90))]
    return np.concatenate([hh, sh, vh, [chroma.mean()], np.array(stats) / 255.0]).astype(np.float32)


def colour_features(img, mask=None, bins=None):
    """img: BGR crop, mask: uint8 (same size) or None -> 1-D feature vector."""
    bins = bins or {"h": 18, "s": 8, "v": 8}
    H, W = img.shape[:2]
    s = MAX_SIDE / max(H, W)
    if s < 1:
        img = cv2.resize(img, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    if mask is None or mask.max() == 0:  # no mask: central ellipse as a stand-in
        m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, (W // 2, H // 2), (int(W * 0.4), int(H * 0.35)), 0, 0, 360, 255, -1)
    else:
        m = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        m = cv2.erode(m, np.ones((5, 5), np.uint8))
    sel = m > 127
    ys, xs = np.nonzero(sel)
    band = np.zeros_like(sel)
    if len(ys):
        y0, y1 = np.percentile(ys, [40, 85]).astype(int)
        x0, x1 = np.percentile(xs, [10, 90]).astype(int)
        band[y0:y1 + 1, x0:x1 + 1] = True
    band &= sel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return np.concatenate([_region_features(hsv, lab, sel, bins), _region_features(hsv, lab, band, bins)])


def features_for_path(crop_path, bins=None):
    img, mask = load_crop(crop_path)
    return colour_features(img, mask, bins)
