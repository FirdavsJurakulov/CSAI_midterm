"""Reads data/labels.csv, makes a leakage-free train/val split, and prepares crops for the backbone."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torchvision import transforms

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load_labels(labels_csv, data_root="data"):
    df = pd.read_csv(labels_csv)
    df["path"] = [str(Path(data_root) / p) for p in df.crop_path]
    df["own"] = df.source.isin(["own", "screen"])
    return df.reset_index(drop=True)


def split(df, val_own=0.3, val_external=0.15, seed=0):
    """Boolean 'is_val' column.

    Own photos are grouped by source photo, so crops from one photo never end up on both sides.
    External photos are one car each, so their session (= source file) is the group.
    """
    is_val = np.zeros(len(df), dtype=bool)
    for mask, frac, groups in [(df.own, val_own, df.source_image), (~df.own, val_external, df.session)]:
        idx = np.flatnonzero(mask)
        gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
        _, val = next(gss.split(idx, groups=groups.iloc[idx]))
        is_val[idx[val]] = True
    return is_val


def letterbox(img, size=224, fill=(124, 116, 104)):
    """Resize keeping aspect ratio and pad to a square: cars are wide, a center crop would cut them."""
    w, h = img.size
    s = size / max(w, h)
    img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def eval_transform(size=224):
    return transforms.Compose([
        transforms.Lambda(lambda im: letterbox(im.convert("RGB"), size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
