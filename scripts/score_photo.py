"""Re-implementation of the exam's per-photo scoring rule, for rehearsals.

    python scripts/score_photo.py runs/predict/predictions.csv reference.csv

reference.csv: photo, brand, colour (one line per car a human can identify). Every photo in the reference is scored;
a photo with no predictions scores 0. Rule (brief, Section 3): 5 points per photo shared equally by its reference cars,
each car's share split 0.6 brand / 0.4 colour, predictions paired with reference cars to maximise the score,
-1 per unpaired prediction, floored at 0, rounded to one decimal.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def photo_score(pred, ref):
    """pred, ref: lists of (brand, colour) -> (score, brand hits, colour hits, extra predictions)."""
    if not ref:
        return 0.0, 0, 0, len(pred)
    share = 5 / len(ref)
    gain = np.array([[share * (0.6 * (pb == rb) + 0.4 * (pc == rc)) for rb, rc in ref] for pb, pc in pred]).reshape(len(pred), len(ref))
    rows, cols = linear_sum_assignment(gain, maximize=True) if len(pred) else ([], [])
    extra = len(pred) - len(rows)
    brand = sum(pred[r][0] == ref[c][0] for r, c in zip(rows, cols))
    colour = sum(pred[r][1] == ref[c][1] for r, c in zip(rows, cols))
    return round(max(0.0, gain[rows, cols].sum() - extra), 1), brand, colour, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("reference")
    args = ap.parse_args()
    pred = pd.read_csv(args.predictions)
    ref = pd.read_csv(args.reference)
    rows = []
    for photo, r in ref.groupby("photo"):
        p = pred[pred.photo == photo]
        s, b, c, e = photo_score(list(zip(p.brand, p.colour)), list(zip(r.brand, r.colour)))
        rows.append((photo, len(r), len(p), b, c, e, s))
    df = pd.DataFrame(rows, columns=["photo", "ref_cars", "predicted", "brand_ok", "colour_ok", "extra", "score"])
    print(df.to_string(index=False))
    n = len(df)
    print(f"\n{n} photos: {df.score.sum():.1f} / {5 * n} points ({df.score.sum() / n:.2f} per photo, "
          f"{100 * df.score.sum() / n / 5:.1f} % of the maximum)")
    print(f"reference cars {df.ref_cars.sum()}: brand right {df.brand_ok.sum()}, colour right {df.colour_ok.sum()}, "
          f"extra predictions {df.extra.sum()}")


if __name__ == "__main__":
    main()
