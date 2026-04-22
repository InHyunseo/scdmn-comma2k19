"""
Add a `ctx` column to frames.csv using mean-brightness thresholds.

    day_clear     mean > 140
    day_overcast  80 < mean <= 140
    night         mean <= 80

Usage:
    python -m scripts.label_context --csv ./data_cache/frames.csv
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


CONTEXT_NAMES = ["day_clear", "day_overcast", "night"]


def classify(mean_b, hi=140, lo=80):
    if mean_b > hi:
        return "day_clear"
    if mean_b > lo:
        return "day_overcast"
    return "night"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default=None,
                    help="default: overwrite input csv")
    ap.add_argument("--hi", type=float, default=140.0)
    ap.add_argument("--lo", type=float, default=80.0)
    ap.add_argument("--recompute", action="store_true",
                    help="recompute ctx even if column exists")
    args = ap.parse_args()

    csv_in = Path(args.csv)
    csv_out = Path(args.out_csv) if args.out_csv else csv_in

    df = pd.read_csv(csv_in)
    if "ctx" in df.columns and not args.recompute:
        print("[label] ctx column already present (use --recompute to overwrite)")
    else:
        ctxs = []
        for p in tqdm(df["image_path"].tolist(), desc="brightness"):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                ctxs.append("day_overcast")
                continue
            mean_b = float(img.mean())
            ctxs.append(classify(mean_b, args.hi, args.lo))
        df["ctx"] = ctxs
        df.to_csv(csv_out, index=False)
        print(f"[label] wrote {csv_out}")

    print("\n=== Context distribution ===")
    counts = df["ctx"].value_counts().reindex(CONTEXT_NAMES, fill_value=0)
    total = int(counts.sum())
    for name, n in counts.items():
        pct = 100.0 * n / max(total, 1)
        print(f"  {name:<14} {n:>8}   ({pct:5.2f}%)")
    print(f"  {'TOTAL':<14} {total:>8}")

    # Warnings
    for name in CONTEXT_NAMES:
        n = int(counts.get(name, 0))
        if n == 0:
            print(f"[warn] context '{name}' has zero samples — consider extra data or relabel thresholds.",
                  file=sys.stderr)
        elif n < 0.05 * total:
            print(f"[warn] context '{name}' is under-represented ({n}/{total}). "
                  f"Consider balanced sampling or oversampling.", file=sys.stderr)


if __name__ == "__main__":
    main()
