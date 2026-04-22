"""
Compare two training runs' per-context MAE and write a markdown summary.

    python -m utils.analyze \
        --single experiments/runs/single_summary.json \
        --sliced experiments/runs/sliced_s0.5_f5_summary.json \
        --out experiments/runs/comma2k19_summary.md
"""
import argparse
import json
from pathlib import Path

from data.comma2k19_dataset import CONTEXT_NAMES


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", type=str, required=True)
    ap.add_argument("--sliced", type=str, required=True)
    ap.add_argument("--out", type=str, default="experiments/runs/comma2k19_summary.md")
    args = ap.parse_args()

    s = load_summary(args.single)["result"]
    d = load_summary(args.sliced)["result"]

    rows = []
    rows.append("# SCDMN-comma2k19 — per-context MAE\n")
    rows.append("| context | Single | SCDMN-Sliced | Δ (Single − Sliced) |")
    rows.append("|---|---:|---:|---:|")
    for ctx in CONTEXT_NAMES:
        a = s["per_context"].get(ctx, float("nan"))
        b = d["per_context"].get(ctx, float("nan"))
        rows.append(f"| {ctx} | {a:.4f} | {b:.4f} | {a - b:+.4f} |")
    rows.append(f"| **OVERALL** | **{s['overall']:.4f}** | **{d['overall']:.4f}** | **{s['overall'] - d['overall']:+.4f}** |")
    rows.append("")
    rows.append("> Positive Δ means SCDMN improves over the single baseline on that context.")
    rows.append("> Success criterion: positive Δ on `night` and `day_overcast`.")

    text = "\n".join(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()
