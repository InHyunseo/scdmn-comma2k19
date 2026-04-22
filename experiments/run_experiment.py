"""
Entry point for SCDMN-comma2k19 experiments.

Usage:
    python -m experiments.run_experiment --model single --epochs 30
    python -m experiments.run_experiment --model sliced --sparsity 0.5 --freeze_epoch 5
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.trainer import train
from data.comma2k19_dataset import CONTEXT_NAMES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--model", type=str, required=True, choices=["single", "sliced"])
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--sparsity", type=float, default=None)
    p.add_argument("--freeze_epoch", type=int, default=None)
    p.add_argument("--csv_path", type=str, default=None)
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--drive_save_dir", type=str, default=None)
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, args):
    if args.epochs is not None:         cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:     cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:             cfg["train"]["lr"] = args.lr
    if args.sparsity is not None:       cfg["model"]["sparsity"] = args.sparsity
    if args.freeze_epoch is not None:   cfg["model"]["freeze_epoch"] = args.freeze_epoch
    if args.csv_path is not None:       cfg["data"]["csv_path"] = args.csv_path
    if args.save_dir is not None:       cfg["log"]["save_dir"] = args.save_dir
    if args.drive_save_dir is not None: cfg["log"]["drive_save_dir"] = args.drive_save_dir
    return cfg


def print_table(run_name, result):
    print(f"\n=== {run_name} — per-context MAE ===")
    print(f"{'context':<16}{'MAE':>10}")
    for ctx in CONTEXT_NAMES:
        v = result["per_context"].get(ctx, float("nan"))
        print(f"{ctx:<16}{v:>10.4f}")
    print(f"{'OVERALL':<16}{result['overall']:>10.4f}")


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    run_name = args.run_name
    if run_name is None:
        if args.model == "single":
            run_name = "single"
        else:
            run_name = f"sliced_s{cfg['model']['sparsity']}_f{cfg['model']['freeze_epoch']}"

    _, history = train(cfg, model_type=args.model, run_name=run_name)

    final = history[-1]
    result = {"overall": final["val_overall"], "per_context": final["per_context"]}
    print_table(run_name, result)

    out_dir = Path(cfg["log"]["save_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{run_name}_summary.json", "w") as f:
        json.dump({"run_name": run_name, "model": args.model, "result": result}, f, indent=2)


if __name__ == "__main__":
    main()
