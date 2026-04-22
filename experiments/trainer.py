"""
Training loop for SCDMN-comma2k19.

- SmoothL1Loss(beta=0.1)
- SGD + Nesterov, CosineAnnealingLR
- Sliced: soft-mask warmup until `freeze_epoch`, then freeze -> hard sliced forward
- Per-context MAE logged each epoch
- Checkpoints saved each epoch to save_dir and (if set) drive_save_dir
"""
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.comma2k19_dataset import (
    Comma2k19Dataset,
    CONTEXT_NAMES,
    NUM_CONTEXTS,
    build_balanced_sampler,
    load_frames_csv,
)
from models import SCDMNSliced, ResNet18Reg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(model_type, sparsity, num_contexts):
    if model_type == "single":
        return ResNet18Reg()
    if model_type == "sliced":
        return SCDMNSliced(num_contexts=num_contexts, sparsity=sparsity)
    raise ValueError(f"unknown model_type: {model_type}")


def model_forward(model, x, ctx):
    out = model(x, ctx_label=ctx)
    return out.squeeze(-1)


def build_loaders(cfg):
    train_df, val_df = load_frames_csv(
        cfg["data"]["csv_path"], val_fraction=cfg["data"]["val_fraction"],
    )
    print(f"[loader] train frames: {len(train_df)}   val frames: {len(val_df)}")
    print(f"[loader] train ctx: {train_df['ctx'].value_counts().to_dict()}")
    print(f"[loader] val   ctx: {val_df['ctx'].value_counts().to_dict()}")

    train_ds = Comma2k19Dataset(train_df, image_size=cfg["data"]["image_size"], augment=True)
    val_ds = Comma2k19Dataset(val_df, image_size=cfg["data"]["image_size"], augment=False)

    if cfg["train"].get("balanced_sampling", False):
        sampler = build_balanced_sampler(train_df)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle, sampler=sampler,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True, persistent_workers=cfg["train"]["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg["train"]["num_workers"] > 0,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = {c: 0 for c in range(NUM_CONTEXTS)}
    abs_err = {c: 0.0 for c in range(NUM_CONTEXTS)}

    for x, y, ctx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        ctx = ctx.to(device, non_blocking=True)
        pred = model_forward(model, x, ctx)
        err = (pred - y).abs()
        for c in range(NUM_CONTEXTS):
            m = (ctx == c)
            if m.any():
                total[c] += int(m.sum().item())
                abs_err[c] += float(err[m].sum().item())

    per_context = {}
    for c in range(NUM_CONTEXTS):
        per_context[CONTEXT_NAMES[c]] = (abs_err[c] / total[c]) if total[c] > 0 else float("nan")
    tot = sum(total.values())
    ovr = sum(abs_err.values()) / tot if tot > 0 else float("nan")
    return {"overall": ovr, "per_context": per_context}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _save_ckpt(model, cfg, epoch, eval_result, save_path, drive_save_dir=None):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg,
        "epoch": epoch,
        "eval": eval_result,
    }, save_path)
    if drive_save_dir:
        drive_dir = Path(drive_save_dir)
        drive_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(save_path, drive_dir / save_path.name)


def train(cfg, model_type, run_name):
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = Path(cfg["log"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / f"{run_name}.log"
    drive_save_dir = cfg["log"].get("drive_save_dir") or None

    def log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"=== Run: {run_name} ===")
    log(f"Model: {model_type}   Device: {device}")

    train_loader, val_loader = build_loaders(cfg)

    model = build_model(model_type, cfg["model"]["sparsity"], cfg["model"]["num_contexts"]).to(device)
    log(f"Params: {count_params(model):,}")

    if model_type == "sliced":
        score_params = list(model.channel_scores.parameters())
        score_ids = {id(p) for p in score_params}
        other = [p for p in model.parameters() if id(p) not in score_ids]
        optim = torch.optim.SGD(
            [
                {"params": other, "lr": cfg["train"]["lr"]},
                {"params": score_params, "lr": cfg["train"]["lr"] * 10.0, "weight_decay": 0.0},
            ],
            lr=cfg["train"]["lr"], momentum=cfg["train"]["momentum"],
            weight_decay=cfg["train"]["weight_decay"], nesterov=True,
        )
    else:
        optim = torch.optim.SGD(
            model.parameters(), lr=cfg["train"]["lr"],
            momentum=cfg["train"]["momentum"],
            weight_decay=cfg["train"]["weight_decay"], nesterov=True,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg["train"]["epochs"])
    criterion = nn.SmoothL1Loss(beta=cfg["train"]["smooth_l1_beta"])

    best_overall = float("inf")
    history = []
    freeze_epoch = cfg["model"]["freeze_epoch"]

    for epoch in range(cfg["train"]["epochs"]):
        if model_type == "sliced" and epoch == freeze_epoch and not model.is_frozen():
            model.freeze_masks()
            log(f"[sliced] froze masks at epoch {epoch}, switching to sliced forward.")

        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_abs = 0.0
        running_total = 0

        for step, (x, y, ctx) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            ctx = ctx.to(device, non_blocking=True)

            pred = model_forward(model, x, ctx)
            loss = criterion(pred, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running_loss += loss.item() * x.size(0)
            running_abs += float((pred - y).abs().sum().item())
            running_total += x.size(0)

            if step % cfg["log"]["log_every"] == 0:
                log(f"  epoch {epoch:03d} step {step:05d}   loss={loss.item():.4f}")

        scheduler.step()

        train_loss = running_loss / max(running_total, 1)
        train_mae = running_abs / max(running_total, 1)
        eval_result = evaluate(model, val_loader, device)
        dt = time.time() - t0

        extra = f"   frozen={model.is_frozen()}" if model_type == "sliced" else ""
        log(
            f"Epoch {epoch:03d}  train_loss={train_loss:.4f}  train_mae={train_mae:.4f}  "
            f"val_overall={eval_result['overall']:.4f}  "
            f"per_ctx={ {k: round(v, 4) for k, v in eval_result['per_context'].items()} }  "
            f"time={dt:.1f}s{extra}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "val_overall": eval_result["overall"],
            "per_context": eval_result["per_context"],
        })

        _save_ckpt(model, cfg, epoch, eval_result,
                   save_dir / f"{run_name}_latest.pt", drive_save_dir)

        if eval_result["overall"] < best_overall:
            best_overall = eval_result["overall"]
            _save_ckpt(model, cfg, epoch, eval_result,
                       save_dir / f"{run_name}_best.pt", drive_save_dir)

    final_path = save_dir / f"{run_name}_final.pt"
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg,
        "history": history,
    }, final_path)
    if drive_save_dir:
        drive_dir = Path(drive_save_dir)
        drive_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(final_path, drive_dir / final_path.name)

    with open(save_dir / f"{run_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    log(f"best overall MAE: {best_overall:.4f}")
    return model, history
