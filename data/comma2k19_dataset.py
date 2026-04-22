"""
CSV-backed comma2k19 dataset. Each row: (image_path, steering, segment_id, ctx).

Train/val split is by segment_id (not frame) to prevent temporal leakage.
Augmentation: color jitter + horizontal flip with steering-sign inversion.
"""
import hashlib
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


CONTEXT_NAMES = ["day_clear", "day_overcast", "night"]
NUM_CONTEXTS = len(CONTEXT_NAMES)
CTX_TO_IDX = {c: i for i, c in enumerate(CONTEXT_NAMES)}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _seg_hash01(seg_id):
    # deterministic [0,1) hash for a stable by-segment split
    h = hashlib.md5(seg_id.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_by_segment(df, val_fraction=0.1):
    scores = df["segment_id"].map(_seg_hash01)
    mask_val = scores < val_fraction
    return df[~mask_val].reset_index(drop=True), df[mask_val].reset_index(drop=True)


class Comma2k19Dataset(Dataset):
    def __init__(self, df, image_size=224, augment=False):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _load(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _augment(self, img, steering):
        # horizontal flip: flip image AND negate steering sign
        if random.random() < 0.5:
            img = img[:, ::-1, :].copy()
            steering = -steering
        # color jitter: brightness / contrast in HSV-V and linear gain
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 2] *= random.uniform(0.7, 1.3)
            hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img, steering

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = self._load(row["image_path"])
        steering = float(row["steering"])
        ctx = CTX_TO_IDX.get(row["ctx"], 1)

        if self.augment:
            img, steering = self._augment(img, steering)

        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))

        return (
            torch.from_numpy(img),
            torch.tensor(steering, dtype=torch.float32),
            torch.tensor(ctx, dtype=torch.long),
        )


def build_balanced_sampler(df):
    """Inverse-frequency weighting over ctx, for a WeightedRandomSampler."""
    counts = df["ctx"].value_counts().to_dict()
    total = sum(counts.values())
    weights = df["ctx"].map(lambda c: total / max(counts.get(c, 1), 1)).values
    weights = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(df), replacement=True)


def load_frames_csv(csv_path, val_fraction=0.1):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["image_path", "steering", "segment_id", "ctx"])
    df = df[df["ctx"].isin(CONTEXT_NAMES)].reset_index(drop=True)
    return split_by_segment(df, val_fraction=val_fraction)
