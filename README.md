# SCDMN-comma2k19

**Scene-Conditional Dynamic Mask Networks — comma2k19 driving-scale validation.**

> Follow-up to [SCDMN-cifar](../SCDMN-cifar) (the controlled CIFAR-10 / CIFAR-10-C
> toy validation). This repo moves the same mechanism onto real driving video —
> comma2k19 front-facing dashcam + CAN steering angle — to check whether
> channel-level context routing still helps once the distribution shift comes
> from actual day / overcast / night lighting rather than synthetic corruptions.

---

## Relationship to SCDMN-cifar

`SCDMN-cifar` answered: *given clean context labels and toy corruptions, does
per-stage channel masking beat a single mixed-context model?* That was a
controlled-setting sanity check.

This repo answers the practical next question: *on real driving data where
the "context" is lighting/weather inferred from the frame, does the same
sliced mask architecture (sparsity 0.5, freeze after warmup) reduce
mode-averaging in the hard contexts (night, overcast)?*

The model code is ported from `SCDMN-cifar` with three adjustments:

- Input 32×32 → 224×224 (driving frames).
- Regression head (steering angle, 1 output + tanh), same as
  `scdmn_sliced_reg.py` in the CIFAR repo.
- `num_contexts = 3` (`day_clear` / `day_overcast` / `night`) instead of 4.

Everything else — SlicedBasicBlock, soft-mask warmup → freeze → hard sliced
forward, per-context top-k channel scores — is the same mechanism.

---

## Data

**comma2k19** (Santana & Hotz, 2018 release) — ~33 hours of California freeway
driving with front dashcam + CAN. We use only a subset (chunk 1 and/or 2,
~10 GB each) for this PoC.

- Official repo: https://github.com/commaai/comma2k19
- Each segment: `video.hevc`, `processed_log/CAN/steering_angle/value`,
  `processed_log/CAN/steering_angle/t`, etc.
- We subsample the video to 10 fps and match steering angle by timestamp.

Context labels are **not** provided. We auto-label by mean frame brightness:

```
day_clear    mean_brightness > 140
day_overcast 80 < mean_brightness <= 140
night        mean_brightness <= 80
```

These are intentionally crude — the point is to get a real lighting-driven
context split, not to nail weather classification.

---

## Pipeline

```
scripts/download_comma2k19.py     # pulls chunk(s) into data_root
scripts/prepare_frames.py         # HEVC -> 10fps JPGs + steering CSV
scripts/label_context.py          # adds ctx column by brightness
                                  #   (produces frames.csv with ctx)
experiments/run_experiment.py     # single baseline OR scdmn_sliced
utils/analyze.py                  # per-context MAE comparison
```

Output CSV schema (`frames.csv`):

| column      | meaning                                  |
|-------------|------------------------------------------|
| image_path  | relative path to the jpg frame           |
| steering    | normalized steering in [-1, 1]           |
| segment_id  | source segment (train/val split key)     |
| ctx         | `day_clear` / `day_overcast` / `night`  |

---

## Running on Colab (recommended)

See `notebooks/colab_runner.ipynb`. It handles: GitHub clone, Drive mount,
requirements install, one-time data prep, single baseline training, SCDMN
training, then the comparison table.

Key Colab defaults:

- T4 / L4 GPU (12–16 GB VRAM) → batch size 64.
- Data and checkpoints live on mounted Drive, so session reconnects don't
  redo the 10 GB download or the HEVC decode.
- One chunk (~10 GB, ~2000 segments × ~1 minute) is enough for a PoC.

Local runs work too — the scripts take `--data_root` and don't depend on Colab.

---

## Local smoke run

```bash
python -m scripts.download_comma2k19 --data_root ./data_cache --chunks 1
python -m scripts.prepare_frames     --data_root ./data_cache --fps 10
python -m scripts.label_context      --data_root ./data_cache

python -m experiments.run_experiment --model single  --epochs 2 --batch_size 64
python -m experiments.run_experiment --model sliced  --epochs 2 --batch_size 64 \
       --sparsity 0.5 --freeze_epoch 1
```

---

## Experiments

1. **Single baseline** — ResNet18 trained on all contexts mixed.
2. **SCDMN-Sliced** — per-context channel slicing, sparsity 0.5, freeze after
   5 warmup epochs.

Reported: per-context MAE (`day_clear` / `day_overcast` / `night`) and overall
MAE. Success = Single MAE > SCDMN MAE on the hard contexts (night, overcast)
by a gap that's credible as "mode averaging relieved."

---

## Repo layout

```
SCDMN-comma2k19/
├── configs/default.yaml
├── data/comma2k19_dataset.py     # CSV-based (image, steering, ctx) dataset
├── models/
│   ├── scdmn_sliced.py           # ported from SCDMN-cifar (224x224, reg head)
│   └── resnet_baseline.py        # ported single-model baseline
├── scripts/
│   ├── download_comma2k19.py
│   ├── prepare_frames.py
│   └── label_context.py
├── experiments/
│   ├── trainer.py
│   └── run_experiment.py
├── notebooks/colab_runner.ipynb
└── utils/analyze.py
```
