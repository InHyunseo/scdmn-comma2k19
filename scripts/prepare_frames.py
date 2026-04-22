"""
Extract 10 fps frames from each comma2k19 segment's video.hevc and match
timestamps to CAN steering angle. Writes:

    <out_dir>/frames/<segment_id>/<frame_idx>.jpg
    <out_dir>/frames.csv  (image_path, steering, segment_id)

Steering angle is stored in radians in processed_log/CAN/steering_angle/value
(with timestamps in .../t). We pick the nearest CAN sample to each frame's
timestamp, clip to +/- steering_clip, and normalize to [-1, 1].

Usage:
    python -m scripts.prepare_frames --data_root ./data_cache --fps 10

Scans all Chunk_*/dongle/route/segment/ dirs under data_root.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def find_segments(data_root):
    # Structure: data_root/Chunk_N/<dongle>/<route>/<seg_idx>/video.hevc
    root = Path(data_root)
    segs = []
    for chunk_dir in sorted(root.glob("Chunk_*")):
        for video in chunk_dir.rglob("video.hevc"):
            segs.append(video.parent)
    return segs


def load_steering(seg_dir):
    """Return (t_array, value_array) or (None, None) if missing."""
    base = Path(seg_dir) / "processed_log" / "CAN" / "steering_angle"
    tp = base / "t"
    vp = base / "value"
    if not tp.exists() or not vp.exists():
        return None, None
    t = np.load(tp, allow_pickle=True) if tp.suffix == ".npy" else np.fromfile(tp, dtype=np.float64)
    v = np.load(vp, allow_pickle=True) if vp.suffix == ".npy" else np.fromfile(vp, dtype=np.float64)
    return t, v


def load_frame_times(seg_dir):
    """
    comma2k19 ships global_pose/frame_times (one timestamp per video frame)
    alongside the video. That file lets us convert frame_idx -> absolute
    time without decoding the whole HEVC just to get PTS.
    Returns np.ndarray of shape (N_frames,) or None.
    """
    candidates = [
        Path(seg_dir) / "global_pose" / "frame_times",
        Path(seg_dir) / "frame_times",
    ]
    for c in candidates:
        if c.exists():
            try:
                return np.load(c, allow_pickle=True) if c.suffix == ".npy" else np.fromfile(c, dtype=np.float64)
            except Exception:
                continue
    return None


def decode_and_save(video_path, frame_indices, out_dir, image_size):
    """Decode only the requested frame indices with PyAV, save as jpg."""
    import av
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(int(i) for i in frame_indices)
    max_wanted = max(wanted) if wanted else -1

    written = {}
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        idx = -1
        for frame in container.decode(stream):
            idx += 1
            if idx > max_wanted:
                break
            if idx not in wanted:
                continue
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
            out_path = out_dir / f"{idx:06d}.jpg"
            cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            written[idx] = str(out_path)
    finally:
        container.close()
    return written


def segment_id(seg_dir, data_root):
    # Use relative path from data_root as stable segment id.
    rel = Path(seg_dir).relative_to(Path(data_root))
    return str(rel).replace("/", "|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data_cache")
    ap.add_argument("--out_csv", type=str, default=None,
                    help="default: <data_root>/frames.csv")
    ap.add_argument("--frames_dir", type=str, default=None,
                    help="default: <data_root>/frames")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--steering_clip", type=float, default=1.0,
                    help="steering_angle is clipped to +/- this (rad) and then divided by it")
    ap.add_argument("--max_segments", type=int, default=None,
                    help="optional cap for smoke tests")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_csv = Path(args.out_csv) if args.out_csv else data_root / "frames.csv"
    frames_dir = Path(args.frames_dir) if args.frames_dir else data_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    segs = find_segments(data_root)
    if args.max_segments:
        segs = segs[: args.max_segments]
    if not segs:
        print(f"[prepare] no segments found under {data_root}", file=sys.stderr)
        sys.exit(1)
    print(f"[prepare] {len(segs)} segments found")

    resume = out_csv.exists()
    f_csv = open(out_csv, "a" if resume else "w", newline="")
    writer = csv.writer(f_csv)
    if not resume:
        writer.writerow(["image_path", "steering", "segment_id"])

    # Skip already-processed segments on resume.
    done_segs = set()
    if resume:
        f_csv.flush()
        with open(out_csv) as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) >= 3:
                    done_segs.add(row[2])

    n_written = 0
    for seg_dir in tqdm(segs, desc="segments"):
        sid = segment_id(seg_dir, data_root)
        if sid in done_segs:
            continue

        t_can, v_can = load_steering(seg_dir)
        if t_can is None or v_can is None or len(t_can) == 0:
            continue
        ft = load_frame_times(seg_dir)
        if ft is None or len(ft) == 0:
            continue

        n_frames = len(ft)
        duration = float(ft[-1] - ft[0])
        if duration <= 0:
            continue
        native_fps = n_frames / duration
        step = max(1, int(round(native_fps / args.fps)))
        frame_idxs = list(range(0, n_frames, step))

        video_path = Path(seg_dir) / "video.hevc"
        seg_out = frames_dir / sid
        try:
            written = decode_and_save(video_path, frame_idxs, seg_out, args.image_size)
        except Exception as e:
            print(f"[warn] decode failed {sid}: {e}", file=sys.stderr)
            continue

        # Match each frame's timestamp to nearest CAN sample.
        t_can = np.asarray(t_can, dtype=np.float64)
        v_can = np.asarray(v_can, dtype=np.float64)
        order = np.argsort(t_can)
        t_can_s = t_can[order]
        v_can_s = v_can[order]

        for idx in frame_idxs:
            if idx not in written:
                continue
            t_frame = float(ft[idx])
            pos = np.searchsorted(t_can_s, t_frame)
            pos = min(max(pos, 0), len(t_can_s) - 1)
            # pick the closer neighbor
            if pos > 0 and abs(t_can_s[pos - 1] - t_frame) < abs(t_can_s[pos] - t_frame):
                pos -= 1
            ang = float(v_can_s[pos])
            ang = max(-args.steering_clip, min(args.steering_clip, ang))
            steering = ang / args.steering_clip  # [-1, 1]

            rel = Path(written[idx]).relative_to(Path.cwd()) if Path(written[idx]).is_absolute() else Path(written[idx])
            writer.writerow([str(written[idx]), f"{steering:.6f}", sid])
            n_written += 1

        f_csv.flush()

    f_csv.close()
    print(f"[done] wrote {n_written} rows -> {out_csv}")


if __name__ == "__main__":
    main()
