"""
Download comma2k19 chunks into data_root.

The official release is distributed as per-chunk archives (Chunk_1.zip ...
Chunk_10.zip, ~10 GB each) hosted on Archive.org. URLs per the official
README at https://github.com/commaai/comma2k19.

Usage:
    python -m scripts.download_comma2k19 --data_root ./data_cache --chunks 1 2

The archive layout after extraction is:
    data_root/
      Chunk_1/
        <dongle_id>/<route_id>/<segment_idx>/
          video.hevc
          processed_log/CAN/steering_angle/value
          processed_log/CAN/steering_angle/t
          ...
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


# Archive.org hosted mirror used by the official repo.
CHUNK_URLS = {
    i: f"https://archive.org/download/comma2k19/Chunk_{i}.zip"
    for i in range(1, 11)
}


def _have(cmd):
    return subprocess.call(
        ["which", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ) == 0


def download(url, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[download] exists, skip: {out_path}")
        return
    print(f"[download] {url} -> {out_path}")
    if _have("aria2c"):
        cmd = [
            "aria2c", "-x", "8", "-s", "8",
            "-d", str(out_path.parent), "-o", out_path.name, url,
        ]
    elif _have("wget"):
        cmd = ["wget", "-c", "-O", str(out_path), url]
    else:
        cmd = ["curl", "-L", "-C", "-", "-o", str(out_path), url]
    subprocess.check_call(cmd)


def unzip(zip_path, out_dir):
    out_dir = Path(out_dir)
    marker = out_dir / f".unzipped_{Path(zip_path).stem}"
    if marker.exists():
        print(f"[unzip] already extracted: {zip_path}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] {zip_path} -> {out_dir}")
    subprocess.check_call(["unzip", "-q", "-o", str(zip_path), "-d", str(out_dir)])
    marker.touch()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data_cache")
    ap.add_argument("--chunks", type=int, nargs="+", default=[1])
    ap.add_argument("--keep_zip", action="store_true",
                    help="keep the zip after extraction (default: delete to save disk)")
    args = ap.parse_args()

    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)

    for c in args.chunks:
        if c not in CHUNK_URLS:
            print(f"[skip] unknown chunk id: {c}", file=sys.stderr)
            continue
        url = CHUNK_URLS[c]
        zip_path = root / f"Chunk_{c}.zip"
        download(url, zip_path)
        unzip(zip_path, root)
        if not args.keep_zip:
            try:
                os.remove(zip_path)
                print(f"[cleanup] removed {zip_path}")
            except OSError:
                pass

    print(f"[done] extracted chunks into {root}")


if __name__ == "__main__":
    main()
