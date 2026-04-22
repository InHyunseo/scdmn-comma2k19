"""
Download comma2k19 chunks into data_root.

Primary source: archive.org item "comma2k19", which hosts per-chunk zip
archives (Chunk_1.zip ... Chunk_10.zip, ~10 GB each) AND the extracted
individual files. archive.org occasionally returns HTTP 503 under load,
especially for the large zips.

Strategy:
    1. Try the chunk zip with retries (exponential backoff).
    2. If that keeps failing, fall back to downloading the chunk's
       individual files via the archive.org metadata API. Slower but
       more reliable — 503s on any single small file can be retried
       without discarding gigabytes of progress.

Usage:
    python -m scripts.download_comma2k19 --data_root ./data_cache --chunks 1
    python -m scripts.download_comma2k19 --data_root ./data_cache --chunks 1 --force_files
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ITEM = "comma2k19"
BASE = f"https://archive.org/download/{ITEM}"
META_URL = f"https://archive.org/metadata/{ITEM}"


def _have(cmd):
    return subprocess.call(
        ["which", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ) == 0


def _http_get_json(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "scdmn-comma2k19/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _curl_status(url, timeout=30):
    """Return HTTP status code via a HEAD request (using curl)."""
    try:
        out = subprocess.check_output(
            ["curl", "-sI", "--max-time", str(timeout), url],
            stderr=subprocess.DEVNULL,
        ).decode(errors="ignore").splitlines()
        if out:
            parts = out[0].split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    except Exception:
        pass
    return 0


def _download_cmd(url, out_path):
    if _have("aria2c"):
        return [
            "aria2c", "-x", "8", "-s", "8",
            "--max-tries=10", "--retry-wait=15",
            "--connect-timeout=30",
            "-d", str(out_path.parent), "-o", out_path.name, url,
        ]
    if _have("wget"):
        return [
            "wget", "-c", "--tries=10", "--waitretry=15",
            "--retry-connrefused", "--read-timeout=60",
            "-O", str(out_path), url,
        ]
    return ["curl", "-L", "--retry", "10", "--retry-delay", "15",
            "-C", "-", "-o", str(out_path), url]


def download_with_retry(url, out_path, max_attempts=6, base_sleep=30):
    """Try to fetch `url` to `out_path`, backing off on transient failures."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[download] exists, skip: {out_path}")
        return True

    for attempt in range(1, max_attempts + 1):
        status = _curl_status(url)
        if status and status != 200 and status != 206:
            print(f"[download] attempt {attempt}: HEAD returned {status} for {url}")
            if status == 503 and attempt < max_attempts:
                sleep_s = base_sleep * (2 ** (attempt - 1))
                print(f"[download]   archive.org busy (503). sleeping {sleep_s}s...")
                time.sleep(sleep_s)
                continue
            # 404 etc: no point retrying.
            if status in (401, 403, 404):
                return False
        cmd = _download_cmd(url, out_path)
        print(f"[download] attempt {attempt}: {' '.join(cmd[:3])} ... {url}")
        rc = subprocess.call(cmd)
        if rc == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return True
        sleep_s = base_sleep * (2 ** (attempt - 1))
        print(f"[download]   failed (rc={rc}). sleeping {sleep_s}s before retry...")
        time.sleep(sleep_s)
    return False


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


def list_chunk_files(chunk_id):
    """
    Hit archive.org metadata API and return (name, size) for every file
    that lives under Chunk_<id>/ inside the item.
    """
    print(f"[meta] fetching file list for chunk {chunk_id}...")
    data = _http_get_json(META_URL)
    files = data.get("files", [])
    prefix = f"Chunk_{chunk_id}/"
    out = []
    for f in files:
        name = f.get("name", "")
        if name.startswith(prefix):
            size = int(f.get("size", 0)) if f.get("size") else 0
            out.append((name, size))
    print(f"[meta] chunk {chunk_id}: {len(out)} files")
    return out


def download_files_fallback(chunk_id, data_root):
    """Download per-file instead of the single zip. Resumable per file."""
    entries = list_chunk_files(chunk_id)
    if not entries:
        print(f"[fallback] metadata returned no files for chunk {chunk_id}", file=sys.stderr)
        return False

    ok_all = True
    for i, (name, size) in enumerate(entries, 1):
        out_path = Path(data_root) / name
        if out_path.exists() and size and out_path.stat().st_size == size:
            continue
        url = f"{BASE}/{name}"
        print(f"[fallback] ({i}/{len(entries)}) {name}  ({size/1e6:.1f} MB)")
        ok = download_with_retry(url, out_path, max_attempts=6, base_sleep=20)
        if not ok:
            print(f"[fallback] failed: {name}", file=sys.stderr)
            ok_all = False
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data_cache")
    ap.add_argument("--chunks", type=int, nargs="+", default=[1])
    ap.add_argument("--keep_zip", action="store_true")
    ap.add_argument("--force_files", action="store_true",
                    help="skip the zip attempt and download individual files directly")
    ap.add_argument("--zip_attempts", type=int, default=4)
    args = ap.parse_args()

    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)

    for c in args.chunks:
        zip_path = root / f"Chunk_{c}.zip"
        zip_url = f"{BASE}/Chunk_{c}.zip"

        got_zip = False
        if not args.force_files:
            got_zip = download_with_retry(
                zip_url, zip_path,
                max_attempts=args.zip_attempts, base_sleep=60,
            )

        if got_zip:
            unzip(zip_path, root)
            if not args.keep_zip:
                try:
                    os.remove(zip_path)
                    print(f"[cleanup] removed {zip_path}")
                except OSError:
                    pass
        else:
            if not args.force_files:
                print(f"[warn] zip path failed for chunk {c} after retries; falling back to per-file download")
            ok = download_files_fallback(c, root)
            if not ok:
                print(f"[error] chunk {c} incomplete", file=sys.stderr)

    print(f"[done] data at {root}")


if __name__ == "__main__":
    main()
