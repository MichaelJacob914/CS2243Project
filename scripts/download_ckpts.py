#!/usr/bin/env python3
"""Download openpi checkpoints from the public GCS bucket into $OPENPI_DATA_HOME.

Stdlib only, so it runs before any venv exists. Files land where openpi's
`download.maybe_download` looks for them ($OPENPI_DATA_HOME/<bucket>/<path>), so openpi
treats them as cached and never re-downloads. Files get today's date (no server mtimes),
which keeps them safe from the netscratch purge; run scripts/fix_mtimes.sh afterwards anyway.

Usage:
    python3 scripts/download_ckpts.py pi0_libero pi0_aloha_sim
    python3 scripts/download_ckpts.py --list pi05_libero
"""

import argparse
import concurrent.futures
import json
import os
import pathlib
import sys
import time
import urllib.parse
import urllib.request

API = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
MEDIA = "https://storage.googleapis.com/{bucket}/{name}"
# Always fetched: openpi's tokenizer (gs://big_vision/paligemma_tokenizer.model).
EXTRA = [("big_vision", "paligemma_tokenizer.model")]


def list_objects(bucket: str, prefix: str) -> list[dict]:
    items, token = [], None
    while True:
        q = {"prefix": prefix, "fields": "items(name,size),nextPageToken", "maxResults": "1000"}
        if token:
            q["pageToken"] = token
        with urllib.request.urlopen(API.format(bucket=bucket) + "?" + urllib.parse.urlencode(q)) as r:
            d = json.load(r)
        items += d.get("items", [])
        token = d.get("nextPageToken")
        if not token:
            return items


def fetch(bucket: str, name: str, size: int, root: pathlib.Path, retries: int = 5) -> str:
    dst = root / bucket / name
    if dst.exists() and dst.stat().st_size == size:
        return f"skip {name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    url = MEDIA.format(bucket=bucket, name=urllib.parse.quote(name))
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(8 << 20):
                    f.write(chunk)
            if tmp.stat().st_size != size:
                raise OSError(f"size {tmp.stat().st_size} != {size}")
            os.replace(tmp, dst)
            return f"ok   {name} ({size / 1e9:.2f} GB)"
        except Exception as e:  # noqa: BLE001
            print(f"retry {attempt}/{retries} {name}: {e}", file=sys.stderr, flush=True)
            time.sleep(10 * attempt)
    raise RuntimeError(f"failed: {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="names under gs://openpi-assets/checkpoints/")
    ap.add_argument("--list", action="store_true", help="only print file count and size")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    root = pathlib.Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()
    jobs = [(b, n, None) for b, n in EXTRA]
    for ck in args.checkpoints:
        objs = list_objects("openpi-assets", f"checkpoints/{ck}/")
        if not objs:
            sys.exit(f"no such checkpoint: gs://openpi-assets/checkpoints/{ck}")
        print(f"{ck}: {len(objs)} files, {sum(int(o['size']) for o in objs) / 1e9:.2f} GB", flush=True)
        # Skip folder-marker objects ("…/", 0 bytes); writing them would create files named like folders.
        jobs += [("openpi-assets", o["name"], int(o["size"])) for o in objs if not o["name"].endswith("/")]
    if args.list:
        return

    # Sizes for the extra files come from a HEAD request.
    sized = []
    for b, n, s in jobs:
        if s is None:
            req = urllib.request.Request(MEDIA.format(bucket=b, name=n), method="HEAD")
            with urllib.request.urlopen(req) as r:
                s = int(r.headers["Content-Length"])
        sized.append((b, n, s))

    print(f"-> {root}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
        for msg in ex.map(lambda j: fetch(*j, root), sized):
            print(msg, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
