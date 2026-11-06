import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

import requests


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_part(url, path, start, end, retries=10):
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return {"part": path.name, "bytes": expected, "status": "reused"}
    for attempt in range(retries):
        try:
            have = path.stat().st_size if path.exists() else 0
            if have > expected:
                raise RuntimeError(f"oversize part {path}: {have}>{expected}")
            with requests.get(url, headers={"Range": f"bytes={start + have}-{end}"},
                              stream=True, timeout=(20, 90)) as response:
                if response.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
                if response.status_code == 200 and have:
                    raise RuntimeError("range resume ignored by server")
                mode = "ab" if have else "wb"
                with open(path, mode) as fh:
                    for block in response.iter_content(1 << 20):
                        if block:
                            fh.write(block)
            if path.stat().st_size == expected:
                return {"part": path.name, "bytes": expected, "status": "downloaded"}
            raise RuntimeError(f"short part {path}: {path.stat().st_size}/{expected}")
        except Exception as exc:
            if attempt + 1 == retries:
                raise
            print(json.dumps({"part": path.name, "attempt": attempt + 1,
                              "error": str(exc)}, ensure_ascii=False), flush=True)
            time.sleep(min(30, 2 ** attempt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--chunk-mib", type=int, default=32)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size == args.expected_size:
        actual = digest(output, "md5")
        if actual != args.expected_md5:
            raise RuntimeError(f"existing output MD5 {actual} != {args.expected_md5}")
        print(json.dumps({"status": "PASS_REUSED", "md5": actual,
                          "sha256": digest(output, "sha256")}), flush=True)
        return
    parts_dir = output.with_suffix(output.suffix + ".parts")
    parts_dir.mkdir(exist_ok=True)
    chunk = args.chunk_mib << 20
    ranges = [(i, start, min(start + chunk - 1, args.expected_size - 1))
              for i, start in enumerate(range(0, args.expected_size, chunk))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_part, args.url,
                               parts_dir / f"{i:05d}.part", start, end)
                   for i, start, end in ranges]
        for future in concurrent.futures.as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    assembling = output.with_suffix(output.suffix + ".assembling")
    with open(assembling, "wb") as dst:
        for i, _, _ in ranges:
            with open(parts_dir / f"{i:05d}.part", "rb") as src:
                for block in iter(lambda: src.read(8 << 20), b""):
                    dst.write(block)
    if assembling.stat().st_size != args.expected_size:
        raise RuntimeError(f"assembled size {assembling.stat().st_size} != {args.expected_size}")
    actual_md5 = digest(assembling, "md5")
    if actual_md5 != args.expected_md5:
        raise RuntimeError(f"assembled MD5 {actual_md5} != {args.expected_md5}")
    os.replace(assembling, output)
    print(json.dumps({"status": "PASS", "bytes": args.expected_size,
                      "md5": actual_md5, "sha256": digest(output, "sha256"),
                      "parts": len(ranges), "workers": args.workers}), flush=True)


if __name__ == "__main__":
    main()
