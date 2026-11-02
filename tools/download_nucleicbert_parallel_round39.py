#!/usr/bin/env python3
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import subprocess
import time
from pathlib import Path

URL = "https://zenodo.org/api/records/16989562/files/secstr.ckpt/content"
EXPECTED_SIZE = 4_970_204_510
EXPECTED_MD5 = "b5b3d40fa07adc3cc8c1942e6ebfe1a5"


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def download_chunk(root, start, end, attempts):
    path = root / "chunks" / f"secstr.{start:010d}-{end:010d}.part"
    expected = end - start + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        size = path.stat().st_size if path.exists() else 0
        if size == expected:
            return {"path": str(path), "start": start, "end": end,
                    "bytes": size, "sha256": digest(path)}
        if size > expected:
            raise RuntimeError(f"CHUNK_OVERRUN {path} {size}>{expected}")
        print(json.dumps({"event": "resume", "attempt": attempt, "start": start, "end": end,
                          "offset": start + size, "chunk_bytes": size, "expected": expected}), flush=True)
        with path.open("ab") as handle:
            completed = subprocess.run(["curl", "-fL", "--connect-timeout", "20", "--max-time", "330",
                                        "--range", f"{start + size}-{end}", URL],
                                       stdout=handle, stderr=subprocess.PIPE, check=False)
        size = path.stat().st_size
        if size > expected:
            raise RuntimeError(f"CHUNK_OVERRUN {path} {size}>{expected}")
        if completed.returncode and attempt == attempts:
            raise RuntimeError(f"CHUNK_DOWNLOAD_FAILED {path}: {completed.stderr.decode(errors='replace')[-2000:]}")
        if completed.returncode:
            time.sleep(5)
    raise RuntimeError(f"CHUNK_INCOMPLETE {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--attempts", type=int, default=2000)
    args = parser.parse_args()
    assert args.workers > 0 and args.attempts > 0
    args.root.mkdir(parents=True, exist_ok=True)
    final = args.root / "secstr.ckpt"
    if final.exists():
        assert final.stat().st_size == EXPECTED_SIZE and digest(final, "md5") == EXPECTED_MD5
        print("NUCLEICBERT_SECSTR_DOWNLOAD_PASS_ALREADY_VERIFIED", flush=True)
        return
    prefix = args.root / "secstr.ckpt.part"
    assert prefix.is_file()
    prefix_size = prefix.stat().st_size
    assert 0 < prefix_size < EXPECTED_SIZE
    prefix_sha256 = digest(prefix)
    remaining = EXPECTED_SIZE - prefix_size
    bounds = [prefix_size + remaining * index // args.workers for index in range(args.workers + 1)]
    jobs = [(bounds[index], bounds[index + 1] - 1) for index in range(args.workers)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_chunk, args.root, start, end, args.attempts) for start, end in jobs]
        chunks = [future.result() for future in futures]
    assert prefix.stat().st_size == prefix_size and digest(prefix) == prefix_sha256
    assembly = args.root / "secstr.ckpt.assembling"
    with assembly.open("wb") as output:
        for path in [prefix] + [Path(chunk["path"]) for chunk in chunks]:
            with path.open("rb") as source:
                for block in iter(lambda: source.read(8 << 20), b""):
                    output.write(block)
    assert assembly.stat().st_size == EXPECTED_SIZE
    md5 = digest(assembly, "md5")
    assert md5 == EXPECTED_MD5, (md5, EXPECTED_MD5)
    sha256 = digest(assembly)
    assembly.replace(final)
    receipt = {"schema_version": 1, "status": "PASS_OFFICIAL_SIZE_MD5_AND_SHA256",
               "url": URL, "bytes": EXPECTED_SIZE, "official_md5": EXPECTED_MD5, "sha256": sha256,
               "workers": args.workers, "prefix": {"path": str(prefix), "bytes": prefix_size,
                                                   "sha256": prefix_sha256},
               "chunks": chunks, "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (args.root / "download_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (args.root / "SHA256SUMS").write_text(f"{sha256}  secstr.ckpt\n")
    print(json.dumps({"status": receipt["status"], "bytes": EXPECTED_SIZE, "md5": md5, "sha256": sha256}, indent=2), flush=True)


if __name__ == "__main__":
    main()
