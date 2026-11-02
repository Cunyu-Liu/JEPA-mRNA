#!/usr/bin/env python3
import argparse
import datetime
import hashlib
import json
from pathlib import Path

EXPECTED_SIZE = 4_970_204_510
EXPECTED_MD5 = "b5b3d40fa07adc3cc8c1942e6ebfe1a5"


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def assemble(root, prefix_sha256, expected_size=EXPECTED_SIZE, expected_md5=EXPECTED_MD5):
    prefix = root / "secstr.ckpt.part"
    final = root / "secstr.ckpt"
    if final.exists():
        assert final.stat().st_size == expected_size and digest(final, "md5") == expected_md5
        return {"status": "PASS_OFFICIAL_SIZE_MD5_AND_SHA256", "bytes": expected_size,
                "md5": expected_md5, "sha256": digest(final), "already_verified": True}
    assert prefix.is_file() and 0 < prefix.stat().st_size < expected_size
    prefix_size = prefix.stat().st_size
    assert digest(prefix) == prefix_sha256
    candidates = {}
    for path in (root / "chunks").glob("secstr.*-*.part"):
        start, end = map(int, path.stem.split(".")[-1].split("-"))
        assert 0 <= start <= end < expected_size and start not in candidates
        if path.stat().st_size == end - start + 1:
            candidates[start] = (path, end)
    first = [(start, value) for start, value in candidates.items()
             if start < prefix_size <= value[1] + 1]
    assert len(first) == 1, f"expected one chunk covering prefix boundary, got {first}"
    cursor = first[0][0]
    chunks = []
    while cursor < expected_size:
        assert cursor in candidates, f"missing or incomplete chunk at {cursor}"
        path, end = candidates[cursor]
        chunks.append({"path": str(path), "start": cursor, "end": end,
                       "skip": prefix_size - cursor if not chunks else 0})
        cursor = end + 1
    assert cursor == expected_size and chunks[0]["skip"] > 0
    assembly = root / "secstr.ckpt.assembling"
    with assembly.open("wb") as output:
        with prefix.open("rb") as source:
            for block in iter(lambda: source.read(8 << 20), b""):
                output.write(block)
        for chunk in chunks:
            with Path(chunk["path"]).open("rb") as source:
                source.seek(chunk["skip"])
                for block in iter(lambda: source.read(8 << 20), b""):
                    output.write(block)
    assert assembly.stat().st_size == expected_size
    assert prefix.stat().st_size == prefix_size and digest(prefix) == prefix_sha256
    md5 = digest(assembly, "md5")
    assert md5 == expected_md5, (md5, expected_md5)
    sha256 = digest(assembly)
    assembly.replace(final)
    receipt = {"schema_version": 1, "status": "PASS_OFFICIAL_SIZE_MD5_AND_SHA256",
               "url": "https://zenodo.org/api/records/16989562/files/secstr.ckpt/content",
               "bytes": expected_size, "official_md5": expected_md5, "sha256": sha256,
               "assembly": "stable prefix plus complete existing chunks; first chunk sliced at new prefix boundary",
               "prefix": {"path": str(prefix), "bytes": prefix_size, "sha256": prefix_sha256},
               "chunks": chunks, "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    temporary = (root / "download_receipt.json").with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(root / "download_receipt.json")
    (root / "SHA256SUMS").write_text(f"{sha256}  secstr.ckpt\n")
    return {key: receipt[key] for key in ("status", "bytes", "official_md5", "sha256")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prefix-sha256", required=True)
    args = parser.parse_args()
    result = assemble(args.root, args.prefix_sha256)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
