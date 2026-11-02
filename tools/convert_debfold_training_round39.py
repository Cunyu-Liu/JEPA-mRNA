#!/usr/bin/env python3
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with zipfile.ZipFile(args.archive) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".ct"))
        assert names
        for name in names:
            text = archive.read(name).decode("utf-8")
            entries = []
            for line in text.splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                fields = line.split()
                if len(fields) >= 2 and fields[1] == "Energy":
                    continue
                assert len(fields) >= 6, (name, line)
                entries.append((int(fields[0]), fields[1], int(fields[4]), int(fields[5])))
            assert [entry[0] for entry in entries] == list(range(1, len(entries) + 1)), name
            sequence = "".join(entry[1] for entry in entries).upper().replace("T", "U")
            assert set(sequence) <= set("ACGUN"), (name, sequence)
            pairs = set()
            for index, _, partner, _ in entries:
                if partner:
                    assert 1 <= partner <= len(entries) and partner != index, (name, index, partner)
                    assert entries[partner - 1][2] == index, (name, index, partner)
                    pairs.add(tuple(sorted((index - 1, partner - 1))))
            rows.append({"name": Path(name).stem, "seq": sequence,
                         "pairs": [list(pair) for pair in sorted(pairs)]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    unique = len({row["seq"] for row in rows})
    manifest = {"schema_version": 1, "status": "PASS", "source": str(args.archive),
                "source_sha256": sha256(args.archive), "n_structures": len(rows),
                "n_unique_sequences": unique, "output": str(args.out),
                "output_sha256": sha256(args.out)}
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
