#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_pickle(args.source)
    train = frame.loc[frame["set"].eq("train")]
    rows = [{"name": str(row.Id), "seq": str(row.sequence).upper().replace("T", "U")}
            for row in train.itertuples()]
    assert len(rows) == len(train) and all(row["seq"] for row in rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"schema_version": 1, "status": "PASS", "source": str(args.source),
                "source_sha256": sha256(args.source), "n_sequences": len(rows),
                "n_unique_sequences": len({row["seq"] for row in rows}),
                "output": str(args.out), "output_sha256": sha256(args.out),
                "selection": "set == train"}
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
