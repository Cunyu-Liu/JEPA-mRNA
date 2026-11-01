#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from benchmark_eternafold_round38 import EXPECTED_SPLIT_COUNTS, EXPECTED_SPLIT_SHA256, canonical, score


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(binary, row, directory, env):
    fasta = directory / "input.fa"
    ct = directory / "output.ct"
    fasta.write_text(f">{row['name']}\n{row['seq']}\n")
    ct.unlink(missing_ok=True)
    command = [str(binary), str(fasta), str(ct)]
    if binary.name == "Fold":
        command.append("--MFE")
    elif binary.name == "MaxExpect":
        command.append("--sequence")
    subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    lines = ct.read_text().splitlines()
    length = len(row["seq"])
    assert len(lines) == length + 1, (binary.name, row["name"], length, len(lines))
    assert lines[0].split()[0] == str(length)
    pairs = []
    for index, line in enumerate(lines[1:], start=1):
        fields = line.split()
        assert len(fields) == 6 and int(fields[0]) == index, (row["name"], index, line)
        assert fields[1].replace("T", "U") == row["seq"][index - 1].replace("T", "U")
        expected_next = 0 if index == length else index + 1
        assert int(fields[2]) == index - 1 and int(fields[3]) == expected_next, (row["name"], index, fields)
        j = int(fields[4]) - 1
        assert -1 <= j < length and int(fields[5]) == index
        if j > index - 1:
            pairs.append([index - 1, j])
    assert len(pairs) == len({tuple(pair) for pair in pairs})
    return {"name": row["name"], "pairs": sorted(pairs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev/packages/rnastructure-6.6/RNAstructure/exe"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--clean-audit", type=Path)
    args = parser.parse_args()
    if not args.smoke:
        assert args.repeats == 3 and args.limit is None
        assert args.data.stem in EXPECTED_SPLIT_COUNTS
    rows = [json.loads(line) for line in args.data.open()]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not args.smoke:
        if args.clean_audit:
            audit = json.loads(args.clean_audit.read_text())
            assert audit["status"] == "PASS" and audit["split_sha256"] == EXPECTED_SPLIT_SHA256[args.data.stem]
            assert audit["clean_sha256"] == sha256(args.data) and len(rows) == audit["clean_n"]
            assert audit["raw_n"] == EXPECTED_SPLIT_COUNTS[args.data.stem]
            assert audit["raw_n"] == audit["clean_n"] + audit["removed_exact"] + audit["removed_homology_only"]
        else:
            assert len(rows) == EXPECTED_SPLIT_COUNTS[args.data.stem]
            assert sha256(args.data) == EXPECTED_SPLIT_SHA256[args.data.stem]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               DATAPATH=str(args.repo.parent / "data_tables"))
    systems = {"fold": args.repo / "Fold", "maxexpect": args.repo / "MaxExpect"}
    for binary in systems.values():
        assert binary.is_file(), binary
    results = {}
    with tempfile.TemporaryDirectory(prefix="rnastructure_round39_") as tmp:
        directory = Path(tmp)
        for name, binary in systems.items():
            repeats = []
            reference = None
            predict(binary, rows[0], directory, env)
            for repeat in range(args.repeats):
                tic = time.perf_counter()
                predictions = [predict(binary, row, directory, env) for row in rows]
                elapsed = time.perf_counter() - tic
                metrics, per_sequence = score(predictions, rows)
                identity = [[p["name"], p["pairs"]] for p in predictions]
                if reference is None:
                    reference = identity
                else:
                    assert identity == reference
                repeats.append({"repeat": repeat, "elapsed_seconds": elapsed,
                                "sequences_per_second": len(rows) / elapsed,
                                "prediction_sha256": hashlib.sha256(canonical(identity).encode()).hexdigest(),
                                "metrics": metrics})
            results[name] = {"binary": str(binary), "binary_sha256": sha256(binary),
                             "repeats": repeats, "per_sequence": per_sequence}
    payload = {"schema_version": 1, "status": "SMOKE_ONLY" if args.smoke else "PASS",
               "system": "RNAstructure", "data": str(args.data),
               "data_sha256": sha256(args.data), "device": "CPU native binaries",
               "n_sequences": len(rows), "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"RNASTRUCTURE_PASS out={args.out}")


if __name__ == "__main__":
    main()
