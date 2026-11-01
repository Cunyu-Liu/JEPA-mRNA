#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import resource
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def predict(binary, params, row, directory, env):
    path = directory / "input.seq"
    path.write_text(row["seq"] + "\n")
    completed = subprocess.run([str(binary), "predict", str(path), "--params", str(params)],
                               check=True, capture_output=True, text=True, env=env)
    lines = completed.stdout.splitlines()
    index = lines.index(">structure")
    structure = lines[index + 1].strip()
    assert len(structure) == len(row["seq"])
    stack = []
    pairs = []
    for position, token in enumerate(structure):
        if token == "(":
            stack.append(position)
        elif token == ")":
            assert stack
            pairs.append([stack.pop(), position])
        else:
            assert token == "."
    assert not stack
    return {"name": row["name"], "pairs": sorted(pairs)}


def score(predictions, rows):
    by_name = {prediction["name"]: prediction for prediction in predictions}
    assert len(by_name) == len(rows)
    tp = fp = fn = 0
    output = []
    for row in rows:
        prediction = by_name[row["name"]]
        pred = {tuple(pair) for pair in prediction["pairs"]}
        true = {tuple(pair) for pair in row["pairs"]}
        a, b, c = len(pred & true), len(pred - true), len(true - pred)
        tp, fp, fn = tp + a, fp + b, fn + c
        output.append({"name": row["name"], "tp": a, "fp": b, "fn": c,
                       "f1": 2 * a / (2 * a + b + c) if 2 * a + b + c else 1.0,
                       "pred_pairs": prediction["pairs"]})
    return {"n_sequences": len(rows), "tp": tp, "fp": fp, "fn": fn,
            "micro_f1": 2 * tp / (2 * tp + fp + fn),
            "macro_f1": float(np.mean([row["f1"] for row in output]))}, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev/third_party/EternaFold"))
    parser.add_argument("--data", type=Path, default=Path("/root/rna_jepa_data/ss_data/jsonl/bprna_ts0.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke:
        assert args.repeats == 3 and args.limit is None
    binary = args.repo / "src/contrafold"
    params = args.repo / "parameters/EternaFoldParams.v1"
    assert binary.is_file() and params.is_file()
    rows = [json.loads(line) for line in args.data.open()]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not args.smoke:
        assert len(rows) == 1288
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    repeats = []
    reference = None
    with tempfile.TemporaryDirectory(prefix="eternafold_round38_") as tmp:
        directory = Path(tmp)
        predict(binary, params, rows[0], directory, env)
        for repeat in range(args.repeats):
            predictions = []
            tic = time.perf_counter()
            for row in rows:
                predictions.append(predict(binary, params, row, directory, env))
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
            print(json.dumps(repeats[-1]), flush=True)
    result = {
        "schema_version": 1, "status": "SMOKE_ONLY" if args.smoke else "PASS",
        "system": "EternaFold", "repository": str(args.repo),
        "commit": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
        "binary": str(binary), "binary_sha256": sha256(binary),
        "parameters": str(params), "parameters_sha256": sha256(params),
        "data": str(args.data), "data_sha256": sha256(args.data),
        "protocol": "raw-sequence end-to-end; official serial CPU binary and MEA decode; single thread",
        "device": "CPU (native method; no GPU timing)", "batch_size": 1,
        "batch_note": "B=2/4 are not applicable to the official serial CLI and are not fabricated",
        "cpu_threads": 1, "n_sequences": len(rows), "repeats": repeats,
        "peak_child_rss_mib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
        "per_sequence": per_sequence,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.out)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
