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


EXPECTED_SPLIT_COUNTS = {
    "bprna_ts0": 1288,
    "bprna_new": 5388,
    "archiveii_embok_clean": 2544,
    "ref_pdb_ts1": 63,
    "ref_pdb_ts2": 39,
    "ref_pdb_ts3": 19,
    "ref_pdb_ts_hard": 28,
    "testsetb": 428,
}
EXPECTED_SPLIT_SHA256 = {
    "bprna_ts0": "f21eb88c1e83fc25bc47cec823e1d4aec441b18a5305f20d5314411667d06882",
    "bprna_new": "271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9",
    "archiveii_embok_clean": "6c1fbaa4684cbe4644acdac2f31ae95e5f9b0eb36871f23d8148006f3d571399",
    "ref_pdb_ts1": "16a6961786f63fa5cd64c790df33f945a66d8ef2175bea8828d9211c334b176a",
    "ref_pdb_ts2": "26e888eda6e8c10786f2fc3ac7cfc10e385035af270796d8e50ce1f516d6ed9d",
    "ref_pdb_ts3": "0224d61efe30cf18e9eb072a7bc6d2f3ca07663c7ae7e27e0a32c4977e0bde6a",
    "ref_pdb_ts_hard": "486b4bc0b3dfea87861b513790483d85c1cb986625711e967d12afedc8c5c806",
    "testsetb": "78bba9255f36b301617c5017bdad79b7329190361c6822d2aec12f0043483a06",
}


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
    command = [str(binary), "predict", str(path)]
    if params is not None:
        command.extend(["--params", str(params)])
    completed = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
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
    parser.add_argument("--native-contrafold", action="store_true")
    parser.add_argument("--clean-audit", type=Path)
    args = parser.parse_args()
    if not args.smoke:
        assert args.repeats == 3 and args.limit is None
    binary = args.repo / "src/contrafold"
    params = None if args.native_contrafold else args.repo / "parameters/EternaFoldParams.v1"
    assert binary.is_file() and (params is None or params.is_file())
    rows = [json.loads(line) for line in args.data.open()]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not args.smoke:
        assert args.data.stem in EXPECTED_SPLIT_COUNTS, f"unregistered split: {args.data.stem}"
        if args.clean_audit:
            audit = json.loads(args.clean_audit.read_text())
            assert audit["status"] == "PASS" and audit["split_sha256"] == EXPECTED_SPLIT_SHA256[args.data.stem]
            assert audit["clean_sha256"] == sha256(args.data) and len(rows) == audit["clean_n"]
            assert audit["raw_n"] == EXPECTED_SPLIT_COUNTS[args.data.stem]
            assert audit["raw_n"] == audit["clean_n"] + audit["removed_exact"] + audit["removed_homology_only"]
        else:
            assert len(rows) == EXPECTED_SPLIT_COUNTS[args.data.stem]
            assert sha256(args.data) == EXPECTED_SPLIT_SHA256[args.data.stem]
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
        "system": "CONTRAfold" if args.native_contrafold else "EternaFold", "repository": str(args.repo),
        "commit": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
        "binary": str(binary), "binary_sha256": sha256(binary),
        "parameters": "native contrafold.params.*" if args.native_contrafold else str(params),
        "parameters_sha256": ({name: sha256(args.repo / "src" / name) for name in
                               ["contrafold.params.complementary", "contrafold.params.noncomplementary", "contrafold.params.profile"]}
                              if args.native_contrafold else sha256(params)),
        "data": str(args.data), "data_sha256": sha256(args.data),
        "protocol": "raw-sequence end-to-end; official serial CPU binary and MEA decode; single thread; " + ("native CONTRAfold" if args.native_contrafold else "EternaFold parameters"),
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
