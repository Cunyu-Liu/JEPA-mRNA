#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from benchmark_eternafold_round38 import EXPECTED_SPLIT_COUNTS, EXPECTED_SPLIT_SHA256, canonical, score


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from clean_speed_round49 import add_clean_speed_args, validate_clean_speed_rows

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev/third_party/BPfold"))
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--clean-audit", type=Path)
    add_clean_speed_args(parser)
    args = parser.parse_args()
    if not args.smoke and not args.clean_speed_protocol:
        assert args.repeats == 3 and args.limit is None
        assert args.data.stem in EXPECTED_SPLIT_COUNTS
    rows = [json.loads(line) for line in args.data.open()]
    if args.limit is not None:
        rows = rows[:args.limit]
    clean_speed = validate_clean_speed_rows(rows, args)
    if not args.smoke and not clean_speed:
        if args.clean_audit:
            audit = json.loads(args.clean_audit.read_text())
            assert audit["status"] == "PASS" and audit["split_sha256"] == EXPECTED_SPLIT_SHA256[args.data.stem]
            assert audit["clean_sha256"] == sha256(args.data) and len(rows) == audit["clean_n"]
            assert audit["raw_n"] == EXPECTED_SPLIT_COUNTS[args.data.stem]
            assert audit["raw_n"] == audit["clean_n"] + audit["removed_exact"] + audit["removed_homology_only"]
        else:
            assert len(rows) == EXPECTED_SPLIT_COUNTS[args.data.stem]
            assert sha256(args.data) == EXPECTED_SPLIT_SHA256[args.data.stem]
    import sys
    import torch
    assert torch.cuda.is_available(), "CUDA_REQUIRED"
    sys.path.insert(0, str(args.repo / "src"))
    from BPfold.predict import BPfold_predict
    predictor = BPfold_predict(str(args.checkpoints))
    assert predictor.device == "cuda", "CPU_FALLBACK_FORBIDDEN"
    assert all(parameter.is_cuda for model in predictor.models for parameter in model.parameters())
    repeats = []
    reference = None
    for repeat in range(args.repeats):
        raw = []
        timed_rows = rows
        if clean_speed:
            raw = list(predictor.predict(input_seqs=[rows[0]["seq"]], batch_size=1, num_workers=1))
            timed_rows = rows[1:]
        tic = time.perf_counter()
        for index, result in enumerate(predictor.predict(input_seqs=[row["seq"] for row in timed_rows], batch_size=1, num_workers=1), start=len(raw) + 1):
            raw.append(result)
            if index % 250 == 0 or index == len(rows):
                print(f"BPFOLD_PROGRESS repeat={repeat} predicted={index}/{len(rows)}", flush=True)
        predictions = []
        for row, result in zip(rows, raw):
            connects = result["connects"]
            pairs = [[index, partner - 1] for index, partner in enumerate(connects) if partner > index + 1]
            predictions.append({"name": row["name"], "pairs": pairs, "CI": result["CI"]})
        elapsed = time.perf_counter() - tic
        metrics, per_sequence = score(predictions, rows)
        identity = [[p["name"], p["pairs"]] for p in predictions]
        if reference is None:
            reference = identity
        else:
            assert identity == reference
        repeats.append({"repeat": repeat, "elapsed_seconds": elapsed,
                        "n_timed": len(timed_rows) if clean_speed else len(rows),
                        "sequences_per_second": len(timed_rows) / elapsed if clean_speed else len(rows) / elapsed,
                        "prediction_sha256": hashlib.sha256(canonical(identity).encode()).hexdigest(),
                        "metrics": metrics})
    checkpoint_files = sorted(args.checkpoints.glob("*.pth"))
    assert len(checkpoint_files) == 6
    payload = {"schema_version": 1, "status": "SMOKE_ONLY" if args.smoke else "PASS",
               "system": "BPfold", "data": str(args.data), "data_sha256": sha256(args.data),
               "repository_commit": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
               "checkpoint_sha256": {path.name: sha256(path) for path in checkpoint_files},
               "device": "cuda", "gpu_name": torch.cuda.get_device_name(0),
               "n_sequences": len(rows), "batch_size": 1, "cached_embeddings": False,
               "warmup_sequences_per_repeat": 1 if clean_speed else 0,
               "clean_speed_protocol": clean_speed, "repeats": repeats,
               "per_sequence": per_sequence}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"BPFOLD_PASS out={args.out}")


if __name__ == "__main__":
    main()
