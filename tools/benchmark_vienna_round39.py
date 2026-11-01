#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

rna_site = os.environ.get("RNA_SITE_PACKAGES")
if rna_site:
    sys.path.insert(0, rna_site)
    import RNA
    sys.path.remove(rna_site)

from benchmark_eternafold_round38 import EXPECTED_SPLIT_COUNTS, EXPECTED_SPLIT_SHA256, canonical, score
from eval.ss.run_baselines import nussinov_turner, vienna_structure
from rnajepa.distill import ThermodynamicTeacher


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
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
    rna = ThermodynamicTeacher._import_rna()
    systems = {
        "vienna_mfe": lambda seq: vienna_structure(seq, "vienna_mfe", rna),
        "vienna_centroid": lambda seq: vienna_structure(seq, "vienna_centroid", rna),
        "vienna_mea": lambda seq: vienna_structure(seq, "vienna_mea", rna),
        "nussinov_turner": nussinov_turner,
    }
    results = {}
    for system, predictor in systems.items():
        repeats = []
        reference = None
        for repeat in range(args.repeats):
            tic = time.perf_counter()
            predictions = [{"name": row["name"], "pairs": predictor(row["seq"])} for row in rows]
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
        results[system] = {"repeats": repeats, "per_sequence": per_sequence}
    payload = {"schema_version": 1, "status": "SMOKE_ONLY" if args.smoke else "PASS",
               "data": str(args.data), "data_sha256": sha256(args.data),
               "device": "CPU native ViennaRNA and project Nussinov-Turner prior",
               "vienna_version": rna.__version__, "leakage": "not_sequence_fitted",
               "n_sequences": len(rows), "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"VIENNA_PASS out={args.out}")


if __name__ == "__main__":
    main()
