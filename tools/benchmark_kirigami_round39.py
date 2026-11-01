#!/usr/bin/env python3
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from benchmark_eternafold_round38 import EXPECTED_SPLIT_COUNTS, EXPECTED_SPLIT_SHA256, canonical, score


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_dot_pairs(dot):
    stacks = {opening: [] for opening in "([{<"}
    closing = {")": "(", "]": "[", "}": "{", ">": "<"}
    pairs = []
    for index, char in enumerate(dot):
        if char in stacks:
            stacks[char].append(index)
        elif char in closing:
            opening = closing[char]
            assert stacks[opening], (dot, index, char)
            pairs.append((stacks[opening].pop(), index))
        else:
            assert char == ".", (dot, index, char)
    assert all(not stack for stack in stacks.values()), dot
    return sorted(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev/third_party/kirigami"))
    parser.add_argument("--weights", type=Path, required=True)
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    if not rows:
        assert args.clean_audit is not None and not args.smoke
        payload = {"schema_version": 1, "status": "EMPTY_AFTER_DELEAK", "system": "kirigami",
                   "data": str(args.data), "data_sha256": sha256(args.data),
                   "clean_audit": str(args.clean_audit), "n_sequences": 0,
                   "f1_mean": None, "repeats": [], "per_sequence": []}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(f"EMPTY_AFTER_DELEAK model=kirigami out={args.out}")
        return
    sys.path.insert(0, str(args.repo))
    from kirigami.utils import _embed_fasta, mat2db, outer_concat
    model = torch.hub.load(str(args.repo), "kirigami", source="local", pretrained=True,
                           checkpoint_path=str(args.weights)).cuda().eval()
    assert next(model.parameters()).is_cuda
    repeats = []
    reference = None
    probability_reference = None
    probabilities = []
    for repeat in range(args.repeats):
        tic = time.perf_counter()
        predictions = []
        probabilities = []
        probability_hasher = hashlib.sha256()
        for row in rows:
            seq = row["seq"].upper().replace("T", "U")
            feat = outer_concat(_embed_fasta(seq)).cuda()
            with torch.no_grad():
                raw = model.model(feat)
                processed = model.post_proc(raw, feat)
                processed[processed < model.threshold.item()] = 0
            probability = raw.squeeze().float().cpu().numpy()
            assert probability.shape == (len(seq), len(seq))
            assert np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
            probability = np.triu(0.5 * (probability + probability.T), k=1).astype(np.float32)
            probabilities.append(probability)
            probability_hasher.update(probability.tobytes())
            dot = mat2db(processed)
            assert len(dot) == len(seq)
            predictions.append({"name": row["name"], "pairs": parse_dot_pairs(dot)})
        elapsed = time.perf_counter() - tic
        metrics, per_sequence = score(predictions, rows)
        identity = [[p["name"], p["pairs"]] for p in predictions]
        probability_hash = probability_hasher.hexdigest()
        if reference is None:
            reference = identity
            probability_reference = probability_hash
        else:
            assert identity == reference and probability_hash == probability_reference
        repeats.append({"repeat": repeat, "elapsed_seconds": elapsed,
                        "sequences_per_second": len(rows) / elapsed,
                        "prediction_sha256": hashlib.sha256(canonical(identity).encode()).hexdigest(),
                        "probability_sha256": probability_hash,
                        "metrics": metrics})
    probability_path = args.out.with_suffix(".probabilities.npz")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(probability_path, names=np.array([row["name"] for row in rows], dtype=object),
                        probabilities=np.array(probabilities, dtype=object))
    payload = {"schema_version": 1, "status": "SMOKE_ONLY" if args.smoke else "PASS",
               "system": "kirigami", "data": str(args.data), "data_sha256": sha256(args.data),
               "clean_audit": str(args.clean_audit) if args.clean_audit else None,
               "repository_commit": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
               "weights": str(args.weights), "weights_sha256": sha256(args.weights),
               "device": "cuda", "threshold": float(model.threshold.item()), "n_sequences": len(rows),
               "environment": {"torch": importlib.metadata.version("torch"),
                               "pytorch-lightning": importlib.metadata.version("pytorch-lightning"),
                               "python": sys.executable},
               "probabilities": str(probability_path), "probabilities_sha256": sha256(probability_path),
               "repeats": repeats, "per_sequence": per_sequence}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"KIRIGAMI_PASS out={args.out}")


if __name__ == "__main__":
    main()
