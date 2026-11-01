#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from efold.api.run import _predict_structure
from efold.core import batch as efold_batch
from efold.core.embeddings import sequence_to_int
from efold.models import create_model


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def decode(model, rows):
    sequences = torch.stack([
        torch.nn.functional.pad(sequence_to_int(row["seq"]), (0, max(map(lambda x: len(x["seq"]), rows)) - len(row["seq"])))
        for row in rows
    ])
    item = efold_batch.Batch(
        sequence=sequences,
        reference=[row["name"] for row in rows],
        length=[len(row["seq"]) for row in rows],
        L=max(len(row["seq"]) for row in rows),
        use_error=False,
        batch_size=len(rows),
        data_types=["sequence"],
        dt_count={"sequence": len(rows)},
    ).to("cuda")
    with torch.inference_mode():
        prediction = model(item)["structure"]
        output = []
        for index, row in enumerate(rows):
            structure = _predict_structure_postprocess(prediction[index:index + 1], sequences[index:index + 1, :len(row["seq"])])
            pairs = (np.stack(np.where(np.triu(structure) == 1))).T.tolist()
            output.append({"name": row["name"], "pairs": pairs})
        return output


def _predict_structure_postprocess(scores, sequence):
    from efold.core.postprocess import Postprocess
    return Postprocess().run(scores.to("cpu"), sequence.to("cpu")).numpy().round()[0]


def score(predictions, rows):
    by_name = {prediction["name"]: prediction for prediction in predictions}
    assert len(by_name) == len(predictions) == len(rows)
    predictions = [by_name.pop(row["name"]) for row in rows]
    assert not by_name
    tp = fp = fn = 0
    per_sequence = []
    for prediction, row in zip(predictions, rows):
        pred = {tuple(pair) for pair in prediction["pairs"]}
        true = {tuple(pair) for pair in row["pairs"]}
        assert all(0 <= i < j < len(row["seq"]) for i, j in pred)
        assert len({i for pair in pred for i in pair}) == len(pred) * 2
        a, b, c = len(pred & true), len(pred - true), len(true - pred)
        tp, fp, fn = tp + a, fp + b, fn + c
        f1 = 2 * a / (2 * a + b + c) if 2 * a + b + c else 1.0
        per_sequence.append({"name": row["name"], "tp": a, "fp": b, "fn": c, "f1": f1,
                             "pred_pairs": prediction["pairs"]})
    return {"n_sequences": len(rows), "tp": tp, "fp": fp, "fn": fn,
            "micro_f1": 2 * tp / (2 * tp + fp + fn),
            "macro_f1": float(np.mean([row["f1"] for row in per_sequence]))}, per_sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev/third_party/eFold"))
    parser.add_argument("--data", type=Path, default=Path("/root/rna_jepa_data/ss_data/jsonl/bprna_ts0.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=[1, 2, 4], required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert args.repeats >= 1
    if not args.smoke:
        assert args.repeats == 3 and args.limit is None
    assert torch.cuda.is_available(), "CUDA_REQUIRED"
    torch.cuda.set_device(0)
    assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4090"
    rows = [json.loads(line) for line in args.data.open()]
    if args.limit is not None:
        rows = rows[:args.limit]
    assert rows and len({row["name"] for row in rows}) == len(rows)
    if not args.smoke:
        assert len(rows) == 1288
    groups = defaultdict(list)
    for row in rows:
        groups[len(row["seq"])].append(row)
    batches = [group[start:start + args.batch_size]
               for _, group in sorted(groups.items())
               for start in range(0, len(group), args.batch_size)]
    weights = args.repo / "efold/resources/efold_weights.pt"
    assert weights.is_file()
    model = create_model(model="efold", ntoken=5, d_model=64, c_z=32, d_cnn=64,
                         num_blocks=4, no_recycles=0, dropout=0, lr=3e-4,
                         weight_decay=0, gamma=0.995)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys, incompatible
    model.eval().to("cuda")
    assert next(model.parameters()).is_cuda
    with torch.inference_mode():
        decode(model, batches[0][:1])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    repeats = []
    reference_pairs = None
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for repeat in range(args.repeats):
        predictions = []
        torch.cuda.synchronize()
        tic = time.perf_counter()
        for group in batches:
            predictions.extend(decode(model, group))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - tic
        metrics, per_sequence = score(predictions, rows)
        digest = hashlib.sha256(canonical(predictions).encode()).hexdigest()
        pair_identity = [[p["name"], p["pairs"]] for p in predictions]
        if reference_pairs is None:
            reference_pairs = pair_identity
        else:
            assert pair_identity == reference_pairs, "REPEAT_PREDICTION_MISMATCH"
        repeats.append({"repeat": repeat, "elapsed_seconds": elapsed,
                        "sequences_per_second": len(rows) / elapsed,
                        "prediction_sha256": digest, "metrics": metrics})
        print(json.dumps(repeats[-1]), flush=True)
    result = {
        "schema_version": 1,
        "status": "SMOKE_ONLY" if args.smoke else "PASS",
        "system": "eFold",
        "repository": str(args.repo),
        "commit": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
        "weights": str(weights), "weights_sha256": sha256(weights),
        "data": str(args.data), "data_sha256": sha256(args.data),
        "protocol": "raw-sequence end-to-end; exact-length groups; official model, constraints, UFold postprocess and Hungarian decode",
        "device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "torch": torch.__version__, "batch_size": args.batch_size, "n_sequences": len(rows),
        "started": started, "repeats": repeats,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
        "per_sequence": per_sequence,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.out)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
