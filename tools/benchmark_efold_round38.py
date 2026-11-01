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


EXPECTED_SPLIT_COUNTS = {
    "bprna_ts0": 1288, "bprna_new": 5388, "archiveii_embok_clean": 2544,
    "ref_pdb_ts1": 63, "ref_pdb_ts2": 39, "ref_pdb_ts3": 19,
    "ref_pdb_ts_hard": 28, "testsetb": 428,
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


def decode(model, rows, reset_state=None):
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
        if reset_state is not None:
            model.load_state_dict(reset_state)
        prediction = model(item)["structure"]
        output = []
        for index, row in enumerate(rows):
            structure = _predict_structure_postprocess(prediction[index:index + 1], sequences[index:index + 1, :len(row["seq"])])
            pairs = (np.stack(np.where(np.triu(structure) == 1))).T.tolist()
            output.append({"name": row["name"], "pairs": pairs})
        return output


def _predict_structure_postprocess(scores, sequence):
    try:
        from efold.core.postprocess import Postprocess
    except ImportError:
        from efold.core.postprocess import postprocess_new_nc
        one_hot = torch.zeros((5, 4), device=scores.device)
        one_hot[1:] = torch.eye(4, device=scores.device)
        embedded = one_hot[sequence].type(torch.long)
        return postprocess_new_nc(scores, embedded, 0.01, 0.1, 100, 1.6, True, 1.5).detach().cpu().numpy().round().astype(int)[0]
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
        assert args.data.stem in EXPECTED_SPLIT_COUNTS, f"unregistered split: {args.data.stem}"
        assert len(rows) == EXPECTED_SPLIT_COUNTS[args.data.stem]
        assert sha256(args.data) == EXPECTED_SPLIT_SHA256[args.data.stem]
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
    legacy_native = args.repo.name == "efold012_wheel"
    reset_state = {key: value.detach().clone() for key, value in model.state_dict().items()} if legacy_native else None
    if legacy_native:
        assert args.batch_size == 1
        model.train().to("cuda")
    else:
        model.eval().to("cuda")
    assert next(model.parameters()).is_cuda
    with torch.inference_mode():
        decode(model, batches[0][:1], reset_state=reset_state)
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
            predictions.extend(decode(model, group, reset_state=reset_state))
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
    efold_version = "0.1.2" if args.repo.name == "efold012_wheel" else "0.1.3"
    protocol = ("raw-sequence end-to-end; exact-length groups; official 0.1.2 model and postprocess_new_nc"
                if efold_version == "0.1.2" else
                "raw-sequence end-to-end; exact-length groups; official 0.1.3 model, constraints, UFold postprocess and Hungarian decode")
    result = {
        "schema_version": 1,
        "status": "SMOKE_ONLY" if args.smoke else "PASS",
        "system": "eFold",
        "efold_version": efold_version,
        "repository": str(args.repo),
        "commit": (subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
                   if (args.repo / ".git").exists() else f"PyPI efold=={efold_version}"),
        "weights": str(weights), "weights_sha256": sha256(weights),
        "data": str(args.data), "data_sha256": sha256(args.data),
        "protocol": protocol,
        "native_protocol_note": ("0.1.2 native API behavior: train mode with per-sequence initial-state reset; identity verification only, not a speed comparison"
                                 if legacy_native else "0.1.3 official eval mode with a single model load"),
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
