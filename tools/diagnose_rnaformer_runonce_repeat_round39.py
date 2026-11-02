#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from speed_end_to_end_4090 import load_models, read_records, run_once, sha256_file


class CaptureRNAformer(torch.nn.Module):
    def __init__(self, model, records, batch_size):
        super().__init__()
        self.model = model
        self.records = records
        self.batch_size = batch_size
        self.raw = [[None for _ in records] for _ in range(3)]
        self.repeat = 0
        self.bucket_offsets = {}

    def forward(self, ids, src_len, pdb_sample):
        output = self.model(ids, src_len, pdb_sample)
        length = int(ids.shape[1])
        start = self.bucket_offsets.get(length, 0)
        indices = [index for index, row in enumerate(self.records) if len(row["seq"]) == length]
        selected = indices[start:start + self.batch_size]
        if len(selected) != len(ids):
            raise RuntimeError(f"capture batch mismatch length={length} start={start} batch={len(ids)} selected={len(selected)}")
        self.bucket_offsets[length] = start + len(selected)
        raw = output[0].detach().float().cpu().numpy()
        for local, index in enumerate(selected):
            self.raw[self.repeat][index] = raw[local].copy()
        return output

    def begin_repeat(self, repeat):
        self.repeat = repeat
        self.bucket_offsets = {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--disable-rotary-cache", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    if args.batch_size < 1 or args.max_sequences < 0:
        raise ValueError("batch-size must be positive and max-sequences nonnegative")
    device = torch.device("cuda:0")
    records = read_records(args.data, args.max_sequences, max_length=None)
    if not records:
        raise RuntimeError("diagnostic requires at least one sequence")
    if len({row["name"] for row in records}) != len(records):
        raise RuntimeError("record names must be unique")
    models = load_models("rnaformer", device, args.weights, args.config)
    original = models["model"]
    rotary_modules = [module for module in original.modules() if type(module).__name__ == "RotaryEmbedding"]
    if args.disable_rotary_cache:
        if not rotary_modules:
            raise RuntimeError("no RotaryEmbedding modules found")
        for module in rotary_modules:
            module.cache_if_possible = False
    capture = CaptureRNAformer(original, records, args.batch_size)
    models = dict(models, model=capture)
    runs = []
    for repeat in range(3):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        capture.begin_repeat(repeat)
        result = run_once(models, records, args.batch_size, device)
        result["predictions"] = [sorted(map(list, pairs)) for pairs in result["predictions"]]
        result["peak_memory_mib"] = torch.cuda.max_memory_allocated() / 2**20
        result["rotary_cache_disabled"] = args.disable_rotary_cache
        result["rotary_embedding_module_count"] = len(rotary_modules)
        runs.append(result)
        print(json.dumps({"repeat": repeat, "f1": result["strict_micro_f1"],
                          "tp": result["tp"], "fp": result["fp"], "fn": result["fn"],
                          "rotary_cache_disabled": args.disable_rotary_cache,
                          "predictions_sha256": result["predictions_sha256"]}), flush=True)
    if any(value is None for repeat in capture.raw for value in repeat):
        raise RuntimeError("not every sequence was captured in every repeat")
    sequence_results = []
    mismatch_arrays = {}
    for index, row in enumerate(records):
        raw = [capture.raw[repeat][index] for repeat in range(3)]
        hashes = [hashlib.sha256(value.tobytes()).hexdigest() for value in raw]
        max_abs_diff = [float(np.max(np.abs(raw[0] - raw[repeat]))) for repeat in (1, 2)]
        raw_equal = all(np.array_equal(raw[0], raw[repeat]) for repeat in (1, 2))
        predictions = [{tuple(pair) for pair in runs[repeat]["predictions"][index]} for repeat in range(3)]
        gt = set(row["gt"])
        counts = [{"tp": len(pred & gt), "fp": len(pred - gt), "fn": len(gt - pred)} for pred in predictions]
        predictions_equal = predictions[0] == predictions[1] == predictions[2]
        item = {"index": index, "name": row["name"], "length": len(row["seq"]),
                "raw_equal": raw_equal, "raw_sha256": hashes, "max_abs_diff_01_02": max_abs_diff,
                "predictions_equal": predictions_equal, "confusion": counts,
                "prediction_n": [len(pred) for pred in predictions]}
        if not raw_equal or not predictions_equal:
            item["predictions"] = [sorted(map(list, pred)) for pred in predictions]
            item["only_in"] = [{"repeat": repeat, "pairs": sorted(map(list, pred - set.union(*(predictions[:repeat] + predictions[repeat + 1:]))))}
                                for repeat, pred in enumerate(predictions)]
            for repeat, value in enumerate(raw):
                mismatch_arrays[f"sequence_{index}_repeat_{repeat}"] = value
        sequence_results.append(item)
    mismatches = [row for row in sequence_results if not row["raw_equal"] or not row["predictions_equal"]]
    payload = {"schema_version": 1, "status": "DIAGNOSTIC", "n": len(records),
               "protocol": {"production_entry": "speed_end_to_end_4090.run_once",
                            "weights": str(args.weights), "weights_sha256": sha256_file(args.weights),
                            "config": str(args.config), "config_sha256": sha256_file(args.config),
                            "data": str(args.data), "data_sha256": sha256_file(args.data),
                            "batch_size": args.batch_size, "max_sequences": args.max_sequences,
                            "repeats": 3, "empty_cache_before_each_repeat": True,
                            "single_model_instance_across_repeats": True,
                            "device": torch.cuda.get_device_name(device), "torch": torch.__version__,
                            "cuda": torch.version.cuda, "training_modules": sum(module.training for module in original.modules())},
               "raw_mismatch_count": sum(not row["raw_equal"] for row in sequence_results),
               "prediction_mismatch_count": sum(not row["predictions_equal"] for row in sequence_results),
               "run_summaries": [{key: value for key, value in run.items() if key != "predictions"} for run in runs],
               "mismatches": mismatches, "sequences": sequence_results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if mismatch_arrays:
        array_path = args.out.with_suffix(".npz")
        np.savez_compressed(array_path, **mismatch_arrays)
        payload["mismatch_arrays"] = str(array_path)
        payload["mismatch_array_keys"] = list(mismatch_arrays)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("n", "raw_mismatch_count", "prediction_mismatch_count", "mismatches")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
