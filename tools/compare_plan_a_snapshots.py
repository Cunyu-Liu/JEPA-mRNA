#!/usr/bin/env python
"""Tensor-level comparison for engineering identity, without computing metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def compare(left, right, path, mismatches, checked):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        checked["tensors"] += 1
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            detail = {"path": path, "shape_left": list(left.shape), "shape_right": list(right.shape),
                      "dtype_left": str(left.dtype), "dtype_right": str(right.dtype)}
            if left.shape == right.shape and left.dtype == right.dtype and left.is_floating_point():
                detail["max_abs"] = float((left - right).abs().max())
            mismatches.append(detail)
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            mismatches.append({"path": path, "keys_only_left": sorted(set(left) - set(right)),
                               "keys_only_right": sorted(set(right) - set(left))})
        for key in sorted(set(left) & set(right), key=str):
            compare(left[key], right[key], f"{path}.{key}", mismatches, checked)
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            mismatches.append({"path": path, "length_left": len(left), "length_right": len(right)})
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f"{path}[{index}]", mismatches, checked)
        return
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        checked["arrays"] += 1
        if not np.array_equal(left, right):
            mismatches.append({"path": path, "kind": "numpy_array"})
        return
    checked["scalars"] += 1
    try:
        equal = bool(left == right)
    except Exception:
        equal = False
    if not equal:
        mismatches.append({"path": path, "left": repr(left)[:300], "right": repr(right)[:300]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-state", action="store_true")
    args = parser.parse_args()
    left = torch.load(args.left, map_location="cpu", weights_only=False, mmap=True)
    right = torch.load(args.right, map_location="cpu", weights_only=False, mmap=True)
    mismatches = []
    checked = {"tensors": 0, "arrays": 0, "scalars": 0}
    if args.training_state:
        keys = ["step", "head", "encoder", "optimizer", "scheduler", "n_unfrozen",
                "stream_state", "dual_view", "dual_view_sigreg_step"]
        for key in keys:
            if key not in left or key not in right:
                mismatches.append({"path": key, "present_left": key in left, "present_right": key in right})
            else:
                compare(left[key], right[key], key, mismatches, checked)
    else:
        compare(left["model"], right["model"], "model", mismatches, checked)
    report = {
        "status": "PASS_TENSOR_IDENTITY" if not mismatches else "FAIL_TENSOR_IDENTITY",
        "left": str(Path(args.left).resolve()),
        "right": str(Path(args.right).resolve()),
        "scope": "training_state" if args.training_state else "deployment_model",
        "checked": checked,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:30],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
