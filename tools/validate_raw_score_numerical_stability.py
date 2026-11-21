#!/usr/bin/env python
"""Independent high-log-partition validation for the v3 stable exact tangent."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from rnajepa.harness import inside_outside, valid_pair_mask
from rnajepa.harness_cuda import cuda_inside_outside, cuda_marginal_tangent
from rnajepa.structured_marginal import _cpu_exact_marginal_vjp, _exact_marginals


def case(length, scale, seed=11):
    sequence = "G" * (length // 2) + "C" * (length - length // 2)
    mask = torch.as_tensor(valid_pair_mask(sequence), dtype=torch.bool)
    base = torch.randn(length, length, generator=torch.Generator().manual_seed(seed),
                       dtype=torch.float64)
    scores = torch.triu(base, diagonal=1) * scale
    labels = torch.zeros(length, length, dtype=torch.float64)
    legal = torch.triu(mask, diagonal=1)
    indices = torch.nonzero(legal)
    labels[indices[::3, 0], indices[::3, 1]] = 1.0
    return scores, mask, labels, legal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"refusing to overwrite {args.out}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA validation requires a GPU")
    rows = []
    for length, scale in [(16, 100.0), (32, 60.0), (64, 30.0), (64, 100.0)]:
        scores, mask, labels, legal = case(length, scale)
        gpu_scores = scores.cuda()
        gpu_mask = mask.cuda()
        log_z, probability = cuda_inside_outside(gpu_scores, gpu_mask)
        cpu_log_z, cpu_probability = inside_outside(scores.numpy(), mask.numpy())
        p = probability[legal.cuda()]
        q = p.detach().requires_grad_(True)
        loss = torch.nn.functional.binary_cross_entropy(
            q.clamp(1e-7, 1.0 - 1e-7), labels[legal].cuda())
        legal_direction = torch.autograd.grad(loss, q)[0]
        direction = torch.zeros_like(gpu_scores)
        direction[legal.cuda()] = legal_direction
        gradient = cuda_marginal_tangent(gpu_scores, gpu_mask, direction)
        expected = torch.as_tensor(cpu_probability[legal.numpy()])
        relative = float((p.cpu() - expected).norm() / expected.norm().clamp_min(1e-30))
        row = {
            "kind": "high_partition_forward_and_tangent",
            "length": length, "scale": scale,
            "cuda_log_partition": float(log_z), "cpu_log_partition": float(cpu_log_z),
            "forward_probability_relative_error": relative,
            "forward_finite": bool(torch.isfinite(probability).all()),
            "tangent_finite": bool(torch.isfinite(gradient).all()),
            "tangent_nonfinite_count": int((~torch.isfinite(gradient)).sum()),
        }
        assert math.isfinite(float(log_z)) and row["forward_finite"] and row["tangent_finite"]
        assert relative < 1e-10
        rows.append(row)
    for scale in [60.0, 100.0]:
        scores, mask, labels, legal = case(12, scale)
        p = _exact_marginals(scores, mask)
        q = p[legal].detach().requires_grad_(True)
        loss = torch.nn.functional.binary_cross_entropy(
            q.clamp(1e-7, 1.0 - 1e-7), labels[legal])
        legal_direction = torch.autograd.grad(loss, q)[0]
        direction = torch.zeros_like(scores)
        direction[legal] = legal_direction
        expected = _cpu_exact_marginal_vjp(scores, mask, direction)
        actual = cuda_marginal_tangent(scores.cuda(), mask.cuda(), direction.cuda()).cpu()
        relative = float((actual - expected)[legal].norm()
                         / expected[legal].norm().clamp_min(1e-30))
        row = {
            "kind": "independent_cpu_dp_vjp",
            "length": 12, "scale": scale,
            "relative_error": relative,
            "absolute_l2_error": float((actual - expected)[legal].norm()),
            "reference_l2": float(expected[legal].norm()),
            "actual_l2": float(actual[legal].norm()),
            "finite": bool(torch.isfinite(actual).all()),
        }
        assert row["finite"] and relative < 1e-8
        rows.append(row)
    output = {
        "status": "PASS",
        "scope": "NUMERICAL_ENGINEERING_VALIDATION_ONLY_NO_SCIENTIFIC_RESULT",
        "dtype": "float64",
        "old_failure_regime": "direct parent exp accumulation overflow and clamp-hidden forward invalidity",
        "rows": rows,
        "max_forward_probability_relative_error": max(
            row["forward_probability_relative_error"] for row in rows
            if row["kind"] == "high_partition_forward_and_tangent"),
        "max_independent_dp_vjp_relative_error": max(
            row["relative_error"] for row in rows if row["kind"] == "independent_cpu_dp_vjp"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
