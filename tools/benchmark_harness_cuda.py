import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import triton

from rnajepa.harness import negative_log_likelihood, valid_pair_mask


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timed_loss(scores, mask, pairs, use_cuda, repeats):
    os.environ["RN_CUDA_INSIDE"] = "1" if use_cuda else "0"
    durations = []
    losses = []
    for _ in range(repeats):
        scores.grad = None
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss = negative_log_likelihood(scores, mask, pairs)
        loss.backward()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        losses.append(float(loss.detach()))
    return losses, scores.grad.detach().clone(), durations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 384, 512])
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    if not torch.cuda.is_available() or args.repeats < 2:
        raise RuntimeError("CUDA and at least two timing repeats are required")
    results = []
    for requested_length in args.lengths:
        record = min(records, key=lambda row: abs(len(row["seq"]) - requested_length))
        sequence = record["seq"].upper().replace("T", "U")
        mask_np = valid_pair_mask(sequence)
        rng = np.random.default_rng(len(sequence))
        scores_np = np.where(mask_np, rng.normal(0.0, 1.5, mask_np.shape), -1.0e4)
        mask = torch.from_numpy(mask_np).cuda()
        scores = torch.tensor(scores_np, dtype=torch.float32, device="cuda", requires_grad=True)
        pairs = [tuple(pair) for pair in record.get("pairs", []) if mask_np[tuple(pair)]]
        timed_loss(scores, mask, pairs, False, 2)
        timed_loss(scores, mask, pairs, True, 2)
        cpu_losses, cpu_grad, cpu_times = timed_loss(scores, mask, pairs, False, args.repeats)
        gpu_losses, gpu_grad, gpu_times = timed_loss(scores, mask, pairs, True, args.repeats)
        loss_error = max(abs(a - b) for a, b in zip(cpu_losses, gpu_losses))
        grad_error = float((cpu_grad - gpu_grad).abs().max())
        if loss_error > 3.0e-5 or grad_error > 3.0e-5:
            raise RuntimeError(f"CUDA equivalence failed at L={len(sequence)}: {loss_error}, {grad_error}")
        result = {
            "requested_length": requested_length,
            "length": len(sequence),
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "n_pairs": len(pairs),
            "repeats": args.repeats,
            "cpu_loss": cpu_losses,
            "cuda_loss": gpu_losses,
            "max_abs_loss_error": loss_error,
            "max_abs_gradient_error": grad_error,
            "cpu_seconds": cpu_times,
            "cuda_seconds": gpu_times,
            "cpu_median_seconds": statistics.median(cpu_times),
            "cuda_median_seconds": statistics.median(gpu_times),
            "speedup": statistics.median(cpu_times) / statistics.median(gpu_times),
            "cpu_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {
        "status": "PASS_EQUIVALENCE_AND_TIMING",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "data": args.data,
        "harness_sha256": sha256(Path(__file__).parents[1] / "src/rnajepa/harness.py"),
        "harness_cuda_sha256": sha256(Path(__file__).parents[1] / "src/rnajepa/harness_cuda.py"),
        "results": results,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)


if __name__ == "__main__":
    main()
