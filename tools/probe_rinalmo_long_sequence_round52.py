import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import xens2_eval
from rinalmo_preflight import load_encoder
from rnajepa.decision_head import FlatDecisionHead
from rnajepa.harness import valid_pair_mask


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=(512, 1022, 1024, 1500, 1869, 3819))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA is required; refusing CPU fallback"
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder, report = load_encoder(device="cpu", dtype=torch.float32)
    head = FlatDecisionHead(d_model=1280, d_z=128, hidden=64, scorer="pairmlp",
                            chunk_size=0, pair_latent_dim=490, jepa_use_ema=False,
                            ripe_mode="r6")
    encoder.load_state_dict(checkpoint["model"]["encoder"], strict=True)
    missing, unexpected = head.load_state_dict(checkpoint["model"]["head"], strict=False)
    assert not missing and not unexpected and head.pair_jepa.teacher is None
    encoder = encoder.to(device).eval()
    head = head.to(device).eval()
    assert next(encoder.parameters()).is_cuda and next(head.parameters()).is_cuda
    rows = []
    for length in args.lengths:
        sequence = "".join(np.random.default_rng(42).choice(list("ACGU"), size=length))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        try:
            with torch.no_grad():
                scores = xens2_eval.plana_scores(encoder, head, sequence, device)
            assert scores.shape == (length, length)
            mask = torch.as_tensor(valid_pair_mask(sequence), device=device)
            assert bool(mask.any())
            assert bool(torch.isfinite(scores[mask]).all())
            rows.append({
                "length": length, "status": "PASS_FULL_LENGTH_CUDA",
                "input_length_preserved": True,
                "output_shape": list(scores.shape),
                "valid_pairs": int(mask.sum()),
                "score_sha256": hashlib.sha256(scores.cpu().numpy().tobytes()).hexdigest(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "wall_seconds_nonformal": time.time() - started,
            })
            print(json.dumps(rows[-1]), flush=True)
            del scores, mask
        except torch.cuda.OutOfMemoryError as error:
            rows.append({
                "length": length, "status": "OOM_STOP",
                "input_length_preserved": True,
                "error": str(error),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "wall_seconds_nonformal": time.time() - started,
            })
            print(json.dumps(rows[-1]), flush=True)
            break
        finally:
            torch.cuda.empty_cache()
    output = {
        "status": "PASS_LONG_SEQUENCE_ENGINEERING_PROBE" if len(rows) == len(args.lengths) else "PARTIAL_LONG_SEQUENCE_ENGINEERING_PROBE",
        "scientific_conclusion": False,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "backbone": report,
        "resolved_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "sequence_source": "deterministic synthetic ACGU, seed42; engineering only, not a biological benchmark",
        "rows": rows,
        "limitations": [
            "This tests full-length forward and memory, not long-range structure accuracy or DP quality.",
            "No truncation, filtering, CPU fallback, teacher forward, or ensemble is used.",
            "The final long-sequence gate still requires real structures, span metrics, and independent prediction equivalence.",
        ],
    }
    path = Path(args.output)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": output["status"], "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
