#!/usr/bin/env python
"""Validate sequence-only deployment from a model-only Plan-A snapshot."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from rnajepa.dual_view import (DualViewJEPA, StructureTokenEncoder,
                               build_structure_example, make_view_masks)  # noqa: E402
from rnajepa.harness import valid_pair_mask  # noqa: E402
from structrfm_decode_round48 import decode_scores  # noqa: E402
from xens2_eval import load_plana, plana_scores  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="/teams/HoulabData_1788502323/liucunyu/rna_jev_data/ss_data/jsonl/bprna_tr1c.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if set(state["model"]) != {"encoder", "head"}:
        raise AssertionError(f"unexpected deployment model groups: {set(state['model'])}")
    forbidden = [key for group in state["model"].values() for key in group
                 if any(token in key.lower() for token in ("dual_view", "structure_encoder", "predictor"))]
    if forbidden:
        raise AssertionError(f"training-only structure modules leaked into deployment: {forbidden[:5]}")
    with open(args.data, encoding="utf-8") as handle:
        record = json.loads(next(handle))
    sequence = record.get("sequence", record.get("seq"))
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("deployment input requires a non-empty sequence/seq field")
    signature = inspect.signature(plana_scores).parameters
    if any(name in signature for name in ("gts", "structure", "dual_view", "pairs", "mask")):
        raise AssertionError("sequence-only deployment signature changed")
    encoder, head = load_plana(args.checkpoint, "cuda")
    assert encoder.training is False and head.training is False
    calls = {name: 0 for name in (
        "DualViewJEPA.forward", "StructureTokenEncoder.forward",
        "build_structure_example", "make_view_masks")}

    def counted(name, function):
        def wrapped(*args, **kwargs):
            calls[name] += 1
            return function(*args, **kwargs)
        return wrapped

    torch.cuda.reset_peak_memory_stats()
    with patch.object(DualViewJEPA, "forward", counted("DualViewJEPA.forward", DualViewJEPA.forward)), \
         patch.object(StructureTokenEncoder, "forward", counted("StructureTokenEncoder.forward", StructureTokenEncoder.forward)), \
         patch("rnajepa.dual_view.build_structure_example", counted("build_structure_example", build_structure_example)), \
         patch("rnajepa.dual_view.make_view_masks", counted("make_view_masks", make_view_masks)):
        score_runs = [plana_scores(encoder, head, sequence, "cuda") for _ in range(3)]
    if any(calls.values()):
        raise AssertionError(f"deployment invoked training-only structure paths: {calls}")
    if not all(torch.equal(score_runs[0], current) for current in score_runs[1:]):
        raise AssertionError("three deterministic deployment forwards differ")
    if not torch.isfinite(score_runs[0][torch.isfinite(score_runs[0])]).all():
        raise AssertionError("non-finite legal deployment scores")
    mask = valid_pair_mask(sequence)
    pairs, crossing_rate, conflicts = decode_scores(score_runs[0], mask, "nussinov")
    if crossing_rate != 0 or conflicts != 0:
        raise AssertionError("exact nested decoder produced crossing/conflicting pairs")
    if "torch_geometric" in sys.modules:
        raise AssertionError("torch-geometric was imported unexpectedly")
    report = {
        "status": "PASS_SEQUENCE_ONLY_DEPLOYMENT",
        "scientific_claim": "none; engineering snapshot load and deterministic forward only",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_step": int(state["step"]),
        "model_groups": sorted(state["model"]),
        "contains_dual_view": False,
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "sequence_length": len(sequence),
        "ground_truth_used_for_inference": False,
        "metrics_computed": False,
        "forward_repetitions": 3,
        "bitwise_deterministic": True,
        "scores_sha256": sha_tensor(score_runs[0]),
        "decoder": "nussinov exact nested DP",
        "decoded_pair_count": len(pairs),
        "crossing_rate": crossing_rate,
        "conflicts": conflicts,
        "training_only_call_counts": calls,
        "torch_geometric_imported": False,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "peak_inference_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "source_sha256": {
            "tools/xens2_eval.py": sha256(REPO / "tools/xens2_eval.py"),
            "src/rnajepa/decision_head.py": sha256(REPO / "src/rnajepa/decision_head.py"),
            "src/rnajepa/harness.py": sha256(REPO / "src/rnajepa/harness.py"),
            "tools/structrfm_decode_round48.py": sha256(REPO / "tools/structrfm_decode_round48.py"),
            "tools/validate_plan_a_deployment.py": sha256(Path(__file__).resolve()),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
