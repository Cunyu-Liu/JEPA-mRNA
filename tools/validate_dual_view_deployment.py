"""P1 deployment audit: original sequence-only inference with zero dual-view calls.

This is an engineering identity check, not an accuracy, convergence, or pilot result.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from rnajepa.decision_head import FlatDecisionHead
from rnajepa.dual_view import (DualViewJEPA, StructureTokenEncoder,
                               build_structure_example, make_view_masks)
from rnajepa.harness import valid_pair_mask
from rinalmo_preflight import load_encoder
from tools import xens2_eval


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="JSONL; first sequence is used")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("FATAL: CUDA unavailable; CPU fallback is forbidden")
    device = "cuda"
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    forbidden_keys = [key for key in checkpoint if "dual" in key.lower() or "structure" in key.lower()]
    if forbidden_keys:
        raise AssertionError(f"deployment checkpoint contains training-only keys: {forbidden_keys}")
    if set(checkpoint["model"]) != {"encoder", "head"}:
        raise AssertionError("deployment model must contain exactly the original encoder and head")
    with Path(args.data).open() as handle:
        sequence = json.loads(next(handle))["seq"]
    encoder, report = load_encoder(device="cpu", dtype=torch.float32)
    encoder.load_state_dict(checkpoint["model"]["encoder"], strict=True)
    meta = checkpoint.get("meta", {})
    scorer = meta.get("scorer", "resnet2d")
    if scorer != "resnet2d":
        raise AssertionError("this audit is bound to the unchanged ResNet2D host")
    head = FlatDecisionHead(d_model=int(report["d_model"]), d_z=128, hidden=64,
                            scorer=scorer, chunk_size=0,
                            pair_latent_dim=int(meta.get("pair_latent_dim", 64)))
    head.load_state_dict(checkpoint["model"]["head"], strict=True)
    encoder = encoder.to(device).eval()
    head = head.to(device).eval()
    signature = inspect.signature(xens2_eval.plana_scores).parameters
    if any(name in signature for name in ("gts", "structure", "dual_view", "pairs", "mask")):
        raise AssertionError("sequence-only deployment signature changed")

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
        first = xens2_eval.plana_scores(encoder, head, sequence, device)
        second = xens2_eval.plana_scores(encoder, head, sequence, device)
    if any(calls.values()):
        raise AssertionError(f"deployment invoked a training-only structure path: {calls}")
    if first.shape != (len(sequence), len(sequence)) or not torch.equal(first, second):
        raise AssertionError("sequence-only deterministic inference mismatch")
    legal = torch.as_tensor(valid_pair_mask(sequence), device=device)
    if not torch.isfinite(first[legal]).all():
        raise AssertionError("non-finite legal-pair deployment scores")
    if "torch_geometric" in sys.modules:
        raise AssertionError("torch-geometric was imported unexpectedly")
    result = {
        "status": "PASS_SEQUENCE_ONLY_DEPLOYMENT",
        "scientific_claim": "none; engineering deployment audit only",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_model_keys": sorted(checkpoint["model"]),
        "dual_view_state_present": False,
        "sequence_length": len(sequence),
        "deterministic_repeat_equal": True,
        "legal_scores_finite": True,
        "training_only_call_counts": calls,
        "torch_geometric_imported": False,
        "gpu": torch.cuda.get_device_name(0),
        "peak_inference_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "source_sha256": {
            "tools/xens2_eval.py": sha256(REPO / "tools/xens2_eval.py"),
            "src/rnajepa/decision_head.py": sha256(REPO / "src/rnajepa/decision_head.py"),
            "src/rnajepa/harness.py": sha256(REPO / "src/rnajepa/harness.py"),
            "tools/structrfm_decode_round48.py": sha256(REPO / "tools/structrfm_decode_round48.py"),
            "tools/validate_dual_view_deployment.py": sha256(Path(__file__).resolve()),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
