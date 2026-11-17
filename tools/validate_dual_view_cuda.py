"""Real-CUDA engineering gate for the training-only dual-view module.

This is not a scientific pilot and reports no accuracy or convergence claim.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from rnajepa.dual_view import (DualViewJEPA, ViewMasks, build_structure_example,
                               collate_structure)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sequence_for_pairs(length, pairs):
    sequence = ["A"] * length
    bases = (("A", "U"), ("G", "C"), ("C", "G"), ("U", "A"), ("G", "U"), ("U", "G"))
    for index, (i, j) in enumerate(pairs):
        sequence[i], sequence[j] = bases[index % len(bases)]
    return "".join(sequence)


def make_example(length, pairs=()):
    return build_structure_example(sequence_for_pairs(length, pairs), pairs)


def masks_for(items, max_length):
    _features, valid = collate_structure(items)
    token_mask = torch.zeros_like(valid)
    sequence_mask = torch.zeros(len(items), max_length, dtype=torch.bool)
    structure_mask = torch.zeros_like(sequence_mask)
    modes = []
    for batch_index, item in enumerate(items):
        if batch_index == 0:  # whole structure, including the empty-helix case
            token_mask[batch_index] = valid[batch_index]
            structure_mask[batch_index, :item.length] = True
            modes.append("whole_structure")
        elif batch_index in (1, 3) and item.helices:  # one partial helix
            for token_index, helix_index in enumerate(item.token_helix_index):
                if helix_index == 0:
                    token_mask[batch_index, token_index] = True
            for i, j in item.helices[0]:
                structure_mask[batch_index, i] = True
                structure_mask[batch_index, j] = True
            modes.append("partial_helix")
        else:
            sequence_mask[batch_index, 2:8] = True
            modes.append("sequence_span")
    return ViewMasks(sequence_mask, token_mask, structure_mask, valid, modes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("FATAL: CUDA unavailable; CPU fallback is forbidden")
    device = torch.device("cuda:0")
    torch.manual_seed(20261116)
    items = [
        make_example(24),
        make_example(28, ((0, 23), (1, 22))),
        make_example(30, ((0, 25), (1, 24), (6, 15), (7, 14))),
        make_example(26, ((0, 21), (1, 20), (5, 12), (6, 11))),
    ]
    lengths = [item.length for item in items]
    max_length = max(lengths)
    report = {
        "status": "RUNNING",
        "scientific_claim": "none; synthetic CUDA engineering gate only",
        "gpu": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "source_sha256": {
            "src/rnajepa/dual_view.py": sha256(REPO / "src/rnajepa/dual_view.py"),
            "tools/train_plan_a.py": sha256(REPO / "tools/train_plan_a.py"),
            "tests/test_dual_view.py": sha256(REPO / "tests/test_dual_view.py"),
            "tools/validate_dual_view_cuda.py": sha256(Path(__file__).resolve()),
        },
        "arms": {},
    }
    for arm in ("H1", "H2", "H3", "H4", "H5", "H6"):
        torch.cuda.reset_peak_memory_stats()
        model = DualViewJEPA(
            input_dimension=32, dimension=64, queries=4, heads=4,
            structure_control="mlp" if arm == "H5" else "transformer",
            mean_structure=arm == "H6", frozen_structure=arm == "H4",
            sigreg_slices=64).to(device)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
        losses = []
        for step in range(3):
            current = items[1:] + items[:1] if arm == "H3" else items
            masks = masks_for(current, max_length)
            hidden = torch.randn(len(items), max_length, 32, device=device)
            output = model(hidden, lengths, current, masks,
                           sigreg_weight=0.0 if arm == "H1" else 0.7)
            if not torch.isfinite(output.loss):
                raise RuntimeError(f"{arm}: non-finite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            missing = [name for name, parameter in model.named_parameters()
                       if parameter.requires_grad and parameter.grad is None]
            if missing:
                raise RuntimeError(f"{arm}: parameters without gradients: {missing}")
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise RuntimeError(f"{arm}: non-finite gradient at step {step}")
            optimizer.step()
            losses.append(float(output.loss.detach()))
        report["arms"][arm] = {
            "forward_backward": "PASS",
            "losses": losses,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
            "mask_modes": masks.modes,
            "structure_prediction_shape": list(output.structure_prediction.shape),
            "masked_structure_tokens": int(output.masks.structure_token_mask.sum()),
            "structure_slots": model.structure_slots,
        }
        del model, optimizer, output, hidden
        torch.cuda.empty_cache()
    # Record profiler-supported FLOPs for the production-width H2 module.  This
    # is an engineering measurement, not a claim about full-host training FLOPs.
    profile_model = DualViewJEPA(input_dimension=32, dimension=256, queries=8, heads=8,
                                 sigreg_slices=1024).to(device)
    profile_masks = masks_for(items, max_length)
    profile_hidden = torch.randn(len(items), max_length, 32, device=device)
    torch.cuda.reset_peak_memory_stats()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 with_flops=True, record_shapes=True) as profiler:
        profile_output = profile_model(profile_hidden, lengths, items, profile_masks,
                                       sigreg_weight=0.7)
        profile_output.loss.backward()
    profiler_flops = sum(int(getattr(event, "flops", 0) or 0)
                         for event in profiler.key_averages())
    if profiler_flops <= 0:
        raise RuntimeError("profiler returned zero FLOPs")
    report["flops"] = {
        "arm": "H2",
        "dimension": 256,
        "structure_slots": profile_model.structure_slots,
        "profiler_reported_forward_backward_flops": profiler_flops,
        "scope": "torch.profiler supported aten operations for synthetic dual-view module only; not full RiNALMo host FLOPs",
        "profiler_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    del profile_model, profile_masks, profile_hidden, profile_output, profiler
    torch.cuda.empty_cache()
    if "torch_geometric" in sys.modules:
        raise RuntimeError("torch-geometric was imported unexpectedly")
    report["torch_geometric_imported"] = False
    report["status"] = "PASS_DUAL_VIEW_CUDA_FORWARD_BACKWARD"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
