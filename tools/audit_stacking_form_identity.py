#!/usr/bin/env python
"""Preflight asset, parameter and analytical-FLOPs audit for stacking-form arms."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from rnajepa.decision_head import FlatDecisionHead
from rnajepa.stacking_closure import MODES


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite identity preflight")
    contract = json.loads(args.contract.read_text())
    train = contract["training"]
    checks = {
        "data_sha256": sha256_file(train["data"]) == train["data_sha256"],
        "shared_init_sha256": sha256_file(train["shared_init_template"].format(seed=0)) == train["shared_init_sha256_seed0"],
        "decoder_sha256": sha256_file(contract["evaluation"]["decoder_source"]) == contract["evaluation"]["decoder_source_sha256"],
        "external_closed": not contract["external_evaluation_allowed"] and not contract["strict_ood_allowed"],
    }
    arms = {}
    reference = None
    for mode in MODES:
        torch.manual_seed(0)
        head = FlatDecisionHead(d_model=1280, d_z=128, scorer=mode, chunk_size=0,
                                resnet_blocks=1, resnet_hidden=128)
        parameters = {name: list(value.shape) for name, value in head.named_parameters()}
        if reference is None:
            reference = parameters
        closure = head.resnet2d
        head_params = sum(value.numel() for value in head.parameters())
        closure_params = sum(value.numel() for value in closure.parameters())
        macs_l500_b4 = 4 * 500 * 500 * (6 * 128 * 128 + 128)
        arms[mode] = {
            "head_params": head_params, "closure_params": closure_params,
            "parameter_names_shapes_match": parameters == reference,
            "closure_macs_B4_L500": macs_l500_b4,
            "flops_definition": "analytical MACs for input/update/head linears only; excludes pair representation, encoder, GELU and products",
            "mode": closure.mode,
        }
    checks["parameter_identity"] = all(arm["parameter_names_shapes_match"] for arm in arms.values())
    checks["parameter_counts_match"] = len({arm["head_params"] for arm in arms.values()}) == 1
    checks["analytical_flops_match"] = len({arm["closure_macs_B4_L500"] for arm in arms.values()}) == 1
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "contract_sha256": sha256_file(args.contract), "arms": arms,
              "code_head": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
