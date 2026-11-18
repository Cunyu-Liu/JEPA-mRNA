#!/usr/bin/env python3
"""Build the single model-only initialization shared by train-on and train-off."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from rnajepa.decision_head import FlatDecisionHead  # noqa: E402
from rnajepa.prior_alignment import sha256_file  # noqa: E402
from rinalmo_preflight import load_encoder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite shared initialization: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    encoder, report = load_encoder(device="cpu", dtype=torch.float32)
    head = FlatDecisionHead(d_model=int(report["d_model"]), d_z=128, hidden=64,
                            scorer="resnet2d", chunk_size=0)
    payload = {
        "format": "prior_alignment_shared_init_v1",
        "seed": int(args.seed),
        "source": "RiNALMo-giga plus newly initialized Plan-A head; no Plan-A training history",
        "model": {"encoder": encoder.state_dict(), "head": head.state_dict()},
        "head_config": {"d_z": 128, "hidden": 64, "scorer": "resnet2d", "chunk_size": 0},
        "backbone_report": report,
        "optimizer": None,
    }
    temporary = args.out.with_suffix(args.out.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, args.out)
    finally:
        if temporary.exists():
            temporary.unlink()
    receipt = {
        "format": "prior_alignment_shared_init_receipt_v1",
        "path": str(args.out), "sha256": sha256_file(args.out),
        "size_bytes": args.out.stat().st_size, "seed": args.seed,
        "source": payload["source"], "head_config": payload["head_config"],
        "backbone": report.get("backbone", report.get("weights")),
    }
    args.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
