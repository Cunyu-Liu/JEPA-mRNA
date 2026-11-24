#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from rnajepa.gec_context import GEC_RESIDUAL_FORMAT_V1, GECResidualV1
from rnajepa.prior_alignment import sha256_file, tensor_state_sha256
from tools.train_plan_a_clean_head_p1 import atomic_torch_save


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-init", required=True)
    parser.add_argument("--gec-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen GEC initialization: {output}")
    config = json.loads(Path(args.gec_config).read_text())
    if config["status"] != "FROZEN_FOR_ENGINEERING_SMOKE_ONLY":
        raise SystemExit("unexpected GEC implementation config status")
    initial = torch.load(args.shared_init, map_location="cpu", weights_only=False)
    if initial.get("format") != "prior_alignment_shared_init_v1":
        raise SystemExit("unexpected base shared initialization")
    if int(initial.get("seed", -1)) != args.seed:
        raise SystemExit("base initialization seed mismatch")
    torch.manual_seed(args.seed)
    residual = GECResidualV1(d_z=config["model"]["d_z"], hidden=config["model"]["residual_hidden"])
    payload = {
        "format": GEC_RESIDUAL_FORMAT_V1,
        "seed": args.seed,
        "d_z": config["model"]["d_z"],
        "hidden": config["model"]["residual_hidden"],
        "gec_config": str(Path(args.gec_config).resolve()),
        "gec_config_sha256": sha256_file(args.gec_config),
        "base_shared_init": str(Path(args.shared_init).resolve()),
        "base_shared_init_sha256": sha256_file(args.shared_init),
        "model": residual.state_dict(),
        "residual_tensor_sha256": tensor_state_sha256(residual.state_dict()),
        "parameter_count": sum(p.numel() for p in residual.parameters()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload, output)
    print(json.dumps({
        "path": str(output),
        "sha256": sha256_file(output),
        "residual_tensor_sha256": payload["residual_tensor_sha256"],
        "parameter_count": payload["parameter_count"],
        "config_sha256": hashlib.sha256(Path(args.gec_config).read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
