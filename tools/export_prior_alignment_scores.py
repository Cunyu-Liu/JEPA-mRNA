#!/usr/bin/env python3
"""Export on/off raw Plan-A scores from one prior-alignment training checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from pair_raw_export import resilient_plana_scores, sha256  # noqa: E402
from xens2_eval import load_plana  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-trajectory", choices=["train_on", "train_off"], required=True)
    parser.add_argument("--contract-mode", choices=["smoke", "pilot", "full"], required=True)
    parser.add_argument("--data", type=Path, default=Path("/root/rna_jepa_data/ss_data/jsonl/bprna_vl0.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--head-chunk-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="engineering smoke only")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.data.name != "bprna_vl0.jsonl":
        raise ValueError("the 2x2 development export is restricted to VL0")
    if args.limit and args.contract_mode != "smoke":
        raise ValueError("limit is permitted only for smoke exports")
    checkpoint_sha = sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    identity = (checkpoint.get("meta") or {}).get("prior_alignment") or {}
    expected = {"training_trajectory": args.train_trajectory, "mode": args.contract_mode}
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError(f"checkpoint identity mismatch for {key}: {identity.get(key)!r} != {value!r}")
    if not identity.get("shared_init_sha256") or not identity.get("contract_sha256"):
        raise ValueError("checkpoint lacks shared-init or contract identity")
    checkpoint_seed = checkpoint.get("meta", {}).get("seed")
    checkpoint_step = int(checkpoint.get("step", -1))
    del checkpoint
    records = []
    with args.data.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            seq = str(row["seq"]).upper().replace("T", "U")
            name = str(row.get("name", row.get("id", f"row:{index + 1}")))
            sequence_sha = hashlib.sha256(seq.encode()).hexdigest()
            records.append({"name": name, "seq": seq, "gt_pairs": row.get("pairs", []),
                            "sequence_sha256": sequence_sha,
                            "selection_sha256": hashlib.sha256(f"bprna_vl0|{name}|{sequence_sha}".encode()).hexdigest()})
    if args.limit:
        records = records[:args.limit]
    split_dir = args.out / "bprna_vl0"
    split_dir.mkdir(parents=True, exist_ok=True)
    encoder, head = load_plana(str(args.checkpoint), args.device)
    head.chunk_size = args.head_chunk_size
    info = {"schema_version": 1, "dataset_id": "bprna_vl0", "n_selected": len(records),
            "n_exported": 0, "n_resumed": 0, "n_skipped_length": 0,
            "train_trajectory": args.train_trajectory, "contract_mode": args.contract_mode,
            "checkpoint": {"path": str(args.checkpoint), "sha256": checkpoint_sha},
            "shared_init_sha256": identity["shared_init_sha256"],
            "contract_sha256": identity["contract_sha256"], "seed": checkpoint_seed,
            "checkpoint_step": checkpoint_step,
            "old_external_panels_used": False, "files": []}
    for record in records:
        seq, name = record["seq"], record["name"]
        if len(seq) > 1024:
            info["n_skipped_length"] += 1
            continue
        path = split_dir / f"{record['sequence_sha256']}.npz"
        if args.resume and path.exists():
            with np.load(path, allow_pickle=False) as old:
                if str(old["seq"]) == seq and "plana_on" in old and "plana_off" in old:
                    info["n_resumed"] += 1
                    continue
        on, on_device = resilient_plana_scores(encoder, head, seq, args.device)
        off, off_device = resilient_plana_scores(encoder, head, seq, args.device, prior_weight_override=0)
        np.savez_compressed(path, plana_on=on.astype(np.float32), plana_off=off.astype(np.float32),
                            name=np.array(name), seq=np.array(seq),
                            sequence_sha256=np.array(record["sequence_sha256"]),
                            selection_sha256=np.array(record["selection_sha256"]),
                            plana_on_device=np.array(on_device), plana_off_device=np.array(off_device),
                            checkpoint_sha256=np.array(checkpoint_sha),
                            train_trajectory=np.array(args.train_trajectory),
                            gt_pairs=np.asarray(record["gt_pairs"], dtype=np.int32).reshape(-1, 2))
        info["n_exported"] += 1
        info["files"].append({"path": str(path), "name": name,
                              "sequence_sha256": record["sequence_sha256"],
                              "on_device": on_device, "off_device": off_device})
        if info["n_exported"] % 10 == 0:
            print(f"[prior-export] {args.train_trajectory} {info['n_exported']}/{len(records)}", flush=True)
    info["code_sha256"] = sha256(Path(__file__).resolve())
    info["git_head"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    (args.out / "export_metadata.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "files"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
