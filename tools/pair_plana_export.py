#!/usr/bin/env python3
"""Export one identity-bound Plan-A member at a time for memory-safe xens2 diagnostics."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

REPO = Path("/home/rna_jev_xens2_dualview_round56")
sys.path.insert(0, str(REPO / "tools"))
from pair_raw_export import (EXPECTED_CHECKPOINT_SHA256, resilient_plana_scores,
                             select_records, sha256)
from xens2_eval import load_plana


def load_one(path: Path, requested_device: str):
    try:
        encoder, head = load_plana(str(path), requested_device)
        return encoder, head, requested_device, None
    except torch.OutOfMemoryError as exc:
        if requested_device == "cpu":
            raise
        torch.cuda.empty_cache()
        encoder, head = load_plana(str(path), "cpu")
        return encoder, head, "cpu", repr(exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", choices=("plana0", "plana1"), required=True)
    ap.add_argument("--datasets", default="bprna_vl0,archiveii_embok_clean,bprna_new,bprna_ts0,chanrg_test,chanrg_gena,chanrg_genc,chanrg_genf,rfam_fam")
    ap.add_argument("--plana-ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--head-chunk-size", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit-per-dataset", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    ckpt = Path(args.plana_ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    digest = sha256(ckpt)
    expected = EXPECTED_CHECKPOINT_SHA256[ckpt.name]
    if digest != expected:
        raise ValueError(f"checkpoint identity mismatch: {digest} != {expected}")
    registry = {r["dataset_id"]: r for r in json.loads((REPO / "manifests/foldnar_v1/data_registry.json").read_text())["rows"]}
    datasets = args.datasets.split(",")
    selected = {d: select_records(registry, d) for d in datasets}
    args.out.mkdir(parents=True, exist_ok=True)
    selection = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                 "member": args.member, "checkpoint_sha256": digest,
                 "rule": "all VL0; other splits first 64 by sha256(dataset_id|name|sequence_sha256)",
                 "datasets": {d: [{k: v for k, v in r.items() if k not in ("seq", "gt_pairs")} for r in rows]
                              for d, rows in selected.items()}}
    (args.out / f"selection_{args.member}.json").write_text(json.dumps(selection, indent=2) + "\n")
    encoder, head, load_device, load_oom = load_one(ckpt, args.device)
    head.chunk_size = args.head_chunk_size
    on_key, off_key = f"{args.member}_on", f"{args.member}_off"
    metadata = {"schema_version": 1, "member": args.member, "requested_device": args.device,
                "load_device": load_device, "load_oom": load_oom,
                "evaluation_head_chunk_size": args.head_chunk_size,
                "checkpoint": {"path": str(ckpt), "sha256": digest}, "datasets": {}}
    for dataset_id, records in selected.items():
        split_dir = args.out / dataset_id
        split_dir.mkdir(parents=True, exist_ok=True)
        info = {"n_selected": len(records), "n_exported": 0, "n_resumed": 0,
                "n_skipped_length": 0, "files": []}
        for record_index, record in enumerate(records):
            if args.limit_per_dataset and record_index >= args.limit_per_dataset:
                break
            seq, name = record["seq"], record["name"]
            if len(seq) > 1024:
                info["n_skipped_length"] += 1
                continue
            out_path = split_dir / f"{record['sequence_sha256']}.npz"
            if args.resume and out_path.exists():
                with np.load(out_path, allow_pickle=False) as old:
                    if str(old["name"]) == name and str(old["seq"]) == seq and on_key in old and off_key in old:
                        info["n_resumed"] += 1
                        continue
            on, on_device = resilient_plana_scores(encoder, head, seq, args.device)
            off, off_device = resilient_plana_scores(encoder, head, seq, args.device, prior_weight_override=0)
            payload = {on_key: on.astype(np.float32), off_key: off.astype(np.float32),
                       "name": np.array(name), "seq": np.array(seq),
                       "sequence_sha256": np.array(record["sequence_sha256"]),
                       "selection_sha256": np.array(record["selection_sha256"]),
                       f"{args.member}_on_device": np.array(on_device),
                       f"{args.member}_off_device": np.array(off_device),
                       "evaluation_head_chunk_size": np.array(args.head_chunk_size),
                       "gt_pairs": np.asarray(record["gt_pairs"], dtype=np.int32).reshape(-1, 2)}
            np.savez_compressed(out_path, **payload)
            info["n_exported"] += 1
            info["files"].append({"path": str(out_path), "name": name,
                                  "sequence_sha256": record["sequence_sha256"],
                                  "on_device": on_device, "off_device": off_device})
            if on_device.startswith("cpu"):
                info["n_cpu_on"] = info.get("n_cpu_on", 0) + 1
            if off_device.startswith("cpu"):
                info["n_cpu_off"] = info.get("n_cpu_off", 0) + 1
            if info["n_exported"] % 10 == 0:
                print(f"[plana-export] {args.member} {dataset_id} {info['n_exported']} exported", flush=True)
        metadata["datasets"][dataset_id] = info
        (args.out / f"metadata_{args.member}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    metadata["code"] = str(Path(__file__).resolve())
    metadata["code_sha256"] = sha256(Path(__file__).resolve())
    metadata["git_head"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    (args.out / f"metadata_{args.member}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"member": args.member, "load_device": load_device,
                      "datasets": {d: {k: v for k, v in x.items() if k != "files"} for d, x in metadata["datasets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
