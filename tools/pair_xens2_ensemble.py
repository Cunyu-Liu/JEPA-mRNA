#!/usr/bin/env python3
"""Combine already exported Plan-A/R2D scores without loading any neural network."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import numpy as np

REPO = Path("/home/rna_jev_xens2_dualview_round56")
MEMBER_KEYS = ("plana0_on", "plana0_off", "plana1_on", "plana1_off", "r2d_on", "r2d_off")


def common_identity(files, default_chunk=None):
    identities = []
    for handle in files:
        chunk = int(handle["evaluation_head_chunk_size"]) if "evaluation_head_chunk_size" in handle else default_chunk
        if chunk is None:
            raise ValueError("evaluation_head_chunk_size missing; pass an explicit default only for legacy smoke files")
        identities.append((str(handle["name"]), str(handle["seq"]), str(handle["sequence_sha256"]),
                           str(handle["selection_sha256"]), handle["gt_pairs"].tolist(), chunk))
    if any(x != identities[0] for x in identities[1:]):
        raise ValueError("member identity/GT/chunk mismatch")
    return identities[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r2d-dir", type=Path, required=True)
    ap.add_argument("--plana0-dir", type=Path, required=True)
    ap.add_argument("--plana1-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", default="bprna_vl0,archiveii_embok_clean,bprna_new,bprna_ts0,chanrg_test,chanrg_gena,chanrg_genc,chanrg_genf,rfam_fam")
    ap.add_argument("--limit-per-dataset", type=int, default=0)
    ap.add_argument("--default-chunk-size", type=int, default=None,
                    help="explicit compatibility fallback for legacy smoke NPZ files only")
    ap.add_argument("--legacy-r2d-device", default=None,
                    help="directory-metadata device label for legacy smoke files lacking per-file device fields")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    selection_path = args.r2d_dir / "selection_r2d.json"
    selection = json.loads(selection_path.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {"schema_version": 1, "members": "xens2_sequential_offline",
                "weights": {"plana0": .35, "plana1": .35, "r2d": .30},
                "sources": {"r2d": str(args.r2d_dir), "plana0": str(args.plana0_dir), "plana1": str(args.plana1_dir)},
                "datasets": {}}
    for dataset_id in args.datasets.split(","):
        records = selection["datasets"][dataset_id]
        split_out = args.out / dataset_id
        split_out.mkdir(parents=True, exist_ok=True)
        info = {"n_selected": len(records), "n_exported": 0, "n_resumed": 0, "files": []}
        for index, record in enumerate(records):
            if args.limit_per_dataset and index >= args.limit_per_dataset:
                break
            seq_hash = record["sequence_sha256"]
            out_path = split_out / f"{seq_hash}.npz"
            if args.resume and out_path.exists():
                info["n_resumed"] += 1
                continue
            paths = [args.plana0_dir / dataset_id / f"{seq_hash}.npz",
                     args.plana1_dir / dataset_id / f"{seq_hash}.npz",
                     args.r2d_dir / dataset_id / f"{seq_hash}.npz"]
            if not all(p.is_file() for p in paths):
                raise FileNotFoundError(f"incomplete member panel: {dataset_id} {seq_hash} {paths}")
            with np.load(paths[0], allow_pickle=False) as p0, np.load(paths[1], allow_pickle=False) as p1, np.load(paths[2], allow_pickle=False) as r2d:
                name, seq, seq_hash2, selection_hash, gt, chunk = common_identity((p0, p1, r2d), args.default_chunk_size)
                arrays = {k: p0[k].astype(np.float32) for k in ("plana0_on", "plana0_off")}
                arrays.update({k: p1[k].astype(np.float32) for k in ("plana1_on", "plana1_off")})
                arrays.update({k: r2d[k].astype(np.float32) for k in ("r2d_on", "r2d_off")})
                for suffix in ("on", "off"):
                    arrays[f"plana_mean_{suffix}"] = (.5 * arrays[f"plana0_{suffix}"] +
                                                       .5 * arrays[f"plana1_{suffix}"]).astype(np.float32)
                    arrays[f"xens2_{suffix}"] = (.70 * arrays[f"plana_mean_{suffix}"] +
                                                    .30 * arrays[f"r2d_{suffix}"]).astype(np.float32)
                payload = dict(arrays)
                payload.update({"name": np.array(name), "seq": np.array(seq),
                                "sequence_sha256": np.array(seq_hash2), "selection_sha256": np.array(selection_hash),
                                "evaluation_head_chunk_size": np.array(chunk),
                                "gt_pairs": np.asarray(gt, dtype=np.int32).reshape(-1, 2)})
                for key in ("plana0_on_device", "plana0_off_device"):
                    payload[key] = p0[key]
                for key in ("plana1_on_device", "plana1_off_device"):
                    payload[key] = p1[key]
                for key in ("r2d_on_device", "r2d_off_device"):
                    if key in r2d:
                        payload[key] = r2d[key]
                    elif args.legacy_r2d_device:
                        payload[key] = np.array(args.legacy_r2d_device)
                    else:
                        raise ValueError(f"{key} missing; use an explicit legacy device label only for directory-verified smoke files")
                np.savez_compressed(out_path, **payload)
            info["n_exported"] += 1
            info["files"].append(str(out_path))
        metadata["datasets"][dataset_id] = info
        (args.out / "metadata_xens2.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"[xens2-offline] {dataset_id}: {info['n_exported']} exported, {info['n_resumed']} resumed", flush=True)
    metadata["git_head"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    (args.out / "metadata_xens2.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
