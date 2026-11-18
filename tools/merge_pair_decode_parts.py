#!/usr/bin/env python3
"""Merge independently computed per-dataset frozen decode partitions without re-tuning."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import pandas as pd

EXPECTED = ["archiveii_embok_clean", "bprna_new", "bprna_ts0", "chanrg_test", "chanrg_gena", "chanrg_genc", "chanrg_genf", "rfam_fam"]
TABLES = ["decode_per_sequence", "decode_summary", "ranking_per_sequence", "ranking_summary"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path("/home/rna_jev_xens2_dualview_round56"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames = {name: [] for name in TABLES}
    routes, part_records = [], []
    frozen_config = raw_root = reused_source = None
    for dataset in EXPECTED:
        part = args.parts_root / dataset
        manifest_path = part / "decode_run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("official") or manifest.get("datasets") != [dataset]:
            raise ValueError(f"invalid partition manifest: {part}")
        frozen = json.loads((part / "frozen_decode_config.json").read_text())
        if frozen.get("external_datasets") != [dataset] or frozen.get("smoke_limit_per_dataset"):
            raise ValueError(f"invalid frozen partition config: {part}")
        if frozen_config is None:
            frozen_config, raw_root, reused_source = frozen["config"], manifest["raw_root"], manifest["reused_frozen_config"]
        elif frozen["config"] != frozen_config or manifest["raw_root"] != raw_root or manifest["reused_frozen_config"] != reused_source:
            raise ValueError(f"partition configuration mismatch: {part}")
        for name in TABLES:
            frame = pd.read_parquet(part / f"{name}.parquet")
            if set(frame.dataset_id.unique()) != {dataset}:
                raise ValueError(f"dataset mismatch in {part}/{name}")
            frames[name].append(frame)
        part_routes = json.loads((part / "route_summary.json").read_text())["rows"]
        if {r["dataset_id"] for r in part_routes} != {dataset}:
            raise ValueError(f"route mismatch: {part}")
        routes.extend(part_routes)
        part_records.append({"dataset_id": dataset, "path": str(part), "manifest_sha256": sha256(manifest_path),
                             "n_decode_rows": manifest["n_decode_rows"], "n_ranking_rows": manifest["n_ranking_rows"]})
    for name in TABLES:
        data = pd.concat(frames[name], ignore_index=True)
        if set(data.dataset_id.unique()) != set(EXPECTED):
            raise ValueError(f"incomplete merged table: {name}")
        data.to_parquet(args.out / f"{name}.parquet", index=False)
    route_keys = {(r["dataset_id"], r["score_variant"]) for r in routes}
    if len(route_keys) != 80 or len(routes) != 80:
        raise ValueError(f"expected 80 dataset-variant routes, found {len(routes)} / {len(route_keys)}")
    frozen_out = {"schema_version": 1, "development_dataset": "bprna_vl0",
                  "selection_policy": "micro F1, then macro F1, then micro precision; no external selection",
                  "temperature_grid": [0.5, 1.0, 2.0, 4.0], "mea_gamma_grid": [0.5, 1.0, 2.0, 4.0, 8.0],
                  "external_datasets": EXPECTED, "config": frozen_config, "reused_frozen_config": reused_source,
                  "smoke_limit_per_dataset": None, "merged_from_independent_dataset_partitions": True}
    (args.out / "frozen_decode_config.json").write_text(json.dumps(frozen_out, indent=2) + "\n")
    (args.out / "route_summary.json").write_text(json.dumps({"schema_version": 1, "rows": routes}, indent=2) + "\n")
    merge_manifest = {"schema_version": 1, "official": True, "raw_root": raw_root, "out": str(args.out), "datasets": EXPECTED,
                      "reused_frozen_config": reused_source, "n_decode_rows": sum(r["n_decode_rows"] for r in part_records),
                      "n_ranking_rows": sum(r["n_ranking_rows"] for r in part_records), "parts": part_records,
                      "code": str(Path(__file__).resolve()), "code_sha256": sha256(Path(__file__).resolve()),
                      "decode_common_sha256": sha256(args.repo / "tools/pair_decode_common.py"),
                      "git_head": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()}
    (args.out / "decode_run_manifest.json").write_text(json.dumps(merge_manifest, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "n_partitions": len(part_records), "n_routes": len(routes),
                      "n_decode_rows": merge_manifest["n_decode_rows"], "n_ranking_rows": merge_manifest["n_ranking_rows"]}, indent=2))


if __name__ == "__main__":
    main()
