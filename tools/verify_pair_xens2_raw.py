#!/usr/bin/env python3
"""Independent full-panel verification of sequential member export and offline fusion."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--ensemble-name", default="xens2")
    args = ap.parse_args()
    ensemble_root = args.root / args.ensemble_name
    failures, rows = [], []
    files = sorted(ensemble_root.glob("*/*.npz"))
    required = {f"{m}_{p}" for m in ("plana0", "plana1", "plana_mean", "r2d", "xens2") for p in ("on", "off")}
    for index, path in enumerate(files, 1):
        dataset, name = path.parent.name, path.stem
        sources = {"plana0": args.root / "plana0" / dataset / path.name,
                   "plana1": args.root / "plana1" / dataset / path.name,
                   "r2d": args.root / dataset / path.name}
        with np.load(path, allow_pickle=False) as e:
            ekeys = set(e.files)
            if not required <= ekeys:
                failures.append({"dataset": dataset, "name": name, "missing_keys": sorted(required - ekeys)})
                continue
            arrays = {k: e[k] for k in required}
            seq = str(e["seq"])
            canonical = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
            legal = np.zeros((len(seq), len(seq)), dtype=bool)
            for i in range(len(seq)):
                for j in range(len(seq)):
                    if abs(j - i) > 3 and (seq[i], seq[j]) in canonical:
                        legal[i, j] = True
            for key, array in arrays.items():
                if not np.isfinite(array[legal]).all() or not np.isneginf(array[~legal]).all():
                    failures.append({"dataset": dataset, "name": name, "check": "invalid_score_mask", "key": key})
            for member, source in sources.items():
                if not source.exists():
                    failures.append({"dataset": dataset, "name": name, "check": "missing_member", "member": member})
                    continue
                with np.load(source, allow_pickle=False) as s:
                    for field in ("name", "seq", "sequence_sha256", "selection_sha256"):
                        if str(s[field]) != str(e[field]):
                            failures.append({"dataset": dataset, "name": name, "check": field, "member": member})
                    if not np.array_equal(s["gt_pairs"], e["gt_pairs"]):
                        failures.append({"dataset": dataset, "name": name, "check": "gt_mismatch", "member": member})
                    for prior in ("on", "off"):
                        key = f"{member}_{prior}"
                        if not np.array_equal(s[key], arrays[key]):
                            failures.append({"dataset": dataset, "name": name, "check": "member_copy", "key": key})
                        if int(s["evaluation_head_chunk_size"]) != int(e["evaluation_head_chunk_size"]):
                            failures.append({"dataset": dataset, "name": name, "check": "chunk_mismatch", "key": key})
            row = {"dataset_id": dataset, "name": name}
            for prior in ("on", "off"):
                mean = .5 * arrays[f"plana0_{prior}"] + .5 * arrays[f"plana1_{prior}"]
                fused = .7 * mean + .3 * arrays[f"r2d_{prior}"]
                mean_error = float(np.max(np.abs(mean[legal] - arrays[f"plana_mean_{prior}"][legal]))) if legal.any() else 0.
                fused_error = float(np.max(np.abs(fused[legal] - arrays[f"xens2_{prior}"][legal]))) if legal.any() else 0.
                row[f"plana_mean_{prior}_max_abs_error"] = mean_error
                row[f"xens2_{prior}_max_abs_error"] = fused_error
                if not np.allclose(mean, arrays[f"plana_mean_{prior}"], atol=1e-6, rtol=1e-6):
                    failures.append({"dataset": dataset, "name": name, "check": "plana_mean_formula", "prior": prior, "max_abs": mean_error})
                if not np.allclose(fused, arrays[f"xens2_{prior}"], atol=1e-6, rtol=1e-6):
                    failures.append({"dataset": dataset, "name": name, "check": "xens2_formula", "prior": prior, "max_abs": fused_error})
            rows.append(row)
        if index % 100 == 0:
            print(f"[verify-xens2] {index}/{len(files)}", flush=True)
    report = {"status": "PASS" if not failures and len(files) == 708 else "FAIL",
              "n_sequences": len(files), "expected_sequences": 708, "n_failed": len(failures), "failures": failures,
              "max_formula_error": max((max(v for k, v in r.items() if k.endswith("max_abs_error")) for r in rows), default=None)}
    out = args.root / "raw_xens2_full_verification.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
