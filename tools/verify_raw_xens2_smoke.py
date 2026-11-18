#!/usr/bin/env python3
"""End-to-end identity checks for recovered xens2 raw export smoke files."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

REPO = Path("/home/rna_jev_xens2_dualview_round56")
ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/pair_coverage_diagnostics_v1")
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))
from rnajepa.harness import nussinov_map, valid_pair_mask
from pair_coverage_audit import load_predictions


def main():
    smoke = ROOT.parent / "pair_coverage_diagnostics_v1_smoke/raw_xens2_sequential/xens2/bprna_new"
    formal = ROOT / "raw_scores_chunk64/bprna_new"
    gpu_r2d = ROOT.parent / "pair_coverage_diagnostics_v1_smoke/chunk64/bprna_new"
    prediction_path = Path("/root/rna_jepa_data/eval_decision/round42_bprna_new/xens2/result.json")
    if not prediction_path.is_file():
        registry = json.loads((REPO / "manifests/foldnar_v1/prediction_registry.json").read_text())["rows"]
        matches = [r for r in registry if r["prediction_id"] == "xens2/bprna_new"]
        if len(matches) != 1:
            raise ValueError(f"frozen xens2/bprna_new prediction identity: {len(matches)} matches")
        prediction_path = Path(matches[0]["path"])
    frozen = load_predictions(prediction_path)
    rows = []
    failures = []
    for path in sorted(smoke.glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            name, seq = str(z["name"]), str(z["seq"])
            mask = valid_pair_mask(seq)
            arrays = {k: np.asarray(z[k]) for k in z.files if k.endswith("_on") or k.endswith("_off")}
            required = {"plana0_on", "plana0_off", "plana1_on", "plana1_off", "r2d_on", "r2d_off", "xens2_on", "xens2_off"}
            if set(arrays) != required:
                failures.append({"name": name, "check": "score_keys", "actual": sorted(arrays)})
                continue
            formula_ok = {}
            for suffix in ("on", "off"):
                expected = 0.7 * (0.5 * arrays[f"plana0_{suffix}"] + 0.5 * arrays[f"plana1_{suffix}"]) + 0.3 * arrays[f"r2d_{suffix}"]
                formula_ok[suffix] = bool(np.allclose(expected, arrays[f"xens2_{suffix}"], atol=1e-6, rtol=1e-6))
            predicted = set(nussinov_map(arrays["xens2_on"], mask))
            expected_pairs = set(frozen[name][0]["pairs"])
            frozen_tp = len(predicted & expected_pairs)
            row = {"name": name, "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
                   "n_smoke_pairs": len(predicted), "n_frozen_pairs": len(expected_pairs),
                   "n_pair_intersection": frozen_tp, "frozen_map_exact": predicted == expected_pairs,
                   "ensemble_formula_exact": formula_ok,
                   "devices": {k: str(z[k]) for k in z.files if k.endswith("_device")}}
            other_path = formal / path.name
            gpu_path = gpu_r2d / path.name
            if not other_path.is_file() or not gpu_path.is_file():
                failures.append({"name": name, "check": "formal_or_gpu_r2d_missing"})
            else:
                with np.load(other_path, allow_pickle=False) as cpu, np.load(gpu_path, allow_pickle=False) as gpu_z:
                    if str(gpu_z["name"]) != name or str(gpu_z["seq"]) != seq:
                        failures.append({"name": name, "check": "gpu_r2d_identity"})
                    for suffix in ("on", "off"):
                        gpu = np.asarray(gpu_z[f"r2d_{suffix}"])
                        cpu_array = np.asarray(cpu[f"r2d_{suffix}"])
                        valid = np.isfinite(gpu) & np.isfinite(cpu_array) & (np.abs(gpu) < 1000) & (np.abs(cpu_array) < 1000)
                        delta = np.abs(gpu[valid] - cpu_array[valid])
                        row[f"r2d_{suffix}_max_abs_cpu_gpu"] = float(delta.max())
                        row[f"r2d_{suffix}_mean_abs_cpu_gpu"] = float(delta.mean())
                        row[f"r2d_{suffix}_map_cpu_gpu_exact"] = set(nussinov_map(gpu, mask)) == set(nussinov_map(cpu_array, mask))
            rows.append(row)
            if not row["frozen_map_exact"] or not all(formula_ok.values()) or not row.get("r2d_on_map_cpu_gpu_exact", False) or not row.get("r2d_off_map_cpu_gpu_exact", False):
                failures.append(row)
    result = {"status": "PASS" if len(rows) == 2 and not failures else "FAIL", "n_sequences": len(rows),
              "frozen_prediction_path": str(prediction_path), "rows": rows, "failures": failures}
    out = ROOT / "raw_xens2_smoke_verification.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
