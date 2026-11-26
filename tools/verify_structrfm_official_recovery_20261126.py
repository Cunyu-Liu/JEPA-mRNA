#!/usr/bin/env python3
import argparse
import ast
import json
from pathlib import Path

from structrfm_official_recovery_20261126 import (
    ASSET_SHA256,
    DATASET_SHA256,
    DEFAULT_ROOT,
    INFERENCE_COMMIT,
    RELEASE_COMMIT,
    inspect_checkpoint,
    load_dataset,
    sha256,
    utc_now,
    validate_source,
    write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev_structrfm_recovery"))
    args = parser.parse_args()
    manifest_path = args.repo / "manifests/foldnar_v2/structrfm_official_recovery_20261126.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["full_ready"] is False and manifest["asset_ready"] is True
    assert manifest["source"]["release_commit"] == RELEASE_COMMIT
    assert manifest["source"]["inference_commit"] == INFERENCE_COMMIT
    assert manifest["checkpoint"]["sha256"] == ASSET_SHA256
    assert manifest["bprna_new_5388_contract"]["predictions"] is None
    assert manifest["bprna_new_5388_contract"]["metrics"] is None
    assert manifest["gpu_execution"]["performed"] is False
    checkpoint = inspect_checkpoint()
    source = validate_source()
    rows = load_dataset()
    assert len(rows) == 5388
    baseline_doc = json.loads((args.repo / "manifests/foldnar_v1/baseline_registry.json").read_text())
    baseline = next(row for row in baseline_doc["rows"] if row["baseline_id"] == "structrfm")
    assert baseline["official_weights"]["sha256"] == ASSET_SHA256
    assert baseline["n_frozen_prediction_rows"] == 0 and baseline["n_blocked_prediction_rows"] == 1
    prediction_doc = json.loads((args.repo / "manifests/foldnar_v1/prediction_registry.json").read_text())
    prediction = next(row for row in prediction_doc["rows"] if row["prediction_id"] == "structrfm/bprna_new")
    assert prediction["status"] == "BLOCKED"
    tools = [args.repo / "tools/structrfm_official_recovery_20261126.py", args.repo / "tools/run_structrfm_bprna_new_gpu_20261126.py", args.repo / "tools/verify_structrfm_official_recovery_20261126.py"]
    test = args.repo / "tests/test_structrfm_official_recovery_20261126.py"
    for path in tools + [test]:
        ast.parse(path.read_text(), filename=str(path))
    rounds = {}
    for key, name in (("round1_asset_cpu", "round1_asset_cpu.log"), ("round2_static_preflight", "round2_static_preflight.log"), ("round3_cpu_tests", "round3_cpu_tests.log")):
        path = args.root / "reports" / name
        assert path.is_file() and path.stat().st_size > 0, name
        rounds[key] = {"log": str(path), "sha256": sha256(path)}
    assert not (args.root / "predictions").exists(), "No GPU prediction artifact may exist in this recovery"
    files = tools + [test, manifest_path, args.root / "release_v0.0.8.json", args.root / "reports/structrfm_official_provenance_20261126.json"]
    report = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "PASS_THREE_ROUNDS_OFFICIAL_ASSET_READY_RUNTIME_AND_PREDICTION_BLOCKED",
        "full_ready": False,
        "full_ready_degree": "OFFICIAL_CHECKPOINT_BYTES_AND_FROZEN_CONTRACT_READY; CPYTHON38_RUNTIME_AND_NEW_GPU_5388_PREDICTIONS_NOT_READY",
        "checkpoint": checkpoint,
        "source": source,
        "dataset_sha256": DATASET_SHA256,
        "n_dataset_sequences": len(rows),
        "historical_prediction_eligible_as_official": False,
        "gpu_inference_performed": False,
        "rounds": rounds,
        "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files],
    }
    output = args.root / "reports/delivery_verification_20261126.json"
    write_json(output, report)
    print(json.dumps({"status": report["status"], "full_ready": False, "report": str(output), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
