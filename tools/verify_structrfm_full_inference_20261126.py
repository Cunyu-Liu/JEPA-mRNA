#!/usr/bin/env python3
import ast
import json
from pathlib import Path

from structrfm_official_recovery_20261126 import DEFAULT_ROOT, HISTORICAL_PREDICTION, HISTORICAL_PREDICTION_SHA256, sha256, utc_now, write_json


def main():
    root = DEFAULT_ROOT
    repo = Path("/home/rna_jev_structrfm_recovery")
    result_path = repo / "manifests/foldnar_v2/structrfm_official_bprna_new_results_20261126.json"
    result = json.loads(result_path.read_text())
    assert result["status"] == "PASS_INDEPENDENT_OFFICIAL_5388_IDENTITY_AND_METRICS"
    assert result["full_ready"] is True and result["metrics"]["coverage"] == 1.0
    assert result["metrics"]["n_dataset_sequences"] == result["metrics"]["n_with_prediction"] == 5388
    assert result["raw_not_clean"] is True and result["summary_only_substitution"] is False
    assert sha256(HISTORICAL_PREDICTION) == HISTORICAL_PREDICTION_SHA256
    predictions = Path(result["predictions"]["path"])
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    assert len(rows) == len({row["name"] for row in rows}) == 5388
    assert all(row["seq"] for row in rows)
    files = [
        repo / "tools/probe_structrfm_longest_gpu_20261126.py",
        repo / "tools/run_structrfm_gated_full_20261126.py",
        repo / "tools/run_structrfm_bprna_new_gpu_20261126.py",
        repo / "tools/score_structrfm_official_bprna_new_20261126.py",
        repo / "tools/verify_structrfm_full_inference_20261126.py",
        repo / "tests/test_structrfm_full_inference_20261126.py",
        repo / "tests/test_structrfm_official_recovery_20261126.py",
        result_path,
    ]
    for path in files:
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
    log = root / "reports/full_cpu_tests.log"
    assert log.is_file() and log.stat().st_size > 0
    report = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "PASS_FULL_5388_GPU_INFERENCE_AND_INDEPENDENT_DELIVERY",
        "full_ready": True,
        "prediction_sha256": sha256(predictions),
        "n_sequences": len(rows),
        "coverage": 1.0,
        "micro_f1": result["metrics"]["micro_f1"],
        "macro_f1": result["metrics"]["macro_f1"],
        "historical_prediction_unchanged": True,
        "raw_not_clean": True,
        "cpu_test_log": {"path": str(log), "sha256": sha256(log)},
        "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files],
    }
    output = root / "reports/full_delivery_verification_20261126.json"
    write_json(output, report)
    print(json.dumps({"status": report["status"], "micro_f1": report["micro_f1"], "macro_f1": report["macro_f1"], "report": str(output), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
