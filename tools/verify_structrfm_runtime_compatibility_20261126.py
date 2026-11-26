#!/usr/bin/env python3
import ast
import json
from pathlib import Path

from structrfm_official_recovery_20261126 import DEFAULT_ROOT, sha256, utc_now, write_json


def main():
    root = DEFAULT_ROOT
    repo = Path("/home/rna_jev_structrfm_recovery")
    manifest_path = repo / "manifests/foldnar_v2/structrfm_runtime_compatibility_20261126.json"
    document = json.loads(manifest_path.read_text())
    assert document["runtime_ready"] is True and document["full_ready"] is False
    assert document["predictions"] is None and document["metrics"] is None
    assert document["rebuilt_runtime"]["sha256"] == sha256(root / "runtime/cp310/interface.cpython-310-x86_64-linux-gnu.so")
    assert document["gpu_readiness"]["full_dataset_inference_performed"] is False
    assert not (root / "predictions").exists()
    logs = {}
    for name in ("runtime_round1_build_differential.log", "runtime_round2_static_cpu_tests.log", "runtime_round3_gpu_readiness.log"):
        path = root / "reports" / name
        assert path.is_file() and path.stat().st_size > 0
        logs[name] = {"path": str(path), "sha256": sha256(path)}
    files = [
        repo / "tools/build_structrfm_interface_cp310_20261126.sh",
        repo / "tools/structrfm_interface_differential_20261126.py",
        repo / "tools/export_structrfm_dp_readiness_params_20261126.py",
        repo / "tools/run_structrfm_bprna_new_gpu_20261126.py",
        repo / "tools/register_structrfm_runtime_compatibility_20261126.py",
        repo / "tools/verify_structrfm_runtime_compatibility_20261126.py",
        repo / "tests/test_structrfm_official_recovery_20261126.py",
        manifest_path,
    ]
    for path in files:
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
    report = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "PASS_RUNTIME_THREE_ROUNDS_NO_FULL_GPU_INFERENCE",
        "runtime_ready": True,
        "full_ready": False,
        "logs": logs,
        "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files],
        "gpu_inference_scope": "one shortest sequence readiness only; no 5388 prediction artifact",
    }
    output = root / "reports/runtime_delivery_verification_20261126.json"
    write_json(output, report)
    print(json.dumps({"status": report["status"], "runtime_ready": True, "full_ready": False, "report": str(output), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
