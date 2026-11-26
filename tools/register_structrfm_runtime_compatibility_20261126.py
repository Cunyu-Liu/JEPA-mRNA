#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

from structrfm_official_recovery_20261126 import DEFAULT_ROOT, sha256, utc_now, write_json

UPSTREAM = Path("/home/rna_jev_structrfm_interface_upstream/mxfold2-v0.1.2")
UPSTREAM_COMMIT = "51b213676708bebd664f0c40873a46e09353e1ee"
INTERFACE = DEFAULT_ROOT / "runtime/cp310/interface.cpython-310-x86_64-linux-gnu.so"
INTERFACE_SHA256 = "616ec59fd9c853bebbcb811eeff8af8ec2ce2b7380971dda9de63b9cc33cb6d0"
OFFICIAL_INTERFACE = Path("/home/rna_jev/third_party/structRFM/tasks/seqcls_ssp/interface.cpython-38-x86_64-linux-gnu.so")
OFFICIAL_INTERFACE_SHA256 = "7d332a297216ecc0d5ccbeec356c42876b543d089cba8846f478ebbaec4ea007"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev_structrfm_recovery"))
    args = parser.parse_args()
    root = args.root
    differential_path = root / "reports/differential/differential_report.json"
    readiness_path = root / "reports/gpu_readiness_one_20261126.json"
    differential = json.loads(differential_path.read_text())
    readiness = json.loads(readiness_path.read_text())
    assert differential["status"] == "PASS_OFFICIAL_CP38_VS_REBUILT_CP310"
    assert differential["n_cases"] == len(differential["checks"]) == 18
    assert differential["pair_and_dot_bracket_exact"] is True
    assert differential["reference_interface_sha256"] == OFFICIAL_INTERFACE_SHA256
    assert differential["candidate_interface_sha256"] == INTERFACE_SHA256 == sha256(INTERFACE)
    assert differential["real_bprna_new_cases"] == 18
    parameter_manifest_path = root / "readiness_params/cases.json"
    parameter_manifest = json.loads(parameter_manifest_path.read_text())
    assert parameter_manifest["status"] == "PASS_18_REAL_MODEL_GPU_FORWARD_PARAMETER_EXPORTS_NO_DP_PREDICTIONS"
    assert parameter_manifest["n_cases"] == 18 and parameter_manifest["full_dataset_inference_performed"] is False
    assert differential["parameter_manifest_sha256"] == sha256(parameter_manifest_path)
    assert sha256(OFFICIAL_INTERFACE) == OFFICIAL_INTERFACE_SHA256
    assert sha256(root / "downloads/cpython-3.8.19+20240726-x86_64-unknown-linux-gnu-install_only.tar.gz") == "e81ea4dd16e6057c8121bdbcb7b64e2956068ca019f244c814bc3ad907cb2765"
    assert readiness["status"] == "PASS_SINGLE_SEQUENCE_GPU_FORWARD_AND_CPU_DP_READINESS"
    assert readiness["n_sequences_processed"] == 1 and readiness["full_dataset_inference_performed"] is False
    assert readiness["prediction_artifact"] is None and readiness["metrics"] is None
    assert readiness["gpu_architecture"] == "sm_120"
    assert readiness["runtime"]["interface"]["sha256"] == INTERFACE_SHA256
    assert subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip() == UPSTREAM_COMMIT
    assert not subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip()
    document = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "OFFICIAL_CP38_DIFFERENTIAL_AND_SM120_SINGLE_SEQUENCE_READINESS_PASS",
        "full_ready": False,
        "runtime_ready": True,
        "ready_scope": "Current CPython 3.10 + torch 2.7.1+cu128 + rebuilt original C++ DP; exact pair/DBN differential against official cp38 interface; one shortest-sequence GPU forward + CPU DP",
        "blocked_scope": "Full 5388 GPU inference and new official predictions were not run in this task",
        "upstream": {"repository": "https://github.com/mxfold/mxfold2", "version": "v0.1.2", "commit": UPSTREAM_COMMIT, "license": "MIT", "source_path": str(UPSTREAM), "source_tree_clean": True, "dp_source_modified": False},
        "official_cp38_reference": {"path": str(OFFICIAL_INTERFACE), "sha256": OFFICIAL_INTERFACE_SHA256, "role": "CPU differential gold standard only, not GPU torch runtime"},
        "cp38_benchmark_interpreter": {"distribution": "python-build-standalone", "release": "20240726", "python": "3.8.19", "license": "PSF-2.0", "official_url": "https://github.com/astral-sh/python-build-standalone/releases/download/20240726/cpython-3.8.19+20240726-x86_64-unknown-linux-gnu-install_only.tar.gz", "archive_path": str(root / "downloads/cpython-3.8.19+20240726-x86_64-unknown-linux-gnu-install_only.tar.gz"), "archive_sha256": "e81ea4dd16e6057c8121bdbcb7b64e2956068ca019f244c814bc3ad907cb2765", "environment_path": str(root / "envs/cp38-standalone")},
        "build_dependencies": {"pybind11": {"version": "2.13.6", "commit": "a2e59f0e7065404b44dfe92a28aca47ba1378dc4", "license": "BSD", "source": "https://github.com/pybind/pybind11"}, "BPfold": {"version": "0.3.1", "license": "MIT", "source": "https://github.com/heqin-zhu/BPfold", "distribution_path": str(root / "pydeps")}},
        "rebuilt_runtime": {"path": str(INTERFACE), "sha256": INTERFACE_SHA256, "python": "3.10.22", "torch": "2.7.1+cu128", "cuda": "12.8", "bpfold": "0.3.1", "arch_list_includes": "sm_120", "build_script": "tools/build_structrfm_interface_cp310_20261126.sh", "mathematical_semantics_modified": False},
        "differential": {"path": str(differential_path), "sha256": sha256(differential_path), "n_cases": 18, "real_bprna_new_cases": 18, "parameter_manifest": str(parameter_manifest_path), "parameter_manifest_sha256": sha256(parameter_manifest_path), "pair_and_dot_bracket_exact": True, "max_energy_delta": differential["max_energy_delta"], "energy_absolute_tolerance": 1e-5},
        "gpu_readiness": {"path": str(readiness_path), "sha256": sha256(readiness_path), **readiness},
        "predictions": None,
        "metrics": None,
        "summary_only_prediction_substitution": False,
    }
    manifest_path = args.repo / "manifests/foldnar_v2/structrfm_runtime_compatibility_20261126.json"
    write_json(manifest_path, document)
    write_json(root / "reports/structrfm_runtime_compatibility_20261126.json", document)

    baseline_path = args.repo / "manifests/foldnar_v1/baseline_registry.json"
    baseline_doc = json.loads(baseline_path.read_text())
    baseline = next(row for row in baseline_doc["rows"] if row["baseline_id"] == "structrfm")
    baseline["inference"]["status"] = "OFFICIAL_CHECKPOINT_AND_RUNTIME_VERIFIED_FULL_GPU_PREDICTION_PENDING"
    baseline["provenance_status"] = "OFFICIAL_WEIGHT_AND_CP38_DIFFERENTIAL_RUNTIME_VERIFIED_PREDICTION_PENDING"
    baseline["structrfm_runtime_compatibility"] = "manifests/foldnar_v2/structrfm_runtime_compatibility_20261126.json"
    write_json(baseline_path, baseline_doc)

    for path in (args.repo / "manifests/foldnar_v2/structrfm_official_recovery_20261126.json", root / "reports/structrfm_official_provenance_20261126.json"):
        original = json.loads(path.read_text())
        original["status"] = "OFFICIAL_CHECKPOINT_AND_RUNTIME_READY_GPU_PREDICTION_PENDING"
        original["runtime_ready"] = True
        original["blocked_scope"] = "Full 5388 GPU inference and new verified-official predictions remain pending; historical weight bytes are unavailable for retrospective linkage"
        original["runtime_compatibility"] = "manifests/foldnar_v2/structrfm_runtime_compatibility_20261126.json"
        original["gpu_execution"]["runtime_requirement"] = "Verified cp310 interface built from unmodified MXfold2 v0.1.2 C++ source; torch 2.7.1+cu128 supports sm_120"
        original["gpu_execution"]["performed"] = True
        original["gpu_execution"]["execution_scope"] = "18 GPU-forward parameter exports plus one end-to-end readiness sequence; no full inference or prediction artifact"
        original["gpu_execution"]["single_sequence_readiness_performed"] = True
        original["gpu_execution"]["full_dataset_inference_performed"] = False
        write_json(path, original)
    print(json.dumps({"status": document["status"], "runtime_ready": True, "full_ready": False, "manifest": str(manifest_path), "sha256": sha256(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
