#!/usr/bin/env python3
import argparse
import ast
import json
import subprocess
from pathlib import Path

from debfold_official_recovery_20261126 import (
    ARCHIVE_NAMES,
    DATASET_SHA256,
    DEFAULT_DATASET,
    DEFAULT_ROOT,
    EXPECTED_ARCHIVE_ROWS,
    EXPECTED_TREE,
    OFFICIAL_COMMIT,
    audit_archive,
    load_dataset,
    sha256,
    utc_now,
    write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo", type=Path, default=Path("/home/rna_jev_debfold_recovery"))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    provenance_path = args.root / "reports/debfold_official_provenance_20261126.json"
    document = json.loads(provenance_path.read_text())
    assert document["status"] == "BLOCKED_OFFICIAL_IMPLEMENTATION_AND_WEIGHTS"
    assert document["full_ready"] is False
    assert document["official_commit"] == OFFICIAL_COMMIT
    assert set(document["official_tree"]) == EXPECTED_TREE
    assert document["official_releases"] == [] and document["official_tags"] == []
    assert document["weights"]["path"] is None and document["weights"]["sha256"] is None
    assert document["implementation"]["local_inference_entrypoint"] is None
    assert document["license"]["spdx"] is None
    assert document["bprna_new_5388_contract"]["predictions"] is None
    assert document["bprna_new_5388_contract"]["metrics"] is None
    assert document["bprna_new_5388_contract"]["coverage"] is None
    assert document["bprna_new_5388_contract"]["official_test_beta_40_is_not_project_5388"] is True
    assert document["dataset_protocol"]["external_test_used_for_tuning"] is False
    assert document["gpu_execution"]["performed"] is False
    dataset = load_dataset(args.dataset)
    assert len(dataset) == 5388 and sha256(args.dataset) == DATASET_SHA256
    unified = json.loads((args.repo / "manifests/foldnar_v1/baseline_registry.json").read_text())
    debfold = next(row for row in unified["rows"] if row["baseline_id"] == "debfold")
    assert debfold["training_data"]["archive_sha256"] == document["official_archives"]["TrainingValidationSet.zip"]["sha256"]
    assert debfold["official_weights"]["sha256"] is None
    assert debfold["license"]["spdx"] is None
    assert debfold["prediction_ids"] == [] and debfold["n_frozen_prediction_rows"] == 0
    source = args.root / "source/DEBFold"
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    for name, expected in EXPECTED_ARCHIVE_ROWS.items():
        archive = source / name
        assert sha256(archive) == document["official_archives"][name]["sha256"]
        assert sha256(args.root / "official_download" / name) == sha256(archive)
        rows = audit_archive(archive)
        assert (len(rows), len({row["seq"] for row in rows})) == expected
    tools = [args.repo / "tools/debfold_official_recovery_20261126.py", args.repo / "tools/run_debfold_bprna_new_gpu_20261126.py", args.repo / "tools/verify_debfold_official_recovery_20261126.py"]
    for path in tools + [args.repo / "tests/test_debfold_official_recovery_20261126.py"]:
        ast.parse(path.read_text(), filename=str(path))
    for log_name in ("round1_download.log", "round2_static_preflight.log", "round3_cpu_unittests.log"):
        assert (args.root / "reports" / log_name).is_file(), log_name
    assert not (args.root / "predictions").exists(), "No GPU prediction artifact should exist in this recovery"
    core_files = tools + [
        args.repo / "tests/test_debfold_official_recovery_20261126.py",
        args.repo / "tools/foldnar_baseline_contract.py",
        args.repo / "manifests/foldnar_v1/baseline_registry.json",
        args.repo / "manifests/foldnar_v2/debfold_official_recovery_20261126.json",
        args.repo / "paper/2026-11-26-foldnar-debfold-official-recovery.md",
        provenance_path,
        args.root / "evidence/download_manifest.json",
        args.root / "paper/PMC11094721.fulltext.xml",
    ]
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "PASS_THREE_ROUNDS_WITH_EXPLICIT_OFFICIAL_ASSET_GAPS",
        "full_ready": False,
        "rounds": {
            "download": "PASS: fixed official commit; GitHub/website archive SHA equality; ZIP CRC and CT parsing; paper full text",
            "environment_static": "PASS: Python AST/compile and fail-closed launcher preflight; no GPU execution",
            "cpu_unit_tests": "PASS: CT, dataset identity, complete per-sequence output, pair validation and launcher boundary tests",
        },
        "official_commit": OFFICIAL_COMMIT,
        "dataset_sha256": DATASET_SHA256,
        "n_dataset_sequences": len(dataset),
        "archive_sha256": {name: sha256(source / name) for name in ARCHIVE_NAMES},
        "gpu_inference_performed": False,
        "external_test_used_for_tuning": False,
        "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in core_files],
    }
    output = args.root / "reports/delivery_verification_20261126.json"
    write_json(output, result)
    print(json.dumps({"status": result["status"], "full_ready": False, "report": str(output), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
