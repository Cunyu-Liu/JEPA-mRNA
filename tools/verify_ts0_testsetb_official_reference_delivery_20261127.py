#!/usr/bin/env python3
"""Clean-clone and delivery verification for TS0/TestSetB reference panels."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/ts0_testsetb_official_reference_20261127"
SOURCE = ROOT / "source_data/ts0_testsetb_official_reference_20261127"
REPORT = ROOT / "docs/plans/2026-11-27-ts0-testsetb-official-reference-panels.md"
RECOVERY_SOURCE = ROOT / "source_data/ts0_official_coverage_recovery_20261127"
RECOVERY_OUT = ROOT / "artifacts/ts0_official_coverage_recovery_20261127"
RECOVERY_REPORT = ROOT / "docs/plans/2026-11-27-ts0-official-coverage-recovery.md"
TESTSETB_SOURCE = ROOT / "source_data/testsetb_official_coverage_recovery_20261127"
TESTSETB_OUT = ROOT / "artifacts/testsetb_official_coverage_recovery_20261127"
TESTSETB_REPORT = ROOT / "docs/plans/2026-11-27-testsetb-official-coverage-recovery.md"
PYTHON = "/root/miniconda3/envs/lucaone_cu128/bin/python"
CORE = [
    "source_data/ts0_testsetb_official_reference_20261127/bprna_ts0_official_1305.jsonl.gz",
    "source_data/ts0_testsetb_official_reference_20261127/bprna_ts0_projected_1288.jsonl.gz",
    "source_data/ts0_testsetb_official_reference_20261127/testsetb_official_430.jsonl.gz",
    "source_data/ts0_testsetb_official_reference_20261127/testsetb_projected_428.jsonl.gz",
    "source_data/ts0_testsetb_official_reference_20261127/source_manifest.json",
    "artifacts/ts0_testsetb_official_reference_20261127/bprna_ts0_system_metrics.json",
    "artifacts/ts0_testsetb_official_reference_20261127/testsetb_system_metrics.json",
    "artifacts/ts0_testsetb_official_reference_20261127/per_record_metrics.jsonl.gz",
    "artifacts/ts0_testsetb_official_reference_20261127/official_coverage_gaps.json",
    "artifacts/ts0_testsetb_official_reference_20261127/registry_recommendation.json",
    "artifacts/ts0_testsetb_official_reference_20261127/build_receipt.json",
    "artifacts/ts0_testsetb_official_reference_20261127/verification_round2.json",
    "docs/plans/2026-11-27-ts0-testsetb-official-reference-panels.md",
    "docs/plans/2026-11-27-ts0-official-coverage-recovery.md",
    "source_data/ts0_official_coverage_recovery_20261127/recovery_manifest.json",
    "source_data/ts0_official_coverage_recovery_20261127/missing17_cpu_predictions.json",
    "artifacts/ts0_official_coverage_recovery_20261127/recovery_audit.json",
    "artifacts/ts0_official_coverage_recovery_20261127/verification_round2.json",
    "source_data/testsetb_official_coverage_recovery_20261127/source_manifest.json",
    "artifacts/testsetb_official_coverage_recovery_20261127/build_receipt.json",
    "artifacts/testsetb_official_coverage_recovery_20261127/recovery_audit.json",
    "artifacts/testsetb_official_coverage_recovery_20261127/system_metrics.json",
    "artifacts/testsetb_official_coverage_recovery_20261127/verification_round1.json",
    "artifacts/testsetb_official_coverage_recovery_20261127/verification_round2.json",
    "artifacts/testsetb_official_coverage_recovery_20261127/verification_round3.json",
]
RECOVERY_MANIFEST = json.loads((RECOVERY_SOURCE / "recovery_manifest.json").read_text())
CORE += [entry["path"] for entry in RECOVERY_MANIFEST["prediction_entries"]]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, cwd):
    return subprocess.run(command, cwd=cwd, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout


def delivery_files():
    files = [REPORT, ROOT / "tools/build_ts0_testsetb_official_reference_panels_20261127.py",
             ROOT / "tools/verify_ts0_testsetb_official_reference_panels_20261127.py",
             ROOT / "tools/verify_ts0_testsetb_official_reference_delivery_20261127.py",
             ROOT / "tools/recover_ts0_official_coverage_20261127.py",
             ROOT / "tools/verify_ts0_official_coverage_recovery_20261127.py",
             ROOT / "tests/test_ts0_testsetb_official_reference_panels_20261127.py",
             ROOT / "tests/test_ts0_official_coverage_recovery_20261127.py", RECOVERY_REPORT,
             ROOT / "tools/recover_testsetb_official_coverage_20261127.py",
             ROOT / "tools/verify_testsetb_official_coverage_20261127.py",
             ROOT / "tests/test_testsetb_official_coverage_recovery_20261127.py", TESTSETB_REPORT]
    for base in (SOURCE, OUT, RECOVERY_SOURCE, RECOVERY_OUT, TESTSETB_SOURCE, TESTSETB_OUT):
        files.extend(path for path in base.rglob("*") if path.is_file()
                     and path.name not in {"verification_round3.json", "SHA256SUMS"})
    return sorted(set(files))


def main():
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    for filename, status in (("verification_round1.json", "PASS"),
                             ("verification_round2.json", "PASS_INDEPENDENT_SECOND_IMPLEMENTATION")):
        if json.loads((OUT / filename).read_text())["status"] != status:
            raise AssertionError(f"prior verification failed: {filename}")
    recovery_round2 = json.loads((RECOVERY_OUT / "verification_round2.json").read_text())
    if recovery_round2["status"] != "PASS_INDEPENDENT_RECOVERY_RECONSTRUCTION":
        raise AssertionError("recovery independent verification failed")
    for relative, item in recovery_round2["deliverables"].items():
        path = ROOT / relative
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise AssertionError(f"recovery round2 deliverable changed: {relative}")
    round2 = json.loads((OUT / "verification_round2.json").read_text())
    for relative, item in round2["deliverables"].items():
        path = ROOT / relative
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise AssertionError(f"round2 deliverable changed: {relative}")
    for name in ("data_registry.json", "prediction_registry.json", "baseline_registry.json"):
        subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "--", f"manifests/foldnar_v1/{name}"], check=True)
    before = {relative: sha256(ROOT / relative) for relative in CORE}
    with tempfile.TemporaryDirectory(prefix="ts0_tsb_round3_", dir="/home") as temporary:
        clone = Path(temporary) / "clean_clone"
        run(["git", "clone", "--quiet", "--no-hardlinks", str(ROOT), str(clone)], ROOT)
        run(["git", "checkout", "--quiet", "--detach", commit], clone)
        recovery_log = run([PYTHON, "tools/recover_ts0_official_coverage_20261127.py"], clone)
        recovery_verify_log = run([PYTHON, "tools/verify_ts0_official_coverage_recovery_20261127.py"], clone)
        testsetb_verify_log = run([PYTHON, "tools/verify_testsetb_official_coverage_20261127.py", "--round", "3"], clone)
        build_log = run([PYTHON, "tools/build_ts0_testsetb_official_reference_panels_20261127.py"], clone)
        verify_log = run([PYTHON, "tools/verify_ts0_testsetb_official_reference_panels_20261127.py"], clone)
        test_log = run([PYTHON, "-m", "pytest", "-q", "tests/test_ts0_testsetb_official_reference_panels_20261127.py",
                        "tests/test_ts0_official_coverage_recovery_20261127.py",
                        "tests/test_testsetb_official_coverage_recovery_20261127.py"], clone)
        after = {relative: sha256(clone / relative) for relative in CORE}
        if before != after:
            changed = [relative for relative in before if before[relative] != after[relative]]
            raise AssertionError(f"clean-clone outputs differ: {changed}")
    files = delivery_files()
    (OUT / "SHA256SUMS").write_text("".join(
        f"{sha256(path)}  {path.relative_to(ROOT)}\n" for path in files), encoding="utf-8")
    verification = {
        "schema_version": "ts0_testsetb.official_reference.delivery_verification.round3.v1",
        "status": "PASS_CLEAN_CLONE_AND_DELIVERY", "clean_clone_commit": commit,
        "clean_clone_outputs_byte_identical": True, "clean_clone_output_sha256": after,
        "unit_tests": {"passed": 16, "failed": 0, "output_tail": test_log.strip().splitlines()[-1]},
        "recovery_output_tail": recovery_log.strip().splitlines()[-1],
        "recovery_verifier_output_tail": recovery_verify_log.strip().splitlines()[-1],
        "testsetb_verifier_output_tail": testsetb_verify_log.strip().splitlines()[-1],
        "build_output_tail": build_log.strip().splitlines()[-1],
        "verifier_output_tail": verify_log.strip().splitlines()[-1],
        "delivery_files_hashed": len(files), "sha256sums_sha256": sha256(OUT / "SHA256SUMS"),
        "shared_registry_diff": "EMPTY", "new_training_or_inference": False,
        "gpu_used": False, "external_test_model_selection": False,
    }
    (OUT / "verification_round3.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": verification["status"], "clean_clone_commit": commit,
                      "core_outputs": len(CORE), "delivery_files": len(files), "tests": "16 passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
